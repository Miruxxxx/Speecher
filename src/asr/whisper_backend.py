from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

from app_config import AsrConfig
from asr.events import ASREvent
from asr.streaming_engine import StreamingASREngine
from asr.whisper_adapter import WhisperAdapter
from audio.buffer import GrowingAudioBuffer

logger = logging.getLogger(__name__)


class WhisperBackend:
    """
    faster-whisper backend: pull-style, LocalAgreement-2 stabilization.

    Thin wrapper around the existing pieces — `WhisperAdapter` decodes,
    `StreamingASREngine` owns the loop. Nothing about their behaviour changes
    here; this class only gives them the `ASRBackend` shape so a second engine
    can sit next to them.
    """

    name = "whisper"

    def __init__(
        self,
        *,
        cfg: AsrConfig,
        audio_buffer: GrowingAudioBuffer,
        latency=None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg = cfg
        self._buffer = audio_buffer
        self._latency = latency
        self._on_status = on_status
        self._transcribe: Optional[WhisperAdapter] = None

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        """Instantiate the model. Runs on the asr-loader thread, not the UI one."""
        cfg = self._cfg
        self._status(f"Загружаю Whisper ({cfg.model}, {cfg.device})…")
        from faster_whisper import WhisperModel  # heavy import, keep it here

        try:
            # Cached models load instantly and survive HF hiccups (renamed
            # repos resolve to a new cache name and re-download; large-file
            # CDN fetches have been seen hanging indefinitely on this box).
            model = WhisperModel(
                cfg.model,
                device=cfg.device,
                compute_type=cfg.compute_type,
                local_files_only=True,
            )
        except Exception:
            self._status(f"Скачиваю модель {cfg.model} с HuggingFace…")
            model = WhisperModel(
                cfg.model, device=cfg.device, compute_type=cfg.compute_type
            )

        self._transcribe = WhisperAdapter(
            model,
            language=cfg.language if cfg.language not in ("", "auto") else None,
            sticky_language=cfg.language == "auto",
            sticky_min_probability=cfg.sticky_min_probability,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": cfg.vad_min_silence_ms},
            beam_size=cfg.beam_size,
            condition_on_previous_text=False,
            no_speech_prob_max=cfg.filters.no_speech_prob_max,
            avg_logprob_min=cfg.filters.avg_logprob_min,
            max_word_repeats=cfg.filters.max_word_repeats,
        )

    # -- loop ------------------------------------------------------------

    def run(
        self,
        *,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
    ) -> None:
        if self._transcribe is None:
            raise RuntimeError("WhisperBackend.run() called before load()")
        cfg = self._cfg
        try:
            engine = StreamingASREngine(
                audio_buffer=self._buffer,
                transcribe_fn=self._transcribe,
                events=events,
                stop_event=stop_event,
                chunk_interval_sec=cfg.chunk_interval_sec,
                min_audio_sec=cfg.min_audio_sec,
                commit_safety_margin_sec=cfg.commit_safety_margin_sec,
                max_buffer_sec=cfg.max_buffer_sec,
                force_commit_keep_tail_words=cfg.force_commit_keep_tail_words,
                silence_rms_threshold=cfg.filters.silence_rms_threshold,
                empty_decodes_before_trim=cfg.filters.empty_decodes_before_trim,
                silence_keep_tail_sec=cfg.filters.silence_keep_tail_sec,
                latency=self._latency,
            )
        except ValueError as exc:
            # Bad tunables in config.toml. Report it the way the engine reports
            # its own failures instead of letting the thread die silently.
            logger.exception("invalid ASR settings")
            events.put(
                ASREvent(type="fatal", source="engine", text=f"invalid ASR settings: {exc}")
            )
            stop_event.set()
            return
        engine.run()

    # -- helpers ---------------------------------------------------------

    def _status(self, msg: str) -> None:
        if self._on_status is not None:
            self._on_status(msg)
