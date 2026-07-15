from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app_config import AppConfig, load_config
from asr.events import ASREvent
from asr.punctuation_backends import create_backend
from asr.punctuation_worker import PunctuationWorker
from asr.streaming_engine import StreamingASREngine
from asr.whisper_adapter import WhisperAdapter
from audio.buffer import GrowingAudioBuffer
from audio.capture_supervisor import CaptureSupervisor
from llm.engine import LLMEngine
from llm.lmstudio_client import LMStudioClient
from store.transcript_store import TranscriptStore
from ui.output_sink import OutputSink
from ui.overlay import Overlay
from utils.latency import LatencyTracker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"


def _run_splitter(
    source: "queue.Queue[ASREvent]",
    sinks: "list[queue.Queue[ASREvent]]",
    store: TranscriptStore,
    stop_event: threading.Event,
) -> None:
    """Fan-out: reads one source queue and copies each event to all sinks.

    Commits are also appended to the TranscriptStore *here*, so the store
    (the source of truth for summaries/questions) never loses words to a
    full UI queue — sinks only render.
    """
    while not stop_event.is_set():
        try:
            ev = source.get(timeout=0.1)
        except queue.Empty:
            continue
        if ev.type == "commit":
            store.append(ev.words)
        for q in sinks:
            try:
                q.put_nowait(ev)
            except queue.Full:
                # Sinks are render-only; a dropped event costs at most a
                # slightly stale view until the next one arrives.
                pass
        source.task_done()


class _Pipeline:
    """Holds pipeline pieces created by the background loader thread."""

    def __init__(self) -> None:
        self.engine_thread: threading.Thread | None = None


def _load_asr_pipeline(
    cfg: AppConfig,
    audio_buffer: GrowingAudioBuffer,
    raw_events: "queue.Queue[ASREvent]",
    stop_event: threading.Event,
    latency: LatencyTracker,
    overlay: Overlay,
    pipeline: _Pipeline,
) -> None:
    """Loads Whisper and starts the engine. Runs in a background thread so
    the overlay window is interactive from the first second."""
    try:
        overlay.post_status(f"Загружаю Whisper ({cfg.asr.model}, {cfg.asr.device})…")
        from faster_whisper import WhisperModel  # heavy import, keep it here

        try:
            # Cached models load instantly and survive HF hiccups (renamed
            # repos resolve to a new cache name and re-download; large-file
            # CDN fetches have been seen hanging indefinitely on this box).
            model = WhisperModel(
                cfg.asr.model,
                device=cfg.asr.device,
                compute_type=cfg.asr.compute_type,
                local_files_only=True,
            )
        except Exception:
            overlay.post_status(f"Скачиваю модель {cfg.asr.model} с HuggingFace…")
            model = WhisperModel(
                cfg.asr.model, device=cfg.asr.device, compute_type=cfg.asr.compute_type
            )
        transcribe_fn = WhisperAdapter(
            model,
            language=cfg.asr.language if cfg.asr.language not in ("", "auto") else None,
            sticky_language=cfg.asr.language == "auto",
            sticky_min_probability=cfg.asr.sticky_min_probability,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": cfg.asr.vad_min_silence_ms},
            beam_size=cfg.asr.beam_size,
            condition_on_previous_text=False,
            no_speech_prob_max=cfg.asr.filters.no_speech_prob_max,
            avg_logprob_min=cfg.asr.filters.avg_logprob_min,
            max_word_repeats=cfg.asr.filters.max_word_repeats,
        )
        engine = StreamingASREngine(
            audio_buffer=audio_buffer,
            transcribe_fn=transcribe_fn,
            events=raw_events,
            stop_event=stop_event,
            chunk_interval_sec=cfg.asr.chunk_interval_sec,
            min_audio_sec=cfg.asr.min_audio_sec,
            commit_safety_margin_sec=cfg.asr.commit_safety_margin_sec,
            max_buffer_sec=cfg.asr.max_buffer_sec,
            force_commit_keep_tail_words=cfg.asr.force_commit_keep_tail_words,
            silence_rms_threshold=cfg.asr.filters.silence_rms_threshold,
            empty_decodes_before_trim=cfg.asr.filters.empty_decodes_before_trim,
            silence_keep_tail_sec=cfg.asr.filters.silence_keep_tail_sec,
            latency=latency,
        )
        if stop_event.is_set():  # app closed while the model was loading
            return
        pipeline.engine_thread = threading.Thread(
            target=engine.run, name="engine", daemon=True
        )
        pipeline.engine_thread.start()
        overlay.post_status("")
    except Exception as exc:
        logger.exception("ASR pipeline failed to load")
        overlay.post_status(f"Ошибка загрузки ASR: {exc}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(CONFIG_PATH)

    app = QApplication(sys.argv)

    stop_event = threading.Event()
    latency = LatencyTracker(window=120, keys=("capture_convert", "whisper"))

    # Engine writes here; splitter fans out to per-consumer queues.
    raw_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=800)
    sink_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=400)
    overlay_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=400)

    audio_buffer = GrowingAudioBuffer(
        sample_rate=cfg.audio.target_sample_rate, max_seconds=cfg.asr.buffer_max_seconds
    )
    store = TranscriptStore()

    capture = CaptureSupervisor(
        audio_buffer=audio_buffer,
        device_hint=cfg.audio.device_hint,
        target_sr=cfg.audio.target_sample_rate,
        events=raw_events,
        stop_event=stop_event,
        frames_per_buffer=cfg.audio.frames_per_buffer,
        max_channels=cfg.audio.max_channels,
        restart_backoff_sec=cfg.audio.restart_backoff_sec,
        latency=latency,
    )

    sink = OutputSink(
        events=sink_events,
        stop_event=stop_event,
        verbose_logs=cfg.ui.console_verbose_logs,
    )

    llm = LLMEngine(LMStudioClient(cfg.llm))
    overlay = Overlay(
        events_queue=overlay_events,
        store=store,
        llm=llm,
        max_recent_chars=cfg.ui.max_recent_chars,
    )

    # When the overlay (last window) closes, stop everything.
    app.aboutToQuit.connect(stop_event.set)

    # Forward a fatal stop_event (set by engine/capture) into Qt's event loop.
    shutdown_check = QTimer()
    shutdown_check.timeout.connect(lambda: app.quit() if stop_event.is_set() else None)
    shutdown_check.start(500)

    splitter_thread = threading.Thread(
        target=_run_splitter,
        args=(raw_events, [sink_events, overlay_events], store, stop_event),
        name="splitter",
        daemon=True,
    )
    sink_thread = threading.Thread(target=sink.run, name="output", daemon=True)

    # Window first, heavy models later: capture starts immediately (buffer
    # caps itself), Whisper loads in the background and reports progress
    # into the overlay's status line.
    llm.start()
    splitter_thread.start()
    sink_thread.start()
    capture.start()

    pipeline = _Pipeline()
    loader_thread = threading.Thread(
        target=_load_asr_pipeline,
        args=(cfg, audio_buffer, raw_events, stop_event, latency, overlay, pipeline),
        name="asr-loader",
        daemon=True,
    )
    loader_thread.start()

    punct_thread: threading.Thread | None = None
    punct_backend = create_backend(cfg.punctuation.backend, cfg.punctuation.language)
    if punct_backend is not None:
        worker = PunctuationWorker(
            store=store,
            backend=punct_backend,
            stop_event=stop_event,
            interval_sec=cfg.punctuation.interval_sec,
            context_words=cfg.punctuation.context_words,
        )
        punct_thread = threading.Thread(target=worker.run, name="punctuation", daemon=True)
        punct_thread.start()

    overlay.show()
    ret = app.exec()

    stop_event.set()
    capture.stop()
    loader_thread.join(timeout=2.0)
    if pipeline.engine_thread is not None:
        pipeline.engine_thread.join(timeout=3.0)
    sink_thread.join(timeout=2.0)
    splitter_thread.join(timeout=2.0)
    if punct_thread is not None:
        punct_thread.join(timeout=2.0)
    llm.stop()
    print(latency.report())

    sys.exit(ret)


if __name__ == "__main__":
    main()
