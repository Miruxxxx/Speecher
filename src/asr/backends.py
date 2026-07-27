from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from app_config import AppConfig
from asr.events import ASREvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EngineProfile:
    """Everything that follows from `asr.engine`, resolved once.

    The choice used to be re-derived wherever it was needed — the frame queue,
    two fields of the session header, whether to start the punctuation worker,
    which language to hand the translator — each with its own
    `cfg.asr.engine == "nemotron"`. Five copies of one decision is how they
    drift: the capture supervisor kept filling a frame queue for a backend that
    had failed to load, because the fallback path knew about the engine but not
    about the queue.
    """

    name: str
    model: str
    language: str
    # Frames are pushed to this backend as they arrive rather than pulled from
    # GrowingAudioBuffer, so it needs a queue and whisper must not get one.
    push_style: bool
    # The model emits its own casing and punctuation; PunctuationWorker would
    # only fight it.
    punctuates_itself: bool

    @classmethod
    def resolve(cls, cfg: AppConfig) -> "EngineProfile":
        """The profile of the *configured* engine."""
        return cls.for_name(cfg, cfg.asr.engine)

    @classmethod
    def for_name(cls, cfg: AppConfig, name: str) -> "EngineProfile":
        """The profile of a named engine, whatever the config asked for.

        Needed because the engine that ends up running is not always the
        configured one: a backend that fails to load falls back to whisper, and
        the session header must then name whisper's model, not nemotron's.
        """
        if (name or "").strip().lower() == "nemotron":
            return cls(
                name="nemotron",
                model=cfg.asr.nemotron.model,
                language=cfg.asr.nemotron.language,
                push_style=True,
                punctuates_itself=True,
            )
        return cls(
            name="whisper",
            model=cfg.asr.model,
            language=cfg.asr.language,
            push_style=False,
            punctuates_itself=False,
        )

# Status strings go to the overlay's status line (see Overlay.post_status).
StatusFn = Callable[[str], None]


class ASRBackend(Protocol):
    """
    One ASR engine, loop included.

    The recognition loop lives *inside* the backend on purpose: Whisper pulls
    snapshots of a growing buffer on its own schedule, while a cache-aware
    streaming model is fed frames as they arrive. Those are different models
    of time; a shared "engine" above them would fit neither.

    `load()` does the expensive work (model download/instantiation) and is
    called from the asr-loader thread. `run()` is the thread body: it must
    return when `stop_event` is set and emit everything through `events`.
    """

    name: str

    def load(self) -> None: ...

    def run(
        self,
        *,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
    ) -> None: ...


def create_asr_backend(
    cfg: AppConfig,
    *,
    audio_buffer,
    latency=None,
    on_status: Optional[StatusFn] = None,
    frame_sink=None,
    engine: Optional[str] = None,
) -> ASRBackend:
    """Pick a backend by `asr.engine`. Unknown names fall back to Whisper.

    `engine` overrides the config, which is how main.py retries with Whisper
    after another backend failed to load.
    """
    from asr.whisper_backend import WhisperBackend  # local: keeps imports flat

    name = (engine if engine is not None else cfg.asr.engine or "").strip().lower()

    if name == "nemotron":
        if frame_sink is None:
            # The supervisor only forwards frames when it was given the queue;
            # without it this backend would sit on silence forever.
            logger.warning("asr.engine='nemotron' needs a frame_sink; using 'whisper'")
        else:
            from asr.nemotron_backend import NemotronBackend

            logger.info("ASR backend: nemotron")
            return NemotronBackend(
                cfg=cfg.asr.nemotron,
                frame_sink=frame_sink,
                latency=latency,
                on_status=on_status,
            )
    elif name != "whisper":
        logger.warning("asr.engine=%r is not available; using 'whisper'", cfg.asr.engine)

    logger.info("ASR backend: whisper")
    return WhisperBackend(
        cfg=cfg.asr, audio_buffer=audio_buffer, latency=latency, on_status=on_status
    )
