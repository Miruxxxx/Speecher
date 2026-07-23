from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional, Protocol

from app_config import AppConfig
from asr.events import ASREvent

logger = logging.getLogger(__name__)

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
