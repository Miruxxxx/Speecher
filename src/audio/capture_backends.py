from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app_config import AppConfig
from asr.events import ASREvent
from audio.capture_base import CaptureSupervisorBase

logger = logging.getLogger(__name__)

# src/audio/capture_backends.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUST_BINARY = (
    PROJECT_ROOT / "native" / "audio_capture" / "target" / "release" / "audio_capture.exe"
)


def resolve_rust_binary(configured: str = "") -> Path:
    """Where the native capture binary is expected to live.

    An empty config value means "the release build in the tree"; a relative
    path is resolved against the repository root, not the current directory,
    so the app behaves the same however it was started.
    """
    configured = (configured or "").strip()
    if not configured:
        return DEFAULT_RUST_BINARY
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_capture_supervisor(
    cfg: AppConfig,
    *,
    audio_buffer,
    events: "queue.Queue[ASREvent]",
    stop_event: threading.Event,
    latency=None,
    frame_sink: "Optional[queue.Queue[np.ndarray]]" = None,
) -> CaptureSupervisorBase:
    """Pick a capture backend by `audio.backend`.

    "rust" needs a binary that someone has built (`cargo build --release` in
    native/audio_capture); a missing one falls back to PortAudio with a warning
    rather than leaving the app deaf.
    """
    audio = cfg.audio
    name = (audio.backend or "").strip().lower()

    if name == "rust":
        binary = resolve_rust_binary(audio.rust_binary)
        if not binary.is_file():
            logger.warning(
                "audio.backend='rust' but %s is missing "
                "(build it: cargo build --release --manifest-path native/audio_capture/Cargo.toml); "
                "using 'pyaudio'",
                binary,
            )
        else:
            from audio.rust_capture import RustCaptureSupervisor

            logger.info("capture backend: rust (%s)", binary)
            return RustCaptureSupervisor(
                audio_buffer=audio_buffer,
                binary_path=binary,
                device_hint=audio.device_hint,
                target_sr=audio.target_sample_rate,
                events=events,
                stop_event=stop_event,
                source=audio.source,
                restart_backoff_sec=audio.restart_backoff_sec,
                latency=latency,
                frame_sink=frame_sink,
            )
    elif name != "pyaudio":
        logger.warning("audio.backend=%r is not available; using 'pyaudio'", audio.backend)

    from audio.capture_supervisor import CaptureSupervisor

    logger.info("capture backend: pyaudio")
    return CaptureSupervisor(
        audio_buffer=audio_buffer,
        device_hint=audio.device_hint,
        target_sr=audio.target_sample_rate,
        events=events,
        stop_event=stop_event,
        frames_per_buffer=audio.frames_per_buffer,
        max_channels=audio.max_channels,
        restart_backoff_sec=audio.restart_backoff_sec,
        latency=latency,
        frame_sink=frame_sink,
    )
