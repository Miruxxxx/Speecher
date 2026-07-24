from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from typing import Optional

import numpy as np
import soxr

from asr.events import ASREvent
from audio.capture_base import CaptureSupervisorBase
from audio.capture_process import run_capture


class CaptureSupervisor(CaptureSupervisorBase):
    """
    Runs the PortAudio loopback capture in a child process and keeps it alive.

    pyaudiowpatch corrupts the heap of its host process after minutes of an
    idle loopback stream (docs/CODE_REVIEW_2026-07-15.md B1). Isolating it in
    a child process turns that crash from "the app silently dies" into "the
    supervisor logs a restart". The child ships raw interleaved float32 frames;
    this thread downmixes to mono, resamples to `target_sr` with a stateful
    soxr stream (no per-chunk filter edge artifacts) and writes into the
    GrowingAudioBuffer.

    The native alternative (audio/rust_capture.py) does that conversion in the
    child and does not have the crash at all; this backend stays as the
    fallback that needs no build step.
    """

    def __init__(
        self,
        *,
        audio_buffer,
        device_hint: str,
        target_sr: int,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
        frames_per_buffer: int = 2048,
        max_channels: int = 2,
        restart_backoff_sec: float = 2.0,
        latency=None,
        frame_sink: "Optional[queue.Queue[np.ndarray]]" = None,
    ) -> None:
        super().__init__(
            audio_buffer=audio_buffer,
            events=events,
            stop_event=stop_event,
            restart_backoff_sec=restart_backoff_sec,
            latency=latency,
            frame_sink=frame_sink,
        )
        self._device_hint = device_hint
        self._target_sr = int(target_sr)
        self._frames_per_buffer = int(frames_per_buffer)
        self._max_channels = int(max_channels)

        self._ctx = mp.get_context("spawn")
        self._child: Optional[mp.process.BaseProcess] = None
        self._mp_stop = None

    # -- one child --------------------------------------------------------

    def _run_one_child(self) -> str:
        """Spawn one child and pump its queue until it dies. Returns death reason."""
        self._mp_stop = self._ctx.Event()
        data_queue = self._ctx.Queue(maxsize=256)
        child = self._ctx.Process(
            target=run_capture,
            args=(
                data_queue,
                self._mp_stop,
                self._device_hint,
                self._frames_per_buffer,
                self._max_channels,
            ),
            name="capture-child",
            daemon=True,
        )
        child.start()
        self._child = child
        self._emit_log(f"capture child started (pid={child.pid})")

        channels = 1
        resampler: Optional[soxr.ResampleStream] = None
        reason = "unknown"
        try:
            while not self._stopping():
                try:
                    kind, payload = data_queue.get(timeout=0.5)
                except queue.Empty:
                    if not child.is_alive():
                        reason = f"exitcode={child.exitcode}"
                        break
                    continue

                if kind == "info":
                    channels = int(payload["channels"])
                    src_sr = int(payload["sample_rate"])
                    if src_sr != self._target_sr:
                        resampler = soxr.ResampleStream(
                            src_sr, self._target_sr, 1, dtype="float32"
                        )
                    else:
                        resampler = None
                    self._emit_log(
                        f"device='{payload['name']}' idx={payload['index']} "
                        f"ch={channels} sr={src_sr}"
                    )
                elif kind == "data":
                    t0 = time.perf_counter()
                    audio = np.frombuffer(payload, dtype=np.float32)
                    if channels > 1:
                        audio = audio.reshape(-1, channels).mean(axis=1)
                    if resampler is not None:
                        audio = resampler.resample_chunk(audio)
                    if len(audio):
                        self._write_audio(audio.astype(np.float32, copy=False))
                    if self._latency is not None:
                        self._latency.mark("capture_convert", time.perf_counter() - t0)
                elif kind == "fatal":
                    reason = f"fatal: {payload}"
                    break
        finally:
            self._terminate_child()
        return reason

    # -- hooks ------------------------------------------------------------

    def _signal_child_stop(self) -> None:
        if self._mp_stop is not None:
            try:
                self._mp_stop.set()
            except Exception:
                pass

    def _terminate_child(self) -> None:
        child = self._child
        self._child = None
        if child is None:
            return
        try:
            if child.is_alive():
                child.terminate()
            child.join(timeout=2.0)
        except Exception:
            pass
