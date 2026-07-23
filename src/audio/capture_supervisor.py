from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from typing import Optional

import numpy as np
import soxr

from asr.events import ASREvent
from audio.capture_process import run_capture


class CaptureSupervisor:
    """
    Runs the PortAudio loopback capture in a child process and keeps it alive.

    pyaudiowpatch corrupts the heap of its host process after minutes of an
    idle loopback stream (docs/CODE_REVIEW_2026-07-15.md §B1). Isolating it in
    a child process turns that crash from "the app silently dies" into "the
    supervisor logs a restart". The child ships raw interleaved float32 frames;
    this thread downmixes to mono, resamples to `target_sr` with a stateful
    soxr stream (no per-chunk filter edge artifacts) and writes into the
    GrowingAudioBuffer.

    Restart policy: unconditional with backoff, except when the child fails
    within `_FAST_FAIL_SEC` of starting `_FAST_FAIL_LIMIT` times in a row —
    that means a persistent setup problem (no loopback device, open error),
    which is reported as fatal and stops the app.
    """

    _FAST_FAIL_SEC = 5.0
    _FAST_FAIL_LIMIT = 3

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
        self._buffer = audio_buffer
        self._device_hint = device_hint
        self._target_sr = int(target_sr)
        self._events = events
        self._stop_event = stop_event
        self._frames_per_buffer = int(frames_per_buffer)
        self._max_channels = int(max_channels)
        self._restart_backoff_sec = float(restart_backoff_sec)
        self._latency = latency
        self._frame_sink = frame_sink
        self._frames_dropped = 0

        self._ctx = mp.get_context("spawn")
        self._thread: Optional[threading.Thread] = None
        self._local_stop = threading.Event()
        self._child: Optional[mp.process.BaseProcess] = None
        self._mp_stop = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._local_stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._local_stop.set()
        if self._mp_stop is not None:
            try:
                self._mp_stop.set()
            except Exception:
                pass
        self._terminate_child()
        t = self._thread
        if t is not None:
            t.join(timeout=3.0)
        self._thread = None

    # -- supervisor loop --------------------------------------------------

    def _run(self) -> None:
        self._emit_log("capture supervisor started")
        fast_fails = 0
        try:
            while not self._stopping():
                started_at = time.monotonic()
                clean_info = self._run_one_child()
                if self._stopping():
                    break

                lived_sec = time.monotonic() - started_at
                if lived_sec < self._FAST_FAIL_SEC:
                    fast_fails += 1
                else:
                    fast_fails = 0
                if fast_fails >= self._FAST_FAIL_LIMIT:
                    self._emit_fatal(
                        f"capture child failed {fast_fails} times within "
                        f"{self._FAST_FAIL_SEC:.0f}s of start (last: {clean_info}); giving up"
                    )
                    self._stop_event.set()
                    return

                self._emit_log(
                    f"capture child died ({clean_info}); restarting in "
                    f"{self._restart_backoff_sec:.1f}s"
                )
                self._local_stop.wait(self._restart_backoff_sec)
        finally:
            self._terminate_child()
            self._emit_log("capture supervisor stopped")

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
                        audio = audio.astype(np.float32, copy=False)
                        self._buffer.write(audio)
                        if self._frame_sink is not None:
                            self._push_frame(audio)
                    if self._latency is not None:
                        self._latency.mark("capture_convert", time.perf_counter() - t0)
                elif kind == "fatal":
                    reason = f"fatal: {payload}"
                    break
        finally:
            self._terminate_child()
        return reason

    # -- helpers --------------------------------------------------------

    def _push_frame(self, audio: np.ndarray) -> None:
        """Hand the mono/resampled chunk to a push-style ASR backend.

        Best effort by design: blocking here would stall the pump that feeds
        the Whisper buffer, so a backend that can't keep up loses frames and
        gets counted instead. The GrowingAudioBuffer path is unaffected.
        """
        try:
            self._frame_sink.put_nowait(audio)
        except queue.Full:
            self._frames_dropped += 1
            if self._frames_dropped % 100 == 1:
                self._emit_log(
                    f"frame_sink full; dropped {self._frames_dropped} chunks so far"
                )

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

    def _stopping(self) -> bool:
        return self._local_stop.is_set() or self._stop_event.is_set()

    def _emit_log(self, msg: str) -> None:
        try:
            self._events.put_nowait(ASREvent(type="log", source="capture", text=str(msg)))
        except queue.Full:
            pass

    def _emit_fatal(self, msg: str) -> None:
        try:
            self._events.put(ASREvent(type="fatal", source="capture", text=str(msg)), timeout=0.5)
        except queue.Full:
            pass
