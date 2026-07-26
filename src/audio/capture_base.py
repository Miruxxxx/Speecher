from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

from asr.events import ASREvent

# What the UI's capture indicator can show (design system 2.1 / 6.5): a wave
# when audio is flowing, a flat line in silence, a struck-through line on error.
# Silence is a normal mode, not a failure: WASAPI loopback delivers nothing at
# all while no application renders audio.
CAPTURE_LIVE = "live"
CAPTURE_SILENT = "silent"
CAPTURE_ERROR = "error"


class CaptureSupervisorBase:
    """Restart policy and event plumbing shared by both capture backends.

    Whatever runs the capture -- the PortAudio child process or the native
    WASAPI binary -- the parent side is the same: keep the child alive, restart
    it with backoff, give up when it dies instantly three times in a row, and
    feed every mono chunk into the GrowingAudioBuffer (plus the optional
    frame_sink of a push-style ASR backend).

    Subclasses implement `_run_one_child()`: start one child, pump it until it
    dies, return a short reason for the log.

    Restart policy: unconditional with backoff, except when the child fails
    within `_FAST_FAIL_SEC` of starting `_FAST_FAIL_LIMIT` times in a row --
    that means a persistent setup problem (no loopback device, open error),
    which is reported as fatal and stops the app.
    """

    _FAST_FAIL_SEC = 5.0
    _FAST_FAIL_LIMIT = 3

    def __init__(
        self,
        *,
        audio_buffer,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
        restart_backoff_sec: float = 2.0,
        latency=None,
        frame_sink: "Optional[queue.Queue[np.ndarray]]" = None,
        sound_rms_threshold: float = 1e-4,
        sound_hold_sec: float = 0.8,
    ) -> None:
        self._buffer = audio_buffer
        self._events = events
        self._stop_event = stop_event
        self._restart_backoff_sec = float(restart_backoff_sec)
        self._latency = latency
        self._frame_sink = frame_sink
        self._frames_dropped = 0

        self._thread: Optional[threading.Thread] = None
        self._local_stop = threading.Event()

        # Capture state for the UI indicator. Written by the supervisor thread,
        # read by the Qt thread; a str/float attribute swap is atomic under the
        # GIL and a tick of staleness costs nothing at a 50 ms poll, so this
        # deliberately takes no lock (locking here would sit in the audio path).
        self._sound_rms_threshold = float(sound_rms_threshold)
        self._sound_hold_sec = float(sound_hold_sec)
        self._last_sound_at = 0.0        # monotonic; 0 = nothing heard yet
        self._child_phase = "starting"   # starting | running | restarting | fatal
        self._last_reason = ""

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._local_stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._local_stop.set()
        self._signal_child_stop()
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
                self._child_phase = "running"
                clean_info = self._run_one_child()
                if self._stopping():
                    break
                self._child_phase = "restarting"
                self._last_reason = str(clean_info)

                lived_sec = time.monotonic() - started_at
                if lived_sec < self._FAST_FAIL_SEC:
                    fast_fails += 1
                else:
                    fast_fails = 0
                if fast_fails >= self._FAST_FAIL_LIMIT:
                    self._child_phase = "fatal"
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

    # -- subclass hooks ---------------------------------------------------

    def _run_one_child(self) -> str:
        raise NotImplementedError

    def _signal_child_stop(self) -> None:
        """Ask the child to exit (best effort) before it gets terminated."""

    def _terminate_child(self) -> None:
        """Kill the current child, if any. Must be safe to call twice."""

    # -- helpers --------------------------------------------------------

    def _write_audio(self, audio: np.ndarray) -> None:
        """Publish one mono chunk at the target rate."""
        if not len(audio):
            return
        # One pass for the indicator before the chunk leaves this thread. The
        # threshold is the same one the ASR silence gate uses, so "звук идёт" in
        # the UI means exactly "loud enough that the engine would decode it".
        rms = float(np.sqrt(np.dot(audio, audio) / len(audio)))
        if rms > self._sound_rms_threshold:
            self._last_sound_at = time.monotonic()
        self._buffer.write(audio)
        if self._frame_sink is not None:
            self._push_frame(audio)

    def capture_state(self) -> tuple[str, str]:
        """(state, reason) for the UI indicator. Safe to call from any thread."""
        phase = self._child_phase
        if phase in ("restarting", "fatal"):
            return CAPTURE_ERROR, self._last_reason
        if phase == "starting":
            return CAPTURE_SILENT, ""
        last = self._last_sound_at
        if last and time.monotonic() - last <= self._sound_hold_sec:
            return CAPTURE_LIVE, ""
        return CAPTURE_SILENT, ""

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
