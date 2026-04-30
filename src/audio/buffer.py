from __future__ import annotations

import threading
from typing import Tuple

import numpy as np


class GrowingAudioBuffer:
    """
    Append-only audio buffer with head-trimming by global timestamp.

    Unlike a sliding ring buffer, this buffer grows as audio arrives and is
    explicitly trimmed from the head once we have committed transcription up
    to a given point. The streaming engine relies on this so that:

      - The model always sees a coherent audio segment starting just before
        the last committed word boundary.
      - The buffer never grows unbounded: trimming after each commit keeps
        memory and decode time in check.

    Timestamps are tracked in "global" seconds: the offset_sec field tells
    callers how many seconds of audio have already been trimmed off the head.
    Combined with the local sample index, this lets the engine convert
    Whisper's local word timestamps into absolute time on the original stream.

    A hard cap (max_seconds) prevents pathological unbounded growth if no
    commits happen for a long time. When the cap is hit, oldest samples are
    dropped and offset_sec is advanced accordingly.
    """

    def __init__(self, sample_rate: int, max_seconds: float = 60.0) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

        self.sample_rate = int(sample_rate)
        self._max_samples = int(float(max_seconds) * self.sample_rate)
        self._samples: np.ndarray = np.empty(0, dtype=np.float32)
        self._offset_sec: float = 0.0
        self._lock = threading.Lock()

    # -- writers --------------------------------------------------------

    def write(self, data: np.ndarray) -> None:
        if data is None or len(data) == 0:
            return
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)

        with self._lock:
            if len(self._samples) == 0:
                self._samples = np.array(data, dtype=np.float32, copy=True)
            else:
                self._samples = np.concatenate((self._samples, data))

            overflow = len(self._samples) - self._max_samples
            if overflow > 0:
                self._samples = self._samples[overflow:]
                self._offset_sec += overflow / self.sample_rate

    # -- readers --------------------------------------------------------

    def snapshot(self) -> Tuple[np.ndarray, float]:
        """Return a copy of the current audio and the head offset in seconds."""
        with self._lock:
            return self._samples.copy(), self._offset_sec

    def duration_sec(self) -> float:
        with self._lock:
            return len(self._samples) / self.sample_rate

    def end_time_sec(self) -> float:
        """Global timestamp of the most recent sample."""
        with self._lock:
            return self._offset_sec + len(self._samples) / self.sample_rate

    # -- trimming -------------------------------------------------------

    def trim_until(self, global_sec: float) -> None:
        """
        Drop samples whose global end-time is before `global_sec`.

        Idempotent: trimming to a point before the current head is a no-op.
        Trimming past the current end empties the buffer.
        """
        with self._lock:
            drop_local_sec = global_sec - self._offset_sec
            if drop_local_sec <= 0:
                return

            drop_samples = int(drop_local_sec * self.sample_rate)
            if drop_samples >= len(self._samples):
                self._offset_sec += len(self._samples) / self.sample_rate
                self._samples = np.empty(0, dtype=np.float32)
                return

            self._samples = self._samples[drop_samples:]
            self._offset_sec += drop_samples / self.sample_rate
