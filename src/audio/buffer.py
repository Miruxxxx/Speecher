from __future__ import annotations

import threading
from typing import Final

import numpy as np


class RingBuffer:
    """
    Thread-safe float32 ring buffer for audio samples.

    Important: tracks how many samples were actually written so readers can
    reliably detect "not enough audio yet" instead of accidentally consuming
    leading zeros.
    """

    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        self.sample_rate: Final[int] = int(sample_rate)
        self.max_samples: Final[int] = int(float(max_seconds) * self.sample_rate)
        self._buffer = np.zeros(self.max_samples, dtype=np.float32)
        self._index = 0  # next write position
        self._filled = 0  # number of valid samples in buffer (<= max_samples)
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        if data is None:
            return
        n = int(len(data))
        if n <= 0:
            return

        with self._lock:
            if n >= self.max_samples:
                self._buffer[:] = data[-self.max_samples :].astype(np.float32, copy=False)
                self._index = 0
                self._filled = self.max_samples
                return

            end = self._index + n
            if end < self.max_samples:
                self._buffer[self._index : end] = data
            else:
                first = self.max_samples - self._index
                self._buffer[self._index :] = data[:first]
                self._buffer[: n - first] = data[first:]
            self._index = (self._index + n) % self.max_samples
            self._filled = min(self.max_samples, self._filled + n)

    def read_last(self, seconds: float) -> np.ndarray:
        n = int(float(seconds) * self.sample_rate)
        if n <= 0:
            return np.empty(0, dtype=np.float32)
        if n > self.max_samples:
            n = self.max_samples

        with self._lock:
            available = min(n, self._filled)
            if available <= 0:
                return np.empty(0, dtype=np.float32)

            start = (self._index - available) % self.max_samples
            if start < self._index:
                return self._buffer[start : self._index].copy()
            return np.concatenate((self._buffer[start:], self._buffer[: self._index]))
