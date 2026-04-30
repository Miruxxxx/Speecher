# latency.py
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple


def _percentile(sorted_vals: List[float], p: float) -> float:
    """
    p in [0..100]. sorted_vals must be sorted ascending.
    Nearest-rank percentile (stable and cheap for small windows).
    """
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    k = max(0, min(k, len(sorted_vals) - 1))
    return sorted_vals[k]


class LatencyTracker:
    """
    Thread-safe rolling window latency tracker (seconds).
    Keys are dynamic; common keys include:
    convert, whisper_fast, text_fast, whisper_refine, text_refine, output.
    """

    def __init__(
        self,
        window: int = 120,
        keys: Iterable[str] = (
            "convert",
            "whisper_fast",
            "text_fast",
            "whisper_refine",
            "text_refine",
            "output",
        ),
    ):
        self.window = int(window)
        self._lock = threading.Lock()
        self._data: Dict[str, Deque[float]] = {k: deque(maxlen=self.window) for k in keys}

    def mark(self, key: str, seconds: float) -> None:
        if seconds is None:
            return
        try:
            v = float(seconds)
        except (TypeError, ValueError):
            return
        if v < 0:
            return

        with self._lock:
            if key not in self._data:
                self._data[key] = deque(maxlen=self.window)
            self._data[key].append(v)

    def _snapshot(self, key: str) -> List[float]:
        with self._lock:
            vals = list(self._data.get(key, ()))
        vals.sort()
        return vals

    def stats_ms(self, key: str) -> Tuple[float, float, float, int]:
        """
        Returns: (mean_ms, p50_ms, p95_ms, n)
        """
        vals = self._snapshot(key)
        n = len(vals)
        if n == 0:
            return 0.0, 0.0, 0.0, 0

        mean = sum(vals) / n
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        return mean * 1000.0, p50 * 1000.0, p95 * 1000.0, n

    def report(self, keys: Iterable[str] | None = None) -> str:
        if keys is None:
            with self._lock:
                keys = list(self._data.keys())
        parts: List[str] = []
        for key in keys:
            mean, p50, p95, n = self.stats_ms(str(key))
            parts.append(f"{key}={mean:.2f}ms (p50={p50:.2f}, p95={p95:.2f}, n={n})")
        return "[LAT] " + " | ".join(parts)
