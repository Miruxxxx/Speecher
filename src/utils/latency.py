from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple


def _percentile(sorted_vals: List[float], p: float) -> float:
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
    """Thread-safe rolling-window latency tracker. Values are seconds."""

    def __init__(self, window: int = 120, keys: Iterable[str] = ()) -> None:
        self.window = int(window)
        self._lock = threading.Lock()
        self._data: Dict[str, Deque[float]] = {k: deque(maxlen=self.window) for k in keys}

    def mark(self, key: str, seconds: float) -> None:
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

    def stats_ms(self, key: str) -> Tuple[float, float, float, int]:
        with self._lock:
            vals = sorted(self._data.get(key, ()))
        n = len(vals)
        if n == 0:
            return 0.0, 0.0, 0.0, 0
        mean = sum(vals) / n
        return mean * 1000.0, _percentile(vals, 50) * 1000.0, _percentile(vals, 95) * 1000.0, n

    def report(self, keys: Iterable[str] | None = None) -> str:
        with self._lock:
            ks = list(keys) if keys is not None else list(self._data.keys())
        parts: List[str] = []
        for k in ks:
            mean, p50, p95, n = self.stats_ms(k)
            parts.append(f"{k}={mean:.1f}ms (p50={p50:.1f}, p95={p95:.1f}, n={n})")
        return "[LAT] " + " | ".join(parts)
