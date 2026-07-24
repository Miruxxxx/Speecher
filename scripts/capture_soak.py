"""Soak test for the capture path: does it survive, and does it keep up?

The failure this is looking for is the historical one from
docs/CODE_REVIEW_2026-07-15.md B1: a capture process that dies with 0xc0000005
after 2-8 minutes of an *idle* loopback stream. So the interesting run is a
long one with the system mostly silent.

It drives the production supervisor with the real config and only replaces the
audio sink with a meter, so nothing grows in memory over half an hour.

Usage (repo root, project venv):

    .venv\\Scripts\\python scripts\\capture_soak.py --seconds 900
    .venv\\Scripts\\python scripts\\capture_soak.py --seconds 900 --source mic

Columns per report line:
    wall     seconds since start
    audio    seconds of audio received (stays flat while nothing plays: WASAPI
             loopback delivers frames only while the endpoint renders)
    rate     audio/wall over the last interval; ~1.0 while sound is playing
    rms      loudness of the last interval, 0.0 = digital silence
    restarts how many times the supervisor had to restart the child
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Make the project's `src` packages importable the same way `python -m src` does.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app_config import load_config  # noqa: E402
from audio.rust_capture import RustCaptureSupervisor, resolve_binary  # noqa: E402


class Meter:
    """Stands in for GrowingAudioBuffer: counts instead of keeping audio."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self.total_samples = 0
        self._window_samples = 0
        self._window_sum_sq = 0.0
        self.peak = 0.0

    def write(self, data: np.ndarray) -> None:
        with self._lock:
            self.total_samples += len(data)
            self._window_samples += len(data)
            self._window_sum_sq += float(np.dot(data, data))
            if len(data):
                self.peak = max(self.peak, float(np.max(np.abs(data))))

    def take_window(self) -> tuple[int, float]:
        with self._lock:
            samples, sum_sq = self._window_samples, self._window_sum_sq
            self._window_samples, self._window_sum_sq = 0, 0.0
        rms = math.sqrt(sum_sq / samples) if samples else 0.0
        return samples, rms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("loopback", "mic"), default=None,
                        help="override audio.source from config/config.toml")
    parser.add_argument("--seconds", type=float, default=900.0)
    parser.add_argument("--report-every", type=float, default=30.0)
    args = parser.parse_args()

    cfg = load_config(ROOT / "config" / "config.toml")
    if args.source:
        cfg.audio.source = args.source

    meter = Meter(cfg.audio.target_sample_rate)
    events: "queue.Queue" = queue.Queue(maxsize=1000)
    stop_event = threading.Event()

    supervisor = RustCaptureSupervisor(
        audio_buffer=meter,
        binary_path=resolve_binary(cfg.audio.binary),
        device_hint=cfg.audio.device_hint,
        target_sr=cfg.audio.target_sample_rate,
        events=events,
        stop_event=stop_event,
        source=cfg.audio.source,
        restart_backoff_sec=cfg.audio.restart_backoff_sec,
    )
    print(
        f"device_hint='{cfg.audio.device_hint}' "
        f"source={cfg.audio.source} target_sr={cfg.audio.target_sample_rate} "
        f"-> {args.seconds:.0f}s soak, report every {args.report_every:.0f}s",
        flush=True,
    )

    restarts = 0
    started = time.monotonic()
    last_report = started
    supervisor.start()
    try:
        while time.monotonic() - started < args.seconds and not stop_event.is_set():
            try:
                ev = events.get(timeout=0.2)
            except queue.Empty:
                ev = None
            if ev is not None:
                if "died" in (ev.text or ""):
                    restarts += 1
                print(f"  [{ev.type}] {ev.text}", flush=True)

            now = time.monotonic()
            if now - last_report >= args.report_every:
                samples, rms = meter.take_window()
                interval = now - last_report
                print(
                    f"wall={now - started:7.1f}s  audio={meter.total_samples / meter.sample_rate:7.1f}s  "
                    f"rate={samples / meter.sample_rate / interval:4.2f}  rms={rms:.5f}  "
                    f"restarts={restarts}",
                    flush=True,
                )
                last_report = now
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        supervisor.stop()
        stop_event.set()

    wall = time.monotonic() - started
    audio_sec = meter.total_samples / meter.sample_rate
    print(
        f"\nDONE  wall={wall:.1f}s  audio={audio_sec:.1f}s  "
        f"peak={meter.peak:.4f}  restarts={restarts}",
        flush=True,
    )
    if restarts:
        print("FAIL: the capture child died at least once", flush=True)
        return 1
    print("OK: the capture child survived the whole run", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
