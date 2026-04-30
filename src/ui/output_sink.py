from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Optional

from asr.events import ASREvent
from transcript_engine import TranscriptEngine, TranscriptUpdate
from utils.latency import LatencyTracker


class ANSI:
    CLEAR_LINE = "\033[2K"
    COL0 = "\r"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _isatty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class OutputSink:
    """
    OutputSink coordinates ASR events, transcript stabilization, and rendering.
    """

    def __init__(
        self,
        *,
        events: queue.Queue[ASREvent],
        stop_event: threading.Event,
        latency: Optional[LatencyTracker] = None,
        final_stability_n: int = 2,
        final_stability_ratio: float = 0.82,
        final_stability_window_words: int = 40,
        min_final_overlap_words: int = 3,
        fuzzy_overlap_min_ratio: float = 0.78,
        max_backtrack_words: int = 48,
        tail_anchor_words: int = 140,
        live_indicator: str = "⚡",
        enable_colors: bool = True,
        idle_finalize_sec: float = 0.8,
    ) -> None:
        self._events = events
        self._stop_event = stop_event
        self._latency = latency

        self._interactive = _isatty()
        self._use_color = enable_colors and self._interactive
        self._live_indicator = live_indicator
        self._idle_finalize_sec = max(0.0, float(idle_finalize_sec))

        self._engine = TranscriptEngine(
            max_overlap_words=max(10, int(tail_anchor_words)),
            final_stability_n=final_stability_n,
            final_stability_ratio=final_stability_ratio,
            final_stability_window_words=final_stability_window_words,
            min_final_overlap_words=min_final_overlap_words,
            fuzzy_overlap_min_ratio=fuzzy_overlap_min_ratio,
            max_backtrack_words=max_backtrack_words,
            tail_anchor_words=tail_anchor_words,
        )

        self._printed_committed_words = 0
        self._live_active = False

    def run(self) -> None:
        last_ev_ts = time.monotonic()
        try:
            while not self._stop_event.is_set():
                try:
                    ev = self._events.get(timeout=0.05)
                except queue.Empty:
                    last_ev_ts = self._maybe_idle_finalize(now=time.monotonic(), last_ev_ts=last_ev_ts)
                    continue

                t0 = time.perf_counter()
                upd = self._engine.update(ev)
                self._render(upd)
                last_ev_ts = time.monotonic()

                if self._latency is not None:
                    self._latency.mark("output", time.perf_counter() - t0)

                self._events.task_done()

        finally:
            self._clear_live()
            self._flush_all()

    def _render(self, upd: TranscriptUpdate) -> None:
        if upd.commit_happened:
            self._flush_committed(upd)

        self._render_live(upd)

    def _flush_committed(self, upd: TranscriptUpdate) -> None:
        pieces = upd.committed_pieces
        start = upd.committed_prev_len
        end = len(pieces)

        if start < self._printed_committed_words:
            self._printed_committed_words = start

        if start >= end:
            self._printed_committed_words = end
            return

        self._clear_live()
        text = "".join(pieces[start:end]).strip()
        if not text:
            self._printed_committed_words = end
            return

        if self._use_color:
            print(f"{ANSI.GREEN}{text}{ANSI.RESET}", flush=True)
        else:
            print(text, flush=True)

        self._printed_committed_words = end

    def _flush_all(self) -> None:
        upd = self._engine.finalize()
        self._flush_committed(upd)
        self._clear_live()

    def _maybe_idle_finalize(self, *, now: float, last_ev_ts: float) -> float:
        if self._idle_finalize_sec <= 0:
            return last_ev_ts
        if (now - last_ev_ts) < self._idle_finalize_sec:
            return last_ev_ts

        upd = self._engine.finalize()
        self._flush_committed(upd)
        self._clear_live()
        return now

    def _render_live(self, upd: TranscriptUpdate) -> None:
        if not self._interactive:
            return

        base_pieces = upd.committed_pieces

        live_tail = "".join(upd.partial_pieces[upd.partial_skip:]).strip()
        base_tail = "".join(base_pieces[self._printed_committed_words:]).strip()

        if not base_tail and not live_tail:
            self._clear_live()
            return

        line = " ".join(x for x in (base_tail, live_tail) if x)

        prefix = f"{self._live_indicator} "
        if self._use_color:
            out = f"{ANSI.DIM}{prefix}{ANSI.YELLOW}{line}{ANSI.RESET}"
        else:
            out = prefix + line

        sys.stdout.write(ANSI.CLEAR_LINE + ANSI.COL0 + out)
        sys.stdout.flush()
        self._live_active = True

    def _clear_live(self) -> None:
        if not self._live_active:
            return
        if self._interactive:
            sys.stdout.write(ANSI.CLEAR_LINE + ANSI.COL0)
            sys.stdout.flush()
        self._live_active = False
