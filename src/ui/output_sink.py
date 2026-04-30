from __future__ import annotations

import queue
import sys
import threading
from typing import List, Optional

from asr.events import ASREvent, Word


class _ANSI:
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
    Consumes ASREvents and writes to stdout.

    Committed events are printed once and never rewritten. Partials are
    drawn on a single live line that gets cleared and redrawn each time.
    The engine is the single source of truth for what is committed: this
    sink does no merging, no overlap detection, no stabilization.
    """

    def __init__(
        self,
        *,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
        live_indicator: str = "⚡",
        enable_colors: bool = True,
        verbose_logs: bool = False,
    ) -> None:
        self._events = events
        self._stop_event = stop_event
        self._live_indicator = live_indicator
        self._verbose_logs = verbose_logs

        self._interactive = _isatty()
        self._use_color = enable_colors and self._interactive
        self._live_active = False

    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    ev = self._events.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._dispatch(ev)
                self._events.task_done()
        finally:
            self._clear_live()

    def _dispatch(self, ev: ASREvent) -> None:
        if ev.type == "commit":
            self._render_commit(ev.words)
        elif ev.type == "partial":
            self._render_partial(ev.words)
        elif ev.type == "log":
            if self._verbose_logs:
                self._print_log(f"[{ev.source}] {ev.text}")
        elif ev.type == "fatal":
            self._print_log(f"[FATAL {ev.source}] {ev.text}")

    # -- rendering ------------------------------------------------------

    def _render_commit(self, words: List[Word]) -> None:
        text = " ".join(w.text for w in words if w.text).strip()
        if not text:
            return
        self._clear_live()
        if self._use_color:
            print(f"{_ANSI.GREEN}{text}{_ANSI.RESET}", flush=True)
        else:
            print(text, flush=True)

    def _render_partial(self, words: List[Word]) -> None:
        if not self._interactive:
            return
        text = " ".join(w.text for w in words if w.text).strip()
        if not text:
            self._clear_live()
            return

        prefix = f"{self._live_indicator} "
        if self._use_color:
            line = f"{_ANSI.DIM}{prefix}{_ANSI.YELLOW}{text}{_ANSI.RESET}"
        else:
            line = prefix + text

        sys.stdout.write(_ANSI.CLEAR_LINE + _ANSI.COL0 + line)
        sys.stdout.flush()
        self._live_active = True

    def _clear_live(self) -> None:
        if not self._live_active:
            return
        if self._interactive:
            sys.stdout.write(_ANSI.CLEAR_LINE + _ANSI.COL0)
            sys.stdout.flush()
        self._live_active = False

    def _print_log(self, msg: str) -> None:
        self._clear_live()
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
