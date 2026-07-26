"""Summaries produced during a session (design system 2.5, brief section 4).

An hour of conversation produces more than ten of these, which is why they live
in a vertical list in their own window rather than in a carousel — and why the
feed never swallows them: a summary appearing inside the transcript would move
it.

In memory only: the transcript itself is what gets journaled, a summary can
always be regenerated from it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class Summary:
    created_at: float   # wall clock
    from_at: float      # start of the summarised window
    to_at: float        # end of it
    text: str
    minutes: float

    def title(self) -> str:
        start = datetime.fromtimestamp(self.from_at).strftime("%H:%M")
        end = datetime.fromtimestamp(self.to_at).strftime("%H:%M")
        return f"Выжимка за {start}–{end}"

    def stamp(self) -> str:
        return datetime.fromtimestamp(self.created_at).strftime("%H:%M")


class SummaryHistory:
    """Thread-safe append-only list. Written from the Qt thread, read there too,
    but the lock keeps it honest if a worker ever adds one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: List[Summary] = []

    def add(self, text: str, minutes: float) -> Summary:
        now = time.time()
        item = Summary(
            created_at=now,
            from_at=now - minutes * 60.0,
            to_at=now,
            text=text.strip(),
            minutes=minutes,
        )
        with self._lock:
            self._items.append(item)
        return item

    def all(self) -> List[Summary]:
        with self._lock:
            return list(self._items)

    def latest(self) -> Optional[Summary]:
        with self._lock:
            return self._items[-1] if self._items else None

    def size(self) -> int:
        with self._lock:
            return len(self._items)
