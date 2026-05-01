from __future__ import annotations

import re
import threading
import time
from typing import List, Optional, Tuple

from asr.events import Word


class TranscriptStore:
    """Thread-safe accumulator for committed ASR words with time-indexed queries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: List[Tuple[Word, float]] = []  # (word, wall_clock_receive_time)

    def append(self, words: List[Word]) -> None:
        t = time.time()
        with self._lock:
            for w in words:
                if w.text.strip():
                    self._entries.append((w, t))

    def all_text(self) -> str:
        with self._lock:
            return " ".join(e[0].text for e in self._entries).strip()

    def recent_text(self, max_chars: int = 800) -> str:
        text = self.all_text()
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def words_since_minutes(self, minutes: float) -> List[Word]:
        cutoff = time.time() - minutes * 60.0
        with self._lock:
            return [w for w, t in self._entries if t >= cutoff]

    def find_last_question(self) -> Optional[str]:
        text = self.all_text()
        if not text:
            return None
        q_idx = text.rfind("?")
        if q_idx == -1:
            return None
        # Walk backwards from q_idx to find sentence start
        search_area = text[:q_idx]
        start = 0
        for ch in ".!?\n":
            pos = search_area.rfind(ch)
            if pos != -1 and pos + 1 > start:
                start = pos + 1
        return text[start : q_idx + 1].strip()

    def context_for_question(self, question: str, max_chars: int = 1500) -> str:
        text = self.all_text()
        if not question:
            return text[-max_chars:]
        pos = text.rfind(question)
        if pos == -1:
            return text[-max_chars:]
        end = pos + len(question)
        start = max(0, end - max_chars)
        return text[start:end].strip()
