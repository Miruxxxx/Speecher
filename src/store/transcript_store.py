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

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_tail(self, n_words: int) -> tuple[int, list[Word]]:
        with self._lock:
            total = len(self._entries)
            start = max(0, total - n_words)
            return start, [e[0] for e in self._entries[start:]]

    def append_to_last_word(self, suffix: str) -> None:
        """Append `suffix` to the most recent word's text in place.

        For punctuation the model emits only after the word was already
        committed (nemotron's terminal '.'/'?'). Keeps the entry count and
        every index unchanged, so it is safe for the same reasons
        `update_word_texts` is.
        """
        if not suffix:
            return
        with self._lock:
            if not self._entries:
                return
            old_word, ts = self._entries[-1]
            self._entries[-1] = (
                Word(text=old_word.text + suffix, start=old_word.start, end=old_word.end),
                ts,
            )

    def attach_punct_at(self, time_sec: float, suffix: str, tol: float = 2.0) -> None:
        """Append `suffix` to the word whose end is closest to `time_sec`.

        Nemotron emits sentence-final '.'/'?' ~1.5 s after the word it closes,
        by which point 1-2 later words are already committed, so the mark can't
        just go on the last entry. It carries its own emission timestamp and
        lands on the word it acoustically belongs to. Edits text in place, so
        the entry count and every index stay put - safe for the same reason
        `update_word_texts` is. `tol` only bounds the backward scan; if nothing
        falls inside it the closest word still gets the mark (never dropped).
        """
        if not suffix:
            return
        with self._lock:
            if not self._entries:
                return
            best_i = len(self._entries) - 1
            best_dist = abs(self._entries[best_i][0].end - time_sec)
            # Punctuation is always recent, so walk back over the tail only and
            # stop once words end well before the mark - older ones can't win.
            for i in range(len(self._entries) - 2, -1, -1):
                w = self._entries[i][0]
                dist = abs(w.end - time_sec)
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
                if w.end < time_sec - tol:
                    break
            old_word, ts = self._entries[best_i]
            self._entries[best_i] = (
                Word(text=old_word.text + suffix, start=old_word.start, end=old_word.end),
                ts,
            )

    def update_word_texts(self, start_index: int, new_texts: list[str]) -> None:
        with self._lock:
            if start_index + len(new_texts) > len(self._entries):
                return
            for i, text in enumerate(new_texts):
                old_word, ts = self._entries[start_index + i]
                self._entries[start_index + i] = (
                    Word(text=text, start=old_word.start, end=old_word.end),
                    ts,
                )
