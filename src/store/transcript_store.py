from __future__ import annotations

import re
import threading
import time
from typing import List, Optional, Protocol, Tuple

from asr.events import Word


class Journal(Protocol):
    """Sink for store mutations (see store/transcript_writer.py)."""

    def record(self, rec: dict) -> None: ...

    def record_many(self, records: List[dict]) -> None: ...


class TranscriptStore:
    """Thread-safe accumulator for committed ASR words with time-indexed queries.

    An optional `Journal` receives every mutation as a record dict. It is
    attached here rather than to the producers because the store is the one
    place all of them meet: the splitter's commits, nemotron's late punctuation
    and the whisper PunctuationWorker's rewrites. Records describe the *applied*
    result (resolved index + final text), so a replay reproduces this store
    exactly. Journal calls happen after the lock is released — disk I/O must
    never block a transcript read.
    """

    def __init__(self, journal: Optional[Journal] = None) -> None:
        self._lock = threading.Lock()
        self._entries: List[Tuple[Word, float]] = []  # (word, wall_clock_receive_time)
        self._journal = journal

    def _emit(self, records: List[dict]) -> None:
        """Hand a whole mutation to the journal in one call.

        One call, not one per record: `record` locks and fsync-less-flushes each
        time, and `load_entries` replays an entire previous session in a single
        mutation. Per-record writes made resuming a 10k-word session ten
        thousand lock/write/flush round trips — on the Qt thread, since the
        resume runs from the session menu.
        """
        if self._journal is None or not records:
            return
        self._journal.record_many(records)

    def append(self, words: List[Word]) -> None:
        t = time.time()
        accepted = [w for w in words if w.text.strip()]
        with self._lock:
            for w in accepted:
                self._entries.append((w, t))
        self._emit(
            [{"t": "w", "text": w.text, "s": w.start, "e": w.end, "at": t} for w in accepted]
        )

    def load_entries(self, entries: List[Tuple[Word, float]]) -> int:
        """Append words replayed from a previous session, keeping their original
        receive times so `words_since_minutes` doesn't report them as just said.

        Journaled like any other append, so the current session's file stays a
        self-contained record of what the store holds.
        """
        accepted = [(w, t) for w, t in entries if w.text.strip()]
        with self._lock:
            self._entries.extend(accepted)
        self._emit(
            [{"t": "w", "text": w.text, "s": w.start, "e": w.end, "at": t} for w, t in accepted]
        )
        return len(accepted)

    def snapshot(self) -> List[Word]:
        """Every word currently held, in order — what the exports write out."""
        with self._lock:
            return [e[0] for e in self._entries]

    def recent_entries(self, max_words: int) -> List[Tuple[Word, float]]:
        """The last `max_words` (word, receive_time) pairs, oldest-first.

        The overlay groups these into timestamped replies (ui/transcript_model);
        it needs the receive times, and it needs only the tail — copying a
        session's worth of entries at every repaint would get slower the longer
        the session runs.
        """
        with self._lock:
            if max_words <= 0 or max_words >= len(self._entries):
                return list(self._entries)
            return list(self._entries[-max_words:])

    def entries_since(self, index: int) -> List[Tuple[Word, float]]:
        """Everything from absolute `index` onward, oldest-first.

        Index-addressed, unlike `recent_entries`, because the translation worker
        must resume exactly where it stopped: a sliding window of the last N
        words regroups as the store grows, and units cut from it are not stable
        (which is precisely how the first version of the feature broke).
        """
        with self._lock:
            if index <= 0:
                return list(self._entries)
            if index >= len(self._entries):
                return []
            return list(self._entries[index:])

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
            index = len(self._entries) - 1
            new_text = old_word.text + suffix
            self._entries[index] = (
                Word(text=new_text, start=old_word.start, end=old_word.end),
                ts,
            )
        self._emit([{"t": "patch", "i": index, "text": new_text}])

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
            new_text = old_word.text + suffix
            self._entries[best_i] = (
                Word(text=new_text, start=old_word.start, end=old_word.end),
                ts,
            )
        self._emit([{"t": "patch", "i": best_i, "text": new_text}])

    def update_word_texts(self, start_index: int, new_texts: list[str]) -> None:
        changed: list[dict] = []
        with self._lock:
            if start_index + len(new_texts) > len(self._entries):
                return
            for i, text in enumerate(new_texts):
                old_word, ts = self._entries[start_index + i]
                if old_word.text == text:
                    # The punctuation worker resubmits its whole 80-word window
                    # every cycle; journaling the untouched majority would cost
                    # hundreds of lines a minute for nothing.
                    continue
                self._entries[start_index + i] = (
                    Word(text=text, start=old_word.start, end=old_word.end),
                    ts,
                )
                changed.append({"t": "patch", "i": start_index + i, "text": text})
        self._emit(changed)
