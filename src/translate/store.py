"""The translated feed: an append-only list of finished lines.

One rule shapes this file, and it is the one the first version broke: **a line
that reached the screen never changes again**. So a row is only added once its
text is final — translated, or deliberately left in the original (already in the
target language, or the model failed on it). There is no "pending" row, because a
row that later mutates re-wraps and shifts everything the reader is looking at.

Rows are keyed by the first word's absolute index in TranscriptStore (see
translate/segments.py) and stay sorted by it, so the journal can replay them and
the feed can render its tail without sorting anything.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from translate.segments import Segment
from ui.transcript_model import Reply

# Why a row reads the way it does. The feed renders all three identically — a
# reader does not care which of them produced the line — but the log and the
# tests do.
TRANSLATED = "translated"
SKIPPED = "skipped"      # already in the target language
FAILED = "failed"        # the model raised or returned nothing; original kept


class Journal(Protocol):
    def record(self, rec: dict) -> None: ...


@dataclass(frozen=True, slots=True)
class Row:
    index: int
    n_words: int
    at: float
    start_sec: float
    end_sec: float
    text: str
    state: str

    def reply(self) -> Reply:
        """The feed renders replies; a row is one, with a final text."""
        return Reply(
            at=self.at, text=self.text, start_sec=self.start_sec, end_sec=self.end_sec
        )


class TranslationStore:
    """Thread-safe append-only rows. Written by the translation worker, read by
    the Qt thread on every repaint."""

    def __init__(self, target: str = "ru", journal: Optional[Journal] = None) -> None:
        self._lock = threading.Lock()
        self._rows: List[Row] = []
        self._target = target
        self._journal = journal

    @property
    def target(self) -> str:
        return self._target

    # -- writing ---------------------------------------------------------

    def add(self, segment: Segment, text: str, state: str) -> Row:
        """Append one finished line. Journaled when it carries a translation."""
        row = Row(
            index=segment.index,
            n_words=segment.n_words,
            at=segment.at,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            text=text,
            state=state,
        )
        with self._lock:
            self._rows.append(row)
        if self._journal is not None and state == TRANSLATED:
            self._journal.record({
                "t": "tr",
                "i": row.index,
                "n": row.n_words,
                "lang": self._target,
                "text": row.text,
            })
        return row

    def load(self, records: Dict[int, tuple[int, str]], entries: List[tuple]) -> int:
        """Rebuild rows from a replayed session.

        The journal stores only `(index, n_words, text)`: the timestamps are
        recoverable from the words themselves, which the same replay just put
        back into the transcript store, so they are not written twice.
        """
        added = 0
        with self._lock:
            for index in sorted(records):
                n_words, text = records[index]
                if not text or index + n_words > len(entries):
                    continue
                first_word, at = entries[index]
                last_word = entries[index + n_words - 1][0]
                self._rows.append(Row(
                    index=index,
                    n_words=n_words,
                    at=at,
                    start_sec=first_word.start,
                    end_sec=last_word.end,
                    text=text,
                    state=TRANSLATED,
                ))
                added += 1
            self._rows.sort(key=lambda r: r.index)
        return added

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    # -- reading ---------------------------------------------------------

    def rows(self) -> List[Row]:
        with self._lock:
            return list(self._rows)

    def replies(self) -> List[Reply]:
        """What the feed renders, oldest first."""
        with self._lock:
            return [row.reply() for row in self._rows]

    def size(self) -> int:
        with self._lock:
            return len(self._rows)

    def skipped_streak(self) -> int:
        """How many of the newest rows in a row went through untranslated.

        A long streak means the speech is already in the target language and
        the feature is a no-op — which looks exactly like a broken translator,
        because the feed fills with the original text. The overlay uses this to
        say so out loud instead of letting the user guess.
        """
        with self._lock:
            count = 0
            for row in reversed(self._rows):
                if row.state != SKIPPED:
                    break
                count += 1
            return count

    def next_index(self) -> int:
        """First word not yet covered by a row — where the worker resumes."""
        with self._lock:
            return self._rows[-1].index + self._rows[-1].n_words if self._rows else 0

    def texts_by_index(self) -> Dict[int, str]:
        """Translated lines keyed by first word index — what an export needs."""
        with self._lock:
            return {r.index: r.text for r in self._rows if r.state == TRANSLATED}
