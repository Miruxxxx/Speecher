"""Cutting the transcript into the units that get translated.

This module exists because the first attempt keyed translations by a reply's
timestamp, and replies are *derived*: `ui.transcript_model.group_entries` regroups
the whole visible window on every repaint, so its blocks depend on how many words
the caller happened to read. Two readers with different window sizes produced
different blocks, and the same translation landed on two of them.

A segment is defined against the **append-only store** instead, and identified by
the absolute index of its first word — the same stability `PunctuationWorker` and
the journal's `patch` records already rely on. Appending words can never change a
segment that was already cut.

The cut itself follows speech, not layout:

* a sentence ends (the ASR model punctuates natively) — one subtitle per
  sentence is the rhythm people already read at;
* or a pause: silence between sentences the model did not mark;
* or a hard word cap, so a monologue without either still reaches the screen.

The last run of words is only cut when it is *closed* — either a later word
already exists (so the boundary is known) or the audio has been quiet long
enough. Everything before it is frozen: cut once, never recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from asr.events import Word

SENTENCE_END = ".!?…"


@dataclass(frozen=True, slots=True)
class Segment:
    """One translation unit, and one line in the translated feed."""

    index: int          # absolute index of the first word in the store
    n_words: int
    at: float           # wall clock of the first word — the gutter timestamp
    start_sec: float    # stream seconds
    end_sec: float
    text: str

    @property
    def next_index(self) -> int:
        return self.index + self.n_words


def _sentence_over(text: str) -> bool:
    return text.rstrip()[-1:] in SENTENCE_END


def cut(
    entries: Sequence[Tuple[Word, float]],
    first_index: int,
    *,
    pause_sec: float = 1.2,
    max_words: int = 40,
    tail_closed: bool = False,
) -> List[Segment]:
    """Segments that are ready, from `entries` starting at absolute `first_index`.

    Returns only *finished* segments; the words after the last one stay in the
    store for the next call. `tail_closed` says the trailing run may be cut too
    (the audio went quiet), which is the only way the last thing said before a
    silence ever reaches the screen.
    """
    segments: List[Segment] = []
    buf: List[Tuple[Word, float]] = []
    start_at = first_index

    for i, (word, received) in enumerate(entries):
        text = word.text.strip()
        if not text:
            continue
        if buf:
            gap = word.start - buf[-1][0].end
            # The boundary belongs *before* this word: the previous one ended a
            # sentence, or this one starts after a silence.
            if _sentence_over(buf[-1][0].text) or gap >= pause_sec:
                segments.append(_make(buf, start_at))
                start_at += len(buf)
                buf = []
        buf.append((word, received))
        if len(buf) >= max_words:
            segments.append(_make(buf, start_at))
            start_at += len(buf)
            buf = []

    if buf and tail_closed:
        segments.append(_make(buf, start_at))
    return segments


def _make(buf: Sequence[Tuple[Word, float]], index: int) -> Segment:
    words = [w for w, _ in buf]
    return Segment(
        index=index,
        n_words=len(buf),
        at=buf[0][1],
        start_sec=words[0].start,
        end_sec=words[-1].end,
        text=" ".join(w.text.strip() for w in words).strip(),
    )


def merge(segments: Sequence[Segment]) -> Segment:
    """Fold consecutive segments into one — how the worker catches up.

    Merging keeps the feed in order and never invents a hole; the price is one
    longer line instead of several short ones, which is the right thing to pay
    when the model is behind the speech.
    """
    first, last = segments[0], segments[-1]
    return Segment(
        index=first.index,
        n_words=sum(s.n_words for s in segments),
        at=first.at,
        start_sec=first.start_sec,
        end_sec=last.end_sec,
        text=" ".join(s.text for s in segments),
    )
