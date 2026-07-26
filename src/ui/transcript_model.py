"""Turning the store's flat word list into the replies the feed renders.

Design system 2.3: one timestamp per reply, in the left gutter, aligned to the
first line and never repeated on wraps. So the view needs replies, not words —
and the split rule lives here, away from Qt, because it is the part worth
testing.

Deliberately pure: input is (word, wall_clock_received_at) pairs exactly as
TranscriptStore keeps them, output is plain dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from asr.events import Word

# A gap this long between two words starts a new reply. Word timestamps are
# global stream seconds, so this is real silence in the captured audio, not
# processing latency.
PAUSE_SEC = 1.2

# A reply longer than this is allowed to end at the next sentence boundary, so
# an uninterrupted monologue still gets time anchors instead of becoming one
# unreadable slab.
SOFT_MAX_CHARS = 180

_SENTENCE_END = ".!?…"


@dataclass(frozen=True, slots=True)
class Reply:
    """One gutter timestamp and the text that belongs to it."""

    at: float          # wall clock (time.time) of the reply's first word
    text: str
    start_sec: float   # stream seconds, first word
    end_sec: float     # stream seconds, last word


def group_entries(
    entries: Sequence[Tuple[Word, float]],
    *,
    pause_sec: float = PAUSE_SEC,
    soft_max_chars: int = SOFT_MAX_CHARS,
) -> List[Reply]:
    """Split committed words into replies. Order is preserved; nothing is dropped."""
    replies: List[Reply] = []
    buf: List[str] = []
    at = 0.0
    start = 0.0
    end = 0.0
    prev_end: float | None = None
    prev_text = ""
    length = 0

    for word, received_at in entries:
        text = word.text.strip()
        if not text:
            continue
        if buf:
            gap = word.start - (prev_end if prev_end is not None else word.start)
            sentence_over = prev_text[-1:] in _SENTENCE_END
            if gap >= pause_sec or (sentence_over and length >= soft_max_chars):
                replies.append(Reply(at, " ".join(buf), start, end))
                buf = []
                length = 0
        if not buf:
            at = received_at
            start = word.start
        buf.append(text)
        length += len(text) + 1
        end = word.end
        prev_end = word.end
        prev_text = text

    if buf:
        replies.append(Reply(at, " ".join(buf), start, end))
    return replies


def tail_by_chars(replies: Sequence[Reply], max_chars: int) -> List[Reply]:
    """The newest replies that fit in `max_chars`, oldest-first.

    The feed only ever shows its bottom, and re-rendering the whole session on
    every commit would get slower the longer the session runs.
    """
    out: List[Reply] = []
    total = 0
    for reply in reversed(replies):
        total += len(reply.text) + 1
        out.append(reply)
        if total >= max_chars:
            break
    out.reverse()
    return out


def replies_from(
    entries: Iterable[Tuple[Word, float]], max_chars: int
) -> List[Reply]:
    """group_entries + tail_by_chars, the combination the view always wants."""
    return tail_by_chars(group_entries(list(entries)), max_chars)
