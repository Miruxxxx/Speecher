from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


ASREventType = Literal["commit", "partial", "amend", "log", "fatal"]


@dataclass(slots=True, frozen=True)
class Word:
    """A single word with absolute (global-stream) timestamps in seconds."""

    text: str
    start: float
    end: float


@dataclass(slots=True)
class ASREvent:
    """
    Event emitted to the UI layer.

      commit  -> these words have been confirmed and will not change
      partial -> live preview, will be replaced on the next event
      amend   -> append `text` to a committed word in place, choosing the word
                 whose end is closest to `time_sec` (used by the nemotron
                 backend for sentence-final punctuation the model emits ~1.5 s
                 after the word, by which point 1-2 later words are already
                 committed; does not shift word indices)
      log     -> informational
      fatal   -> unrecoverable error; should trigger shutdown
    """

    type: ASREventType
    source: str = ""
    text: str = ""
    words: List[Word] = field(default_factory=list)
    # Emission time (global stream seconds) of an `amend` mark; ignored by
    # other event types.
    time_sec: float = 0.0
