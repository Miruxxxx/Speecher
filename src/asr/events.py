from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


ASREventType = Literal["commit", "partial", "log", "fatal"]


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
      log     -> informational
      fatal   -> unrecoverable error; should trigger shutdown
    """

    type: ASREventType
    source: str = ""
    text: str = ""
    words: List[Word] = field(default_factory=list)
