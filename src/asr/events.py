from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ASREventType = Literal["partial", "final", "log", "fatal"]


@dataclass(slots=True)
class ASREvent:
    type: ASREventType
    source: str = ""
    text: str = ""
    words: list[str] | None = None
    pieces: list[str] | None = None
    created_t: float | None = None

