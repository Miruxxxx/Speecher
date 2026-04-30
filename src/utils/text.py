from __future__ import annotations

import re
import unicodedata


_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)


def normalize_word(text: str) -> str:
    """
    Normalize a word for equality comparison across ASR hypotheses.

    Strips punctuation, casefolds, and collapses Unicode compatibility
    forms so that "World," "world" and "world" all compare equal.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _PUNCT_RE.sub("", t)
    return t.casefold().strip()
