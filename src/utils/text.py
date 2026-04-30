from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Sequence

# ---------------------------------------------------------------------------
# Legacy helpers (kept for compatibility)
# ---------------------------------------------------------------------------

def strip_overlap(prev: str, curr: str, min_overlap_chars: int = 20) -> str:
    """Remove duplicated prefix in curr if it matches a suffix of prev."""
    prev = (prev or "").strip()
    curr = (curr or "").strip()

    if not prev or not curr:
        return curr

    max_len = min(len(prev), len(curr))
    prev_l = prev.casefold()
    curr_l = curr.casefold()

    for size in range(max_len, min_overlap_chars - 1, -1):
        if prev_l[-size:] == curr_l[:size]:
            return curr[size:].lstrip()

    return curr


def strip_word_overlap(prev: str, curr: str, min_words: int = 3) -> str:
    """Remove duplicated word prefix in curr if it matches word suffix of prev."""
    prev_words = (prev or "").strip().split()
    curr_words = (curr or "").strip().split()

    max_overlap = min(len(prev_words), len(curr_words))
    for size in range(max_overlap, min_words - 1, -1):
        if [w.casefold() for w in prev_words[-size:]] == [w.casefold() for w in curr_words[:size]]:
            return " ".join(curr_words[size:]).lstrip()

    return curr

# ---------------------------------------------------------------------------
# Streaming-optimized helpers (word-level de-dup + stability)
# ---------------------------------------------------------------------------

# Matches "words" across Latin/Cyrillic/etc. (unicode letters + digits), optional apostrophe part.
# [^\W_] == any unicode "word" char except underscore.
_word_re = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)


def words(text: str) -> List[str]:
    """Tokenize to words, stripping punctuation. Returns [] on None/empty."""
    return _word_re.findall(text or "")


def word_pieces(text: str) -> tuple[List[str], List[str]]:
    """
    Split text into (words, pieces) aligned by word index.

    - words: matched word strings (no punctuation)
    - pieces: slices of the original text starting at each word and ending
      right before the next word (keeps punctuation/whitespace between words)

    Joining pieces reconstructs the word-bearing portion of the original text
    with punctuation preserved.
    """
    text = text or ""
    matches = list(_word_re.finditer(text))
    if not matches:
        return [], []

    ws: List[str] = []
    pcs: List[str] = []
    for i, m in enumerate(matches):
        ws.append(m.group(0))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pcs.append(text[start:end])

    return ws, pcs


def _norm(w: str) -> str:
    return (w or "").casefold()


def overlap_suffix_prefix(a: List[str], b: List[str], max_k: int = 80) -> int:
    """
    Find maximum k such that suffix(a, k) == prefix(b, k) (case-insensitive).
    Used to remove repeats under sliding windows.
    """
    max_k = min(max_k, len(a), len(b))
    if max_k <= 0:
        return 0

    a_norm = [_norm(x) for x in a]
    b_norm = [_norm(x) for x in b]

    for k in range(max_k, 0, -1):
        if a_norm[-k:] == b_norm[:k]:
            return k
    return 0


def common_prefix_len(a: List[str], b: List[str], max_k: int | None = None) -> int:
    """Length of the common prefix of a and b (case-insensitive)."""
    n = min(len(a), len(b))
    if max_k is not None:
        n = min(n, max_k)
    for i in range(n):
        if _norm(a[i]) != _norm(b[i]):
            return i
    return n


def similarity_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """Case-insensitive similarity between two word sequences."""
    if not a or not b:
        return 0.0

    return SequenceMatcher(
        a=[_norm(x) for x in a],
        b=[_norm(x) for x in b],
    ).ratio()


def find_tail_anchor(
    committed_words: Sequence[str],
    final_words: Sequence[str],
    *,
    tail_anchor_words: int,
    min_overlap_words: int,
    fuzzy_overlap_min_ratio: float,
    max_backtrack_words: int,
) -> tuple[int, int, float] | None:
    """
    Find where `final_words` should replace the recent committed tail.

    Returns `(anchor_index, overlap_words, score)` where:
    - `anchor_index` is the committed word index to replace from
    - `overlap_words` is the matched prefix length in `final_words`
    - `score` is 1.0 for exact matches or the fuzzy similarity ratio
    """
    committed = list(committed_words)
    final = list(final_words)
    if not committed or not final:
        return None

    tail_anchor_words = max(1, int(tail_anchor_words))
    min_overlap_words = max(1, int(min_overlap_words))
    max_backtrack_words = max(min_overlap_words, int(max_backtrack_words))

    committed_len = len(committed)
    search_start = max(
        0,
        committed_len - tail_anchor_words,
        committed_len - max_backtrack_words,
    )

    exact_best: tuple[int, int] | None = None
    fuzzy_best: tuple[float, int, int] | None = None

    for anchor in range(search_start, committed_len):
        max_k = min(len(final), committed_len - anchor)
        if max_k < min_overlap_words:
            continue

        exact_k = common_prefix_len(committed[anchor : anchor + max_k], final, max_k=max_k)
        if exact_k >= min_overlap_words:
            candidate = (exact_k, anchor)
            if exact_best is None or candidate > exact_best:
                exact_best = candidate

        for k in range(min_overlap_words, max_k + 1):
            ratio = similarity_ratio(committed[anchor : anchor + k], final[:k])
            candidate = (ratio, k, anchor)
            if fuzzy_best is None or candidate > fuzzy_best:
                fuzzy_best = candidate

    if exact_best is not None:
        overlap_words, anchor_index = exact_best
        return anchor_index, overlap_words, 1.0

    if fuzzy_best is None:
        return None

    ratio, overlap_words, anchor_index = fuzzy_best
    if ratio < float(fuzzy_overlap_min_ratio):
        return None

    return anchor_index, overlap_words, ratio
