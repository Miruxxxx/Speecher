from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from asr.events import ASREvent
from utils.text import (
    common_prefix_len,
    find_tail_anchor,
    overlap_suffix_prefix,
    similarity_ratio,
    word_pieces,
)


# ============================================================
# Public data model
# ============================================================

@dataclass(slots=True)
class TranscriptUpdate:
    committed_words: List[str]
    committed_pieces: List[str]

    partial_words: List[str]
    partial_pieces: List[str]
    partial_skip: int

    commit_happened: bool
    committed_prev_len: int


@dataclass(slots=True)
class _MergeResult:
    accepted: bool
    changed: bool
    changed_prefix_len: int


# ============================================================
# Engine
# ============================================================

class TranscriptEngine:
    """
    Minimal, deterministic transcript stabilizer.

    Rules:
    - `final` -> commits only after stabilizing, then replaces a safe recent tail
    - `partial` -> preview only, never mutates committed
    """

    def __init__(
        self,
        *,
        max_overlap_words: int | None = None,
        final_stability_n: int = 2,
        final_stability_ratio: float = 0.82,
        final_stability_window_words: int = 40,
        min_final_overlap_words: int = 3,
        fuzzy_overlap_min_ratio: float = 0.78,
        max_backtrack_words: int = 48,
        tail_anchor_words: int | None = None,
    ) -> None:
        base_tail_words = tail_anchor_words if tail_anchor_words is not None else max_overlap_words
        if base_tail_words is None:
            base_tail_words = 80

        self.max_overlap_words = max(10, int(base_tail_words))
        self.final_stability_n = max(1, int(final_stability_n))
        self.final_stability_ratio = float(final_stability_ratio)
        self.final_stability_window_words = max(1, int(final_stability_window_words))
        self.min_final_overlap_words = max(1, int(min_final_overlap_words))
        self.fuzzy_overlap_min_ratio = float(fuzzy_overlap_min_ratio)
        self.max_backtrack_words = max(self.min_final_overlap_words, int(max_backtrack_words))
        self.tail_anchor_words = self.max_overlap_words

        self._committed_words: List[str] = []
        self._committed_pieces: List[str] = []

        self._partial_words: List[str] = []
        self._partial_pieces: List[str] = []
        self._recent_final_words: List[List[str]] = []
        self._pending_final_words: List[str] = []
        self._pending_final_pieces: List[str] = []

    # --------------------------------------------------------

    def update(self, ev: ASREvent) -> TranscriptUpdate:
        if ev.type == "partial":
            return self._on_partial(ev)

        if ev.type == "final":
            return self._on_final(ev)

        # log / fatal / unknown
        return self._snapshot(False, len(self._committed_words))

    # --------------------------------------------------------

    def finalize(self) -> TranscriptUpdate:
        """
        On shutdown or idle finalize: try to commit the latest pending final,
        then discard live partials.
        """
        commit = False
        changed_prefix_len = len(self._committed_words)

        if self._pending_final_words:
            result = self._accept_final_candidate(
                self._pending_final_words,
                self._pending_final_pieces,
            )
            self._clear_pending_final()
            self._reset_final_history()
            if result.accepted:
                changed_prefix_len = result.changed_prefix_len
                commit = result.changed

        self._partial_words.clear()
        self._partial_pieces.clear()
        return self._snapshot(commit, changed_prefix_len)

    # ========================================================
    # Event handlers
    # ========================================================

    def _on_partial(self, ev: ASREvent) -> TranscriptUpdate:
        self._partial_words, self._partial_pieces = _extract(ev)
        return self._snapshot(False, len(self._committed_words))

    def _on_final(self, ev: ASREvent) -> TranscriptUpdate:
        new_words, new_pieces = _extract(ev)

        if not new_words:
            return self._snapshot(False, len(self._committed_words))

        self._pending_final_words = list(new_words)
        self._pending_final_pieces = list(new_pieces)

        if not self._committed_words:
            self._committed_words = list(new_words)
            self._committed_pieces = list(new_pieces)
            self._partial_words.clear()
            self._partial_pieces.clear()
            self._clear_pending_final()
            self._reset_final_history(new_words)
            return self._snapshot(True, 0)

        self._remember_final(new_words)
        if not self._latest_final_is_stable():
            return self._snapshot(False, len(self._committed_words))

        result = self._accept_final_candidate(new_words, new_pieces)
        if not result.accepted:
            return self._snapshot(False, len(self._committed_words))

        self._partial_words.clear()
        self._partial_pieces.clear()
        self._clear_pending_final()
        self._reset_final_history(new_words)
        return self._snapshot(result.changed, result.changed_prefix_len)

    def _accept_final_candidate(self, new_words: List[str], new_pieces: List[str]) -> _MergeResult:
        if not self._committed_words:
            self._committed_words = list(new_words)
            self._committed_pieces = list(new_pieces)
            return _MergeResult(accepted=True, changed=True, changed_prefix_len=0)

        anchor = find_tail_anchor(
            self._committed_words,
            new_words,
            tail_anchor_words=self.tail_anchor_words,
            min_overlap_words=self.min_final_overlap_words,
            fuzzy_overlap_min_ratio=self.fuzzy_overlap_min_ratio,
            max_backtrack_words=self.max_backtrack_words,
        )
        if anchor is None:
            return _MergeResult(accepted=False, changed=False, changed_prefix_len=len(self._committed_words))

        anchor_index, _overlap_words, _score = anchor
        merged_words = self._committed_words[:anchor_index] + list(new_words)
        merged_pieces = self._committed_pieces[:anchor_index] + list(new_pieces)
        changed_prefix_len = common_prefix_len(self._committed_words, merged_words)

        if merged_words == self._committed_words and merged_pieces == self._committed_pieces:
            return _MergeResult(
                accepted=True,
                changed=False,
                changed_prefix_len=len(self._committed_words),
            )

        self._committed_words = merged_words
        self._committed_pieces = merged_pieces
        return _MergeResult(
            accepted=True,
            changed=True,
            changed_prefix_len=changed_prefix_len,
        )

    def _latest_final_is_stable(self) -> bool:
        if self.final_stability_n <= 1:
            return True
        if len(self._recent_final_words) < self.final_stability_n:
            return False

        finals = self._recent_final_words[-self.final_stability_n :]
        current_tail = finals[-1][-self.final_stability_window_words :]
        for prev in finals[:-1]:
            prev_tail = prev[-self.final_stability_window_words :]
            if similarity_ratio(prev_tail, current_tail) < self.final_stability_ratio:
                return False
        return True

    def _remember_final(self, words: List[str]) -> None:
        self._recent_final_words.append(list(words))
        max_hist = max(1, self.final_stability_n)
        if len(self._recent_final_words) > max_hist:
            self._recent_final_words = self._recent_final_words[-max_hist:]

    def _reset_final_history(self, words: List[str] | None = None) -> None:
        if words:
            self._recent_final_words = [list(words)]
            return
        self._recent_final_words.clear()

    def _clear_pending_final(self) -> None:
        self._pending_final_words.clear()
        self._pending_final_pieces.clear()

    # ========================================================
    # Snapshot
    # ========================================================

    def _snapshot(self, commit: bool, prev_len: int) -> TranscriptUpdate:
        skip = overlap_suffix_prefix(
            self._committed_words[-self.tail_anchor_words :],
            self._partial_words,
            max_k=self.tail_anchor_words,
        )

        return TranscriptUpdate(
            committed_words=list(self._committed_words),
            committed_pieces=list(self._committed_pieces),
            partial_words=list(self._partial_words),
            partial_pieces=list(self._partial_pieces),
            partial_skip=int(skip),
            commit_happened=bool(commit),
            committed_prev_len=int(prev_len),
        )


# ============================================================
# Helpers
# ============================================================

def _extract(ev: ASREvent) -> Tuple[List[str], List[str]]:
    if ev.words and ev.pieces and len(ev.words) == len(ev.pieces):
        return list(ev.words), list(ev.pieces)

    if ev.text:
        return word_pieces(ev.text)

    return [], []
