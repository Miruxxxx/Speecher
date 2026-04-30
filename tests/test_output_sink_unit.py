import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


from asr.events import ASREvent  # noqa: E402
from transcript_engine import TranscriptEngine  # noqa: E402


class TestTranscriptEngine(unittest.TestCase):
    def _make_engine(self, **kwargs) -> TranscriptEngine:
        params = {
            "max_overlap_words": 80,
            "final_stability_n": 1,
            "final_stability_ratio": 0.82,
            "final_stability_window_words": 40,
            "min_final_overlap_words": 3,
            "fuzzy_overlap_min_ratio": 0.78,
            "max_backtrack_words": 48,
            "tail_anchor_words": 80,
        }
        params.update(kwargs)
        return TranscriptEngine(**params)

    def test_first_final_commits(self) -> None:
        eng = self._make_engine(final_stability_n=2)
        u = eng.update(ASREvent(type="final", text="hello world"))
        self.assertTrue(u.commit_happened)
        self.assertEqual(u.committed_words, ["hello", "world"])

    def test_second_final_requires_stability_then_appends_by_overlap(self) -> None:
        eng = self._make_engine(final_stability_n=2)
        eng.update(ASREvent(type="final", text="hello world this"))
        u = eng.update(ASREvent(type="final", text="hello world this is fine"))
        self.assertFalse(u.commit_happened)
        self.assertEqual(" ".join(u.committed_words), "hello world this")

        u = eng.update(ASREvent(type="final", text="hello world this is fine"))
        self.assertTrue(u.commit_happened)
        self.assertEqual(" ".join(u.committed_words), "hello world this is fine")

    def test_final_without_overlap_is_ignored(self) -> None:
        eng = self._make_engine()
        eng.update(ASREvent(type="final", text="hello world this"))
        u = eng.update(ASREvent(type="final", text="totally different words"))
        self.assertFalse(u.commit_happened)
        self.assertEqual(u.committed_words, ["hello", "world", "this"])

    def test_partial_never_mutates_committed_and_skip_is_reported(self) -> None:
        eng = self._make_engine()
        eng.update(ASREvent(type="final", text="hello world"))
        u = eng.update(ASREvent(type="partial", text="world again"))
        self.assertEqual(u.committed_words, ["hello", "world"])
        self.assertEqual(u.partial_words, ["world", "again"])
        self.assertEqual(u.partial_skip, 1)

    def test_finalize_discards_partial(self) -> None:
        eng = self._make_engine()
        eng.update(ASREvent(type="final", text="hello world"))
        eng.update(ASREvent(type="partial", text="world again"))
        u = eng.finalize()
        self.assertEqual(u.partial_words, [])
        self.assertEqual(u.partial_pieces, [])
        self.assertFalse(u.commit_happened)

    def test_finalize_commits_pending_final_after_silence(self) -> None:
        eng = self._make_engine(final_stability_n=2)
        eng.update(ASREvent(type="final", text="hello world this is"))

        u = eng.update(ASREvent(type="final", text="world this is more words"))
        self.assertFalse(u.commit_happened)
        self.assertEqual(" ".join(u.committed_words), "hello world this is")

        u = eng.finalize()
        self.assertTrue(u.commit_happened)
        self.assertEqual(" ".join(u.committed_words), "hello world this is more words")

    def test_duplicate_tail_is_rewritten_when_stable(self) -> None:
        eng = self._make_engine(final_stability_n=2)
        eng.update(ASREvent(type="final", text="hello world this is a test test case"))

        u = eng.update(ASREvent(type="final", text="this is a test case for drift"))
        self.assertFalse(u.commit_happened)
        self.assertEqual(
            " ".join(u.committed_words),
            "hello world this is a test test case",
        )

        u = eng.update(ASREvent(type="final", text="this is a test case for drift"))
        self.assertTrue(u.commit_happened)
        self.assertEqual(
            " ".join(u.committed_words),
            "hello world this is a test case for drift",
        )
        self.assertEqual(u.committed_prev_len, 6)

    def test_noisy_final_does_not_poison_future_alignment(self) -> None:
        eng = self._make_engine(final_stability_n=2)
        eng.update(ASREvent(type="final", text="hello world this is"))

        u = eng.update(ASREvent(type="final", text="broken noisy guess words"))
        self.assertFalse(u.commit_happened)

        u = eng.update(ASREvent(type="final", text="hello world this is more words"))
        self.assertFalse(u.commit_happened)
        self.assertEqual(" ".join(u.committed_words), "hello world this is")

        u = eng.update(ASREvent(type="final", text="hello world this is more words"))
        self.assertTrue(u.commit_happened)
        self.assertEqual(" ".join(u.committed_words), "hello world this is more words")

    def test_identical_stable_final_clears_partial_without_duplicate_commit(self) -> None:
        eng = self._make_engine(final_stability_n=2)
        eng.update(ASREvent(type="final", text="hello world this"))
        eng.update(ASREvent(type="partial", text="hello world this again"))

        u = eng.update(ASREvent(type="final", text="hello world this"))
        self.assertFalse(u.commit_happened)
        self.assertEqual(u.partial_words, [])
        self.assertEqual(u.partial_pieces, [])
        self.assertEqual(u.committed_words, ["hello", "world", "this"])


if __name__ == "__main__":
    unittest.main()
