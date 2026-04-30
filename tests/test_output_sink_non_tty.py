import io
import os
import queue
import sys
import threading
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


from asr.events import ASREvent  # noqa: E402
from ui.output_sink import OutputSink  # noqa: E402


class TestOutputSinkNonTTY(unittest.TestCase):
    def _make_sink(self) -> OutputSink:
        return OutputSink(
            events=queue.Queue(),
            stop_event=threading.Event(),
            enable_colors=True,
            live_indicator="LIVE",
        )

    def test_render_live_writes_no_control_sequences_when_not_tty(self) -> None:
        fake_stdout = io.StringIO()
        with patch("ui.output_sink._isatty", return_value=False), patch("sys.stdout", fake_stdout):
            sink = self._make_sink()
            upd = sink._engine.update(ASREvent(type="partial", text="hello world"))
            sink._render_live(upd)

        out = fake_stdout.getvalue()
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\r", out)

    def test_non_tty_stream_contains_no_ansi_or_carriage_return(self) -> None:
        fake_stdout = io.StringIO()
        with patch("ui.output_sink._isatty", return_value=False), patch("sys.stdout", fake_stdout):
            sink = self._make_sink()
            sink._render(sink._engine.update(ASREvent(type="final", text="hello world")))
            sink._render(sink._engine.update(ASREvent(type="partial", text="hello world again")))
            sink._flush_all()

        out = fake_stdout.getvalue()
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\r", out)
        self.assertIn("hello world", out)

    def test_clear_live_resets_flag_when_not_tty(self) -> None:
        with patch("ui.output_sink._isatty", return_value=False):
            sink = self._make_sink()
            sink._live_active = True
            sink._clear_live()
        self.assertFalse(sink._live_active)

    def test_idle_finalize_commits_pending_tail_after_silence(self) -> None:
        fake_stdout = io.StringIO()
        with patch("ui.output_sink._isatty", return_value=False), patch("sys.stdout", fake_stdout):
            sink = OutputSink(
                events=queue.Queue(),
                stop_event=threading.Event(),
                enable_colors=False,
                final_stability_n=10,
                idle_finalize_sec=0.8,
            )
            sink._render(sink._engine.update(ASREvent(type="final", text="hello world")))

            new_ts = sink._maybe_idle_finalize(now=10.0, last_ev_ts=9.1)
            self.assertEqual(new_ts, 10.0)

        out = fake_stdout.getvalue()
        self.assertIn("hello world", out)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\r", out)

    def test_non_tty_renders_rewritten_tail_from_first_changed_word(self) -> None:
        fake_stdout = io.StringIO()
        with patch("ui.output_sink._isatty", return_value=False), patch("sys.stdout", fake_stdout):
            sink = OutputSink(
                events=queue.Queue(),
                stop_event=threading.Event(),
                enable_colors=False,
                final_stability_n=1,
                min_final_overlap_words=3,
                fuzzy_overlap_min_ratio=0.78,
                max_backtrack_words=48,
                tail_anchor_words=80,
            )
            sink._render(sink._engine.update(ASREvent(type="final", text="hello world this is a test test case")))
            sink._render(sink._engine.update(ASREvent(type="final", text="this is a test case for drift")))

        out = fake_stdout.getvalue()
        self.assertIn("hello world this is a test test case", out)
        self.assertIn("case for drift", out)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\r", out)


if __name__ == "__main__":
    unittest.main()
