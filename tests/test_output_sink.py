import io
import queue
import threading
from contextlib import redirect_stderr, redirect_stdout

from asr.events import ASREvent, Word
from ui.output_sink import OutputSink


def _make_sink(events=None, verbose=False):
    return OutputSink(
        events=events or queue.Queue(),
        stop_event=threading.Event(),
        verbose_logs=verbose,
    )


def _words(*texts):
    return [Word(text=t, start=0.0, end=0.0) for t in texts]


def test_commit_prints_plain_text_without_ansi_when_not_tty():
    out = io.StringIO()  # StringIO has no isatty=True => non-interactive
    with redirect_stdout(out):
        sink = _make_sink()
        sink._dispatch(ASREvent(type="commit", words=_words("hello", "world")))
    assert out.getvalue() == "hello world\n"


def test_partial_writes_nothing_when_not_tty():
    out = io.StringIO()
    with redirect_stdout(out):
        sink = _make_sink()
        sink._dispatch(ASREvent(type="partial", words=_words("live", "tail")))
    assert out.getvalue() == ""


def test_empty_commit_is_skipped():
    out = io.StringIO()
    with redirect_stdout(out):
        sink = _make_sink()
        sink._dispatch(ASREvent(type="commit", words=_words("  ")))
    assert out.getvalue() == ""


def test_fatal_goes_to_stderr_regardless_of_verbosity():
    err = io.StringIO()
    with redirect_stderr(err):
        sink = _make_sink(verbose=False)
        sink._dispatch(ASREvent(type="fatal", source="capture", text="boom"))
    assert "boom" in err.getvalue()


def test_log_suppressed_unless_verbose():
    err = io.StringIO()
    with redirect_stderr(err):
        _make_sink(verbose=False)._dispatch(
            ASREvent(type="log", source="engine", text="chatter")
        )
    assert err.getvalue() == ""

    err2 = io.StringIO()
    with redirect_stderr(err2):
        _make_sink(verbose=True)._dispatch(
            ASREvent(type="log", source="engine", text="chatter")
        )
    assert "chatter" in err2.getvalue()
