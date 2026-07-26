"""The log file, and the session header telling the truth about the engine.

Both exist because of the same afternoon: nemotron had been silently falling
back to whisper on every start, and there was no way to find out. The status
message that announced it clears itself in seconds, `logging` wrote to stderr
only (so a run started from a shortcut left nothing), and the session header —
the thing every export and every journal file carries — named the *configured*
engine, written before the model had even tried to load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app_config import AppConfig, LoggingConfig, load_config
from asr.events import Word
from store.session_io import export_markdown, read_session
from store.transcript_store import TranscriptStore
from store.transcript_writer import TranscriptWriter
import logging_setup


# ----------------------------------------------------------------------
# Log file
# ----------------------------------------------------------------------


@pytest.fixture
def clean_logging():
    """Leave the root logger exactly as it was — other tests share it."""
    root = logging.getLogger()
    before = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if getattr(handler, "_speecher", False):
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = before[0]
    root.setLevel(before[1])


def test_messages_reach_a_file(tmp_path: Path, clean_logging):
    path = logging_setup.setup(LoggingConfig(dir=str(tmp_path / "logs")))
    assert path is not None and path.exists()
    logging.getLogger("test").warning("nemotron не загрузился")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "nemotron не загрузился" in path.read_text(encoding="utf-8")


def test_empty_dir_means_console_only(tmp_path: Path, clean_logging):
    assert logging_setup.setup(LoggingConfig(dir="")) is None


def test_setup_twice_does_not_double_every_line(tmp_path: Path, clean_logging):
    cfg = LoggingConfig(dir=str(tmp_path / "logs"))
    path = logging_setup.setup(cfg)
    logging_setup.setup(cfg)
    logging.getLogger("test").warning("однажды")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert path.read_text(encoding="utf-8").count("однажды") == 1


def test_an_unopenable_log_is_not_fatal(tmp_path: Path, clean_logging):
    """A diagnostic aid that can stop the app is worse than none."""
    blocker = tmp_path / "logs"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    assert logging_setup.setup(LoggingConfig(dir=str(blocker))) is None
    logging.getLogger("test").info("всё ещё работает")


def test_banner_names_the_config_and_the_log(tmp_path: Path, clean_logging, caplog):
    path = logging_setup.setup(LoggingConfig(dir=str(tmp_path / "logs")))
    with caplog.at_level(logging.INFO):
        logging_setup.banner(path, Path("config/config.toml"))
    assert "config.toml" in caplog.text


def test_logging_config_defaults_to_a_file(tmp_path: Path):
    """Console-only was the old behaviour; the file has to be the default."""
    assert AppConfig().logging.dir
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.logging.dir == "storage/logs"
    assert cfg.logging.backups >= 1


# ----------------------------------------------------------------------
# The session header
# ----------------------------------------------------------------------


def _wired(tmp_path: Path, **meta):
    writer = TranscriptWriter(tmp_path, meta=meta)
    assert writer.start()
    return writer, TranscriptStore(journal=writer)


def _say(store: TranscriptStore) -> None:
    store.append([Word(text="привет", start=0.0, end=0.5)])


def test_update_meta_corrects_the_header_after_a_fallback(tmp_path: Path):
    writer, store = _wired(tmp_path, engine="nemotron", model="nvidia/x")
    _say(store)
    # …two seconds later the loader finds out what really happened.
    writer.update_meta(engine="whisper", model="large-v3-turbo", engine_requested="nemotron")
    writer.close()

    path = next(tmp_path.glob("*.jsonl"))
    meta, entries = read_session(path)
    assert meta["engine"] == "whisper"
    assert meta["engine_requested"] == "nemotron"
    assert meta["model"] == "large-v3-turbo"
    # The words are untouched by a header correction.
    assert [e.word.text for e in entries] == ["привет"]


def test_correction_keeps_the_start_time(tmp_path: Path):
    writer, store = _wired(tmp_path, engine="nemotron")
    _say(store)
    started = json.loads(next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    writer.update_meta(engine="whisper")
    writer.close()

    meta, _ = read_session(next(tmp_path.glob("*.jsonl")))
    assert meta["started"] == started["started"]


def test_update_meta_before_recording_is_kept_for_the_first_write(tmp_path: Path):
    writer = TranscriptWriter(tmp_path, meta={"engine": "nemotron"})
    writer.update_meta(engine="whisper")
    assert writer.start()
    _say(TranscriptStore(journal=writer))
    writer.close()
    meta, _ = read_session(next(tmp_path.glob("*.jsonl")))
    assert meta["engine"] == "whisper"


def test_update_meta_with_nothing_writes_nothing(tmp_path: Path):
    writer, store = _wired(tmp_path, engine="whisper")
    _say(store)
    before = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
    writer.update_meta()
    writer.close()
    assert next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8") == before


def test_export_header_would_name_the_engine_that_ran(tmp_path: Path):
    """The header is generated from meta, so the correction has to reach it."""
    writer, store = _wired(tmp_path, engine="nemotron", model="nvidia/x")
    _say(store)
    writer.update_meta(engine="whisper", model="large-v3-turbo")
    writer.close()

    meta, entries = read_session(next(tmp_path.glob("*.jsonl")))
    out = export_markdown([e.word for e in entries], tmp_path / "out.md", meta=meta)
    text = out.read_text(encoding="utf-8")
    assert "whisper" in text
    assert "nemotron" not in text
