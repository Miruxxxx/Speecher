"""Journal round-trips: whatever the store ends up holding, a replay of its
JSONL must reproduce — including punctuation applied after the fact."""

from __future__ import annotations

import json

from asr.events import Word
from store import session_io
from store.transcript_store import TranscriptStore
from store.transcript_writer import TranscriptWriter
from translate.segments import Segment
from translate.store import TRANSLATED, TranslationStore


def _w(text: str, start: float = 0.0, end: float = 0.0) -> Word:
    return Word(text=text, start=start, end=end)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _wired(tmp_path, **meta):
    writer = TranscriptWriter(tmp_path, meta=meta or {"engine": "nemotron"})
    assert writer.start()
    return writer, TranscriptStore(journal=writer)


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------


def test_meta_is_first_line_and_words_follow(tmp_path):
    writer, store = _wired(tmp_path, engine="nemotron", model="m")
    store.append([_w("привет", 0.1, 0.4), _w("мир", 0.5, 0.9)])
    writer.close()

    recs = _lines(writer.path)
    assert recs[0]["t"] == "meta"
    assert recs[0]["version"] == 1
    assert recs[0]["engine"] == "nemotron"
    assert [r["text"] for r in recs if r["t"] == "w"] == ["привет", "мир"]
    assert recs[1]["s"] == 0.1 and recs[1]["e"] == 0.4


def test_blank_words_are_not_journaled(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("a"), _w("   "), _w("b")])
    writer.close()
    assert [r["text"] for r in _lines(writer.path) if r["t"] == "w"] == ["a", "b"]


def test_empty_session_removes_its_own_file(tmp_path):
    writer = TranscriptWriter(tmp_path)
    assert writer.start()
    path = writer.path
    assert path.exists()
    writer.close()
    assert not path.exists()


def test_pause_stops_writing_and_start_continues_same_file(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("один")])
    path = writer.path

    writer.pause()
    assert not writer.enabled
    store.append([_w("невидимый")])  # dropped: recording is off

    assert writer.start()
    assert writer.path == path, "resuming must continue the same session file"
    store.append([_w("два")])
    writer.close()

    assert [r["text"] for r in _lines(path) if r["t"] == "w"] == ["один", "два"]
    # The store keeps everything; only the journal has the gap.
    assert store.all_text() == "один невидимый два"


def test_write_failure_disables_journal_and_reports_once(tmp_path):
    errors: list[str] = []
    writer = TranscriptWriter(tmp_path, on_error=errors.append)
    assert writer.start()
    store = TranscriptStore(journal=writer)

    writer._fh.close()  # simulate the handle dying under us
    store.append([_w("одно")])
    store.append([_w("другое")])  # already disabled, must stay silent

    assert writer.failed and not writer.enabled
    assert len(errors) == 1
    # A broken journal must not cost the store any words.
    assert store.all_text() == "одно другое"


def test_error_handler_installed_late_still_gets_the_message(tmp_path):
    writer = TranscriptWriter(tmp_path / "nope" / "\0bad")
    writer.start()
    assert writer.failed
    seen: list[str] = []
    writer.set_error_handler(seen.append)
    assert len(seen) == 1


def test_disabled_journal_never_writes(tmp_path):
    writer = TranscriptWriter(tmp_path)
    store = TranscriptStore(journal=writer)  # start() never called
    store.append([_w("тишина")])
    assert writer.path is None
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
# Patches: punctuation applied after the word was committed
# ----------------------------------------------------------------------


def test_amend_is_journaled_as_resolved_index(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("встреча", 1.0, 1.5), _w("сегодня", 3.0, 3.4)])
    # Nemotron's late '?' belongs to the earlier word by timestamp.
    store.attach_punct_at(1.5, "?")
    writer.close()

    patches = [r for r in _lines(writer.path) if r["t"] == "patch"]
    assert patches == [{"t": "patch", "i": 0, "text": "встреча?"}]


def test_punctuation_worker_rewrites_journal_only_changed_words(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("привет"), _w("мир"), _w("как")])
    store.update_word_texts(0, ["Привет,", "мир", "как"])  # only index 0 differs
    writer.close()

    patches = [r for r in _lines(writer.path) if r["t"] == "patch"]
    assert patches == [{"t": "patch", "i": 0, "text": "Привет,"}]


def test_append_to_last_word_is_journaled(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("да"), _w("конечно")])
    store.append_to_last_word(".")
    writer.close()

    patches = [r for r in _lines(writer.path) if r["t"] == "patch"]
    assert patches == [{"t": "patch", "i": 1, "text": "конечно."}]


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------


def test_replay_reproduces_the_store_text(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("это", 0.0, 0.3), _w("вопрос", 0.4, 0.9)])
    store.attach_punct_at(0.9, "?")
    store.append([_w("да", 2.0, 2.2)])
    store.update_word_texts(0, ["Это", "вопрос?", "да"])
    writer.close()

    meta, entries = session_io.read_session(writer.path)
    assert meta["engine"] == "nemotron"
    assert " ".join(e.word.text for e in entries) == store.all_text()
    assert entries[1].word.start == 0.4 and entries[1].word.end == 0.9
    assert all(e.received_at > 0 for e in entries)


def test_replay_skips_broken_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"t":"meta","version":1}\n'
        '{"t":"w","text":"ок","s":0,"e":1,"at":5}\n'
        '{"t":"w","text":"обрез\n'          # truncated by a crash mid-line
        '{"t":"patch","i":99,"text":"x"}\n'  # index that never existed
        '{"t":"w","text":"дальше","s":1,"e":2,"at":6}\n',
        encoding="utf-8",
    )
    _, entries = session_io.read_session(path)
    assert [e.word.text for e in entries] == ["ок", "дальше"]


def test_resumed_words_keep_their_original_receive_time(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("старое", 0.0, 0.5)])
    writer.close()
    _, entries = session_io.read_session(writer.path)

    # Age the replayed entry: a resumed session must not look like fresh speech
    # to `words_since_minutes` (that is what Sum 5min reads).
    aged = [(e.word, e.received_at - 3600.0) for e in entries]
    fresh_store = TranscriptStore()
    assert fresh_store.load_entries(aged) == 1
    fresh_store.append([_w("новое")])
    assert [w.text for w in fresh_store.words_since_minutes(5.0)] == ["новое"]
    assert fresh_store.all_text() == "старое новое"


def test_resumed_words_are_journaled_into_the_new_session(tmp_path):
    writer, store = _wired(tmp_path)
    store.load_entries([(_w("перенос", 1.0, 1.4), 111.0)])
    writer.close()
    recs = [r for r in _lines(writer.path) if r["t"] == "w"]
    assert recs == [{"t": "w", "text": "перенос", "s": 1.0, "e": 1.4, "at": 111.0}]


def test_latest_session_picks_the_newest_name(tmp_path):
    for name in ("2026-07-01_100000.jsonl", "2026-07-24_235959.jsonl", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert session_io.latest_session(tmp_path).name == "2026-07-24_235959.jsonl"
    assert session_io.latest_session(tmp_path / "missing") is None


def test_snapshot_returns_every_word_in_order(tmp_path):
    store = TranscriptStore()
    store.append([_w("a"), _w("b")])
    store.append([_w("c")])
    assert [w.text for w in store.snapshot()] == ["a", "b", "c"]


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def _spoken() -> list[Word]:
    # Two utterances separated by a 4 s pause -> two paragraphs.
    return [
        _w("Первая", 0.0, 0.5),
        _w("фраза.", 0.6, 1.1),
        _w("Вторая", 5.0, 5.4),
        _w("фраза.", 5.5, 6.0),
    ]


def test_export_text_splits_paragraphs_on_pauses(tmp_path):
    out = session_io.export_text(_spoken(), tmp_path / "t.txt")
    assert out.read_text(encoding="utf-8") == "Первая фраза.\n\nВторая фраза.\n"


def test_export_markdown_has_header_and_timestamps(tmp_path):
    out = session_io.export_markdown(
        _spoken(),
        tmp_path / "t.md",
        {"started": "2026-07-24T19:30:00", "engine": "nemotron", "model": "m"},
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Транскрипт 2026-07-24 19:30:00")
    assert "движок: `nemotron`" in text
    assert "слов: 4" in text
    assert "**[00:00]** Первая фраза." in text
    assert "**[00:05]** Вторая фраза." in text


def test_export_markdown_uses_hours_past_one_hour(tmp_path):
    out = session_io.export_markdown([_w("поздно", 3725.0, 3725.5)], tmp_path / "t.md")
    assert "**[1:02:05]** поздно" in out.read_text(encoding="utf-8")


def test_export_handles_empty_transcript(tmp_path):
    assert session_io.export_text([], tmp_path / "e.txt").read_text(encoding="utf-8") == ""
    md = session_io.export_markdown([], tmp_path / "e.md").read_text(encoding="utf-8")
    assert "слов: 0" in md


def test_translations_share_the_session_file(tmp_path):
    """The `tr` record lives in the same journal and survives a replay."""
    writer, store = _wired(tmp_path)
    store.append([_w("hello", 0.0, 0.4), _w("there", 0.5, 0.9)])
    translations = TranslationStore(target="ru", journal=writer)
    translations.add(
        Segment(index=0, n_words=2, at=1.0, start_sec=0.0, end_sec=0.9, text="hello there"),
        "привет", TRANSLATED,
    )
    writer.close()

    recs = _lines(writer.path)
    assert {"t": "tr", "i": 0, "n": 2, "lang": "ru", "text": "привет"} in recs
    # A reader that does not know the record still gets the words.
    _, entries = session_io.read_session(writer.path)
    assert [e.word.text for e in entries] == ["hello", "there"]
    assert session_io.read_translations(writer.path) == {0: (2, "привет")}


def test_replayed_translations_take_their_times_from_the_words(tmp_path):
    """The journal does not repeat timestamps — the words already carry them."""
    writer, store = _wired(tmp_path)
    store.append([_w("hello", 3.0, 3.4), _w("there", 3.5, 3.9)])
    translations = TranslationStore(target="ru", journal=writer)
    translations.add(
        Segment(index=0, n_words=2, at=1.0, start_sec=3.0, end_sec=3.9, text="hello there"),
        "привет", TRANSLATED,
    )
    writer.close()

    _, entries = session_io.read_session(writer.path)
    pairs = [(e.word, e.received_at) for e in entries]
    replayed = TranslationStore()
    assert replayed.load(session_io.read_translations(writer.path), pairs) == 1
    row = replayed.rows()[0]
    assert (row.start_sec, row.end_sec) == (3.0, 3.9)
    assert row.at == entries[0].received_at
    assert replayed.next_index() == 2


def test_read_translations_of_a_session_without_any(tmp_path):
    writer, store = _wired(tmp_path)
    store.append([_w("слово", 0.0, 0.4)])
    writer.close()
    assert session_io.read_translations(writer.path) == {}


def test_export_markdown_glosses_paragraphs_with_translations(tmp_path):
    out = session_io.export_markdown(
        _spoken(),
        tmp_path / "t.md",
        {"translate_target": "ru"},
        # Keyed by first word index; the second paragraph holds two segments.
        translations={0: "First phrase.", 2: "Second", 3: "phrase."},
    )
    text = out.read_text(encoding="utf-8")
    assert "перевод: `ru`" in text
    assert "**[00:00]** Первая фраза.\n\n> First phrase." in text
    assert "**[00:05]** Вторая фраза.\n\n> Second phrase." in text


def test_export_markdown_without_translations_is_unchanged(tmp_path):
    text = session_io.export_markdown(_spoken(), tmp_path / "t.md").read_text(
        encoding="utf-8"
    )
    assert "\n> " not in text  # the export's own HTML comment ends in ">"


def test_suggest_export_name_follows_the_session_file(tmp_path):
    source = tmp_path / "2026-07-24_193000.jsonl"
    assert session_io.suggest_export_name({}, source, "md") == "2026-07-24_193000.md"
    name = session_io.suggest_export_name({"started": "2026-07-24T19:30:00"}, None, "txt")
    assert name.endswith(".txt") and "2026" in name
