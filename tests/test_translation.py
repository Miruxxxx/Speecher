"""Live translation: segmentation, the append-only feed, the worker's loop and
the language heuristics. No model is loaded — the backend is a fake that speaks
the same protocol, exactly as tests/fake_capture.py does for the capture binary.

The rule most of these tests defend: **a line that reached the feed never
changes again**. The first version of this feature mutated transcript lines in
place (original -> dimmed -> translation) and the result was unreadable.
"""

from __future__ import annotations

import threading
import time

import pytest

from asr.events import Word
from store.transcript_store import TranscriptStore
from translate import languages
from translate.backends import split_sentences
from translate.segments import cut, merge
from translate.store import FAILED, SKIPPED, TRANSLATED, TranslationStore
from translate.worker import TranslationWorker


class FakeBackend:
    """Marks the text instead of translating it, and can be told to fail."""

    name = "fake"

    def __init__(self, fail: bool = False, empty: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.loaded = False
        self._fail = fail
        self._empty = empty

    def load(self) -> None:
        self.loaded = True

    def translate(self, text: str, source: str, target: str) -> str:
        self.calls.append((text, source, target))
        if self._fail:
            raise RuntimeError("backend is down")
        if self._empty:
            return ""
        return f"[{target}] {text}"


def _entries(*spec: tuple[str, float]) -> list[tuple[Word, float]]:
    """(text, start_sec) pairs -> store entries; every word is 0.3 s long."""
    return [
        (Word(text=text, start=start, end=start + 0.3), 1_700_000_000.0 + start)
        for text, start in spec
    ]


def _store(*spec: tuple[str, float]) -> TranscriptStore:
    store = TranscriptStore()
    for word, _received in _entries(*spec):
        store.append([word])
    return store


def _worker(store: TranscriptStore, translations: TranslationStore, backend, **kw):
    kw.setdefault("target", "ru")
    kw.setdefault("close_after_sec", 0.0)  # nothing is "still growing" in a test
    return TranslationWorker(
        store=store,
        translations=translations,
        backend=backend,
        stop_event=threading.Event(),
        **kw,
    )


# -- segmentation -----------------------------------------------------------


def test_a_sentence_is_a_segment():
    segments = cut(_entries(("Hello", 0.0), ("there.", 0.4), ("Next", 0.8)), 0,
                   tail_closed=False)
    assert [s.text for s in segments] == ["Hello there."]
    assert segments[0].index == 0 and segments[0].n_words == 2


def test_a_pause_ends_a_segment_without_punctuation():
    segments = cut(_entries(("one", 0.0), ("two", 0.4), ("three", 5.0)), 0,
                   tail_closed=False)
    assert [s.text for s in segments] == ["one two"]


def test_the_growing_tail_is_not_cut_until_the_audio_goes_quiet():
    entries = _entries(("still", 0.0), ("speaking", 0.4))
    assert cut(entries, 0, tail_closed=False) == []
    closed = cut(entries, 0, tail_closed=True)
    assert [s.text for s in closed] == ["still speaking"]


def test_speech_without_punctuation_or_pauses_still_reaches_the_screen():
    entries = _entries(*[(f"w{i}", i * 0.4) for i in range(25)])
    segments = cut(entries, 0, max_words=10, tail_closed=False)
    assert [s.n_words for s in segments] == [10, 10]


def test_indices_are_absolute_and_contiguous():
    """The worker resumes by index; a gap or an overlap would repeat speech."""
    entries = _entries(("a.", 0.0), ("b.", 0.4), ("c.", 0.8))
    segments = cut(entries, 40, tail_closed=True)
    assert [(s.index, s.n_words) for s in segments] == [(40, 1), (41, 1), (42, 1)]
    assert segments[-1].next_index == 43


def test_merge_keeps_the_span_and_the_order():
    segments = cut(_entries(("a.", 0.0), ("b.", 0.4), ("c.", 0.8)), 7, tail_closed=True)
    one = merge(segments)
    assert one.index == 7 and one.n_words == 3
    assert one.text == "a. b. c."
    assert one.start_sec == segments[0].start_sec
    assert one.end_sec == segments[-1].end_sec


# -- languages --------------------------------------------------------------


def test_script_detection_separates_the_alphabets():
    assert languages.detect_script("привет как дела") == "cyrillic"
    assert languages.detect_script("hello there") == "latin"
    assert languages.detect_script("12:30 — 15%") == ""
    assert languages.looks_like("привет", "ru")
    assert not languages.looks_like("hello", "ru")


def test_locales_normalize_to_table_codes():
    assert languages.normalize_code("en-US") == "en"
    assert languages.normalize_code("RU") == "ru"
    # "auto" is not a language; callers read the empty string as "unknown".
    assert languages.normalize_code("auto") == ""
    assert languages.normalize_code("klingon") == ""


def test_long_segment_is_cut_into_sentences_without_losing_words():
    text = "Первое предложение. Второе предложение! И третье?"
    assert split_sentences(text, 220) == [
        "Первое предложение.", "Второе предложение!", "И третье?"
    ]
    long_one = " ".join(["слово"] * 100)
    pieces = split_sentences(long_one, 60)
    assert len(pieces) > 1
    assert all(len(p) <= 60 for p in pieces)
    assert " ".join(pieces) == long_one


# -- the feed ---------------------------------------------------------------


def test_rows_are_append_only_and_final():
    store = _store(("Hello.", 0.0), ("Bye.", 5.0))
    translations = TranslationStore()
    worker = _worker(store, translations, FakeBackend())

    worker._tick()
    first = translations.replies()
    assert [r.text for r in first] == ["[ru] Hello.", "[ru] Bye."]

    # Nothing the pipeline does afterwards may rewrite a line that is on screen.
    store.append([Word(text="More.", start=9.0, end=9.3)])
    worker._tick()
    after = translations.replies()
    assert [r.text for r in after[:2]] == [r.text for r in first]
    assert after[2].text == "[ru] More."


def test_a_line_carries_the_timestamp_of_its_first_word():
    store = _store(("Hello", 4.0), ("there.", 4.4))
    translations = TranslationStore()
    _worker(store, translations, FakeBackend())._tick()

    row = translations.rows()[0]
    assert row.start_sec == 4.0 and row.end_sec == 4.7
    assert row.at == translations.replies()[0].at


def test_next_index_walks_forward_and_never_repeats():
    store = _store(("a.", 0.0), ("b.", 1.0))
    translations = TranslationStore()
    worker = _worker(store, translations, FakeBackend())

    worker._tick()
    assert translations.next_index() == 2
    worker._tick()  # nothing new
    assert translations.size() == 2


def test_translating_never_consumes_the_transcript():
    """The source window renders the store, so translation must not empty it.

    An earlier version fed that window only the words the model had not reached
    yet, and it wiped itself every couple of seconds. The store keeps every
    word regardless of how far the worker has walked.
    """
    store = _store(("Done.", 0.0), ("Still", 5.0), ("going", 5.4))
    translations = TranslationStore()
    _worker(store, translations, FakeBackend(), close_after_sec=60.0)._tick()

    assert [r.text for r in translations.replies()] == ["[ru] Done."]
    assert [w.text for w, _ in store.recent_entries(400)] == ["Done.", "Still", "going"]


# -- worker -----------------------------------------------------------------


def test_target_language_text_is_never_sent_to_the_model():
    store = _store(("Привет.", 0.0), ("Hello.", 5.0))
    translations = TranslationStore()
    backend = FakeBackend()
    _worker(store, translations, backend, target="ru")._tick()

    rows = translations.rows()
    assert rows[0].state == SKIPPED and rows[0].text == "Привет."
    assert backend.calls == [("Hello.", "en", "ru")]
    assert rows[1].state == TRANSLATED


def test_a_configured_source_beats_the_script_guess():
    store = _store(("Guten", 0.0), ("Tag.", 0.4))
    backend = FakeBackend()
    _worker(_store(("Guten", 0.0), ("Tag.", 0.4)), TranslationStore(), backend,
            source="de", target="ru")._tick()
    assert backend.calls == [("Guten Tag.", "de", "ru")]
    _ = store


def test_same_script_guard_is_off_when_the_source_is_known():
    """de -> en is Latin to Latin; the guard would have skipped every line."""
    backend = FakeBackend()
    _worker(_store(("Guten", 0.0), ("Tag.", 0.4)), TranslationStore(), backend,
            source="de", target="en")._tick()
    assert backend.calls == [("Guten Tag.", "de", "en")]


def test_the_asr_language_is_used_when_the_config_says_auto():
    backend = FakeBackend()
    _worker(_store(("Guten", 0.0), ("Tag.", 0.4)), TranslationStore(), backend,
            source="auto", asr_language="de")._tick()
    assert backend.calls == [("Guten Tag.", "de", "ru")]


def test_backlog_is_merged_not_dropped():
    """Falling behind costs one longer line, never a hole in the feed."""
    store = _store(*[(f"w{i}.", i * 2.0) for i in range(6)])
    translations = TranslationStore()
    backend = FakeBackend()
    worker = _worker(store, translations, backend, max_pending=2)

    worker._step()
    assert backend.calls == [("w0. w1. w2. w3. w4.", "en", "ru")]
    assert translations.next_index() == 5
    worker._step()
    assert translations.rows()[-1].text == "[ru] w5."
    # Every word is on the feed exactly once, in order.
    assert translations.next_index() == 6


@pytest.mark.parametrize("backend", [FakeBackend(fail=True), FakeBackend(empty=True)])
def test_a_broken_backend_shows_the_original_instead_of_a_hole(backend):
    store = _store(("Hello.", 0.0))
    translations = TranslationStore()
    _worker(store, translations, backend)._tick()

    row = translations.rows()[0]
    assert row.state == FAILED and row.text == "Hello."


def test_a_backend_that_cannot_load_disables_the_feature_not_the_app():
    class Broken(FakeBackend):
        def load(self) -> None:
            raise RuntimeError("no such model")

    seen: list[tuple[str, str]] = []
    worker = TranslationWorker(
        store=_store(("Hello.", 0.0)),
        translations=TranslationStore(),
        backend=Broken(),
        stop_event=threading.Event(),
        on_status=lambda text, detail="": seen.append((text, detail)),
    )
    worker.run()  # returns instead of looping

    assert not worker.enabled
    assert "no such model" in worker.failure
    assert seen and seen[0][0] == "Перевод"


def test_disabled_worker_does_not_touch_the_model():
    store = _store(("Hello.", 0.0))
    translations = TranslationStore()
    backend = FakeBackend()
    stop = threading.Event()
    worker = TranslationWorker(
        store=store,
        translations=translations,
        backend=backend,
        stop_event=stop,
        close_after_sec=0.0,
        poll_sec=0.01,
        enabled=False,
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    time.sleep(0.1)
    assert backend.calls == []

    worker.set_enabled(True)
    deadline = time.time() + 2.0
    while not backend.calls and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=2.0)
    assert backend.calls == [("Hello.", "en", "ru")]


# -- journal ----------------------------------------------------------------


def test_rows_are_journaled_by_index_and_replayed():
    records: list[dict] = []

    class Journal:
        def record(self, rec: dict) -> None:
            records.append(rec)

    store = _store(("Hello", 0.0), ("there.", 0.4))
    translations = TranslationStore(target="ru", journal=Journal())
    _worker(store, translations, FakeBackend())._tick()

    assert records == [
        {"t": "tr", "i": 0, "n": 2, "lang": "ru", "text": "[ru] Hello there."}
    ]

    replayed = TranslationStore()
    assert replayed.load({0: (2, "[ru] Hello there.")}, _entries(("Hello", 0.0), ("there.", 0.4))) == 1
    row = replayed.rows()[0]
    assert row.text == "[ru] Hello there." and row.n_words == 2
    assert replayed.next_index() == 2


def test_a_skipped_line_is_not_journaled():
    records: list[dict] = []

    class Journal:
        def record(self, rec: dict) -> None:
            records.append(rec)

    store = _store(("Привет.", 0.0))
    translations = TranslationStore(target="ru", journal=Journal())
    _worker(store, translations, FakeBackend())._tick()
    # The line is the original: replaying it would add nothing the words do not
    # already say.
    assert records == []


def test_skipped_streak_counts_only_the_newest_run():
    """A feed filling with pass-through lines looks like a broken translator;
    the overlay uses this to say the speech is already in the target language."""
    from translate.segments import Segment
    from translate.store import FAILED, SKIPPED, TRANSLATED, TranslationStore

    store = TranslationStore("ru")

    def add(state, index):
        segment = Segment(
            index=index, n_words=1, at=0.0, start_sec=0.0, end_sec=1.0, text="x"
        )
        store.add(segment, f"строка {index}", state)

    assert store.skipped_streak() == 0
    add(TRANSLATED, 0)
    assert store.skipped_streak() == 0
    add(SKIPPED, 1)
    add(SKIPPED, 2)
    assert store.skipped_streak() == 2
    # Anything that is not a pass-through ends the run.
    add(FAILED, 3)
    assert store.skipped_streak() == 0
