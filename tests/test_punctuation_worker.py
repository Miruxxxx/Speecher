import threading

from asr.events import Word
from asr.punctuation_worker import PunctuationWorker
from store.transcript_store import TranscriptStore


class FakeBackend:
    name = "fake"

    def __init__(self, transform=None, load_error=None):
        self._transform = transform
        self._load_error = load_error
        self.load_calls = 0
        self.restore_calls = 0

    def load(self):
        self.load_calls += 1
        if self._load_error:
            raise self._load_error

    def restore(self, text):
        self.restore_calls += 1
        return self._transform(text) if self._transform else text


def _store_with(*texts):
    store = TranscriptStore()
    store.append([Word(text=t, start=i, end=i + 0.5) for i, t in enumerate(texts)])
    return store


def _worker(store, backend):
    return PunctuationWorker(
        store=store, backend=backend, stop_event=threading.Event(),
        interval_sec=0.01, context_words=80,
    )


def test_tick_applies_punctuation_preserving_word_count():
    store = _store_with("привет", "как", "дела")
    backend = FakeBackend(transform=lambda t: "Привет, как дела?")
    worker = _worker(store, backend)
    backend.load()
    worker._tick()
    assert store.all_text() == "Привет, как дела?"


def test_tick_skips_on_word_count_mismatch():
    store = _store_with("one", "two")
    backend = FakeBackend(transform=lambda t: "one two three")
    worker = _worker(store, backend)
    backend.load()
    worker._tick()
    assert store.all_text() == "one two"  # unchanged


def test_tick_strips_previous_punctuation_before_restore():
    store = _store_with("Привет,", "мир!")
    seen = {}
    backend = FakeBackend(transform=lambda t: seen.setdefault("input", t) or t)
    worker = _worker(store, backend)
    backend.load()
    worker._tick()
    assert seen["input"] == "Привет мир"  # edges stripped, case kept


def test_tick_reasserts_whisper_question_mark():
    # Backend (silero-like) drops the "?" and ends the sentence with ".".
    store = _store_with("во", "сколько", "начнётся", "встреча?")
    backend = FakeBackend(transform=lambda t: t + ".")
    worker = _worker(store, backend)
    backend.load()
    worker._tick()
    assert store.all_text() == "во сколько начнётся встреча?"


def test_question_mark_reassertion_is_idempotent():
    store = _store_with("сколько", "это", "стоит?")
    backend = FakeBackend(transform=lambda t: t + ".")
    worker = _worker(store, backend)
    backend.load()
    worker._tick()
    assert store.all_text() == "сколько это стоит?"
    worker._last_seen_size = -1  # force a re-run over the now-punctuated words
    worker._tick()
    assert store.all_text() == "сколько это стоит?"


def test_keep_question_mark_helper():
    k = PunctuationWorker._keep_question_mark
    assert k("встреча?", "встреча.") == "встреча?"  # "." → "?"
    assert k("встреча?", "встреча") == "встреча?"    # nothing → "?"
    assert k("встреча?", "встреча?") == "встреча?"   # already "?"
    assert k("дела", "дела?") == "дела?"             # model-added "?" left alone
    assert k("мир", "мир.") == "мир."                # statements untouched


def test_tick_skips_when_store_unchanged():
    store = _store_with("hello", "world")
    backend = FakeBackend()
    worker = _worker(store, backend)
    backend.load()
    worker._tick()
    worker._tick()  # no new words appended in between
    assert backend.restore_calls == 1


def test_run_disables_itself_when_backend_load_fails():
    store = _store_with("a", "b")
    backend = FakeBackend(load_error=RuntimeError("no model"))
    worker = _worker(store, backend)
    worker.run()  # must return quietly, not raise
    assert backend.load_calls == 1
    assert backend.restore_calls == 0


def test_run_stops_on_stop_event():
    store = _store_with("a", "b")
    backend = FakeBackend()
    stop = threading.Event()
    worker = PunctuationWorker(store=store, backend=backend, stop_event=stop,
                               interval_sec=0.01, context_words=10)
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
