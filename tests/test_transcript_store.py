from asr.events import Word
from store.transcript_store import TranscriptStore


def _w(text: str, start: float = 0.0, end: float = 0.0) -> Word:
    return Word(text=text, start=start, end=end)


def test_append_filters_blank_words():
    store = TranscriptStore()
    store.append([_w("hello"), _w("  "), _w("world")])
    assert store.size() == 2
    assert store.all_text() == "hello world"


def test_recent_text_returns_tail_chars():
    store = TranscriptStore()
    store.append([_w("a" * 50), _w("b" * 50)])
    text = store.recent_text(max_chars=30)
    assert len(text) == 30
    assert set(text) == {"b"}


def test_get_tail_returns_start_index_and_words():
    store = TranscriptStore()
    store.append([_w(f"w{i}") for i in range(10)])
    start, words = store.get_tail(3)
    assert start == 7
    assert [w.text for w in words] == ["w7", "w8", "w9"]

    start_all, words_all = store.get_tail(100)
    assert start_all == 0
    assert len(words_all) == 10


def test_update_word_texts_rewrites_in_place_preserving_times():
    store = TranscriptStore()
    store.append([_w("hello", 1.0, 2.0), _w("world", 2.0, 3.0)])
    store.update_word_texts(0, ["Hello,", "world!"])
    assert store.all_text() == "Hello, world!"
    _, words = store.get_tail(2)
    assert words[0].start == 1.0 and words[0].end == 2.0


def test_update_word_texts_out_of_range_is_noop():
    store = TranscriptStore()
    store.append([_w("hello")])
    store.update_word_texts(0, ["a", "b"])  # longer than store
    assert store.all_text() == "hello"


def test_find_last_question_extracts_last_sentence():
    store = TranscriptStore()
    for token in "First point. Then some words. What time is it? Sure thing.".split():
        store.append([_w(token)])
    assert store.find_last_question() == "What time is it?"


def test_find_last_question_none_without_question_mark():
    store = TranscriptStore()
    store.append([_w("no"), _w("questions"), _w("here.")])
    assert store.find_last_question() is None


def test_words_since_minutes_uses_receive_time():
    store = TranscriptStore()
    store.append([_w("old")])
    # Age the first entry artificially (receive time is store-internal).
    with store._lock:
        word, t = store._entries[0]
        store._entries[0] = (word, t - 600.0)
    store.append([_w("fresh")])

    recent = store.words_since_minutes(5.0)
    assert [w.text for w in recent] == ["fresh"]


def test_context_for_question_window_ends_at_question():
    store = TranscriptStore()
    for token in "one two three why not? four five".split():
        store.append([_w(token)])
    ctx = store.context_for_question("why not?", max_chars=100)
    assert ctx.endswith("why not?")
    assert "four" not in ctx
