"""Reply grouping for the feed's timestamp gutter (ui/transcript_model.py)."""

from __future__ import annotations

from asr.events import Word
from ui.transcript_model import PAUSE_SEC, group_entries, replies_from, tail_by_chars


def entries(*specs):
    """(text, start, end, received_at) tuples -> what the store keeps."""
    return [(Word(text=t, start=s, end=e), at) for t, s, e, at in specs]


def test_words_without_a_pause_are_one_reply():
    replies = group_entries(
        entries(("Согласен.", 0.0, 0.4, 100.0), ("Главное", 0.5, 0.9, 100.1))
    )
    assert len(replies) == 1
    assert replies[0].text == "Согласен. Главное"
    assert replies[0].at == 100.0  # the reply is stamped by its first word


def test_a_long_pause_starts_a_new_reply():
    replies = group_entries(
        entries(
            ("раз", 0.0, 0.4, 100.0),
            ("два", 0.4 + PAUSE_SEC + 0.1, 5.0, 102.0),
        )
    )
    assert [r.text for r in replies] == ["раз", "два"]
    assert replies[1].at == 102.0


def test_sentence_end_splits_only_past_the_soft_limit():
    long_sentence = entries(
        *[
            (f"слово{i}.", i * 0.5, i * 0.5 + 0.4, 100.0 + i * 0.5)
            for i in range(40)
        ]
    )
    replies = group_entries(long_sentence, soft_max_chars=100)
    assert len(replies) > 1
    # Nothing is lost or reordered by the split.
    assert " ".join(r.text for r in replies) == " ".join(
        w.text for w, _ in long_sentence
    )


def test_blank_words_are_ignored():
    replies = group_entries(entries(("  ", 0.0, 0.1, 100.0), ("да", 0.2, 0.4, 100.1)))
    assert [r.text for r in replies] == ["да"]


def test_tail_by_chars_keeps_the_newest_replies_in_order():
    replies = group_entries(
        entries(
            ("первое", 0.0, 0.4, 100.0),
            ("второе", 10.0, 10.4, 110.0),
            ("третье", 20.0, 20.4, 120.0),
        )
    )
    tail = tail_by_chars(replies, max_chars=8)
    assert [r.text for r in tail] == ["второе", "третье"]


def test_replies_from_is_grouping_plus_tail():
    data = entries(
        ("первое", 0.0, 0.4, 100.0),
        ("второе", 10.0, 10.4, 110.0),
    )
    assert replies_from(data, 1000) == group_entries(data)


def test_empty_input():
    assert group_entries([]) == []
