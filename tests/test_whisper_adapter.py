from types import SimpleNamespace

import numpy as np

from asr.whisper_adapter import WhisperAdapter


def _word(text, start=0.0, end=0.5):
    return SimpleNamespace(word=text, start=start, end=end)


def _segment(words, no_speech_prob=0.0, avg_logprob=-0.3):
    return SimpleNamespace(words=words, no_speech_prob=no_speech_prob,
                           avg_logprob=avg_logprob)


def _info(language="en", probability=0.95):
    return SimpleNamespace(language=language, language_probability=probability)


class FakeModel:
    def __init__(self, results):
        # results: list of (segments, info) returned per call
        self.results = list(results)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


AUDIO = np.zeros(16000, dtype=np.float32)


def test_extracts_words_and_clamps_bad_timestamps():
    model = FakeModel([([
        _segment([_word(" hello ", 0.0, 0.4), _word("world", 1.0, 0.2)]),
    ], _info())])
    adapter = WhisperAdapter(model, language="en")
    words = adapter(AUDIO)
    assert [w.text for w in words] == ["hello", "world"]
    assert words[1].end == words[1].start  # end < start clamped


def test_drops_segments_by_no_speech_prob_and_logprob():
    model = FakeModel([([
        _segment([_word("real")], no_speech_prob=0.1, avg_logprob=-0.3),
        _segment([_word("ghost1")], no_speech_prob=0.95, avg_logprob=-0.3),
        _segment([_word("ghost2")], no_speech_prob=0.1, avg_logprob=-2.5),
    ], _info())])
    adapter = WhisperAdapter(model, language="en",
                             no_speech_prob_max=0.9, avg_logprob_min=-1.5)
    words = adapter(AUDIO)
    assert [w.text for w in words] == ["real"]


def test_collapses_hallucination_repeats():
    repeats = [_word("Thanks.", i * 0.1, i * 0.1 + 0.05) for i in range(12)]
    model = FakeModel([([_segment([_word("ok")] + repeats)], _info())])
    adapter = WhisperAdapter(model, language="en", max_word_repeats=4)
    words = adapter(AUDIO)
    assert [w.text for w in words] == ["ok"] + ["Thanks."] * 4


def test_sticky_language_pins_after_confident_decode():
    model = FakeModel([
        ([_segment([_word("привет")])], _info(language="ru", probability=0.92)),
        ([_segment([_word("мир")])], _info(language="ru", probability=0.5)),
    ])
    adapter = WhisperAdapter(model, language=None, sticky_language=True,
                             sticky_min_probability=0.7)
    adapter(AUDIO)
    assert adapter.pinned_language == "ru"
    adapter(AUDIO)
    assert model.calls[0]["language"] is None
    assert model.calls[1]["language"] == "ru"


def test_sticky_language_not_pinned_on_low_confidence_or_empty():
    model = FakeModel([
        ([_segment([_word("hm")])], _info(language="de", probability=0.3)),
        ([_segment([])], _info(language="pt", probability=0.99)),  # no words
    ])
    adapter = WhisperAdapter(model, language=None, sticky_language=True,
                             sticky_min_probability=0.7)
    adapter(AUDIO)
    assert adapter.pinned_language is None
    adapter(AUDIO)
    assert adapter.pinned_language is None


def test_fixed_language_is_never_overridden():
    model = FakeModel([([_segment([_word("hi")])],
                        _info(language="ru", probability=0.99))])
    adapter = WhisperAdapter(model, language="en")
    adapter(AUDIO)
    assert model.calls[0]["language"] == "en"
    assert adapter.pinned_language == "en"
