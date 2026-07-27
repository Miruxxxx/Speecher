import pytest

from asr.punctuation_backends import SileroBackend, create_backend


class _FakeModel:
    def __init__(self):
        self.seen = None

    def enhance_text(self, text, lang):
        self.seen = (text, lang)
        return text  # echo back so we can inspect what silero received


def test_silero_lowercases_input_before_model():
    # silero-te corrupts capitalized input ("Мы" -> "&Ы") / raises IndexError,
    # and the worker feeds it its own prior (capitalized) output; restore must
    # lowercase first.
    b = SileroBackend(language="ru")
    b._model = _FakeModel()
    b.restore("Мы Обсудили ПЛАН")
    assert b._model.seen == ("мы обсудили план", "ru")


def test_create_backend_off_returns_none():
    assert create_backend("off", "ru") is None
    assert create_backend("", "ru") is None


def test_create_backend_silero():
    b = create_backend("silero", "en")
    assert isinstance(b, SileroBackend)
    assert b._language == "en"


def test_create_backend_unknown_returns_none():
    assert create_backend("bogus", "ru") is None


def test_create_backend_deepmultilingual_is_gone():
    # Removed in phase 3 (no Russian, and its package drags transformers below
    # the >=5.13 nemotron needs). An old config naming it must degrade to
    # "punctuation off", not crash.
    assert create_backend("deepmultilingual", "ru") is None


# -- language follows the speech -------------------------------------------


def test_auto_language_follows_the_script_of_the_window():
    """A fixed `language` is silently harmful once `asr.language = "auto"`.

    The user's config said "ru" while they listened to English: silero would
    then run the Russian model over English text. The script is the one signal
    available without another model, and it separates the case that corrupts
    words (Cyrillic vs Latin) from the case that only costs commas (en vs de).
    """
    from asr.punctuation_backends import SileroBackend

    b = SileroBackend(language="auto")
    assert b.language_for("привет как дела вообще") == "ru"
    assert b.language_for("how often have you had that dream") == "en"
    # Digits and punctuation alone say nothing; English is the safer default
    # because it is what the Latin branch mostly sees.
    assert b.language_for("12:30 — 15%") == "en"


def test_a_pinned_language_is_never_second_guessed():
    from asr.punctuation_backends import SileroBackend

    b = SileroBackend(language="de")
    assert b.language_for("привет как дела") == "de"
    assert b.language_for("how are you") == "de"


def test_auto_is_accepted_by_the_factory_and_by_load_validation():
    from asr.punctuation_backends import SileroBackend, create_backend

    assert isinstance(create_backend("silero", "auto"), SileroBackend)
    # A real language still validates; an invented one still raises. `load()`
    # is where that happens, so check the guard directly rather than importing
    # torch here.
    assert SileroBackend(language="auto")._language == "auto"
    with pytest.raises(ValueError):
        b = SileroBackend(language="klingon")
        b.load()
