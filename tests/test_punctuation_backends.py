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
