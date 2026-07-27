import queue
import threading
from pathlib import Path

import pytest

from app_config import AppConfig, load_config
from asr.backends import create_asr_backend
from asr.nemotron_backend import WordAssembler
from asr.whisper_backend import WhisperBackend
from audio.buffer import GrowingAudioBuffer

FRAME = 0.08  # one subsampled encoder frame, seconds


def _assemble(pieces, frame_sec=FRAME, idle_flush_sec=100.0):
    """Feed (piece, frame) pairs and return every word, flush included."""
    a = WordAssembler(frame_sec, idle_flush_sec=idle_flush_sec)
    words = []
    for piece, frame in pieces:
        words.extend(a.push(piece, frame))
    words.extend(a.flush())
    return words


def _texts(words):
    return [w.text for w in words]


# -- token -> word grouping ----------------------------------------------


def test_leading_space_starts_a_new_word():
    words = _assemble([(" об", 0), ("суд", 1), ("им", 2), (" арх", 5), ("итектуру", 6)])
    assert _texts(words) == ["обсудим", "архитектуру"]


def test_lone_space_token_does_not_glue_words():
    # Dropping "empty" pieces is the classic bug: it yields "обсудимархитектуру".
    words = _assemble([(" обсудим", 0), (" ", 4), ("архитектуру", 5)])
    assert _texts(words) == ["обсудим", "архитектуру"]


def test_punctuation_sticks_to_the_previous_word():
    words = _assemble([(" встреча", 0), (",", 1), (" завтра", 4), (".", 5)])
    assert _texts(words) == ["встреча,", "завтра."]


def test_first_piece_without_leading_space_still_opens_a_word():
    words = _assemble([("Привет", 0), (" мир", 3)])
    assert _texts(words) == ["Привет", "мир"]


def test_orphan_punctuation_is_dropped():
    # The pause punctuation arrives after an idle flush already committed the
    # word it belongs to; alone it is not a word.
    a = WordAssembler(FRAME, idle_flush_sec=0.4)
    a.push(" сервиса", 0)
    assert _texts(a.advance(5)) == ["сервиса"]

    assert a.push(",", 40) == []
    assert a.flush() == []


def test_punctuation_after_a_lone_space_is_dropped_too():
    words = _assemble([(" слово", 0), (" ", 1), (".", 2)])
    assert _texts(words) == ["слово"]


def test_empty_pieces_are_ignored():
    words = _assemble([("", 0), (" слово", 1), ("", 2)])
    assert _texts(words) == ["слово"]


# -- timestamps -----------------------------------------------------------


def test_timestamps_are_stream_seconds_from_frame_indices():
    words = _assemble([(" раз", 10), ("два", 11), (" три", 25)])

    assert words[0].start == pytest.approx(10 * FRAME)
    # end is the emission point of the last piece + one frame, not the
    # acoustic end of the word.
    assert words[0].end == pytest.approx(12 * FRAME)
    assert words[1].start == pytest.approx(25 * FRAME)


def test_words_come_out_in_order_with_growing_timestamps():
    words = _assemble([(" a", 0), (" b", 7), (" c", 100)])
    starts = [w.start for w in words]
    assert starts == sorted(starts)


# -- flushing -------------------------------------------------------------


def test_word_is_emitted_only_once_the_next_one_starts():
    a = WordAssembler(FRAME, idle_flush_sec=100.0)

    assert a.push(" первое", 0) == []
    assert _texts(a.push(" второе", 5)) == ["первое"]
    assert _texts(a.flush()) == ["второе"]


def test_silence_closes_a_pending_word():
    a = WordAssembler(FRAME, idle_flush_sec=0.4)  # 5 frames
    a.push(" хвост", 10)

    assert a.advance(14) == []
    assert _texts(a.advance(15)) == ["хвост"]
    assert a.advance(30) == []  # nothing left pending


def test_flush_is_idempotent():
    a = WordAssembler(FRAME)
    a.push(" слово", 0)

    assert _texts(a.flush()) == ["слово"]
    assert a.flush() == []


# -- backend selection ----------------------------------------------------


def _cfg(engine: str) -> AppConfig:
    cfg = AppConfig()
    cfg.asr.engine = engine
    return cfg


def test_nemotron_engine_selects_the_nemotron_backend():
    from asr.nemotron_backend import NemotronBackend

    backend = create_asr_backend(
        _cfg("nemotron"),
        audio_buffer=GrowingAudioBuffer(sample_rate=16000),
        frame_sink=queue.Queue(),
    )

    assert isinstance(backend, NemotronBackend)
    assert backend.name == "nemotron"


def test_nemotron_without_frame_sink_falls_back_to_whisper():
    backend = create_asr_backend(
        _cfg("nemotron"), audio_buffer=GrowingAudioBuffer(sample_rate=16000)
    )

    assert isinstance(backend, WhisperBackend)


def test_engine_argument_overrides_the_config():
    backend = create_asr_backend(
        _cfg("nemotron"),
        audio_buffer=GrowingAudioBuffer(sample_rate=16000),
        frame_sink=queue.Queue(),
        engine="whisper",
    )

    assert isinstance(backend, WhisperBackend)


def test_run_before_load_raises():
    from asr.nemotron_backend import NemotronBackend

    backend = NemotronBackend(cfg=AppConfig().asr.nemotron, frame_sink=queue.Queue())

    with pytest.raises(RuntimeError):
        backend.run(events=queue.Queue(), stop_event=threading.Event())


# -- mel chunking ---------------------------------------------------------


class _FakeProcessor:
    """Stands in for AutoProcessor: records the blocks it was handed."""

    def __init__(self) -> None:
        self.blocks: list[int] = []

    def __call__(self, block, **_kw):
        self.blocks.append(len(block))
        return {"input_features": block}


def _chunker(frames: "queue.Queue", stop: threading.Event, *, need: int = 100):
    """A loaded-enough NemotronBackend to pull mel chunks from."""
    from asr.nemotron_backend import NemotronBackend

    backend = NemotronBackend(cfg=AppConfig().asr.nemotron, frame_sink=frames)
    backend._processor = _FakeProcessor()
    backend._first_chunk_samples = need
    backend._chunk_samples = need
    return backend, backend._mel_chunks(stop, _StaleSink())


class _StaleSink:
    def flush_if_stale(self, _sec: float) -> None:
        pass


class _StopOnGet(queue.Queue):
    """Delivers one chunk and stops the app in the same breath.

    This is the shutdown race exactly as it happens: the wait loop leaves
    because the chunk it just read satisfied `need`, and stop_event is set by
    the time the next line looks at it.
    """

    def __init__(self, chunk, stop: threading.Event) -> None:
        super().__init__()
        self._stop = stop
        self.put(chunk)

    def get(self, *a, **kw):
        item = super().get(*a, **kw)
        self._stop.set()
        return item


def test_shutdown_with_a_whole_chunk_buffered_ends_cleanly():
    """A full chunk in hand when stop arrives must not be padded to a negative
    length. Audio arrives in blocks bigger than one mel chunk, so `pending`
    routinely overshoots `need` — and then `np.zeros(need - len(pending))` is
    negative and raises, which run() reports as a fatal engine error on what is
    an ordinary window close."""
    import numpy as np

    stop = threading.Event()
    frames = _StopOnGet(np.zeros(250, dtype=np.float32), stop)
    backend, chunks = _chunker(frames, stop, need=100)

    # Must not raise: the buffered chunk comes out whole, then the stream ends.
    assert len(next(chunks)) == 100
    with pytest.raises(StopIteration):
        next(chunks)
    assert backend._processor.blocks == [100]


def test_short_tail_is_still_zero_padded_at_shutdown():
    """The padding itself must survive the fix: a partial final chunk is padded
    rather than dropped, so the last words spoken are not cut off."""
    import numpy as np

    stop = threading.Event()
    frames = _StopOnGet(np.zeros(40, dtype=np.float32), stop)
    backend, chunks = _chunker(frames, stop, need=100)

    assert len(next(chunks)) == 100      # 40 real + 60 zeros
    assert backend._processor.blocks == [100]


# -- config ---------------------------------------------------------------


def test_nemotron_defaults():
    nem = AppConfig().asr.nemotron
    assert nem.model == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    # fp16, not fp32: half the VRAM at no cost in speed. Measured on probe.wav
    # (64 s, 3 runs each) — median per-chunk decode 37.8-39.1 ms for fp16 vs
    # 43.2-44.3 ms for fp32, same transcript. See app_config.NemotronConfig.
    assert nem.dtype == "float16"
    assert nem.lookahead_tokens == 6


def test_unsupported_lookahead_falls_back(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[asr.nemotron]\nlookahead_tokens = 7\n", encoding="utf-8")

    assert load_config(p).asr.nemotron.lookahead_tokens == 6


def test_supported_lookahead_is_kept(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[asr.nemotron]\nlookahead_tokens = 13\n", encoding="utf-8")

    assert load_config(p).asr.nemotron.lookahead_tokens == 13


def test_unknown_dtype_falls_back(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[asr.nemotron]\ndtype = "int8"\n', encoding="utf-8")

    assert load_config(p).asr.nemotron.dtype == "float32"


def test_dtype_is_normalized(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[asr.nemotron]\ndtype = " Float16 "\n', encoding="utf-8")

    assert load_config(p).asr.nemotron.dtype == "float16"


def test_nemotron_engine_survives_normalization(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[asr]\nengine = "Nemotron"\n', encoding="utf-8")

    assert load_config(p).asr.engine == "nemotron"


# -- engine profile -------------------------------------------------------


def test_engine_profile_answers_every_question_the_wiring_asks():
    """One resolved object instead of five `cfg.asr.engine == "nemotron"` checks
    scattered through main(). The frame-queue bug came from two of those copies
    disagreeing."""
    from asr.backends import EngineProfile

    cfg = AppConfig()
    cfg.asr.engine = "nemotron"
    nemo = EngineProfile.resolve(cfg)
    assert (nemo.name, nemo.push_style, nemo.punctuates_itself) == ("nemotron", True, True)
    assert nemo.model == cfg.asr.nemotron.model
    assert nemo.language == cfg.asr.nemotron.language

    cfg.asr.engine = "whisper"
    whisper = EngineProfile.resolve(cfg)
    assert (whisper.name, whisper.push_style, whisper.punctuates_itself) == (
        "whisper", False, False,
    )
    assert whisper.model == cfg.asr.model


def test_engine_profile_can_describe_the_engine_that_actually_loaded():
    """After a fallback the running engine is not the configured one, and the
    session header has to name the model that produced the words."""
    from asr.backends import EngineProfile

    cfg = AppConfig()
    cfg.asr.engine = "nemotron"          # what was asked for
    actual = EngineProfile.for_name(cfg, "whisper")   # what loaded
    assert actual.name == "whisper"
    assert actual.model == cfg.asr.model
    assert actual.punctuates_itself is False          # so the worker starts
