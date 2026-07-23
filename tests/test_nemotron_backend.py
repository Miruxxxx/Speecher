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


# -- config ---------------------------------------------------------------


def test_nemotron_defaults():
    nem = AppConfig().asr.nemotron
    assert nem.model == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert nem.dtype == "float32"       # fp16 decodes ~2x slower, see docs
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
