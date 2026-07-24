import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from app_config import AppConfig, load_config
from asr.backends import create_asr_backend
from asr.whisper_backend import WhisperBackend
from audio.buffer import GrowingAudioBuffer
from audio.capture_supervisor import CaptureSupervisor


def _backend(engine: str) -> WhisperBackend:
    cfg = AppConfig()
    cfg.asr.engine = engine
    return create_asr_backend(cfg, audio_buffer=GrowingAudioBuffer(sample_rate=16000))


def test_default_engine_is_whisper():
    assert AppConfig().asr.engine == "whisper"
    assert isinstance(_backend("whisper"), WhisperBackend)


def test_unknown_engine_falls_back_to_whisper():
    b = _backend("vosk")
    assert isinstance(b, WhisperBackend)
    assert b.name == "whisper"


def test_config_normalizes_unknown_engine(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[asr]\nengine = "Vosk"\n', encoding="utf-8")
    assert load_config(p).asr.engine == "whisper"


def test_config_keeps_known_engine_case_insensitively(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[asr]\nengine = " Whisper "\n', encoding="utf-8")
    assert load_config(p).asr.engine == "whisper"


def test_run_before_load_raises():
    with pytest.raises(RuntimeError):
        _backend("whisper").run(events=queue.Queue(), stop_event=threading.Event())


def test_bad_tunables_report_fatal_instead_of_dying():
    cfg = AppConfig()
    cfg.asr.min_audio_sec = -1.0
    backend = WhisperBackend(
        cfg=cfg.asr, audio_buffer=GrowingAudioBuffer(sample_rate=16000)
    )
    backend._transcribe = lambda audio: []  # skip load(); no model needed here
    events: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    backend.run(events=events, stop_event=stop)

    assert stop.is_set()
    assert events.get_nowait().type == "fatal"


# -- capture_supervisor frame_sink ---------------------------------------


# -- nemotron WordAssembler: late punctuation -----------------------------


def _words(items):
    """Feed (piece, frame) pairs, return (emitted word texts, late puncts)."""
    from asr.nemotron_backend import WordAssembler

    late: list[str] = []
    a = WordAssembler(0.08, on_late_punct=lambda text, t: late.append(text))
    out = []
    for piece, frame in items:
        out += a.push(piece, frame)
    out += a.flush()
    return [w.text for w in out], late


def test_punctuation_attaches_when_word_still_open():
    # '.' arrives before the next word's leading space -> sticks to the word.
    texts, late = _words([(" встреча", 10), ("?", 10), (" Да", 12)])
    assert texts[0] == "встреча?"
    assert late == []


def test_late_punctuation_after_flush_is_reported_not_dropped():
    from asr.nemotron_backend import WordAssembler

    late: list[tuple[str, float]] = []
    a = WordAssembler(0.08, idle_flush_sec=2.0, on_late_punct=lambda text, t: late.append((text, t)))

    out = a.push(" встреча", 100)
    assert out == []
    # A pause: 25+ blank frames close the word before the '?' is emitted.
    for f in range(101, 135):
        out += a.advance(f)
    assert [w.text for w in out] == ["встреча"]  # committed without the mark

    # The late '?' now has no word to attach to -> surfaced as late punctuation,
    # carrying its own emission time (frame 128 * 0.08 s) so it can be bound by
    # timestamp downstream rather than pinned to whatever word is last.
    assert a.push("?", 128) == []
    assert late == [("?", 128 * 0.08)]


def test_append_to_last_word_amends_in_place():
    from asr.events import Word
    from store.transcript_store import TranscriptStore

    store = TranscriptStore()
    store.append([Word("привет", 0.0, 0.3), Word("встреча", 0.4, 0.8)])
    store.append_to_last_word("?")

    assert store.all_text() == "привет встреча?"
    assert store.size() == 2  # no new entry, indices intact


def test_attach_punct_at_binds_by_timestamp_not_last_word():
    from asr.events import Word
    from store.transcript_store import TranscriptStore

    store = TranscriptStore()
    store.append([Word("встреча", 8.0, 9.0)])
    store.append([Word("Давайте", 9.0, 9.3)])
    store.append([Word("зафиксируем", 9.5, 10.0)])
    # '?' is emitted for 'встреча' (end 9.0) though two later words already
    # landed. It must find 'встреча' by time, not fall on the last word.
    store.attach_punct_at(9.1, "?")

    assert store.all_text() == "встреча? Давайте зафиксируем"
    assert store.size() == 3  # in place, indices intact


def test_attach_punct_at_falls_back_to_closest_when_outside_tol():
    from asr.events import Word
    from store.transcript_store import TranscriptStore

    store = TranscriptStore()
    store.append([Word("привет", 0.0, 0.3), Word("мир", 0.4, 0.8)])
    # A mark whose time is far past every word still lands, on the closest
    # (last) word - it is never dropped.
    store.attach_punct_at(50.0, ".")

    assert store.all_text() == "привет мир."


def _supervisor(**kwargs) -> CaptureSupervisor:
    return CaptureSupervisor(
        audio_buffer=GrowingAudioBuffer(sample_rate=16000),
        device_hint="",
        target_sr=16000,
        events=queue.Queue(),
        stop_event=threading.Event(),
        **kwargs,
    )


def test_frame_sink_receives_chunks():
    sink: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
    sup = _supervisor(frame_sink=sink)
    chunk = np.zeros(160, dtype=np.float32)

    sup._push_frame(chunk)

    assert sink.get_nowait() is chunk


def test_frame_sink_full_drops_and_counts():
    sink: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
    events: "queue.Queue" = queue.Queue()
    sup = CaptureSupervisor(
        audio_buffer=GrowingAudioBuffer(sample_rate=16000),
        device_hint="",
        target_sr=16000,
        events=events,
        stop_event=threading.Event(),
        frame_sink=sink,
    )
    chunk = np.zeros(160, dtype=np.float32)

    for _ in range(3):
        sup._push_frame(chunk)

    assert sink.qsize() == 1
    assert sup._frames_dropped == 2
    assert any(ev.type == "log" for ev in list(events.queue))
