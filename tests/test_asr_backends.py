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
