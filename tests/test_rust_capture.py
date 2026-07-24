import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from app_config import AppConfig
from audio.buffer import GrowingAudioBuffer
from audio.capture_backends import (
    DEFAULT_RUST_BINARY,
    create_capture_supervisor,
    resolve_rust_binary,
)
from audio.rust_capture import RustCaptureSupervisor

FAKE = Path(__file__).with_name("fake_capture.py")


class FakeBinarySupervisor(RustCaptureSupervisor):
    """The real supervisor, talking to tests/fake_capture.py instead of the exe.

    Only the executable is swapped: the arguments, the two pipes, the JSON
    parsing and the restart policy under test are the production ones.
    """

    def _command(self):
        return [sys.executable, str(FAKE), *super()._command()[1:]]


def make_supervisor(mode, monkeypatch, **kwargs):
    monkeypatch.setenv("FAKE_CAPTURE_MODE", mode)
    audio_buffer = GrowingAudioBuffer(sample_rate=16000, max_seconds=10.0)
    events: "queue.Queue" = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    supervisor = FakeBinarySupervisor(
        audio_buffer=audio_buffer,
        binary_path=FAKE,
        device_hint="",
        target_sr=16000,
        events=events,
        stop_event=stop_event,
        **kwargs,
    )
    return supervisor, audio_buffer, events, stop_event


def wait_until(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def drain(events):
    out = []
    while True:
        try:
            out.append(events.get_nowait())
        except queue.Empty:
            return out


def test_audio_reaches_buffer_and_frame_sink(monkeypatch):
    frame_sink: "queue.Queue" = queue.Queue(maxsize=64)
    supervisor, audio_buffer, events, _ = make_supervisor(
        "stream", monkeypatch, frame_sink=frame_sink
    )
    supervisor.start()
    try:
        assert wait_until(lambda: len(audio_buffer.snapshot()[0]) >= 1000)
    finally:
        supervisor.stop()

    samples, _ = audio_buffer.snapshot()
    assert len(samples) == 1000
    assert samples.dtype == np.float32
    assert np.allclose(samples, 0.5)
    assert not frame_sink.empty()

    texts = [ev.text for ev in drain(events)]
    assert any("Fake Endpoint" in t for t in texts), texts
    assert any("sr=48000" in t and "16000" in t for t in texts), texts


def test_a_sample_split_across_reads_is_not_lost(monkeypatch):
    supervisor, audio_buffer, _, _ = make_supervisor("trickle", monkeypatch)
    supervisor.start()
    try:
        assert wait_until(lambda: len(audio_buffer.snapshot()[0]) == 1)
    finally:
        supervisor.stop()

    samples, _ = audio_buffer.snapshot()
    assert samples.tolist() == [0.75]


def test_repeated_fatal_start_gives_up_and_stops_the_app(monkeypatch):
    supervisor, _, events, stop_event = make_supervisor(
        "fatal", monkeypatch, restart_backoff_sec=0.05
    )
    supervisor.start()
    try:
        assert wait_until(stop_event.is_set)
    finally:
        supervisor.stop()

    fatals = [ev.text for ev in drain(events) if ev.type == "fatal"]
    assert fatals, "the supervisor must report why it gave up"
    assert "no loopback endpoint" in fatals[-1]


def test_non_json_stderr_is_still_logged(monkeypatch):
    supervisor, _, events, _ = make_supervisor("garbage", monkeypatch)
    supervisor.start()
    try:
        assert wait_until(
            lambda: any("panicked" in ev.text for ev in list(events.queue))
        )
    finally:
        supervisor.stop()


def test_stop_ends_a_child_that_is_waiting_on_silence(monkeypatch):
    # "silent" never writes to stdout, like an idle loopback endpoint: the only
    # way out is closing stdin, which is what stop() does.
    supervisor, _, events, _ = make_supervisor("silent", monkeypatch)
    supervisor.start()
    try:
        assert wait_until(lambda: supervisor._proc is not None)
        proc = supervisor._proc
        assert wait_until(
            lambda: any("Fake Endpoint" in ev.text for ev in list(events.queue))
        )
    finally:
        supervisor.stop()

    assert proc.poll() is not None, "child survived stop()"
    assert supervisor._thread is None


def test_missing_binary_falls_back_to_pyaudio(tmp_path):
    cfg = AppConfig()
    cfg.audio.backend = "rust"
    cfg.audio.rust_binary = str(tmp_path / "not-built-yet.exe")
    supervisor = create_capture_supervisor(
        cfg,
        audio_buffer=GrowingAudioBuffer(sample_rate=16000),
        events=queue.Queue(),
        stop_event=threading.Event(),
    )
    assert type(supervisor).__name__ == "CaptureSupervisor"


def test_existing_binary_selects_the_rust_backend(tmp_path):
    binary = tmp_path / "audio_capture.exe"
    binary.write_bytes(b"")
    cfg = AppConfig()
    cfg.audio.backend = "rust"
    cfg.audio.rust_binary = str(binary)
    cfg.audio.source = "mic"
    supervisor = create_capture_supervisor(
        cfg,
        audio_buffer=GrowingAudioBuffer(sample_rate=16000),
        events=queue.Queue(),
        stop_event=threading.Event(),
    )
    assert isinstance(supervisor, RustCaptureSupervisor)
    assert supervisor._command()[1:] == [
        "--source",
        "mic",
        "--sample-rate",
        "16000",
        "--device-hint",
        "HyperX",
    ]


def test_default_backend_is_pyaudio():
    cfg = AppConfig()
    supervisor = create_capture_supervisor(
        cfg,
        audio_buffer=GrowingAudioBuffer(sample_rate=16000),
        events=queue.Queue(),
        stop_event=threading.Event(),
    )
    assert type(supervisor).__name__ == "CaptureSupervisor"


def test_binary_path_resolution():
    assert resolve_rust_binary("") == DEFAULT_RUST_BINARY
    assert DEFAULT_RUST_BINARY.name == "audio_capture.exe"
    # A relative path follows the repository, not the current directory.
    assert resolve_rust_binary("native/x.exe").is_absolute()
    assert resolve_rust_binary(r"C:\tools\x.exe") == Path(r"C:\tools\x.exe")
