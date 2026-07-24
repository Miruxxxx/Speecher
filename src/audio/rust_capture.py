from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from asr.events import ASREvent
from audio.capture_base import CaptureSupervisorBase

# Keep the console window of the child from flashing when the app is started
# from a shortcut. Windows-only flag, and so is the whole capture path.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# One read is at most this many bytes (~128 ms at 16 kHz mono f32). The read
# returns as soon as anything is available, so this only caps the batch size.
_READ_BYTES = 8192

# src/audio/rust_capture.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = (
    PROJECT_ROOT / "native" / "audio_capture" / "target" / "release" / "audio_capture.exe"
)
BUILD_HINT = "cargo build --release --manifest-path native/audio_capture/Cargo.toml"


def resolve_binary(configured: str = "") -> Path:
    """Where the capture binary is expected to live.

    An empty config value means "the release build in the tree"; a relative
    path is resolved against the repository root, not the current directory,
    so the app behaves the same however it was started.
    """
    configured = (configured or "").strip()
    if not configured:
        return DEFAULT_BINARY
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class RustCaptureSupervisor(CaptureSupervisorBase):
    """
    Runs native/audio_capture (WASAPI, Rust) as the capture child.

    The child does the conversion work too: it already delivers mono float32 at
    the target rate, so this side only turns bytes into arrays. See
    native/audio_capture/src/main.rs for the protocol:

      stdout  raw little-endian f32, mono, target rate -- nothing else
      stderr  one JSON object per line: info / log / fatal
      stdin   held open here; closing it is how the child is asked to stop
              (it blocks waiting for audio and cannot poll an event)
      exit    0 = clean, 1 = setup failure, 2 = device failure mid-stream

    A child that dies is restarted by the base class with backoff; three deaths
    within seconds of starting are treated as a permanent problem and stop the
    app, which is also how a missing binary surfaces.
    """

    def __init__(
        self,
        *,
        audio_buffer,
        binary_path: Path | str,
        device_hint: str,
        target_sr: int,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
        source: str = "loopback",
        restart_backoff_sec: float = 2.0,
        latency=None,
        frame_sink: "Optional[queue.Queue[np.ndarray]]" = None,
    ) -> None:
        super().__init__(
            audio_buffer=audio_buffer,
            events=events,
            stop_event=stop_event,
            restart_backoff_sec=restart_backoff_sec,
            latency=latency,
            frame_sink=frame_sink,
        )
        self._binary = Path(binary_path)
        self._device_hint = device_hint or ""
        self._target_sr = int(target_sr)
        self._source = source or "loopback"

        self._proc: Optional[subprocess.Popen] = None
        self._last_fatal: Optional[str] = None

    def _command(self) -> List[str]:
        return [
            str(self._binary),
            "--source",
            self._source,
            "--sample-rate",
            str(self._target_sr),
            "--device-hint",
            self._device_hint,
        ]

    # -- one child --------------------------------------------------------

    def _run_one_child(self) -> str:
        self._last_fatal = None
        if not self._binary.is_file():
            # There is no second capture path to fall back to, so say exactly
            # what is missing and how to get it instead of retrying blindly.
            self._emit_fatal(
                f"нативный бинарь захвата не найден: {self._binary}\nсобрать: {BUILD_HINT}"
            )
            self._stop_event.set()
            return f"binary missing: {self._binary}"
        try:
            # bufsize=0: read() must return as soon as a packet lands, not when
            # some Python-side buffer happens to fill.
            proc = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:
            return f"cannot start {self._binary}: {exc!r}"

        self._proc = proc
        self._emit_log(f"capture child started (pid={proc.pid})")
        err_thread = threading.Thread(
            target=self._pump_stderr, args=(proc,), name="capture-stderr", daemon=True
        )
        err_thread.start()

        leftover = b""
        read_error = None
        try:
            while not self._stopping():
                chunk = proc.stdout.read(_READ_BYTES)
                if not chunk:
                    break  # child exited or closed the pipe
                t0 = time.perf_counter()
                if leftover:
                    chunk = leftover + chunk
                # A read can split a sample; carry the tail to the next one.
                usable = len(chunk) - len(chunk) % 4
                leftover = chunk[usable:]
                if usable:
                    self._write_audio(np.frombuffer(chunk[:usable], dtype=np.float32))
                if self._latency is not None:
                    self._latency.mark("capture_convert", time.perf_counter() - t0)
        except (OSError, ValueError) as exc:
            # ValueError: the pipe was closed under us by stop().
            read_error = exc
        finally:
            self._terminate_child()
            err_thread.join(timeout=1.0)
            self._close_pipes(proc)

        if read_error is not None:
            return f"stdout read failed: {read_error!r}"
        if self._last_fatal:
            return f"fatal: {self._last_fatal}"
        return f"exitcode={proc.returncode}"

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        """Turn the child's JSON lines into capture events."""
        stream = proc.stderr
        if stream is None:
            return
        while True:
            try:
                raw = stream.readline()
            except (OSError, ValueError):
                return
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Anything not JSON is still worth seeing (a panic, say).
                self._emit_log(f"capture child: {line}")
                continue
            kind = msg.get("kind")
            if kind == "info":
                self._emit_log(
                    f"device='{msg.get('device')}' source={msg.get('source')} "
                    f"ch={msg.get('input_channels')} sr={msg.get('input_sample_rate')} "
                    f"fmt={msg.get('input_format')} -> mono {msg.get('output_sample_rate')}"
                )
            elif kind == "fatal":
                self._last_fatal = str(msg.get("text"))
                self._emit_log(f"capture child fatal: {self._last_fatal}")
            else:
                self._emit_log(f"capture child: {msg.get('text', line)}")

    # -- hooks ------------------------------------------------------------

    def _signal_child_stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._close_stdin(proc)

    def _terminate_child(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        # Closing stdin is the polite stop: the child notices the EOF within
        # one event wait (500 ms) and shuts the stream down itself.
        self._close_stdin(proc)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass

    @staticmethod
    def _close_stdin(proc: subprocess.Popen) -> None:
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass

    @staticmethod
    def _close_pipes(proc: subprocess.Popen) -> None:
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe is not None and not pipe.closed:
                    pipe.close()
            except OSError:
                pass
