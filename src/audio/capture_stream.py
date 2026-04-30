from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

from asr.events import ASREvent
from .devices import pick_loopback_device


class LoopbackStreamCapture:
    """Capture system audio via WASAPI loopback into a GrowingAudioBuffer."""

    def __init__(
        self,
        *,
        audio_buffer,
        name_hint: str,
        target_sr: int,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
        frames_per_buffer: int = 2048,
        max_channels: int = 2,
        latency=None,
    ) -> None:
        self._buffer = audio_buffer
        self._name_hint = name_hint
        self._target_sr = int(target_sr)
        self._events = events
        self._stop_event = stop_event
        self._frames_per_buffer = int(frames_per_buffer)
        self._max_channels = int(max_channels)
        self._latency = latency

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._local_stop = threading.Event()

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._local_stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._local_stop.set()
        s = self._stream
        if s is not None:
            # Close the stream from the outside to unblock a hung read().
            try:
                s.stop_stream()
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass
            self._stream = None
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    # -- inner loop -----------------------------------------------------

    def _run(self) -> None:
        self._emit_log("capture started")
        self._pa = pyaudio.PyAudio()
        try:
            try:
                idx, name, ch, sr = pick_loopback_device(self._pa, self._name_hint)
            except Exception as exc:
                self._emit_fatal(f"pick_loopback_device failed: {exc!r}")
                self._stop_event.set()
                return

            channels = max(1, min(int(ch), self._max_channels))
            src_sr = int(sr)
            self._emit_log(f"device='{name}' idx={idx} ch={channels} sr={src_sr}")

            try:
                self._stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=src_sr,
                    input=True,
                    input_device_index=idx,
                    frames_per_buffer=self._frames_per_buffer,
                )
            except Exception as exc:
                self._emit_fatal(f"stream open failed: {exc!r}")
                self._stop_event.set()
                return

            while not self._stopping():
                try:
                    raw = self._stream.read(self._frames_per_buffer, exception_on_overflow=False)
                except Exception:
                    if self._stopping():
                        break
                    raise

                t0 = time.perf_counter()
                audio = np.frombuffer(raw, dtype=np.float32)
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                if src_sr != self._target_sr:
                    audio = resample_poly(audio, self._target_sr, src_sr).astype(np.float32, copy=False)
                if self._latency is not None:
                    self._latency.mark("capture_convert", time.perf_counter() - t0)

                self._buffer.write(audio)

        except Exception as exc:
            self._emit_fatal(f"capture loop failed: {exc!r}")
            self._stop_event.set()
        finally:
            self._cleanup()
            self._emit_log("capture stopped")

    # -- helpers --------------------------------------------------------

    def _stopping(self) -> bool:
        return self._local_stop.is_set() or self._stop_event.is_set()

    def _cleanup(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _emit_log(self, msg: str) -> None:
        try:
            self._events.put_nowait(ASREvent(type="log", source="capture", text=str(msg)))
        except queue.Full:
            pass

    def _emit_fatal(self, msg: str) -> None:
        try:
            self._events.put(ASREvent(type="fatal", source="capture", text=str(msg)), timeout=0.5)
        except queue.Full:
            pass
