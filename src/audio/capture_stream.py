from __future__ import annotations

import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

from asr.events import ASREvent
from .devices import pick_loopback_device


class LoopbackStreamCapture:
    def __init__(
        self,
        ring_buffer,
        name_hint: str,
        frames_per_buffer: int = 2048,
        max_channels: int = 2,
        target_sr: int | None = None,
        latency=None,  # LatencyTracker (duck-typed)
        events=None,  # queue.Queue[ASREvent] (duck-typed)
        stop_event: threading.Event | None = None,
    ):
        self.ring_buffer = ring_buffer
        self.name_hint = name_hint
        self.frames_per_buffer = frames_per_buffer
        self.max_channels = max_channels
        self.target_sr = target_sr
        self.latency = latency
        self.events = events
        self.app_stop_event = stop_event

        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._src_sr: int | None = None
        self._channels: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Try to unblock a blocking `read()` quickly.
        stream = self._stream
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            self._stream = None

        t = self._thread
        if t:
            t.join(timeout=2.0)
        # Cleanup is handled in the worker thread, but keep a safety net
        # in case the thread got stuck and we force-closed the stream above.
        if t and t.is_alive():
            self._cleanup()
        self._thread = None

    def _run(self) -> None:
        self._emit_log("thread started")
        self._pa = pyaudio.PyAudio()

        try:
            try:
                idx, name, ch, sr = pick_loopback_device(self._pa, self.name_hint)
            except Exception as e:
                self._emit_fatal(f"pick_loopback_device failed: {e!r}")
                self._request_app_stop()
                return

            channels = max(1, min(int(ch), self.max_channels))
            self._src_sr = int(sr)
            self._channels = channels
            self._emit_log(f"device='{name}' idx={idx} ch={channels} sr={sr}")

            try:
                self._stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    input=True,
                    input_device_index=idx,
                    frames_per_buffer=self.frames_per_buffer,
                )
            except Exception as e:
                self._emit_fatal(f"stream open failed: {e!r}")
                self._request_app_stop()
                return

            while not self._stop_event.is_set() and not self._is_app_stopping():
                try:
                    data = self._stream.read(self.frames_per_buffer, exception_on_overflow=False)
                except Exception as e:
                    # If we are stopping, stream teardown can race with a blocking read().
                    if self._stop_event.is_set() or self._is_app_stopping():
                        break
                    raise

                t0 = time.perf_counter()

                audio = np.frombuffer(data, dtype=np.float32)
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)

                if self.target_sr and self._src_sr and self._src_sr != self.target_sr:
                    audio = resample_poly(audio, self.target_sr, self._src_sr).astype(np.float32, copy=False)

                if self.latency is not None:
                    self.latency.mark("convert", time.perf_counter() - t0)

                self.ring_buffer.write(audio)

        except Exception as e:
            self._emit_fatal(f"capture loop failed: {e!r}")
            self._request_app_stop()
        finally:
            self._cleanup()
            self._emit_log("thread stopped")

    def _cleanup(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _emit_log(self, msg: str) -> None:
        if self.events is None:
            return
        try:
            self.events.put_nowait(ASREvent(type="log", source="capture", text=str(msg)))
        except Exception:
            pass

    def _emit_fatal(self, msg: str) -> None:
        if self.events is None:
            return
        try:
            self.events.put_nowait(ASREvent(type="fatal", source="capture", text=str(msg)))
        except Exception:
            pass

    def _is_app_stopping(self) -> bool:
        return bool(self.app_stop_event is not None and self.app_stop_event.is_set())

    def _request_app_stop(self) -> None:
        if self.app_stop_event is not None:
            self.app_stop_event.set()
