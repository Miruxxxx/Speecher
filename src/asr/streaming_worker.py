from __future__ import annotations

import queue
import time
from typing import Any, Dict, Iterable, Tuple

from scipy.signal import resample_poly

from asr.events import ASREvent
from utils.text import word_pieces


def _unwrap_segments(result) -> Iterable[Any]:
    if isinstance(result, tuple) and len(result) == 2:
        segments, _info = result
        return segments
    return result


def _segments_to_text(segments: Iterable[Any]) -> str:
    parts: list[str] = []
    for seg in segments:
        t = getattr(seg, "text", None)
        if t is None:
            t = str(seg)
        t = (t or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


class FastASRWorker:
    def __init__(
        self,
        *,
        ring_buffer,
        model,
        events: queue.Queue[ASREvent],
        stop_event,
        input_sr: int,
        target_sr: int = 16000,
        window_sec: float = 3.0,
        step_sec: float = 0.25,
        language: str | None = "en",
        vad_filter: bool = False,
        min_audio_sec: float = 1.0,
        transcribe_kwargs: Dict[str, Any] | None = None,
        latency=None,  # LatencyTracker (duck-typed)
    ) -> None:
        self.ring_buffer = ring_buffer
        self.model = model
        self.events = events
        self.stop_event = stop_event

        self.input_sr = int(input_sr)
        self.target_sr = int(target_sr)
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.language = language
        self.vad_filter = bool(vad_filter)
        self.min_audio_sec = float(min_audio_sec)
        self.transcribe_kwargs = dict(transcribe_kwargs or {})
        self.latency = latency

        self._running = False
        self._last_key: str = ""
        self._base_kwargs: Dict[str, Any] = {
            "language": self.language,
            "vad_filter": self.vad_filter,
            **self.transcribe_kwargs,
        }
        self._fast_kwargs: Dict[str, Any] = {
            **self._base_kwargs,
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        }

    def run(self) -> None:
        self._running = True
        self._emit_log("thread started")

        min_samples = int(self.min_audio_sec * self.input_sr)
        next_tick = time.perf_counter()

        try:
            while self._running and not self.stop_event.is_set():
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_tick = max(next_tick + self.step_sec, now + self.step_sec)

                audio = self.ring_buffer.read_last(self.window_sec)
                if len(audio) < min_samples:
                    continue
                if self.input_sr != self.target_sr:
                    audio = resample_poly(audio, self.target_sr, self.input_sr).astype("float32", copy=False)

                t_w0 = time.perf_counter()
                try:
                    result = self.model.transcribe(audio, **self._fast_kwargs)
                except TypeError:
                    result = self.model.transcribe(audio, **self._base_kwargs)
                whisper_s = time.perf_counter() - t_w0
                if self.latency is not None:
                    self.latency.mark("whisper_fast", whisper_s)

                t_text0 = time.perf_counter()
                text = _segments_to_text(_unwrap_segments(result))
                ws, pcs = word_pieces(text)

                key = " ".join(ws).casefold() if ws else ""
                if key != self._last_key:
                    self._last_key = key
                    self._emit_partial(text=text, words=ws, pieces=pcs)

                if self.latency is not None:
                    self.latency.mark("text_fast", time.perf_counter() - t_text0)

        except Exception as e:
            self._emit_fatal(f"fast_asr crashed: {e!r}")
            self.stop_event.set()
        finally:
            self._emit_log("thread stopped")

    def stop(self) -> None:
        self._running = False

    def _emit_partial(self, *, text: str, words: list[str], pieces: list[str]) -> None:
        try:
            self.events.put_nowait(
                ASREvent(type="partial", source="fast", text=text, words=words, pieces=pieces, created_t=time.time())
            )
        except queue.Full:
            # Partials are allowed to drop under backpressure to minimize latency.
            pass
        except Exception:
            pass

    def _emit_log(self, msg: str) -> None:
        try:
            self.events.put_nowait(ASREvent(type="log", source="fast", text=str(msg)))
        except Exception:
            pass

    def _emit_fatal(self, msg: str) -> None:
        try:
            self.events.put_nowait(ASREvent(type="fatal", source="fast", text=str(msg)))
        except Exception:
            pass


class RefinerASRWorker:
    def __init__(
        self,
        *,
        ring_buffer,
        model,
        events: queue.Queue[ASREvent],
        stop_event,
        input_sr: int,
        target_sr: int = 16000,
        window_sec: float = 8.0,
        every_sec: float = 1.0,
        language: str | None = "en",
        vad_filter: bool = False,
        min_audio_sec: float = 2.0,
        soft_replace_last_n_words: int = 40,
        transcribe_kwargs: Dict[str, Any] | None = None,
        latency=None,  # LatencyTracker (duck-typed)
    ) -> None:
        self.ring_buffer = ring_buffer
        self.model = model
        self.events = events
        self.stop_event = stop_event

        self.input_sr = int(input_sr)
        self.target_sr = int(target_sr)
        self.window_sec = float(window_sec)
        self.every_sec = float(every_sec)
        self.language = language
        self.vad_filter = bool(vad_filter)
        self.min_audio_sec = float(min_audio_sec)
        self.soft_replace_last_n_words = int(soft_replace_last_n_words)
        self.transcribe_kwargs = dict(transcribe_kwargs or {})
        self.latency = latency

        self._running = False
        self._last_key: str = ""
        self._base_kwargs: Dict[str, Any] = {
            "language": self.language,
            "vad_filter": self.vad_filter,
            **self.transcribe_kwargs,
        }
        self._refine_kwargs: Dict[str, Any] = {
            **self._base_kwargs,
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "condition_on_previous_text": True,
        }

    def run(self) -> None:
        self._running = True
        self._emit_log("thread started")

        min_samples = int(self.min_audio_sec * self.input_sr)
        next_tick = time.perf_counter()

        try:
            while self._running and not self.stop_event.is_set():
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_tick = max(next_tick + self.every_sec, now + self.every_sec)

                audio = self.ring_buffer.read_last(self.window_sec)
                if len(audio) < min_samples:
                    continue
                if self.input_sr != self.target_sr:
                    audio = resample_poly(audio, self.target_sr, self.input_sr).astype("float32", copy=False)

                t_w0 = time.perf_counter()
                try:
                    result = self.model.transcribe(audio, **self._refine_kwargs)
                except TypeError:
                    result = self.model.transcribe(audio, **self._base_kwargs)
                whisper_s = time.perf_counter() - t_w0
                if self.latency is not None:
                    self.latency.mark("whisper_refine", whisper_s)

                t_text0 = time.perf_counter()
                text = _segments_to_text(_unwrap_segments(result))
                ws, pcs = word_pieces(text)

                key = " ".join(ws).casefold() if ws else ""
                # Emit finals even when unchanged so the downstream stabilizer
                # can observe "same hypothesis twice" and commit.
                if ws:
                    self._last_key = key
                    self._emit_final(text=text, words=ws, pieces=pcs)

                if self.latency is not None:
                    self.latency.mark("text_refine", time.perf_counter() - t_text0)

        except Exception as e:
            self._emit_fatal(f"refiner crashed: {e!r}")
            self.stop_event.set()
        finally:
            self._emit_log("thread stopped")

    def stop(self) -> None:
        self._running = False

    def _emit_final(self, *, text: str, words: list[str], pieces: list[str]) -> None:
        # Finals should try harder to deliver.
        ev = ASREvent(type="final", source="refine", text=text, words=words, pieces=pieces, created_t=time.time())
        try:
            self.events.put(ev, timeout=0.25)
        except Exception:
            pass

    def _emit_log(self, msg: str) -> None:
        try:
            self.events.put_nowait(ASREvent(type="log", source="refine", text=str(msg)))
        except Exception:
            pass

    def _emit_fatal(self, msg: str) -> None:
        try:
            self.events.put_nowait(ASREvent(type="fatal", source="refine", text=str(msg)))
        except Exception:
            pass
