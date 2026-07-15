from __future__ import annotations

import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from asr.events import ASREvent, Word
from audio.buffer import GrowingAudioBuffer
from utils.text import normalize_word


# A transcribe function takes a mono float32 audio array (sampled at the
# engine's sample_rate) and returns a flat list of Word objects whose
# timestamps are LOCAL to that audio (start at zero). The engine handles
# converting them to global timestamps via the audio buffer's offset.
TranscribeFn = Callable[[np.ndarray], List[Word]]


class StreamingASREngine:
    """
    Streaming ASR with LocalAgreement-2 stabilization.

    Algorithm (Macháček et al., "Turning Whisper into Real-Time Transcription
    System", Interspeech 2023):

      1. Every `chunk_interval_sec` seconds, snapshot the entire (un-trimmed)
         audio buffer and run Whisper on it with word-level timestamps.
      2. Compare the new hypothesis to the previous one. The longest common
         word prefix (case- and punctuation-insensitive) is the "agreed"
         portion: two independent decodes produced the same words in the
         same positions, so we treat them as stable and commit them.
      3. Trim the audio buffer to just before the last committed word's end
         time. Reset the previous hypothesis so the next iteration starts
         a fresh comparison on a smaller buffer.
      4. The unagreed tail of the latest hypothesis is the live partial.

    Why this avoids the bugs in the original two-window pipeline:

      - One worker, one model: no shared mutable state across threads.
      - Words are committed only after Whisper produces them twice in a row
         at the same position, so transient hallucinations never stick.
      - The buffer is actively trimmed, so Whisper never re-decodes the
         same audio many times. Repeats can't accumulate.
      - There are no fuzzy text-overlap heuristics. Equality of normalized
         word strings, in order, is the entire merge rule.

    A safety mechanism (`max_buffer_sec`) force-commits stable-looking
    prefixes when the buffer grows too large despite no agreement (this
    happens with continuous speech where the tail keeps shifting).
    """

    def __init__(
        self,
        *,
        audio_buffer: GrowingAudioBuffer,
        transcribe_fn: TranscribeFn,
        events: "queue.Queue[ASREvent]",
        stop_event: threading.Event,
        chunk_interval_sec: float = 1.0,
        min_audio_sec: float = 1.0,
        commit_safety_margin_sec: float = 0.1,
        max_buffer_sec: float = 25.0,
        force_commit_keep_tail_words: int = 6,
        silence_rms_threshold: float = 1e-4,
        empty_decodes_before_trim: int = 3,
        silence_keep_tail_sec: float = 2.0,
        latency=None,
    ) -> None:
        if chunk_interval_sec <= 0:
            raise ValueError("chunk_interval_sec must be positive")
        if min_audio_sec <= 0:
            raise ValueError("min_audio_sec must be positive")
        if force_commit_keep_tail_words < 1:
            raise ValueError("force_commit_keep_tail_words must be >= 1")
        if max_buffer_sec <= min_audio_sec:
            raise ValueError("max_buffer_sec must exceed min_audio_sec")

        self._buffer = audio_buffer
        self._transcribe = transcribe_fn
        self._events = events
        self._stop_event = stop_event
        self._latency = latency

        self._chunk_interval_sec = float(chunk_interval_sec)
        self._min_audio_samples = int(min_audio_sec * audio_buffer.sample_rate)
        self._commit_safety_margin_sec = float(commit_safety_margin_sec)
        self._max_buffer_sec = float(max_buffer_sec)
        self._force_commit_keep_tail_words = int(force_commit_keep_tail_words)
        self._silence_rms_threshold = float(silence_rms_threshold)
        self._empty_decodes_before_trim = max(1, int(empty_decodes_before_trim))
        self._silence_keep_tail_sec = float(silence_keep_tail_sec)

        self._prev_hypothesis: List[Word] = []
        self._empty_decodes = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._emit_log("engine started")
        next_tick = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                # Anchor next_tick to "now + interval" so we never accumulate
                # backlog if a single iteration was slow.
                next_tick = max(next_tick + self._chunk_interval_sec,
                                time.perf_counter() + self._chunk_interval_sec * 0.25)

                try:
                    self._iterate()
                except Exception as exc:  # noqa: BLE001 - intentional broad catch
                    self._emit_fatal(f"engine iteration failed: {exc!r}")
                    self._stop_event.set()
                    return
        finally:
            self._emit_log("engine stopped")

    # ------------------------------------------------------------------
    # One transcription cycle
    # ------------------------------------------------------------------

    def _iterate(self) -> None:
        audio, offset_sec = self._buffer.snapshot()
        if len(audio) < self._min_audio_samples:
            return

        # Silence gate: don't burn a GPU decode on an all-quiet buffer.
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < self._silence_rms_threshold:
            self._register_empty_decode()
            return

        t0 = time.perf_counter()
        local_words = self._transcribe(audio)
        if self._latency is not None:
            self._latency.mark("whisper", time.perf_counter() - t0)

        if not local_words:
            # Nothing recognized this round (silence / VAD filtered everything).
            # Don't reset prev_hypothesis - silence is not disagreement.
            self._register_empty_decode()
            return
        self._empty_decodes = 0

        current = [
            Word(text=w.text, start=w.start + offset_sec, end=w.end + offset_sec)
            for w in local_words
        ]

        agreed = self._longest_agreed_prefix(self._prev_hypothesis, current)

        if agreed:
            self._commit_words(agreed)
            trim_to = agreed[-1].end - self._commit_safety_margin_sec
            self._buffer.trim_until(trim_to)
            # After trimming, current's timestamps past `agreed` are still
            # valid (they were absolute), but the previous-hypothesis baseline
            # no longer corresponds to the same audio span. Reset it; we'll
            # rebuild it on the next iteration. This costs at most one cycle
            # of latency.
            self._prev_hypothesis = []
            partial_tail = current[len(agreed):]
            if partial_tail:
                self._emit_partial(partial_tail)
            return

        # No agreement: the new hypothesis becomes our next baseline.
        self._prev_hypothesis = current
        self._emit_partial(current)

        # Safety valve: continuous speech can keep the tail shifting forever.
        # If the buffer grows too long, commit everything except a small tail
        # of unstable words and trim aggressively.
        if self._buffer.duration_sec() > self._max_buffer_sec:
            self._force_commit(current)

    # ------------------------------------------------------------------
    # Silence bookkeeping
    # ------------------------------------------------------------------

    def _register_empty_decode(self) -> None:
        """
        Without this, silence never trims the buffer: it grows to the hard cap
        and every cycle re-decodes the whole window for nothing. After a few
        consecutive empty decodes we keep only a short tail.
        """
        self._empty_decodes += 1
        if self._empty_decodes < self._empty_decodes_before_trim:
            return
        self._empty_decodes = 0
        keep = max(self._silence_keep_tail_sec, self._min_audio_samples / self._buffer.sample_rate)
        trim_to = self._buffer.end_time_sec() - keep
        self._buffer.trim_until(trim_to)
        self._prev_hypothesis = []

    # ------------------------------------------------------------------
    # LocalAgreement-2 prefix
    # ------------------------------------------------------------------

    @staticmethod
    def _longest_agreed_prefix(prev: List[Word], curr: List[Word]) -> List[Word]:
        """
        Length of the longest leading run where prev[i] and curr[i] are the
        same normalized word. We return the slice from `curr` because its
        timestamps reflect the most recent decode.
        """
        if not prev or not curr:
            return []

        agreed: List[Word] = []
        for p, c in zip(prev, curr):
            if normalize_word(p.text) and normalize_word(p.text) == normalize_word(c.text):
                agreed.append(c)
            else:
                break
        return agreed

    # ------------------------------------------------------------------
    # Forced commit when buffer is dangerously long
    # ------------------------------------------------------------------

    def _force_commit(self, current: List[Word]) -> None:
        keep = self._force_commit_keep_tail_words
        if len(current) <= keep:
            # Hypothesis is short but buffer is long => model is producing
            # almost nothing (sustained noise/silence beyond VAD threshold).
            # Drop the front of the buffer to recover.
            drop_to = self._buffer.end_time_sec() - self._max_buffer_sec / 2.0
            self._buffer.trim_until(drop_to)
            self._prev_hypothesis = []
            return

        to_commit = current[:-keep]
        self._commit_words(to_commit)
        trim_to = to_commit[-1].end - self._commit_safety_margin_sec
        self._buffer.trim_until(trim_to)
        self._prev_hypothesis = []
        self._emit_log(
            f"force-commit: buffer={self._buffer.duration_sec():.1f}s, "
            f"committed={len(to_commit)} words, kept_tail={keep}"
        )

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _commit_words(self, words: List[Word]) -> None:
        text = self._render_text(words)
        ev = ASREvent(type="commit", source="engine", text=text, words=list(words))
        try:
            self._events.put(ev, timeout=0.5)
        except queue.Full:
            self._emit_log("event queue full on commit; dropped")

    def _emit_partial(self, words: List[Word]) -> None:
        text = self._render_text(words)
        ev = ASREvent(type="partial", source="engine", text=text, words=list(words))
        try:
            self._events.put_nowait(ev)
        except queue.Full:
            # Partials are allowed to drop under backpressure.
            pass

    def _emit_log(self, msg: str) -> None:
        try:
            self._events.put_nowait(ASREvent(type="log", source="engine", text=str(msg)))
        except queue.Full:
            pass

    def _emit_fatal(self, msg: str) -> None:
        try:
            self._events.put(ASREvent(type="fatal", source="engine", text=str(msg)), timeout=0.5)
        except queue.Full:
            pass

    @staticmethod
    def _render_text(words: List[Word]) -> str:
        return " ".join(w.text for w in words if w.text).strip()
