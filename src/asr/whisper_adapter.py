from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from asr.events import Word
from utils.text import normalize_word

logger = logging.getLogger(__name__)


class WhisperAdapter:
    """
    Wraps a `faster_whisper.WhisperModel` to expose the simple
    `(audio: np.ndarray) -> list[Word]` contract the engine wants.

    Word timestamps are required: without them LocalAgreement-2 still works
    on text alone, but we lose the ability to trim the audio buffer
    accurately, which is what stops Whisper from looping over the same span.

    Language handling:
      - language="ru"/"en"/... — fixed for the session.
      - language=None + sticky_language=True — auto-detect until the first
        decode whose detection confidence >= sticky_min_probability, then pin
        that language for the rest of the session. Per-cycle re-detection on
        1-second snapshots is unstable (commits wrong-alphabet garbage), so
        this is the default mode.
      - language=None + sticky_language=False — re-detect every call.

    Hallucination guards:
      - segments with no_speech_prob > no_speech_prob_max are dropped;
      - segments with avg_logprob < avg_logprob_min are dropped;
      - runs of the same normalized word longer than max_word_repeats are
        collapsed (Whisper loops on steady noise).
    """

    def __init__(
        self,
        model: Any,  # faster_whisper.WhisperModel; not type-imported to keep this testable
        *,
        language: Optional[str] = None,
        sticky_language: bool = True,
        sticky_min_probability: float = 0.7,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None,
        beam_size: int = 5,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.6,
        no_speech_prob_max: float = 0.9,
        avg_logprob_min: float = -1.5,
        max_word_repeats: int = 6,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = model
        self._sticky = sticky_language and language is None
        self._sticky_min_probability = float(sticky_min_probability)
        self._pinned_language: Optional[str] = language
        self._no_speech_prob_max = float(no_speech_prob_max)
        self._avg_logprob_min = float(avg_logprob_min)
        self._max_word_repeats = max(1, int(max_word_repeats))
        self._kwargs: Dict[str, Any] = {
            "language": language,
            "beam_size": beam_size,
            "word_timestamps": True,
            "vad_filter": vad_filter,
            "condition_on_previous_text": condition_on_previous_text,
            "no_speech_threshold": no_speech_threshold,
        }
        if vad_parameters:
            self._kwargs["vad_parameters"] = vad_parameters
        if extra_kwargs:
            self._kwargs.update(extra_kwargs)

    @property
    def pinned_language(self) -> Optional[str]:
        return self._pinned_language

    def __call__(self, audio: np.ndarray) -> List[Word]:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        segments, info = self._model.transcribe(audio, **self._kwargs)

        words: List[Word] = []
        for seg in segments:
            no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
            if no_speech > self._no_speech_prob_max:
                continue
            if avg_logprob < self._avg_logprob_min:
                continue
            seg_words = getattr(seg, "words", None) or []
            for w in seg_words:
                text = (getattr(w, "word", "") or "").strip()
                if not text:
                    continue
                start = float(getattr(w, "start", 0.0) or 0.0)
                end = float(getattr(w, "end", start) or start)
                # Whisper sometimes emits end < start on edge cases; clamp.
                if end < start:
                    end = start
                words.append(Word(text=text, start=start, end=end))

        words = self._collapse_repeats(words)

        # Pin the language only when the decode actually produced speech:
        # detection confidence on silence/noise is meaningless.
        if self._sticky and self._pinned_language is None and words:
            lang = getattr(info, "language", None)
            prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            if lang and prob >= self._sticky_min_probability:
                self._pinned_language = lang
                self._kwargs["language"] = lang
                logger.info("language pinned: %s (p=%.2f)", lang, prob)

        return words

    def _collapse_repeats(self, words: List[Word]) -> List[Word]:
        if len(words) <= self._max_word_repeats:
            return words
        out: List[Word] = []
        run_norm = ""
        run_len = 0
        for w in words:
            norm = normalize_word(w.text)
            if norm and norm == run_norm:
                run_len += 1
                if run_len > self._max_word_repeats:
                    continue
            else:
                run_norm = norm
                run_len = 1
            out.append(w)
        return out
