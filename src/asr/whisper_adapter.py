from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from asr.events import Word


class WhisperAdapter:
    """
    Wraps a `faster_whisper.WhisperModel` to expose the simple
    `(audio: np.ndarray) -> list[Word]` contract the engine wants.

    Word timestamps are required: without them LocalAgreement-2 still works
    on text alone, but we lose the ability to trim the audio buffer
    accurately, which is what stops Whisper from looping over the same span.
    """

    def __init__(
        self,
        model: Any,  # faster_whisper.WhisperModel; not type-imported to keep this testable
        *,
        language: Optional[str] = None,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None,
        beam_size: int = 5,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.6,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = model
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

    def __call__(self, audio: np.ndarray) -> List[Word]:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        segments, _info = self._model.transcribe(audio, **self._kwargs)

        words: List[Word] = []
        for seg in segments:
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
        return words
