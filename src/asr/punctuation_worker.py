from __future__ import annotations

import logging
import re
import threading

from asr.punctuation_backends import PunctuationBackend
from store.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

# Punctuation the models re-add; stripped from the input so repeated passes
# over the same words stay idempotent.
_EDGE_PUNCT_RE = re.compile(r"^[.,!?;:…]+|[.,!?;:…]+$")
# Trailing punctuation on a single word, used to reassert Whisper's "?".
_TRAILING_PUNCT_RE = re.compile(r"[.,!?;:…]+$")


class PunctuationWorker:
    """
    Periodically re-punctuates the tail of the transcript store in place.

    Relies on two contracts:
      - TranscriptStore is append-only, so the start index from get_tail()
        stays valid while the (slow) model call runs;
      - the backend must not change the number of whitespace-separated words;
        ticks where it does are skipped (checked, logged).

    The backend is loaded lazily inside run() so a missing/broken punctuation
    model degrades to "no punctuation" instead of failing app startup.
    """

    def __init__(
        self,
        store: TranscriptStore,
        backend: PunctuationBackend,
        stop_event: threading.Event,
        interval_sec: float = 12.0,
        context_words: int = 80,
    ) -> None:
        self._store = store
        self._backend = backend
        self._stop_event = stop_event
        self._interval_sec = float(interval_sec)
        self._context_words = int(context_words)
        self._last_seen_size = -1

    def run(self) -> None:
        try:
            self._backend.load()
        except Exception:
            logger.exception(
                "PunctuationWorker: backend %r failed to load; punctuation disabled",
                self._backend.name,
            )
            return
        logger.info(
            "PunctuationWorker started (backend=%s, interval=%.1fs, context=%d words)",
            self._backend.name,
            self._interval_sec,
            self._context_words,
        )
        while True:
            self._stop_event.wait(timeout=self._interval_sec)
            if self._stop_event.is_set():
                break
            try:
                self._tick()
            except Exception:
                logger.exception("PunctuationWorker: unexpected error in tick; continuing")
        logger.info("PunctuationWorker stopped")

    def _tick(self) -> None:
        size = self._store.size()
        if size == self._last_seen_size:
            return  # nothing new since the last pass
        self._last_seen_size = size

        start_index, words = self._store.get_tail(self._context_words)
        if len(words) < 2:
            return
        raw_text = " ".join(_EDGE_PUNCT_RE.sub("", w.text) or w.text for w in words)
        try:
            punctuated: str = self._backend.restore(raw_text)
        except Exception:
            logger.exception("PunctuationWorker: restore raised")
            return
        new_texts = punctuated.split()
        if len(new_texts) != len(words):
            logger.warning(
                "PunctuationWorker: word count mismatch (before=%d, after=%d); skipping",
                len(words),
                len(new_texts),
            )
            return
        # Silero predicts terminal punctuation from text alone and routinely
        # turns Whisper's "?" into "." (verified for ru, even with a question
        # word present). Whisper's question marks come from prosody and are more
        # reliable, so reassert them: any word that ended a question before the
        # pass still ends one after it. Idempotent — the "?" is re-stripped by
        # _EDGE_PUNCT_RE on the next tick before restore runs again.
        new_texts = [
            self._keep_question_mark(old.text, new)
            for old, new in zip(words, new_texts)
        ]
        self._store.update_word_texts(start_index, new_texts)

    @staticmethod
    def _keep_question_mark(old_text: str, new_text: str) -> str:
        if old_text.endswith("?") and not new_text.endswith("?"):
            return _TRAILING_PUNCT_RE.sub("", new_text) + "?"
        return new_text
