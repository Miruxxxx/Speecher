from __future__ import annotations

import logging
import threading

from store.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)


class PunctuationWorker:
    def __init__(
        self,
        store: TranscriptStore,
        model,
        stop_event: threading.Event,
        interval_sec: float = 12.0,
        context_words: int = 80,
    ) -> None:
        self._store = store
        self._model = model
        self._stop_event = stop_event
        self._interval_sec = float(interval_sec)
        self._context_words = int(context_words)

    def run(self) -> None:
        logger.info(
            "PunctuationWorker started (interval=%.1fs, context=%d words)",
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
        start_index, words = self._store.get_tail(self._context_words)
        if len(words) < 2:
            return
        raw_text = " ".join(w.text for w in words)
        try:
            punctuated: str = self._model.restore_punctuation(raw_text)
        except Exception:
            logger.exception("PunctuationWorker: restore_punctuation raised")
            return
        new_texts = punctuated.split()
        if len(new_texts) != len(words):
            logger.warning(
                "PunctuationWorker: word count mismatch (before=%d, after=%d); skipping",
                len(words),
                len(new_texts),
            )
            return
        self._store.update_word_texts(start_index, new_texts)
