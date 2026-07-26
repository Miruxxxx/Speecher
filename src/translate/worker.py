"""The thread that keeps the translated feed fed.

It walks the transcript store forward by word index and never looks back: cut a
segment, translate it, append the finished line, advance. Nothing already on the
screen is ever revisited — which is also why a late `amend` from nemotron or a
rewrite from PunctuationWorker costs nothing here: they can only touch words the
worker has not reached yet, and a subtitle that is already read is not worth
re-flowing for a comma.

Falling behind is handled by *merging*, not dropping: consecutive waiting
segments become one longer line. Order is preserved, no speech disappears, and
the latency stays bounded — which is what a reader notices.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from store.transcript_store import TranscriptStore
from translate import languages
from translate.backends import TranslationBackend
from translate.segments import Segment, cut, merge
from translate.store import FAILED, SKIPPED, TRANSLATED, TranslationStore

logger = logging.getLogger(__name__)

# Script -> the language we assume when nothing else says what is being spoken.
_SCRIPT_DEFAULT = {"cyrillic": "ru", "latin": "en", "han": "zh", "kana": "ja"}

# How many segments one tick may turn into lines before going back to sleep.
# Bounded so a long backlog cannot hold the loop (and the stop event) hostage.
_MAX_PER_TICK = 4


class TranslationWorker:
    def __init__(
        self,
        store: TranscriptStore,
        translations: TranslationStore,
        backend: TranslationBackend,
        stop_event: threading.Event,
        *,
        target: str = "ru",
        source: str = "auto",
        asr_language: str = "auto",
        close_after_sec: float = 1.5,
        pause_sec: float = 1.2,
        max_words_per_segment: int = 40,
        max_pending: int = 4,
        poll_sec: float = 0.4,
        skip_same_script: bool = True,
        enabled: bool = True,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._store = store
        self._translations = translations
        self._backend = backend
        self._stop_event = stop_event
        self._target = languages.normalize_code(target) or "ru"
        self._source = languages.normalize_code(source)  # "" means auto
        self._asr_language = languages.normalize_code(asr_language)
        self._close_after_sec = float(close_after_sec)
        self._pause_sec = float(pause_sec)
        self._max_words = max(4, int(max_words_per_segment))
        self._max_pending = max(1, int(max_pending))
        self._poll_sec = float(poll_sec)
        self._skip_same_script = bool(skip_same_script)
        self._on_status = on_status
        self._enabled = threading.Event()
        if enabled:
            self._enabled.set()
        self._failure = ""

    # -- control (Qt thread) ---------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    @property
    def target(self) -> str:
        return self._target

    @property
    def failure(self) -> str:
        """Why translation is unavailable, or "" while it is fine."""
        return self._failure

    def set_enabled(self, on: bool) -> None:
        if on:
            self._enabled.set()
        else:
            self._enabled.clear()

    # -- worker ----------------------------------------------------------

    def run(self) -> None:
        try:
            self._backend.load()
        except Exception as exc:
            logger.exception("translate: backend %r failed to load", self._backend.name)
            self._failure = str(exc)
            self._enabled.clear()
            self._status("Перевод", f"Модель перевода не загрузилась: {exc}")
            return
        logger.info(
            "TranslationWorker started (backend=%s, target=%s, source=%s)",
            self._backend.name, self._target, self._source or "auto",
        )
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._poll_sec)
            if self._stop_event.is_set():
                break
            if not self._enabled.is_set():
                continue
            try:
                self._tick()
            except Exception:
                logger.exception("translate: unexpected error in tick; continuing")
        logger.info("TranslationWorker stopped")

    def _tick(self) -> None:
        for _ in range(_MAX_PER_TICK):
            if self._stop_event.is_set() or not self._enabled.is_set():
                return
            if not self._step():
                return

    def _step(self) -> bool:
        """Turn the next ready segment into a line. False when there is none."""
        index = self._translations.next_index()
        entries = self._store.entries_since(index)
        if not entries:
            return False
        # The trailing words may still be growing; they are only cut once the
        # audio has been quiet, so a line never appears half-said.
        idle = time.time() - entries[-1][1]
        ready = cut(
            entries,
            index,
            pause_sec=self._pause_sec,
            max_words=self._max_words,
            tail_closed=idle >= self._close_after_sec,
        )
        if not ready:
            return False
        if len(ready) > self._max_pending:
            # Behind the speech: one longer line instead of a growing queue.
            extra = len(ready) - self._max_pending + 1
            segment = merge(ready[:extra])
            logger.info("translate: merged %d segments to catch up", extra)
        else:
            segment = ready[0]
        self._emit(segment)
        return True

    def _emit(self, segment: Segment) -> None:
        source = self._resolve_source(segment.text)
        if not source or source == self._target or self._same_language(segment.text):
            self._translations.add(segment, segment.text, SKIPPED)
            return
        started = time.perf_counter()
        try:
            text = self._backend.translate(segment.text, source, self._target)
        except Exception:
            logger.exception("translate: backend raised on segment %d", segment.index)
            self._translations.add(segment, segment.text, FAILED)
            return
        text = text.strip()
        if not text:
            self._translations.add(segment, segment.text, FAILED)
            return
        self._translations.add(segment, text, TRANSLATED)
        logger.debug(
            "translate: %s->%s %d words in %.2fs",
            source, self._target, segment.n_words, time.perf_counter() - started,
        )

    # -- language resolution ---------------------------------------------

    def _auto_source(self) -> bool:
        """Nothing pins the source: neither the config nor the ASR language.

        `normalize_code` maps "auto" (and any unknown locale) to "", so both
        fields being empty is exactly the "we are guessing" case.
        """
        return not self._source and not self._asr_language

    def _resolve_source(self, text: str) -> str:
        """Configured source, else the ASR's pinned language, else the script."""
        if self._source:
            return self._source
        if self._asr_language:
            return self._asr_language
        return _SCRIPT_DEFAULT.get(languages.detect_script(text), "")

    def _same_language(self, text: str) -> bool:
        """The cheap guard against translating the target language into itself.

        Only trusted when nobody told us the source language: the script test
        cannot tell German from English, so with a configured source we rely on
        `source == target` alone.
        """
        if not self._skip_same_script or not self._auto_source():
            return False
        return languages.looks_like(text, self._target)

    def _status(self, text: str, detail: str = "") -> None:
        if self._on_status is not None:
            self._on_status(text, detail)
