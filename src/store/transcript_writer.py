from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Bumped when the record vocabulary changes in a way old readers can't handle.
JOURNAL_VERSION = 1

# What a failing journal can raise. ValueError is not decoration: writing to a
# handle that was closed under us raises ValueError("I/O operation on closed
# file"), and a path with an embedded NUL makes mkdir raise ValueError too.
# Both must switch the journal off, never reach the splitter thread.
_IO_ERRORS = (OSError, ValueError)


class TranscriptWriter:
    """Append-only JSONL journal of every mutation the TranscriptStore makes.

    The store is the single source of truth, so the journal hangs off *it*
    rather than off the splitter: words appended by the splitter, nemotron's
    late `amend` marks and the whisper PunctuationWorker's in-place rewrites
    all land in the same file without each producer knowing about persistence.

    Records are outcomes, not intents — a rewrite is journaled as the resolved
    `(index, new_text)` pair the store actually applied, so a replay can never
    diverge from what the store held. Vocabulary:

        {"t":"meta","version":1,"started":"...","engine":"nemotron", ...}
        {"t":"w","text":"привет","s":1.23,"e":1.51,"at":1753382400.5}
        {"t":"patch","i":12,"text":"привет,"}
        {"t":"resumed","from":"2026-07-24_1830.jsonl","words":420}
        {"t":"tr","i":12,"n":7,"lang":"ru","text":"привет, как дела"}

    The `tr` record is the one exception to "every record is a store mutation":
    it comes from the TranslationStore and describes seven words starting at
    index 12, not a change to them. It is keyed by index like `patch` and for
    the same reason — the store is append-only, so an index is the one handle
    that never moves. Readers that predate it skip it.

    Every record is flushed to the OS immediately (not fsync'd): the failure
    mode this survives is the process dying, which is exactly the history this
    project has (docs/CODE_REVIEW_2026-07-15.md §B1).

    Writes come from the splitter and punctuation threads at once, so `record`
    holds a lock — but the store calls it *outside* its own lock, so transcript
    reads never queue behind disk I/O. I/O errors disable the journal and are
    reported once through `on_error`; they never propagate into the caller.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        meta: Optional[dict] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._dir = Path(directory)
        self._meta = dict(meta or {})
        self._on_error = on_error
        self._lock = threading.Lock()
        self._fh = None
        self._path: Optional[Path] = None
        self._words_written = 0
        self._failed = False
        self._failure: Optional[str] = None

    def update_meta(self, **fields) -> None:
        """Correct the session header after the fact.

        The meta line is written when recording starts, but the truth about
        which ASR engine actually runs is only known a couple of seconds later,
        once the model has loaded or failed and fallen back. Writing the
        *configured* engine and leaving it there means every exported
        transcript names an engine that may not have produced a word of it.

        Replay takes the **last** meta record it sees (`store/session_io.py`),
        so appending a corrected one is enough — no new record type, no journal
        version bump, and an older reader still gets a valid header.
        """
        if not fields:
            return
        with self._lock:
            self._meta.update(fields)
            if self._fh is None:
                return  # not recording; the next start() writes the merged meta
            self._write_locked({
                "t": "meta",
                "version": JOURNAL_VERSION,
                **self._meta,
            })

    def set_error_handler(self, handler: Callable[[str], None]) -> None:
        """Install (or replace) the failure callback.

        A failure can happen before the UI exists — the very first `start()`
        may hit a bad path — so a message already recorded is replayed into the
        new handler instead of being lost.
        """
        self._on_error = handler
        if self._failure is not None:
            handler(self._failure)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        """File this session writes to; None until the first successful open."""
        return self._path

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._fh is not None

    @property
    def failed(self) -> bool:
        """True once an I/O error switched the journal off for good."""
        return self._failed

    def start(self) -> bool:
        """Open (or reopen) this session's file. Returns True if recording.

        Reopening after a pause appends to the same file, so pausing and
        resuming the toggle produces one continuous session rather than
        fragments the export would have to stitch back together.
        """
        with self._lock:
            if self._failed or self._fh is not None:
                return self._fh is not None
            first_open = self._path is None
            if first_open:
                stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                self._path = self._dir / f"{stamp}.jsonl"
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                self._fh = open(self._path, "a", encoding="utf-8", newline="\n")
            except _IO_ERRORS as exc:
                self._fh = None
                self._fail(f"не удалось открыть {self._path}: {exc}")
                return False
            if first_open:
                # Kept in _meta rather than only in the record: update_meta
                # rewrites the whole header, and a correction must not drop the
                # start time when the caller did not supply one.
                self._meta.setdefault(
                    "started", datetime.now().isoformat(timespec="seconds")
                )
                self._write_locked({
                    "t": "meta",
                    "version": JOURNAL_VERSION,
                    **self._meta,
                })
            else:
                self._write_locked({"t": "resumed_recording", "at": time.time()})
        logger.info("transcript journal: recording to %s", self._path)
        return True

    def pause(self) -> None:
        """Stop recording but keep the path, so `start` continues the file."""
        with self._lock:
            self._close_locked()

    def close(self) -> None:
        """Final close. An empty session (meta only) removes its own file."""
        with self._lock:
            self._close_locked()
            if self._path is not None and self._words_written == 0:
                try:
                    self._path.unlink(missing_ok=True)
                    logger.info("transcript journal: no words, removed %s", self._path)
                except OSError:
                    pass

    def _close_locked(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.close()
        except _IO_ERRORS:
            pass
        self._fh = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, rec: dict) -> None:
        """Journal one store mutation. Never raises — a broken disk must not
        take down the transcript pipeline that feeds the UI."""
        with self._lock:
            if self._fh is None:
                return
            if rec.get("t") == "w":
                self._words_written += 1
            self._write_locked(rec)

    def _write_locked(self, rec: dict) -> None:
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
        except _IO_ERRORS as exc:
            self._close_locked()
            self._fail(f"запись прервана: {exc}")

    def _fail(self, message: str) -> None:
        """Switch the journal off permanently and report once."""
        if self._failed:
            return
        self._failed = True
        self._failure = message
        logger.error("transcript journal disabled: %s", message)
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:
                logger.exception("transcript journal: on_error callback failed")
