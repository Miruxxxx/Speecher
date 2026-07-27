"""The session as an object: its data, its journal, and what the menu does to it.

Split out of `Overlay`, which had grown to own the window *and* everything the
window could be asked to do. This half is the part that has nothing to do with
pixels: which words the session holds, whether it is being recorded, what its
header says, and the four menu entries that read or write it (export, copy,
resume, reveal in Explorer).

Keeping it separate buys two things beyond size. The header is now owned in one
place — `meta` is mutated by exactly this class, where before the overlay held
the dict and the writer held a copy of it. And the operations are testable
without constructing a window, which the overlay's own tests cannot do at all
(building one registers global hotkeys on the machine running them).

It is a QObject rather than a plain class because the overlay has to re-render
when a resume drops several thousand words into the store, and a signal is how
that crosses back without this file knowing what a transcript view is.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from store import session_io
from store.transcript_store import TranscriptStore
from store.transcript_writer import TranscriptWriter
from translate.store import TranslationStore
from ui.hotkeys import describe
from ui.widgets.indicators import REC_FAILED, REC_ON, REC_PAUSED

logger = logging.getLogger(__name__)


class SessionController(QObject):
    """The transcript, the journal and the session header, plus the menu's verbs.

    Every method here is safe to call from the Qt thread only — they open
    dialogs and touch the journal.
    """

    # Short label + full sentence, exactly what Overlay.post_status takes.
    status = pyqtSignal(str, str)
    # The store changed underneath the view (a resume). Re-render.
    changed = pyqtSignal()

    def __init__(
        self,
        *,
        store: TranscriptStore,
        translations: TranslationStore,
        writer: Optional[TranscriptWriter],
        session_dir: Optional[Path],
        meta: dict,
        record_hotkey: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._translations = translations
        self._writer = writer
        self._session_dir = Path(session_dir) if session_dir else None
        self._meta = dict(meta)
        self._record_hotkey = record_hotkey
        # The widget dialogs are parented to; None is legal and centres them.
        self._parent_widget = parent
        # Filled once the loader knows which engine really came up. The
        # requested one is kept only when it differs — i.e. after a fallback.
        self._engine_actual = ""
        self._engine_requested = ""

    # -- session header --------------------------------------------------

    @property
    def meta(self) -> dict:
        """The export/journal header. A copy: nobody edits this behind our back."""
        return dict(self._meta)

    @property
    def engine_actual(self) -> str:
        return self._engine_actual

    @property
    def engine_requested(self) -> str:
        """The configured engine, when it is *not* the one that runs. Else ""."""
        return self._engine_requested

    def set_engine(self, actual: str, model: str, *, requested: str = "") -> None:
        """Record which ASR engine is really running. Called from the loader.

        Until this existed the session header carried the *configured* engine,
        written before the model had even tried to load — so after a fallback
        every exported transcript named an engine that produced none of it. The
        journal header is corrected too (TranscriptWriter.update_meta), and the
        session menu grows a line saying what is actually running, because the
        status message that announced the fallback is gone in seconds.
        """
        self._engine_actual = actual
        self._engine_requested = requested if requested and requested != actual else ""
        fields = {"engine": actual, "model": model}
        if self._engine_requested:
            fields["engine_requested"] = self._engine_requested
        self._meta.update(fields)
        if self._writer is not None:
            self._writer.update_meta(**fields)

    def engine_caption(self) -> str:
        if self._engine_requested:
            return f"Движок: {self._engine_actual} (вместо {self._engine_requested})"
        return f"Движок: {self._engine_actual}"

    # -- journal ---------------------------------------------------------

    @property
    def writer_path(self) -> Optional[Path]:
        return self._writer.path if self._writer is not None else None

    def journal_state(self) -> str:
        if self._writer is None:
            return REC_PAUSED
        if self._writer.failed:
            return REC_FAILED
        return REC_ON if self._writer.enabled else REC_PAUSED

    def journal_tooltip(self, state: str) -> str:
        if self._writer is None:
            return "Персист выключен в конфиге: [storage] enabled = false"
        where = str(self._writer.path) if self._writer.path else "—"
        if state == REC_FAILED:
            return "Журнал отключён из-за ошибки записи — подробности в логе"
        if state == REC_ON:
            return f"Пишется в {where}\n{describe(self._record_hotkey)} — пауза"
        return f"Запись на паузе\n{describe(self._record_hotkey)} — продолжить"

    def toggle_recording(self) -> None:
        if self._writer is None or self._writer.failed:
            self.status.emit("Журнал", "Запись на диск недоступна")
            return
        if self._writer.enabled:
            self._writer.pause()
            self.status.emit("Пауза", "Запись транскрипта на паузе")
        elif self._writer.start():
            self.status.emit("Запись", f"Пишется в {self._writer.path}")

    def pause_recording(self) -> None:
        if self._writer is None or not self._writer.enabled:
            return
        self._writer.pause()
        self.status.emit("Пауза", "Запись транскрипта на паузе")

    # -- resume ----------------------------------------------------------

    def latest_session(self) -> Optional[Path]:
        if self._session_dir is None:
            return None
        return session_io.latest_session(self._session_dir)

    def can_resume(self) -> bool:
        """Resuming into a non-empty store would interleave two streams whose
        word timestamps both start at zero, so it is offered only on a clean
        start — which is also the only moment it is useful."""
        last = self.latest_session()
        return last is not None and last != self.writer_path and self._store.size() == 0

    def resume(self, path: Path) -> None:
        try:
            _meta, entries = session_io.read_session(path)
        except OSError as exc:
            self._warn("Не удалось прочитать сессию", f"{path}\n\n{exc}")
            return
        pairs = [(e.word, e.received_at) for e in entries]
        loaded = self._store.load_entries(pairs)
        # Translations come back with the words: re-running the model over a
        # replayed session would cost minutes and produce the same text. Their
        # timestamps are recomputed from the words, which is why the entries go
        # in too.
        restored = self._translations.load(session_io.read_translations(path), pairs)
        # The export header keeps *this* session's start time; the origin of the
        # resumed words is recorded separately.
        self._meta["resumed_from"] = path.name
        logger.info(
            "resumed session %s (%d words, %d translations)", path, loaded, restored
        )
        self.changed.emit()
        self.status.emit(f"{loaded} слов", f"Загружено из {path.name}")

    # -- export ----------------------------------------------------------

    def export(self, suffix: str) -> None:
        words = self._store.snapshot()
        if not words:
            return
        default_dir = self._session_dir or Path.cwd()
        suggested = default_dir / session_io.suggest_export_name(
            self._meta, self.writer_path, suffix
        )
        label = "Markdown (*.md)" if suffix == "md" else "Текст (*.txt)"
        path, _ = QFileDialog.getSaveFileName(
            self._parent_widget, "Экспорт транскрипта", str(suggested),
            f"{label};;Все файлы (*)",
        )
        if not path:
            return
        try:
            if suffix == "md":
                session_io.export_markdown(
                    words, path, self._meta,
                    translations=self._translations.texts_by_index(),
                )
            else:
                # Plain text stays the transcript and nothing else; a bilingual
                # .txt has no way to say which line is which.
                session_io.export_text(words, path)
        except OSError as exc:
            self._warn("Экспорт не удался", f"{path}\n\n{exc}")
            return
        self.status.emit("Экспорт", f"Экспортировано в {Path(path).name}")

    def copy_transcript(self) -> None:
        text = " ".join(w.text for w in self._store.snapshot()).strip()
        if text:
            QApplication.clipboard().setText(text)
            self.status.emit("Копия", "Транскрипт скопирован в буфер обмена")

    def open_session_dir(self) -> None:
        if self._session_dir is None:
            return
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            # Explorer wants a native path and returns 1 even on success.
            subprocess.Popen(["explorer", str(self._session_dir)])
        except OSError as exc:
            self._warn("Не удалось открыть папку", f"{self._session_dir}\n\n{exc}")

    @property
    def has_session_dir(self) -> bool:
        return self._session_dir is not None

    # -- helpers ---------------------------------------------------------

    def _warn(self, title: str, body: str) -> None:
        QMessageBox.warning(self._parent_widget, title, body)
