from __future__ import annotations

import queue
import threading
from typing import Optional

from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QInputDialog,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QSizeGrip,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asr.events import ASREvent
from llm.engine import LLMEngine, LLMTask
from store.transcript_store import TranscriptStore


_TRANSCRIPT_CSS = (
    "QTextEdit { background: transparent; color: #d4d4d4; border: none; "
    "selection-background-color: rgba(100,100,200,120); }"
)
_LLM_CSS = (
    "QTextEdit { background: transparent; color: #7ecfff; border: none; }"
)
_BTN_CSS = (
    "QPushButton { background: rgba(55,55,75,180); color: #d4d4d4; "
    "border: 1px solid rgba(140,140,170,100); border-radius: 4px; "
    "padding: 3px 10px; font-size: 11px; }"
    "QPushButton:hover { background: rgba(75,75,105,210); }"
    "QPushButton:pressed { background: rgba(35,35,55,230); }"
    "QPushButton:disabled { color: rgba(140,140,140,120); "
    "background: rgba(40,40,55,120); }"
)
_CLOSE_CSS = (
    "QPushButton { background: rgba(160,45,45,160); color: #ffffff; "
    "border: none; border-radius: 3px; padding: 0; font-size: 11px; "
    "min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px; }"
    "QPushButton:hover { background: rgba(210,65,65,210); }"
)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Overlay(QWidget):
    """
    Always-on-top frameless overlay showing live ASR transcript + LLM results.
    Drag anywhere on the window to reposition; resize via the bottom-right grip.
    """

    _llm_token_sig = pyqtSignal(str)
    _llm_done_sig = pyqtSignal()
    _llm_error_sig = pyqtSignal(str)
    _status_sig = pyqtSignal(str)
    _lm_health_sig = pyqtSignal(bool)

    def __init__(
        self,
        events_queue: "queue.Queue[ASREvent]",
        store: TranscriptStore,
        llm: LLMEngine,
        max_recent_chars: int = 700,
        max_summary_chars: int = 6000,
    ) -> None:
        super().__init__()
        self._events = events_queue
        self._store = store
        self._llm = llm
        self._max_recent_chars = max_recent_chars
        self._max_summary_chars = max_summary_chars
        self._partial = ""
        self._llm_output = ""
        self._llm_busy = False
        self._fatal_active = False
        self._drag_pos: Optional[QPoint] = None

        self._setup_window()
        self._build_layout()
        self._connect_signals()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_events)
        self._poll_timer.start(50)

        # The punctuation worker rewrites committed words in place without
        # emitting events; re-render periodically so its edits become visible.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.timeout.connect(self._refresh_transcript)
        self._repaint_timer.start(3000)

        # Re-probe LM Studio periodically so the banner tracks the server going
        # up or down after startup, not just its state 500 ms in.
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_lm_availability)
        self._health_timer.start(20000)
        QTimer.singleShot(500, self._check_lm_availability)

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(360, 200)
        self.resize(520, 340)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        mono = QFont("Consolas", 10)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 6)
        root.setSpacing(5)

        # LM Studio warning banner (hidden when server is available)
        self._lm_warning = QLabel("⚠ LM Studio недоступен — запустите сервер на :1234")
        self._lm_warning.setStyleSheet(
            "QLabel { color: #ffaa33; background: rgba(80,50,0,160); "
            "padding: 3px 8px; border-radius: 3px; font-size: 10px; }"
        )
        self._lm_warning.setVisible(False)
        root.addWidget(self._lm_warning)

        # App status line (model loading progress, load errors)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            "QLabel { color: #9fd49f; background: rgba(30,60,30,150); "
            "padding: 3px 8px; border-radius: 3px; font-size: 10px; }"
        )
        self._status_lbl.setVisible(False)
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        # Title bar / drag handle
        title_row = QHBoxLayout()
        title_row.setContentsMargins(2, 0, 0, 0)
        title_lbl = QLabel("🎤 Speecher")
        title_lbl.setStyleSheet("color: rgba(190,190,210,160); font-size: 11px;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setStyleSheet(_CLOSE_CSS)
        close_btn.clicked.connect(self.close)
        title_row.addWidget(close_btn)
        root.addLayout(title_row)

        # Committed transcript + live partial
        self._transcript_view = QTextEdit()
        self._transcript_view.setReadOnly(True)
        self._transcript_view.setStyleSheet(_TRANSCRIPT_CSS)
        self._transcript_view.setFont(mono)
        self._transcript_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        root.addWidget(self._transcript_view, stretch=3)

        # LLM result (hidden until a button is pressed)
        self._llm_view = QTextEdit()
        self._llm_view.setReadOnly(True)
        self._llm_view.setStyleSheet(_LLM_CSS)
        self._llm_view.setFont(mono)
        self._llm_view.setVisible(False)
        root.addWidget(self._llm_view, stretch=2)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_last_q = QPushButton("Last ?")
        self._btn_ai = QPushButton("Last ? + AI")
        self._btn_sum5 = QPushButton("Sum 5min")
        self._btn_sumx = QPushButton("Sum Xmin…")
        for btn in (self._btn_last_q, self._btn_ai, self._btn_sum5, self._btn_sumx):
            btn.setStyleSheet(_BTN_CSS)
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

        # Resize grip
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch()
        grip = QSizeGrip(self)
        grip.setFixedSize(14, 14)
        grip_row.addWidget(
            grip,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        )
        root.addLayout(grip_row)

    def _connect_signals(self) -> None:
        self._btn_last_q.clicked.connect(self._on_last_q)
        self._btn_ai.clicked.connect(self._on_last_q_ai)
        self._btn_sum5.clicked.connect(self._on_sum5)
        self._btn_sumx.clicked.connect(self._on_sumx)

        self._llm_token_sig.connect(self._on_llm_token)
        self._llm_done_sig.connect(self._on_llm_done)
        self._llm_error_sig.connect(self._on_llm_error)
        self._status_sig.connect(self._on_status)
        self._lm_health_sig.connect(self._on_lm_health)

    # ------------------------------------------------------------------
    # Background painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(18, 18, 28, 215)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 10, 10)

    # ------------------------------------------------------------------
    # Dragging
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if self._drag_pos is not None and (ev.buttons() & Qt.MouseButton.LeftButton):
            self.move(ev.globalPosition().toPoint() - self._drag_pos)
            ev.accept()

    def mouseReleaseEvent(self, _ev) -> None:
        self._drag_pos = None

    def closeEvent(self, event) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().quit()
        event.accept()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            self.close()

    # ------------------------------------------------------------------
    # ASR event polling (Qt main thread via QTimer)
    # ------------------------------------------------------------------

    def _poll_events(self) -> None:
        for _ in range(30):
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            self._handle_asr(ev)
            self._events.task_done()

    def _handle_asr(self, ev: ASREvent) -> None:
        if ev.type == "commit":
            # The splitter already appended the words to the store (so a
            # dropped UI event can't lose transcript data); just re-render.
            self._partial = ""
            self._refresh_transcript()
        elif ev.type == "partial":
            self._partial = ev.text
            self._refresh_transcript()
        elif ev.type == "amend":
            # The splitter already appended the punctuation to the last word
            # in the store; just re-render so it shows without waiting for the
            # repaint timer.
            self._refresh_transcript()
        elif ev.type == "fatal":
            # Show the reason before the app tears down. main's shutdown timer
            # defers to fatal_active() so the process doesn't quit underneath
            # this dialog (its stop_event is already set). Guard re-entry: more
            # fatal events may be queued behind this one.
            if not self._fatal_active:
                self._fatal_active = True
                QTimer.singleShot(0, lambda: self._on_fatal(f"[{ev.source}] {ev.text}"))

    def fatal_active(self) -> bool:
        """True once a fatal error is being surfaced; main defers its quit."""
        return self._fatal_active

    def _on_fatal(self, msg: str) -> None:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        self._poll_timer.stop()
        self._repaint_timer.stop()
        self._health_timer.stop()
        QMessageBox.critical(
            self,
            "Speecher — фатальная ошибка",
            f"Пайплайн остановлен и приложение закроется:\n\n{msg}",
        )
        QApplication.instance().quit()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _refresh_transcript(self) -> None:
        committed = _esc(self._store.recent_text(self._max_recent_chars))
        html = f'<span style="color:#d4d4d4">{committed}</span>'
        if self._partial:
            html += f'<br><span style="color:#ffdd55">⚡ {_esc(self._partial)}</span>'
        self._transcript_view.setHtml(html)
        sb = self._transcript_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_llm_output(self, text: str) -> None:
        self._llm_output = text
        self._llm_view.setVisible(bool(text))
        self._llm_view.setPlainText(text)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_last_q(self) -> None:
        q = self._store.find_last_question()
        self._show_llm_output(
            f"[?] {q}" if q else "[No question found in transcript]"
        )

    def _on_last_q_ai(self) -> None:
        if self._llm_busy:
            return
        q = self._store.find_last_question()
        if not q:
            self._show_llm_output("[No question found in transcript]")
            return
        ctx = self._store.context_for_question(q, max_chars=1500)
        prompt = (
            "You are given a speech transcript excerpt. "
            "Answer the question that appears at the end concisely and helpfully. "
            "Reply in the same language as the question.\n\n"
            f"Transcript:\n{ctx}\n\nQuestion: {q}"
        )
        self._start_llm(prompt)

    def _on_sum5(self) -> None:
        self._do_summarize(5.0)

    def _on_sumx(self) -> None:
        minutes, ok = QInputDialog.getDouble(
            self,
            "Summary Duration",
            "Summarize last N minutes:",
            value=10.0,
            min=1.0,
            max=120.0,
            decimals=0,
        )
        if ok:
            self._do_summarize(minutes)

    def _do_summarize(self, minutes: float) -> None:
        if self._llm_busy:
            return
        words = self._store.words_since_minutes(minutes)
        text = " ".join(w.text for w in words).strip()
        if not text:
            self._show_llm_output(
                f"[No transcript in the last {int(minutes)} minutes]"
            )
            return
        # Keep the most recent text within budget so a long window can't
        # silently overflow the local model's context (LM Studio would drop
        # the middle without telling us).
        truncated = len(text) > self._max_summary_chars
        if truncated:
            text = text[-self._max_summary_chars:]
        prompt = (
            f"Summarize the following speech transcript from the last {int(minutes)} "
            "minutes. Focus on key points, decisions, and important information. "
            "Reply in the same language as the transcript.\n\n"
            + ("[earlier text omitted]\n" if truncated else "")
            + text
        )
        self._start_llm(prompt)

    # ------------------------------------------------------------------
    # Status line (thread-safe entry point for loader threads)
    # ------------------------------------------------------------------

    def post_status(self, text: str) -> None:
        """Show text in the status line; empty string hides it. Any thread."""
        self._status_sig.emit(text)

    def _on_status(self, text: str) -> None:
        self._status_lbl.setText(text)
        self._status_lbl.setVisible(bool(text))

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _check_lm_availability(self) -> None:
        # A concurrent health probe mid-generation could flip the banner on a
        # transient miss while tokens are clearly flowing; skip it while busy.
        if self._llm_busy:
            return
        # models.list() is a network call — keep it off the UI thread.
        def probe() -> None:
            self._lm_health_sig.emit(self._llm.is_available())

        threading.Thread(target=probe, name="lm-health", daemon=True).start()

    def _on_lm_health(self, ok: bool) -> None:
        self._lm_warning.setVisible(not ok)

    def _start_llm(self, prompt: str) -> None:
        # No availability pre-check here: it would block the UI thread.
        # The LLM thread checks and reports back through on_error.
        self._llm_busy = True
        self._set_buttons_enabled(False)
        self._show_llm_output("…")
        self._llm.submit(
            LLMTask(
                type="answer",
                prompt=prompt,
                on_token=self._llm_token_sig.emit,
                on_done=self._llm_done_sig.emit,
                on_error=self._llm_error_sig.emit,
            )
        )

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self._btn_last_q, self._btn_ai, self._btn_sum5, self._btn_sumx):
            btn.setEnabled(enabled)

    def _on_llm_token(self, token: str) -> None:
        self._lm_warning.setVisible(False)  # tokens flowing => server is alive
        if self._llm_output == "…":
            self._llm_output = ""
        self._llm_output += token
        self._llm_view.setVisible(True)
        self._llm_view.setPlainText(self._llm_output)
        sb = self._llm_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_llm_done(self) -> None:
        self._llm_busy = False
        self._set_buttons_enabled(True)

    def _on_llm_error(self, msg: str) -> None:
        self._llm_busy = False
        self._set_buttons_enabled(True)
        if "недоступен" in msg:
            self._lm_warning.setVisible(True)
        # Keep any tokens already streamed (a mid-stream failure shouldn't wipe
        # a half-written answer); append the error instead of replacing.
        partial = self._llm_output if self._llm_output not in ("", "…") else ""
        prefix = partial + "\n\n" if partial else ""
        self._show_llm_output(f"{prefix}[LLM Error] {msg}")
