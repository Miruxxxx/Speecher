from __future__ import annotations

import logging
import queue
from dataclasses import fields
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from app_config import AppConfig
from asr.events import ASREvent
from audio import capture_base
from llm.engine import LLMEngine
from llm.summaries import SummaryHistory
from store.transcript_store import TranscriptStore
from store.transcript_writer import TranscriptWriter
from translate import languages
from translate.store import TranslationStore
from ui import transcript_model
from ui.hotkeys import HotkeyManager, describe
from ui.llm_presenter import LlmPresenter
from ui.session import SessionController
from ui.theme.fonts import load_app_fonts, qcolor
from ui.theme.styles import menu_qss, tooltip_qss
from ui.theme.tokens import (
    COMPACT,
    Border,
    Color,
    LayoutMode,
    Motion,
    NORMAL,
    PANELS_INLINE,
    Radius,
    Space,
    panel_mode,
)
from ui.widgets.buttons import ActionButton
from ui.widgets.cards import AnswerCard, CapturePlate
from ui.widgets.indicators import (
    ERROR,
    LIVE,
    REC_FAILED,
    REC_ON,
    SILENT,
    capture_caption,
    capture_tooltip,
    disk_caption,
    disk_dot,
    llm_caption,
    llm_dot,
    llm_tooltip,
)
from ui.widgets.titlebar import TitleBar
from ui.widgets.transcript import TranscriptView
from ui.windows.history import HistoryWindow
from ui.windows.notification import AdviceWindow, FatalWindow, Notification
from ui.windows.onboarding import CheatSheetWindow, OnboardingWindow
from ui.windows.settings import SettingsWindow
from ui.windows.side import SidePanel, answer_panel, summary_panel
from ui.windows.source import SourceWindow

logger = logging.getLogger(__name__)

# How often the capture indicator is re-read. Cheap (two attribute reads) and
# far below the 200 ms animation step, so the wave never looks stale.
_CAPTURE_POLL_MS = 250

_MENU_MIN_W = 216

# The source window is a feed that runs alongside, not a peek at what the model
# has not reached yet: nothing is dropped once it is on screen, and it keeps
# more than it can show so scrolling up actually finds something.
_SOURCE_WORDS = 400
_SOURCE_CHARS = 2000

# Consecutive pass-through lines after which the overlay admits the
# translation has nothing to translate. Three is past coincidence — a single
# Russian sentence inside an English talk is normal and must not trigger it.
_SKIP_WARN_ROWS = 3


def minutes_caption(minutes: int) -> str:
    """"за 5 минут" — generated, never written by hand (6.6)."""
    if minutes <= 0:
        return "интервал не задан"
    tail = minutes % 10
    hundred = minutes % 100
    if tail == 1 and hundred != 11:
        word = "минуту"
    elif tail in (2, 3, 4) and hundred not in (12, 13, 14):
        word = "минуты"
    else:
        word = "минут"
    return f"за {minutes} {word}"


class Overlay(QWidget):
    """Always-on-top frameless overlay, laid out per the design system.

    Everything the window shows is one of the six screens in section 4; the
    acceptance criterion that shaped the code is the no-shift table: the answer
    card and the capture plate are overlays inside the transcript area, the
    status message has a permanently reserved zone, and the indicator icon slot
    is a fixed 14x14 whatever it draws.
    """

    _status_sig = pyqtSignal(str, str)
    _journal_error_sig = pyqtSignal(str)
    _advice_sig = pyqtSignal(str, str)   # (title, body) — worker thread -> Qt

    def __init__(
        self,
        events_queue: "queue.Queue[ASREvent]",
        store: TranscriptStore,
        llm: LLMEngine,
        cfg: AppConfig,
        config_path: Path,
        *,
        writer: Optional[TranscriptWriter] = None,
        session_dir: Optional[Path] = None,
        capture=None,
        summaries: Optional[SummaryHistory] = None,
        translations: Optional[TranslationStore] = None,
    ) -> None:
        super().__init__()
        self._events = events_queue
        self._store = store
        self._llm = llm
        self._cfg = cfg
        self._config_path = Path(config_path)
        self._writer = writer
        self._capture = capture
        self._summaries = summaries or SummaryHistory()
        self._session_dir = Path(session_dir) if session_dir else None
        self._translations = translations or TranslationStore(cfg.translate.target)
        # The session's data, its journal and its header. Everything the menu
        # does that is not about pixels lives there — see ui/session.py.
        self._session = SessionController(
            store=store,
            translations=self._translations,
            writer=writer,
            session_dir=self._session_dir,
            meta={},
            record_hotkey=cfg.hotkeys.toggle_recording,
            parent=self,
        )
        self._session.status.connect(self.post_status)
        self._session.changed.connect(self._on_session_changed)
        # The LLM state machine. It owns "is a task running / is the provider
        # reachable / what has streamed so far" and never touches a widget; the
        # handlers below turn its transitions into card and indicator updates.
        self._llm_ui = LlmPresenter(
            llm=llm,
            store=store,
            cfg=cfg,
            summaries=self._summaries,
            is_visible=self.isVisible,
            minutes_caption=minutes_caption,
            parent=self,
        )
        # Set by main once the worker exists; None means the feature is off in
        # the config and the toggle says so instead of doing nothing.
        self._translator = None
        self._source_window: Optional[SourceWindow] = None
        self._rows_shown = 0
        # The user closed the source window by hand: translation stays on, but
        # it does not reappear on the next toggle until asked for.
        self._source_dismissed = False
        # Reset whenever translation is switched on; see _warn_if_nothing_to_translate.
        self._skip_warned = False
        self._max_recent_chars = cfg.ui.max_recent_chars
        self._max_words = max(120, cfg.ui.max_recent_chars // 4)

        self._mode: LayoutMode = COMPACT if cfg.ui.compact else NORMAL
        self._partial = ""
        self._fatal_active = False
        self._capture_state = SILENT

        # Where an answer/summary is shown: the in-feed card, or the two windows
        # beside the overlay (§7.9). Built on demand — in the default mode they
        # are never created at all.
        self._panels = panel_mode(cfg.ui.panels)
        self._answer_panel: Optional[SidePanel] = None
        self._summary_panel: Optional[SidePanel] = None
        # Panels hidden by Alt+Space, so the same key brings back exactly the
        # ones that were open — and not the ones the user had closed.
        self._panels_hidden: list = []

        self._settings_window: Optional[SettingsWindow] = None
        self._history_window: Optional[HistoryWindow] = None
        self._about_window: Optional[OnboardingWindow] = None
        self._cheatsheet_window: Optional[CheatSheetWindow] = None
        self._fatal_window: Optional[FatalWindow] = None
        self._advice_window: Optional[AdviceWindow] = None

        load_app_fonts()
        self._setup_window()
        self._build_layout()
        self._connect_signals()
        self._apply_mode(self._mode, animate=False)
        self._refresh_indicators()
        self._refresh_actions()

        self._notification = Notification()
        self._notification.activated.connect(self._on_notification_clicked)

        self._hotkeys = HotkeyManager(hotkey_bindings(cfg.hotkeys), self)
        self._hotkeys.triggered.connect(self._on_hotkey)
        self._hotkeys.start()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_events)
        self._poll_timer.start(50)

        # The punctuation worker rewrites committed words in place without
        # emitting events; re-render periodically so its edits become visible.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.timeout.connect(self._refresh_transcript)
        self._repaint_timer.start(3000)

        self._capture_timer = QTimer(self)
        self._capture_timer.timeout.connect(self._refresh_capture)
        self._capture_timer.start(_CAPTURE_POLL_MS)

        if not cfg.ui.onboarding_shown:
            QTimer.singleShot(700, self._show_onboarding)

    # ------------------------------------------------------------------
    # Window setup and layout
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        # Per-pixel alpha only so the corners can be round; the backdrop itself
        # is fully opaque — no glass, no blur (brief section 3).
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(tooltip_qss())
        self.setMinimumSize(COMPACT.width, COMPACT.height)
        self.resize(self._mode.width, self._mode.height)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._title = TitleBar(self._mode, self)
        self._title.menu_requested.connect(self._on_session_menu)
        self._title.close_requested.connect(self.close)
        self._title.mode_toggled.connect(self.toggle_mode)
        root.addWidget(self._title)

        content = QWidget(self)
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(
            self._mode.padding, self._mode.padding, self._mode.padding, self._mode.padding
        )
        self._content_layout.setSpacing(self._mode.padding)

        # The feed area is a plain container so the cards can be its children
        # and overlay the text without ever entering the layout.
        self._feed_area = QWidget(content)
        feed_layout = QVBoxLayout(self._feed_area)
        feed_layout.setContentsMargins(0, 0, 0, 0)
        self._transcript = TranscriptView(self._mode, self._feed_area)
        feed_layout.addWidget(self._transcript)
        self._content_layout.addWidget(self._feed_area, 1)

        self._answer = AnswerCard(self._mode, self._feed_area)
        # Both cards share the slot at the bottom of the feed; when the answer
        # goes away an unresolved capture failure has to come back.
        self._answer.closed.connect(self._restore_capture_plate)
        self._plate = CapturePlate(self._mode, self._feed_area)
        self._plate.settings_requested.connect(self._open_settings)

        # Three actions and nothing else: the session menu lives in the title
        # bar, and a second entry into it down here bought nothing.
        self._action_row = QHBoxLayout()
        self._action_row.setSpacing(self._mode.action_gap)
        self._btn_answer = ActionButton("answer", "Ответить", "на последний ?")
        self._btn_summary = ActionButton("summary", "Выжимка", "")
        self._btn_question = ActionButton("question", "Последний вопрос", "показать")
        for button in (self._btn_answer, self._btn_summary, self._btn_question):
            self._action_row.addWidget(button, 1)
        self._content_layout.addLayout(self._action_row)
        root.addWidget(content, 1)

    def _connect_signals(self) -> None:
        self._btn_answer.clicked.connect(self._llm_ui.answer)
        self._btn_summary.clicked.connect(lambda: self._llm_ui.summarize())
        self._btn_question.clicked.connect(self._llm_ui.last_question)

        self._llm_ui.state_changed.connect(self._on_llm_state)
        self._llm_ui.answer_started.connect(self._on_llm_started)
        self._llm_ui.answer_updated.connect(self._on_llm_updated)
        self._llm_ui.answer_finished.connect(self._on_llm_finished)
        self._llm_ui.answer_failed.connect(self._on_llm_failed)
        self._llm_ui.summary_added.connect(self._on_summary_added)
        self._llm_ui.question_ready.connect(self._on_question_ready)
        self._llm_ui.answer_dismissed.connect(self._on_llm_dismissed)
        self._llm_ui.notify.connect(self._on_llm_notify)
        self._llm_ui.status.connect(self.post_status)

        self._status_sig.connect(self._on_status)
        self._journal_error_sig.connect(self._on_journal_error)
        self._advice_sig.connect(self._on_advice)

    # ------------------------------------------------------------------
    # Layout modes (section 3)
    # ------------------------------------------------------------------

    def toggle_mode(self) -> None:
        self._apply_mode(COMPACT if self._mode.name == "normal" else NORMAL, animate=True)

    def _apply_mode(self, mode: LayoutMode, *, animate: bool) -> None:
        self._mode = mode
        self._title.set_mode(mode)
        self._content_layout.setContentsMargins(
            mode.padding, mode.padding, mode.padding, mode.padding
        )
        self._content_layout.setSpacing(mode.padding)
        self._action_row.setSpacing(mode.action_gap)
        self._transcript.set_mode(mode)
        self._answer.set_mode(mode)
        self._plate.set_mode(mode)

        # Compact keeps all three actions and drops their captions (6.3).
        compact = not mode.with_labels
        for button in (self._btn_answer, self._btn_summary, self._btn_question):
            button.set_layout(
                icon_only=compact, height=mode.action_h, icon=mode.icon
            )
        self._refresh_actions()

        target = QRect(self.x(), self.y(), mode.width, mode.height)
        if animate:
            self._mode_anim = QPropertyAnimation(self, b"geometry", self)
            self._mode_anim.setDuration(Motion.MODE_SWITCH_MS)
            self._mode_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._mode_anim.setStartValue(self.geometry())
            self._mode_anim.setEndValue(target)
            self._mode_anim.start()
        else:
            self.resize(mode.width, mode.height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._answer.reposition()
        self._plate.reposition()
        # Compact/normal changes the overlay's width, so the window docked to
        # its right edge is no longer at that edge.
        self._reposition_panels()

    # ------------------------------------------------------------------
    # Background painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(qcolor(Color.BORDER))
        pen.setWidthF(Border.WIDTH)
        p.setPen(pen)
        p.setBrush(qcolor(Color.BG_WINDOW))
        p.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
            Radius.WINDOW, Radius.WINDOW,
        )
        p.end()

    def closeEvent(self, event) -> None:
        from PyQt6.QtWidgets import QApplication

        self._hotkeys.stop()
        # Release the prober before the widget goes: it emits a signal bound to
        # this object, and a probe in flight would otherwise land on a
        # half-destroyed window.
        self._llm_ui.stop()
        self._notification.hide()
        if self._source_window is not None:
            self._source_window.hide()
        for panel in self._panel_windows():
            panel.hide()
        QApplication.instance().quit()
        event.accept()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            # Esc closes what is on top of the feed, never the window (section 5).
            if self._answer.isVisible():
                self._answer.fade_out()
            return
        if ev.key() == Qt.Key.Key_C and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self._transcript.textCursor().hasSelection():
                self._transcript.copy()
            else:
                self._session.copy_transcript()
            return
        super().keyPressEvent(ev)

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
        # Translated lines are produced by a worker thread, not by an event, and
        # waiting for the 3 s repaint would hold a finished line off screen for
        # longer than it took to translate. Comparing one integer at 50 ms is
        # cheaper than a queue and cannot drop anything.
        if self.translating() and self._translations.size() != self._rows_shown:
            self._rows_shown = self._translations.size()
            self._refresh_transcript()

    def _handle_asr(self, ev: ASREvent) -> None:
        if ev.type == "commit":
            # The splitter already appended the words to the store (so a
            # dropped UI event can't lose transcript data); just re-render.
            self._partial = ""
            self._refresh_transcript()
            self._refresh_actions()
        elif ev.type == "partial":
            self._partial = ev.text
            self._refresh_transcript()
        elif ev.type == "amend":
            self._refresh_transcript()
        elif ev.type == "fatal":
            if not self._fatal_active:
                self._fatal_active = True
                QTimer.singleShot(0, lambda: self._on_fatal(f"[{ev.source}] {ev.text}"))

    def fatal_active(self) -> bool:
        """True once a fatal error is being surfaced; main defers its quit."""
        return self._fatal_active

    def _on_fatal(self, msg: str) -> None:
        from PyQt6.QtWidgets import QApplication

        self._poll_timer.stop()
        self._repaint_timer.stop()
        self._capture_timer.stop()
        self._fatal_window = FatalWindow(
            f"Пайплайн остановлен, приложение закроется.\n\n{msg}",
            QApplication.instance().quit,
        )
        geo = self.frameGeometry()
        self._fatal_window.move(
            geo.center().x() - self._fatal_window.width() // 2,
            max(0, geo.center().y() - self._fatal_window.height() // 2),
        )
        self._fatal_window.show()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_transcript(self) -> None:
        if self.translating():
            self._refresh_translated()
            return
        entries = self._store.recent_entries(self._max_words)
        replies = transcript_model.replies_from(entries, self._max_recent_chars)
        self._transcript.set_content(replies, self._partial)

    def _refresh_translated(self) -> None:
        """Two feeds while translating: finished lines here, the original there.

        The main feed is append-only — a line is added already translated and is
        never touched again, which is the whole point of the split.

        The source window is the ordinary transcript, not the untranslated
        remainder: lines stay after the model has consumed them, so it reads as
        a feed that simply runs alongside. Showing only what is still pending
        made it empty itself every couple of seconds.
        """
        rows = self._translations.replies()
        self._transcript.set_content(
            transcript_model.tail_by_chars(rows, self._max_recent_chars), ""
        )
        self._warn_if_nothing_to_translate()
        if self._source_window is not None and self._source_window.isVisible():
            entries = self._store.recent_entries(_SOURCE_WORDS)
            self._source_window.set_content(
                transcript_model.replies_from(entries, _SOURCE_CHARS), self._partial
            )

    def _warn_if_nothing_to_translate(self) -> None:
        """Say it when the translation is running but has nothing to do.

        Speech already in the target language is passed through as the original
        (translate/store.py: SKIPPED), so the feed fills with the untranslated
        transcript and the window looks like it is ignoring the setting. Said
        once per switch-on — a status line that repeats every repaint would be
        worse than the confusion it is fixing.
        """
        if self._skip_warned or self._translations.skipped_streak() < _SKIP_WARN_ROWS:
            return
        self._skip_warned = True
        target = languages.caption(self._translations.target)
        self.post_status(
            "Перевод",
            f"Речь уже на {target} — строки идут как есть, переводить нечего",
        )

    def _refresh_capture(self) -> None:
        if self._capture is None:
            return
        state, reason = self._capture.capture_state()
        state = {
            capture_base.CAPTURE_LIVE: LIVE,
            capture_base.CAPTURE_SILENT: SILENT,
            capture_base.CAPTURE_ERROR: ERROR,
        }.get(state, SILENT)
        if state == self._capture_state:
            return
        self._capture_state = state
        self._title.capture.icon_widget().set_state(state)
        self._title.capture.set_caption(capture_caption(state))
        self._title.capture.setToolTip(capture_tooltip(state, reason))
        if state == ERROR:
            self._plate.set_reason(reason)
            if not self._answer.isVisible():
                self._plate.fade_in()
        else:
            self._plate.fade_out()

    def _restore_capture_plate(self) -> None:
        if self._capture_state == ERROR and not self._answer.isVisible():
            self._plate.fade_in()

    def _refresh_indicators(self) -> None:
        state = self._session.journal_state()
        self._title.disk.set_caption(disk_caption(state))
        color, filled = disk_dot(state)
        self._title.disk.icon_widget().set_dot(color, filled)
        self._title.disk.setToolTip(self._session.journal_tooltip(state))

        self._title.llm.set_caption(llm_caption(self._llm_ui.state))
        color, filled, pulsing = llm_dot(self._llm_ui.state)
        self._title.llm.icon_widget().set_dot(color, filled, pulsing)
        self._title.llm.setToolTip(llm_tooltip(self._llm_ui.state))

        self._title.capture.icon_widget().set_state(self._capture_state)
        self._title.capture.set_caption(capture_caption(self._capture_state))
        self._title.capture.setToolTip(capture_tooltip(self._capture_state))
        self._title.set_mode(self._mode, force=True)

    def _refresh_actions(self) -> None:
        has_words = self._store.size() > 0
        online = self._llm_ui.online
        model_ready = online and not self._llm_ui.busy
        interval = self._cfg.ui.summary_interval_min

        answer_ok = model_ready and has_words
        self._btn_answer.setEnabled(answer_ok)
        self._btn_answer.set_sub("на последний ?" if online else "модель офлайн")
        self._btn_summary.setEnabled(model_ready and has_words)
        self._btn_summary.set_sub(
            minutes_caption(interval) if online else "модель офлайн"
        )
        self._btn_question.setEnabled(has_words)
        self._btn_question.set_sub("показать")

        self._btn_answer.setToolTip(self._action_tooltip(
            "Ответить на последний вопрос", self._cfg.hotkeys.answer, answer_ok, has_words
        ))
        self._btn_summary.setToolTip(self._action_tooltip(
            f"Выжимка {minutes_caption(interval)}" if interval else "Выжимка за всю сессию",
            self._cfg.hotkeys.summary, model_ready and has_words, has_words,
        ))
        self._btn_question.setToolTip(self._action_tooltip(
            "Показать последний вопрос", self._cfg.hotkeys.last_question, has_words, has_words
        ))

    def _action_tooltip(self, title: str, combo: str, enabled: bool, has_words: bool) -> str:
        lines = [title]
        if combo:
            lines.append(describe(combo))
        if not enabled:
            lines.append("нет транскрипта" if not has_words else "модель офлайн")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Session menu (2.5 / 4.5)
    # ------------------------------------------------------------------

    def set_session_meta(self, meta: dict) -> None:
        """Engine/model info for the export header. Any thread (dict swap)."""
        self._session._meta = dict(meta)

    def set_engine(self, actual: str, model: str, *, requested: str = "") -> None:
        """Which ASR engine really came up. Called from the loader thread."""
        self._session.set_engine(actual, model, requested=requested)

    def _on_session_changed(self) -> None:
        """A resume dropped a previous session into the store."""
        self._refresh_transcript()
        self._refresh_actions()

    def _on_session_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(menu_qss())
        menu.setMinimumWidth(_MENU_MIN_W)

        has_words = self._store.size() > 0

        # Which engine is actually running. A disabled row, like the broken
        # journal below: this is the state of the session, not a control. It
        # exists because the fallback used to be announced only by a status
        # message that clears itself, and the app then ran for hours on an
        # engine the user had not chosen.
        if self._session.engine_actual:
            act_engine = menu.addAction(self._session.engine_caption())
            act_engine.setEnabled(False)
            if self._session.engine_requested:
                act_engine.setToolTip(
                    f"{self._session.engine_requested} не загрузился — причина в логе"
                )
            menu.addSeparator()

        # The journal toggle has to live somewhere clickable: the disk pill is an
        # indicator, not a control, and Alt+R is not guaranteed — Windows may
        # already own the combination.
        journal = self._session.journal_state()
        if journal == REC_FAILED:
            act_record = menu.addAction("Запись сорвалась — журнал отключён")
            act_record.setEnabled(False)
        elif journal == REC_ON:
            act_record = menu.addAction("Остановить запись сессии")
        else:
            act_record = menu.addAction("Продолжить запись сессии")
            act_record.setEnabled(self._writer is not None)

        act_translate = menu.addAction(self._translation_caption())
        act_translate.setEnabled(
            self._translator is not None and not self._translator.failure
        )
        act_source = None
        if self.translating():
            source_open = (
                self._source_window is not None and self._source_window.isVisible()
            )
            act_source = menu.addAction(
                "Скрыть окно оригинала" if source_open else "Показать окно оригинала"
            )

        menu.addSeparator()
        act_md = menu.addAction("Экспорт в Markdown…")
        act_txt = menu.addAction("Экспорт в текст…")
        act_copy = menu.addAction("Копировать транскрипт")
        for action in (act_md, act_txt, act_copy):
            action.setEnabled(has_words)

        menu.addSeparator()
        last = self._session.latest_session()
        resumable = self._session.can_resume()
        # Not hidden when unavailable — disabled with the reason (4.5).
        caption = "Загрузить сессию…"
        if not resumable:
            reason = "идёт запись" if has_words else "нет сохранённых сессий"
            caption = f"{caption} ({reason})"
        act_resume = menu.addAction(caption)
        act_resume.setEnabled(resumable)
        act_open = menu.addAction("Открыть папку сессии…")
        act_open.setEnabled(self._session.has_session_dir)

        menu.addSeparator()
        act_history = menu.addAction("История выжимок…")
        act_history.setEnabled(self._summaries.size() > 0)
        act_keys = menu.addAction("Хоткеи…")
        act_settings = menu.addAction("Настройки…")
        act_about = menu.addAction("О программе…")

        anchor = self._title.menu_button
        point = anchor.mapToGlobal(anchor.rect().bottomRight())
        chosen = menu.exec(point + _menu_offset(menu.sizeHint().width()))
        if chosen is None:
            return
        if chosen is act_record:
            self.toggle_recording()
        elif chosen is act_translate:
            self.toggle_translation()
        elif act_source is not None and chosen is act_source:
            if self._source_window is not None and self._source_window.isVisible():
                self._source_window.close()
            else:
                self.show_source_window()
        elif chosen is act_md:
            self._session.export("md")
        elif chosen is act_txt:
            self._session.export("txt")
        elif chosen is act_copy:
            self._session.copy_transcript()
        elif chosen is act_resume and last is not None:
            self._session.resume(last)
        elif chosen is act_open:
            self._session.open_session_dir()
        elif chosen is act_history:
            self._open_history()
        elif chosen is act_keys:
            self._open_cheatsheet()
        elif chosen is act_settings:
            self._open_settings()
        elif chosen is act_about:
            self._show_onboarding(intro=False)

    def _translation_caption(self) -> str:
        """Menu entries state why they are unavailable rather than hiding (4.5)."""
        target = languages.caption(self._cfg.translate.target)
        if self._translator is None:
            return "Перевод на лету (выключен в конфиге)"
        if self._translator.failure:
            return "Перевод на лету (модель не загрузилась)"
        if self._translator.enabled:
            return "Выключить перевод на лету"
        return f"Перевод на лету → {target}"

    # ------------------------------------------------------------------
    # Satellite windows
    # ------------------------------------------------------------------

    def _open_settings(self) -> None:
        if self._settings_window is None:
            self._settings_window = SettingsWindow(
                self._cfg,
                self._config_path,
                None,
                hotkey_failures=self._hotkeys.failures(),
                on_status=self.post_status,
            )
            self._settings_window.applied.connect(self._on_settings_applied)
        self._settings_window.set_hotkey_failures(self._hotkeys.failures())
        # The window is built once and reused, so its controls must be re-synced
        # with anything that can change behind its back. Translation is the one
        # that can: Alt+T and the session menu flip it for the session without
        # touching the config the window was built from.
        self._settings_window.set_translate_enabled(
            self.translating(),
            available=self._translator is not None and not self._translator.failure,
        )
        self._settings_window.center_on(self)
        self._settings_window.show()
        self._settings_window.raise_()
        self._settings_window.activateWindow()

    def _on_settings_applied(self, changes: dict) -> None:
        self._llm_ui.apply_auto_summary()
        self.set_panels_mode(changes.get("panels", self._panels))
        self._hotkeys.rebind(changes.get("hotkeys", {}))
        client = changes.get("llm_client")
        if client is not None:
            # A new provider/key/model takes effect on the next task; re-probe
            # so the title-bar indicator stops showing the old one's state.
            self._llm.set_client(client)
            self._llm_ui.reset_provider()
        if self._translator is not None and not self._translator.failure:
            self._translator.set_enabled(bool(changes.get("translate_enabled")))
            self._apply_translation_ui()
        if self._settings_window is not None:
            self._settings_window.set_hotkey_failures(self._hotkeys.failures())
        self._refresh_actions()
        self._refresh_indicators()
        if changes.get("needs_restart"):
            self.post_status("Перезапуск", "Часть настроек применится после перезапуска")

    def _open_history(self) -> None:
        if self._history_window is None:
            self._history_window = HistoryWindow(None)
        self._history_window.set_summaries(self._summaries.all())
        self._history_window.center_on(self)
        self._history_window.show()
        self._history_window.raise_()
        self._history_window.activateWindow()

    def _open_cheatsheet(self) -> None:
        if self._cheatsheet_window is None:
            self._cheatsheet_window = CheatSheetWindow(self._cfg.hotkeys, None)
        self._cheatsheet_window.center_on(self)
        self._cheatsheet_window.show()
        self._cheatsheet_window.raise_()

    def _show_onboarding(self, intro: bool = True) -> None:
        self._about_window = OnboardingWindow(self._cfg.hotkeys, None, intro=intro)
        self._about_window.center_on(self)
        self._about_window.show()
        self._about_window.raise_()
        if intro:
            self._remember_onboarding()

    def _remember_onboarding(self) -> None:
        from app_config import save_overrides

        self._cfg.ui.onboarding_shown = True
        try:
            save_overrides(self._config_path, {"ui": {"onboarding_shown": True}})
        except OSError as exc:
            logger.warning("cannot persist onboarding_shown: %s", exc)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_question_ready(self, question: str) -> None:
        """The presenter found the last question — instantly from a '?', or via
        the model when the engine never emitted one."""
        if not self.isVisible():
            # Triggered by a hotkey with the window hidden: the card has nowhere
            # to appear, so the notification carries the answer (2.6).
            self._notification.announce(
                "Последний вопрос",
                question,
                icon="question",
            )
            return
        if self._panels == PANELS_INLINE:
            self._plate.fade_out()
        self._target("question").show_text("Последний вопрос", question)

    # ------------------------------------------------------------------
    # LLM presentation — the state lives in ui/llm_presenter.py; these are the
    # widget reactions to it, and nothing else.
    # ------------------------------------------------------------------

    def _target(self, kind: str = ""):
        """The card or one of the two side windows — whatever the mode says.

        Both have the same surface (`begin`/`set_answer`/`show_text`/`finish`/
        `fail`/`fade_out`), so every handler below is written once and does not
        know which one it got. `kind` is the presenter's: only "summary" goes
        right, everything else — an answer, a found question, a failure — is on
        the left, because they are one conversation.
        """
        if self._panels == PANELS_INLINE:
            return self._answer
        if kind == "summary":
            if self._summary_panel is None:
                self._summary_panel = summary_panel(self._panels, self)
            return self._summary_panel
        if self._answer_panel is None:
            self._answer_panel = answer_panel(self._panels, self)
        return self._answer_panel

    def _panel_windows(self) -> list:
        return [p for p in (self._answer_panel, self._summary_panel) if p is not None]

    def set_panels_mode(self, mode: str) -> None:
        """Applied live from the settings window."""
        mode = panel_mode(mode)
        if mode == self._panels:
            return
        self._panels = mode
        for panel in self._panel_windows():
            if mode == PANELS_INLINE:
                panel.hide()
            else:
                panel.set_panel_mode(mode)
                panel.place_beside(force=True)

    def _reposition_panels(self) -> None:
        for panel in self._panel_windows():
            if panel.isVisible():
                panel.place_beside()

    def moveEvent(self, event) -> None:
        # The side windows are docked to the overlay's edges, so dragging the
        # overlay takes them along — unless the user has moved one by hand.
        super().moveEvent(event)
        self._reposition_panels()

    def _on_llm_state(self, _state: str) -> None:
        self._refresh_indicators()
        self._refresh_actions()

    def _on_llm_started(self, placeholder: str) -> None:
        if self._panels == PANELS_INLINE:
            # The card shares its slot with the capture plate; a side window
            # does not, and an unresolved capture failure must stay visible.
            self._plate.fade_out()
        self._target(self._llm_ui.kind).begin(placeholder)
        self._refresh_actions()

    def _on_llm_updated(self, text: str) -> None:
        self._target(self._llm_ui.kind).set_answer(text)

    def _on_llm_finished(self) -> None:
        self._target(self._llm_ui.kind).finish()
        self._refresh_actions()

    def _on_llm_failed(self, text: str, empty: bool) -> None:
        # `empty` says nothing had streamed yet, so the card shows the failure
        # on its own instead of appending it to a half-written answer.
        target = self._target(self._llm_ui.kind)
        if empty:
            target.fail(text)
        else:
            target.set_answer(text)
        self._refresh_actions()

    def _on_llm_dismissed(self) -> None:
        """The task produced nothing worth showing — take the surface away.

        Deliberately not through `_target`: that would build a side window in
        order to hide it.
        """
        if self._panels == PANELS_INLINE:
            self._answer.fade_out()
            return
        panel = (
            self._summary_panel if self._llm_ui.kind == "summary" else self._answer_panel
        )
        if panel is not None:
            panel.fade_out()

    def _on_summary_added(self) -> None:
        if self._history_window is not None and self._history_window.isVisible():
            self._history_window.set_summaries(self._summaries.all())

    def _on_llm_notify(self, title: str, hotkey_action: str, icon: str) -> None:
        """The presenter says what happened; the window knows which key opens it."""
        combo = getattr(self._cfg.hotkeys, hotkey_action, "")
        self._notification.announce(
            title, f"Клик или {describe(combo)} — открыть", icon=icon
        )

    def _on_notification_clicked(self) -> None:
        self._open_history()

    # ------------------------------------------------------------------
    # Status line (thread-safe entry point for loader threads)
    # ------------------------------------------------------------------

    def post_status(self, text: str, detail: str = "") -> None:
        """Show text in the title bar's status zone. Any thread.

        The zone is narrow (what the three labelled pills leave of 180 px), so
        callers pass a short label and the full sentence as `detail` — that is
        what the tooltip shows, per 2.6's "полный текст в тултипе".
        """
        self._status_sig.emit(text, detail)

    def _on_status(self, text: str, detail: str = "") -> None:
        if text:
            self._title.status.show_message(text, detail=detail)
        else:
            self._title.status.clear()

    def post_advice(self, message: str, title: str = "Не хватило видеопамяти") -> None:
        """Surface a "working, but not as configured" state. Any thread.

        Distinct from `post_status`, which is a line that clears itself in
        seconds: this is for a downgrade the user has to actually know about,
        such as live translation loading on the CPU because the GPU had no room
        left beside the ASR model.
        """
        self._advice_sig.emit(title, message)

    def _on_advice(self, title: str, message: str) -> None:
        self._advice_window = AdviceWindow(title, message)
        geo = self.frameGeometry()
        self._advice_window.move(
            geo.center().x() - self._advice_window.width() // 2,
            max(0, geo.center().y() - self._advice_window.height() // 2),
        )
        self._advice_window.show()
        self._advice_window.raise_()

    def post_journal_error(self, message: str) -> None:
        """Transcript journal died (called from the writing thread). Any thread."""
        self._journal_error_sig.emit(message)

    def _on_journal_error(self, message: str) -> None:
        self._refresh_indicators()
        self._title.status.show_message(
            "Журнал", detail=f"Запись отключена: {message}", error=True
        )

    # ------------------------------------------------------------------
    # Recording / hotkeys
    # ------------------------------------------------------------------

    def toggle_recording(self) -> None:
        self._session.toggle_recording()
        self._refresh_indicators()

    # ------------------------------------------------------------------
    # Live translation
    # ------------------------------------------------------------------

    def set_translator(self, worker) -> None:
        """Hand over the TranslationWorker main started. Any thread (one store)."""
        self._translator = worker
        if worker is not None and worker.enabled:
            QTimer.singleShot(0, self._apply_translation_ui)

    def translating(self) -> bool:
        return self._translator is not None and self._translator.enabled

    def toggle_translation(self) -> None:
        """Alt+T. Session-scoped, like the ⏺ toggle: `[translate] enabled` in
        the config stays whatever the user set it to."""
        if self._translator is None:
            reason = "Перевод выключен в config.toml ([translate] backend)"
            self.post_status("Перевод", reason)
            return
        if self._translator.failure:
            self.post_status("Перевод", f"Недоступен: {self._translator.failure}")
            return
        on = not self._translator.enabled
        self._translator.set_enabled(on)
        if on:
            self._source_dismissed = False  # an explicit "on" means show both
        target = languages.caption(self._translator.target)
        self.post_status(
            "Перевод",
            f"Перевод на {target} включён" if on else "Перевод выключен, показан оригинал",
        )
        self._apply_translation_ui()

    def _apply_translation_ui(self) -> None:
        """Switch the window between one feed and two."""
        on = self.translating()
        if not on:
            # Cleared while off, so the "nothing to translate" notice is shown
            # once per switch-on rather than once per session.
            self._skip_warned = False
        if on and self._cfg.translate.show_source and not self._source_dismissed:
            self._open_source_window()
        elif self._source_window is not None:
            self._source_window.hide()
        # The caret is the sign that words are still arriving, and while
        # translating they arrive in the *other* window.
        self._transcript.set_caret_visible(not on)
        self._rows_shown = self._translations.size()
        self._refresh_transcript()

    def _open_source_window(self) -> None:
        if self._source_window is None:
            self._source_window = SourceWindow(parent=None)
            self._source_window.closed.connect(self._on_source_closed)
            self._source_window.dock_under(self)
        self._source_window.show()
        # Never in front of the overlay it belongs to: this window is a glance,
        # and it must not cover the feed the user is actually reading.
        self.raise_()

    def _on_source_closed(self) -> None:
        self._source_dismissed = True

    def show_source_window(self) -> None:
        """Menu entry: bring the live original back after it was closed."""
        self._source_dismissed = False
        self._open_source_window()

    def pause_recording(self) -> None:
        self._session.pause_recording()
        self._refresh_indicators()

    def toggle_visibility(self) -> None:
        """Alt+Space and nothing else (6.4)."""
        if self.isVisible():
            self._fade_window(0.0, hide=True)
            # The source window is part of this window, not a separate app: it
            # goes away with it and comes back with it. Same for the two side
            # panels: hiding the overlay must not leave an answer on screen.
            if self._source_window is not None:
                self._source_window.hide()
            self._panels_hidden = [p for p in self._panel_windows() if p.isVisible()]
            for panel in self._panels_hidden:
                panel.hide()
        else:
            self.setWindowOpacity(0.0)
            self.show()
            self.raise_()
            self._fade_window(1.0, hide=False)
            self._notification.dismiss()
            if self.translating():
                self._apply_translation_ui()
            for panel in self._panels_hidden:
                panel.appear()
            self._panels_hidden = []

    def _fade_window(self, target: float, *, hide: bool) -> None:
        self._window_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._window_anim.setDuration(Motion.WINDOW_TOGGLE_MS)
        self._window_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._window_anim.setStartValue(self.windowOpacity())
        self._window_anim.setEndValue(target)
        if hide:
            self._window_anim.finished.connect(self.hide)
        self._window_anim.start()

    def _on_hotkey(self, action: str) -> None:
        if action == "toggle_window":
            self.toggle_visibility()
            return
        if action == "toggle_recording":
            self.toggle_recording()
            return
        if action == "pause_recording":
            self.pause_recording()
            return
        if action == "history":
            self._notification.dismiss()
            self._open_history()
            return
        if action == "translate":
            self.toggle_translation()
            return
        # The remaining actions produce a result; when the window is hidden the
        # result announces itself when it is ready (2.6), not when the key is
        # pressed — so nothing extra happens here.
        if action == "summary":
            self._llm_ui.summarize()
        elif action == "answer":
            self._llm_ui.answer()
        elif action == "last_question":
            self._llm_ui.last_question()


def hotkey_bindings(hotkeys) -> dict:
    """Field name -> combination. `vars()` does not work on a slotted dataclass."""
    return {f.name: getattr(hotkeys, f.name) for f in fields(hotkeys)}


def _menu_offset(menu_width: int) -> QPoint:
    """Menu drops 6 px below the button and sits 8 px from the right edge (4.5)."""
    return QPoint(-menu_width + Space.S2, 6)
