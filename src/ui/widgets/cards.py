"""Overlay cards inside the transcript area (design system 2.4, 4.4).

Both the model's answer and the capture-error plate live in the same slot: the
bottom of the transcript area, as an *overlay* rather than an element of the
flow. That is the whole point — the feed is never re-laid-out when a card
appears or disappears, which is the acceptance criterion in section 4.

Height grows upward from the bottom edge as the stream arrives, capped by the
mode (160 px normal, 96 px compact); past the cap the body scrolls internally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui import markup
from ui.theme import icons
from ui.theme.fonts import css, qcolor, qfont
from ui.theme.styles import label_qss, scrollbar_qss
from ui.theme.tokens import (
    Border,
    Color,
    LayoutMode,
    Motion,
    NORMAL,
    Radius,
    Space,
    Type,
)
from ui.widgets.buttons import IconButton, TextButton

_HEADER_ICON = 18
_CLOSE_BOX = 20


class _OverlayCard(QWidget):
    """Shared frame: card background, strong border, bottom-anchored geometry,
    120 ms opacity fade. No shadow — a frameless window cannot afford one, so
    the border does the separating."""

    closed = pyqtSignal()

    def __init__(self, mode: LayoutMode, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mode = mode
        self.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation, True)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(Motion.CARD_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.finished.connect(self._after_fade)
        self.hide()

    def set_mode(self, mode: LayoutMode) -> None:
        self._mode = mode
        self._relayout()
        self.reposition()

    def _relayout(self) -> None:
        """Re-apply mode-dependent paddings. Overridden by subclasses."""

    # -- show / hide -----------------------------------------------------

    def fade_in(self) -> None:
        self.reposition()
        self.show()
        self.raise_()
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(1.0)
        self._fade.start()

    def fade_out(self) -> None:
        if not self.isVisible():
            return
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _after_fade(self) -> None:
        if self._effect.opacity() <= 0.01:
            self.hide()
            self.closed.emit()

    # -- geometry --------------------------------------------------------

    def content_height(self) -> int:
        raise NotImplementedError

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        height = min(self.content_height(), self._mode.answer_max_h)
        self.setGeometry(0, parent.height() - height, parent.width(), height)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(qcolor(Color.BORDER_STRONG))
        pen.setWidthF(Border.WIDTH)
        p.setPen(pen)
        p.setBrush(qcolor(Color.BG_CARD))
        p.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), Radius.PANEL, Radius.PANEL
        )
        p.end()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.fade_out()
            return
        super().keyPressEvent(event)


class AnswerCard(_OverlayCard):
    """The model's answer, section 2.4.

    Header (icon plate, title, time, close), body, nothing else: no ratings, no
    like buttons — there is nowhere to send that signal (brief section 8).
    """

    def __init__(self, mode: LayoutMode = NORMAL, parent: Optional[QWidget] = None) -> None:
        super().__init__(mode, parent)
        self._streaming = False

        root = QVBoxLayout(self)
        root.setContentsMargins(
            mode.answer_pad_h, mode.answer_pad_v, mode.answer_pad_h, mode.answer_pad_v
        )
        root.setSpacing(Space.S2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(Space.S2)
        self._icon = QLabel()
        self._icon.setFixedSize(_HEADER_ICON, _HEADER_ICON)
        self._icon.setPixmap(icons.pixmap("answer", _HEADER_ICON, Color.ACCENT_ON))
        self._icon.setStyleSheet(
            f"QLabel {{ background: {Color.ACCENT_SUBTLE};"
            f" border-radius: {Radius.PLATE}px; }}"
        )
        header.addWidget(self._icon)

        self._title = QLabel("Ответ модели")
        self._title.setFont(qfont(Type.PANEL_TITLE))
        self._title.setStyleSheet(label_qss(Type.PANEL_TITLE))
        header.addWidget(self._title)
        header.addStretch()

        self._time = QLabel("")
        self._time.setFont(qfont(Type.TIMESTAMP))
        self._time.setStyleSheet(label_qss(Type.TIMESTAMP))
        header.addWidget(self._time)

        self._close = IconButton("close", _CLOSE_BOX, 12, self)
        self._close.setToolTip("Закрыть (Esc)")
        self._close.clicked.connect(self.fade_out)
        header.addWidget(self._close)
        root.addLayout(header)

        self._body = QTextEdit()
        self._body.setReadOnly(True)
        self._body.setFrameStyle(0)
        self._body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body.setStyleSheet(
            f"QTextEdit {{ background: transparent; border: none; padding: 0;"
            f" {css(Type.ANSWER)}"
            f" selection-background-color: {Color.ACCENT_ACTIVE}; }}"
            + scrollbar_qss()
        )
        root.addWidget(self._body)

        self._autoclose = QTimer(self)
        self._autoclose.setSingleShot(True)
        self._autoclose.timeout.connect(self.fade_out)

    def _relayout(self) -> None:
        m = self._mode
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(
                m.answer_pad_h, m.answer_pad_v, m.answer_pad_h, m.answer_pad_v
            )
        self._time.setVisible(m.with_labels)

    # -- content ---------------------------------------------------------

    def begin(self, title: str = "Думаю…") -> None:
        """Loading state: title changes, body is one secondary line, no spinner."""
        self._streaming = True
        self._autoclose.stop()
        self._title.setText(title)
        self._time.setText(datetime.now().strftime("%H:%M:%S") if self._mode.with_labels else "")
        self._set_body("Собираю ответ…", Color.TEXT_SECONDARY)
        self.fade_in()

    def set_answer(self, text: str) -> None:
        self._title.setText("Ответ модели")
        self._set_body(text, Color.TEXT_PRIMARY, markdown=True)
        self.reposition()

    def show_text(self, title: str, text: str) -> None:
        """Same card, no model involved — "показать последний вопрос" uses it."""
        self._streaming = False
        self._title.setText(title)
        self._time.setText(datetime.now().strftime("%H:%M:%S") if self._mode.with_labels else "")
        self._set_body(text, Color.TEXT_PRIMARY)
        self.fade_in()
        self._autoclose.start(Motion.ANSWER_AUTOCLOSE_MS)

    def finish(self) -> None:
        """Stream ended: the card closes itself 30 s later unless dismissed."""
        self._streaming = False
        self._autoclose.start(Motion.ANSWER_AUTOCLOSE_MS)

    def fail(self, message: str) -> None:
        self._streaming = False
        self._title.setText("Модель не ответила")
        self._set_body(message, Color.STATE_ERROR)
        self.reposition()
        self.fade_in()
        self._autoclose.start(Motion.ANSWER_AUTOCLOSE_MS)

    def is_streaming(self) -> bool:
        return self._streaming

    def _set_body(self, text: str, color: str, *, markdown: bool = False) -> None:
        self._body.setStyleSheet(
            f"QTextEdit {{ background: transparent; border: none; padding: 0;"
            f" {css(Type.ANSWER, with_color=False)} color: {color};"
            f" selection-background-color: {Color.ACCENT_ACTIVE}; }}"
            + scrollbar_qss()
        )
        # Only a model's answer goes through the markdown renderer. Our own
        # strings ("Собираю ответ…", an error, the last question the user
        # asked) are plain text and must stay literal — a question containing
        # an asterisk is not emphasis.
        if markdown:
            self._body.setHtml(markup.to_html(text))
        else:
            self._body.setPlainText(text)
        bar = self._body.verticalScrollBar()
        bar.setValue(bar.maximum())

    def content_height(self) -> int:
        m = self._mode
        doc = self._body.document()
        doc.setTextWidth(max(40, self.width() - 2 * m.answer_pad_h))
        body = doc.size().height()
        chrome = 2 * m.answer_pad_v + max(_HEADER_ICON, self._close.height()) + Space.S2
        return int(min(m.answer_max_h, chrome + body + 2))


class CapturePlate(_OverlayCard):
    """Capture failure, section 4.4 — same slot as the answer card, same rule:
    it must not move the feed."""

    settings_requested = pyqtSignal()

    def __init__(self, mode: LayoutMode = NORMAL, parent: Optional[QWidget] = None) -> None:
        super().__init__(mode, parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(
            mode.answer_pad_h, mode.answer_pad_v, mode.answer_pad_h, mode.answer_pad_v
        )
        root.setSpacing(Space.S1)

        head = QHBoxLayout()
        head.setSpacing(Space.S2)
        icon = QLabel()
        icon.setFixedSize(_HEADER_ICON, _HEADER_ICON)
        icon.setPixmap(icons.pixmap("warning", _HEADER_ICON, Color.STATE_ERROR))
        head.addWidget(icon)
        self._title = QLabel("Звук с устройства вывода не приходит")
        self._title.setFont(qfont(Type.PANEL_TITLE))
        self._title.setStyleSheet(label_qss(Type.PANEL_TITLE))
        head.addWidget(self._title)
        head.addStretch()
        root.addLayout(head)

        self._hint = QLabel("Проверьте настройки звука · транскрипт сохранён")
        self._hint.setFont(qfont(Type.BUTTON_SUB))
        self._hint.setStyleSheet(label_qss(Type.BUTTON_SUB))
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        row = QHBoxLayout()
        row.setContentsMargins(0, Space.S1, 0, 0)
        self._settings = TextButton("Настройки…", self, style=Type.MENU_ITEM)
        self._settings.clicked.connect(self.settings_requested.emit)
        row.addWidget(self._settings)
        row.addStretch()
        root.addLayout(row)

    def _relayout(self) -> None:
        m = self._mode
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(
                m.answer_pad_h, m.answer_pad_v, m.answer_pad_h, m.answer_pad_v
            )
        self._hint.setVisible(m.with_labels)

    def set_reason(self, reason: str) -> None:
        self.setToolTip(reason)

    def content_height(self) -> int:
        m = self._mode
        height = 2 * m.answer_pad_v + _HEADER_ICON + Space.S1 + self._settings.height()
        if m.with_labels:
            height += self._hint.sizeHint().height() + Space.S1
        return int(min(m.answer_max_h, height + Space.S1))
