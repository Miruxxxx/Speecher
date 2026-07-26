"""The original transcript, in its own small window (design system 7.8).

While the translation is on, the main feed shows *only* finished translated
lines — that is what makes it readable: a line appears once and never changes.
The speech itself keeps running here, as an ordinary feed: the same widget, the
same grouping, the same stickiness. Lines stay after the model has translated
them, so this window can be read, scrolled back and glanced at like any other
transcript.

An earlier version showed only the words the model had not consumed yet, and it
emptied itself every couple of seconds — a window that blanks while you watch it
is worse than no window.

Built like the overlay and not like the settings panel: `Qt.Tool`, never takes
focus, always on top, dragged by its own header. It is watched out of the corner
of an eye while another window has the keyboard.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui.theme.fonts import qcolor, qfont
from ui.theme.styles import label_qss, tooltip_qss
from ui.theme.tokens import (
    Border,
    COMPACT,
    Color,
    Radius,
    SOURCE_H,
    SOURCE_W,
    Size,
    Space,
    Type,
)
from ui.transcript_model import Reply
from ui.widgets.buttons import IconButton
from ui.widgets.transcript import TranscriptView

_HEADER_H = 24


class SourceWindow(QWidget):
    """Compact live transcript in the language actually being spoken."""

    closed = pyqtSignal()

    def __init__(self, title: str = "Оригинал", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setWindowTitle(title)
        self.setFixedSize(SOURCE_W, SOURCE_H)
        self.setStyleSheet(tooltip_qss())
        self._drag: Optional[QPoint] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.S2, Space.S1, Space.S2, Space.S2)
        root.setSpacing(Space.S1)

        head = QHBoxLayout()
        head.setContentsMargins(Space.S1, 0, 0, 0)
        head.setSpacing(Space.S2)
        caption = QLabel(title)
        caption.setFont(qfont(Type.STATUS))
        caption.setStyleSheet(label_qss(Type.STATUS))
        head.addWidget(caption)
        head.addStretch()
        close = IconButton("close", Size.HIT_MIN, 10, self)
        close.setToolTip("Скрыть окно оригинала")
        close.clicked.connect(self.close)
        head.addWidget(close)
        root.addLayout(head)

        # Compact metrics: this window is a glance, not a reading surface.
        self._view = TranscriptView(COMPACT, self)
        root.addWidget(self._view, 1)

    # -- content ---------------------------------------------------------

    def set_content(self, replies: Sequence[Reply], partial: str = "") -> None:
        self._view.set_content(replies, partial)

    # -- chrome ----------------------------------------------------------

    def dock_under(self, other: Optional[QWidget]) -> None:
        """Sit right below the overlay, aligned to its left edge."""
        if other is None:
            return
        geo = other.frameGeometry()
        self.move(geo.left(), geo.bottom() + Space.S2)

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

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= _HEADER_H:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)
            event.accept()

    def mouseReleaseEvent(self, _event) -> None:
        self._drag = None

    def closeEvent(self, event) -> None:
        # Hiding this window must not stop the translation: the user just wants
        # the screen back.
        self.closed.emit()
        event.accept()
