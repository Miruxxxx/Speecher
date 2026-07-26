"""Summary history and the single-summary viewer (design system 2.5).

A vertical list in its own window, not a carousel inside the overlay: an hour
of talking produces more than ten entries and arrow-stepping through them is
punishment. Each entry is a card with a 44 px gutter (accent marker + time), a
title and two preview lines — truncation is legal here, because a preview is
recognised, not read (6.1).
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm.summaries import Summary
from ui import markup
from ui.theme.fonts import css, qcolor, qfont
from ui.theme.styles import label_qss, scrollbar_qss
from ui.theme.tokens import (
    Border,
    Color,
    HISTORY_H,
    HISTORY_W,
    Radius,
    Space,
    Type,
)
from ui.widgets.buttons import TextButton
from ui.windows.base import Panel

_GUTTER = 44
_MARKER = 6


class _Entry(QWidget):
    """One history card: hover -> surface.hover + border.strong, press -> active."""

    opened = pyqtSignal(object)

    def __init__(self, summary: Summary, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._summary = summary
        self._hover = False
        self._selected = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 9, 10, 9)
        root.setSpacing(0)

        gutter = QLabel(summary.stamp())
        gutter.setFont(qfont(Type.TIMESTAMP))
        gutter.setStyleSheet(label_qss(Type.TIMESTAMP))
        gutter.setFixedWidth(_GUTTER - _MARKER - Space.S1)
        gutter.setAlignment(Qt.AlignmentFlag.AlignTop)
        root.addSpacing(_MARKER + Space.S1)
        root.addWidget(gutter)
        root.addSpacing(Space.S2)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        title = QLabel(summary.title())
        title.setFont(qfont(Type.HISTORY_TITLE))
        title.setStyleSheet(label_qss(Type.HISTORY_TITLE))
        text.addWidget(title)
        preview = QLabel(summary.text)
        preview.setFont(qfont(Type.HISTORY_PREVIEW))
        preview.setStyleSheet(label_qss(Type.HISTORY_PREVIEW))
        preview.setWordWrap(True)
        preview.setMaximumHeight(int(Type.HISTORY_PREVIEW.line_px * 2))
        text.addWidget(preview)
        root.addLayout(text, 1)

    def summary(self) -> Summary:
        return self._summary

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.opened.emit(self._summary)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.opened.emit(self._summary)
        event.accept()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = Color.BG_SURFACE_HOVER if self._hover else Color.BG_CARD
        border = Color.ACCENT if self._selected else (
            Color.BORDER_STRONG if self._hover else Color.BORDER
        )
        pen = QPen(qcolor(border))
        pen.setWidthF(Border.WIDTH)
        p.setPen(pen)
        p.setBrush(qcolor(background))
        p.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), Radius.PANEL, Radius.PANEL
        )
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(Color.ACCENT))
        p.drawRoundedRect(QRectF(10, 12, _MARKER, _MARKER), _MARKER / 2, _MARKER / 2)
        p.end()


class SummaryViewer(Panel):
    """One summary, full text, selectable — the "editor" an entry opens into."""

    def __init__(self, summary: Summary, parent: Optional[QWidget] = None) -> None:
        super().__init__(summary.title(), (HISTORY_W, 320), parent)
        root = QVBoxLayout(self.body())
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(Space.S3)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setFrameStyle(0)
        # Summaries are model output too: the same asterisks, the same lists.
        view.setHtml(markup.to_html(summary.text))
        view.setStyleSheet(
            f"QTextEdit {{ background: {Color.BG_CARD}; border: {Border.WIDTH}px solid"
            f" {Color.BORDER}; border-radius: {Radius.PANEL}px; padding: 10px 12px;"
            f" {css(Type.ANSWER)} }}" + scrollbar_qss()
        )
        root.addWidget(view, 1)

        row = QHBoxLayout()
        row.addStretch()
        copy = TextButton("Копировать", self, style=Type.MENU_ITEM)
        copy.clicked.connect(lambda: self._copy(summary.text))
        row.addWidget(copy)
        root.addLayout(row)

    @staticmethod
    def _copy(text: str) -> None:
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText(text)


class HistoryWindow(Panel):
    """The list itself. Rebuilt on open — a session's worth of entries is small."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("История выжимок", (HISTORY_W, HISTORY_H), parent)
        self._viewer: Optional[SummaryViewer] = None
        self._entries: List[_Entry] = []

        root = QVBoxLayout(self.body())
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(Space.S2)

        self._empty = QLabel("Выжимок пока нет")
        self._empty.setFont(qfont(Type.BUTTON_SUB))
        self._empty.setStyleSheet(label_qss(Type.BUTTON_SUB))
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._empty)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameStyle(0)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._holder = QWidget()
        self._list = QVBoxLayout(self._holder)
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(Space.S2)
        self._list.addStretch()
        self._scroll.setWidget(self._holder)
        root.addWidget(self._scroll, 1)

    def set_summaries(self, summaries: List[Summary]) -> None:
        for entry in self._entries:
            entry.setParent(None)
            entry.deleteLater()
        self._entries.clear()
        for summary in reversed(summaries):  # newest first
            entry = _Entry(summary, self._holder)
            entry.opened.connect(self.open_summary)
            self._list.insertWidget(self._list.count() - 1, entry)
            self._entries.append(entry)
        self._empty.setVisible(not summaries)
        self._scroll.setVisible(bool(summaries))

    def open_summary(self, summary: Summary) -> None:
        for entry in self._entries:
            entry.set_selected(entry.summary() is summary)
        self._viewer = SummaryViewer(summary, None)
        self._viewer.center_on(self)
        self._viewer.show()
        self._viewer.raise_()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._entries:
            self.open_summary(self._entries[0].summary())
            return
        super().keyPressEvent(event)
