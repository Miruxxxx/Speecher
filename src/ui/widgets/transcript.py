"""The transcript feed (design system 2.3).

One read-only QTextEdit for the whole feed, with the timestamps in a two-column
table of Qt's HTML subset — exactly as the spec prescribes, because a gutter
made of separate widgets cannot keep a timestamp on the baseline of a wrapped
paragraph.

Three behaviours carry most of the design's intent:

* Wrapping, never truncation (6.1). Six whole lines beat six cut ones.
* The caret is painted, not inserted into the document: it is the only sign of
  life when the engine emits no provisional text, and re-rendering the document
  twice a second to blink it would fight the selection and the scrollbar.
* Auto-scroll sticks to the bottom until the wheel is touched; after that the
  feed stays where the user left it and a "down" button offers the way back.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from PyQt6.QtCore import QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFontMetricsF, QPainter, QTextLayout, QTextOption
from PyQt6.QtWidgets import QTextEdit, QWidget

from ui.theme import icons
from ui.theme.fonts import css, qcolor, qfont
from ui.theme.styles import scrollbar_qss
from ui.theme.tokens import (
    Color,
    LayoutMode,
    MAX_LINES_PER_REPLY,
    Motion,
    NORMAL,
    Space,
    TextStyle,
    Type,
)
from ui.transcript_model import Reply
from ui.widgets.buttons import IconButton

EMPTY_TITLE = "Слушаю систему…"
EMPTY_HINT = "Здесь появится транскрипт"
_EMPTY_ICON = 26

# How long the scrollbar handle stays visible after the last scroll (2.3:
# "появляется только при прокрутке").
_SCROLLBAR_FADE_MS = 1200

_CARET_W = 1.5


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def wrap_chunks(text: str, style: TextStyle, width: float, max_lines: int) -> List[str]:
    """Split `text` so no chunk needs more than `max_lines` lines at `width`.

    Section 2.3 caps a reply at four lines and continues it as a new,
    timestamp-less block. Measured with the same font the cell renders in, so
    the cut lands where Qt would have wrapped anyway.
    """
    if not text or width <= 0 or max_lines <= 0:
        return [text] if text else []
    font = qfont(style)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WordWrap)

    chunks: List[str] = []
    rest = text
    # Bounded: every pass either consumes the whole remainder or cuts at a
    # position > 0, so `rest` strictly shrinks.
    while rest:
        layout = QTextLayout(rest, font)
        layout.setTextOption(option)
        layout.beginLayout()
        cut = -1
        count = 0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            count += 1
            if count == max_lines:
                cut = line.textStart() + line.textLength()
        layout.endLayout()
        if count <= max_lines or cut <= 0 or cut >= len(rest):
            chunks.append(rest)
            break
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return chunks


class TranscriptView(QTextEdit):
    """The feed itself. Owns stickiness, the caret and the empty state."""

    sticky_changed = pyqtSignal(bool)

    def __init__(self, mode: LayoutMode = NORMAL, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mode = mode
        self._sticky = True
        self._caret_on = True
        self._caret_visible = True
        self._empty = True
        self._aligning = False
        self._replies: Sequence[Reply] = ()
        self._partial = ""

        self.setReadOnly(True)
        self.setFrameStyle(0)
        self.setViewportMargins(0, 0, 0, 0)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._apply_style(visible_bar=False)

        self._caret_timer = QTimer(self)
        self._caret_timer.setInterval(Motion.CARET_BLINK_MS // 2)
        self._caret_timer.timeout.connect(self._blink)
        self._caret_timer.start()

        self._bar_timer = QTimer(self)
        self._bar_timer.setSingleShot(True)
        self._bar_timer.timeout.connect(lambda: self._apply_style(visible_bar=False))

        self._to_bottom = IconButton("chevron_down", 24, 14, self)
        self._to_bottom.setToolTip("Вернуться к последним словам")
        self._to_bottom.clicked.connect(self.stick_to_bottom)
        self._to_bottom.hide()

        self.verticalScrollBar().valueChanged.connect(self._on_scrolled)

    # -- appearance ------------------------------------------------------

    def _apply_style(self, *, visible_bar: bool) -> None:
        handle = Color.BORDER_STRONG if visible_bar else "transparent"
        self.setStyleSheet(
            f"QTextEdit {{ background: {Color.BG_WINDOW}; border: none; padding: 0;"
            f" {css(self._mode.transcript)}"
            f" selection-background-color: {Color.ACCENT_ACTIVE};"
            f" selection-color: {Color.TEXT_PRIMARY}; }}"
            + scrollbar_qss().replace(Color.BORDER_STRONG, handle)
        )

    def set_mode(self, mode: LayoutMode) -> None:
        if mode.name == self._mode.name:
            return
        self._mode = mode
        self._apply_style(visible_bar=False)
        self.refresh()

    def set_caret_visible(self, visible: bool) -> None:
        self._caret_visible = visible
        self.viewport().update()

    # -- content ---------------------------------------------------------

    def set_content(self, replies: Sequence[Reply], partial: str = "") -> None:
        self._replies = list(replies)
        self._partial = partial
        self.refresh()

    def refresh(self) -> None:
        was_sticky = self._sticky
        keep = self.verticalScrollBar().value()
        self._empty = not self._replies and not self._partial
        if self._empty:
            self.setHtml("")
            # The empty screen is centred in the whole area, so it must not
            # inherit the bottom-alignment margin.
            self.setViewportMargins(0, 0, 0, 0)
        else:
            self.setHtml(self._build_html())
            self._align_bottom()
        if was_sticky:
            self._scroll_to_bottom()
        else:
            self.verticalScrollBar().setValue(min(keep, self.verticalScrollBar().maximum()))
        self.viewport().update()

    def _align_bottom(self) -> None:
        """Keep the feed pinned to the bottom edge (section 3).

        Text arrives at the bottom and rises; starting a short transcript at the
        top would make every new line move the whole block. A top viewport
        margin equal to the free space keeps the last line where the eye left it
        from the very first word.
        """
        if self._aligning:
            return
        self._aligning = True
        try:
            self.setViewportMargins(0, 0, 0, 0)
            free = self.viewport().height() - self.document().size().height()
            self.setViewportMargins(0, int(max(0.0, free)), 0, 0)
        finally:
            self._aligning = False

    def _text_width(self) -> float:
        m = self._mode
        # The document is laid out inside the viewport; the gutter and the gap
        # are table columns, so what is left for the text column is the rest.
        return max(40.0, self.viewport().width() - m.gutter_w - m.gutter_gap - 2)

    def _build_html(self) -> str:
        m = self._mode
        width = self._text_width()
        text_style = m.transcript
        rows: List[str] = []
        last = len(self._replies) - 1
        for i, reply in enumerate(self._replies):
            stamp = datetime.fromtimestamp(reply.at).strftime(m.ts_format) if reply.at else ""
            chunks = wrap_chunks(reply.text, text_style, width, MAX_LINES_PER_REPLY)
            for j, chunk in enumerate(chunks):
                body = _esc(chunk)
                if i == last and j == len(chunks) - 1 and self._partial:
                    body += (
                        f' <span style="{css(m.preliminary)}">{_esc(self._partial)}</span>'
                    )
                rows.append(self._row(stamp if j == 0 else "", body))
        if not rows and self._partial:
            rows.append(
                self._row("", f'<span style="{css(m.preliminary)}">{_esc(self._partial)}</span>')
            )
        return (
            f'<table cellspacing="0" cellpadding="0" width="{int(width + m.gutter_w + m.gutter_gap)}">'
            + "".join(rows)
            + "</table>"
        )

    def _row(self, stamp: str, body_html: str) -> str:
        m = self._mode
        pad = f"padding-bottom:{Space.S2}px;"
        stamp_html = (
            f'<span style="{css(m.timestamp)}">{stamp}</span>' if stamp else ""
        )
        line = f"line-height:{int(m.transcript.line_px)}px;"
        return (
            "<tr>"
            f'<td width="{m.gutter_w}" style="{pad}">{stamp_html}</td>'
            f'<td width="{m.gutter_gap}" style="{pad}"></td>'
            f'<td style="{pad}"><p style="{line} margin:0;">{body_html}</p></td>'
            "</tr>"
        )

    # -- stickiness ------------------------------------------------------

    def is_sticky(self) -> bool:
        return self._sticky

    def stick_to_bottom(self) -> None:
        if not self._sticky:
            self._sticky = True
            self.sticky_changed.emit(True)
        self._to_bottom.hide()
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _unstick(self) -> None:
        if self._sticky:
            self._sticky = False
            self.sticky_changed.emit(False)
        self._to_bottom.show()
        self._place_button()

    def wheelEvent(self, event) -> None:
        bar = self.verticalScrollBar()
        if event.angleDelta().y() > 0 and bar.maximum() > 0:
            self._unstick()
        super().wheelEvent(event)
        if bar.value() >= bar.maximum() - 1:
            self.stick_to_bottom()

    def _on_scrolled(self, _value: int) -> None:
        self._apply_style(visible_bar=True)
        self._bar_timer.start(_SCROLLBAR_FADE_MS)

    # -- geometry --------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_button()
        if not self._empty:
            self.refresh()

    def _place_button(self) -> None:
        margin = Space.S1
        size = self._to_bottom.size()
        self._to_bottom.move(
            self.width() - size.width() - margin - 4,
            self.height() - size.height() - margin,
        )

    def sizeHint(self) -> QSize:
        return QSize(self._mode.width, self._mode.height)

    # -- painting --------------------------------------------------------

    def _blink(self) -> None:
        self._caret_on = not self._caret_on
        if self._caret_visible:
            self.viewport().update()

    def paintEvent(self, event) -> None:
        if self._empty:
            self._paint_empty()
            return
        super().paintEvent(event)
        if not (self._caret_visible and self._caret_on and self._sticky):
            return
        cursor = self.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        rect = self.cursorRect(cursor)
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(Color.TEXT_PRELIMINARY))
        height = self._mode.transcript.size_px
        p.drawRect(
            QRectF(rect.x(), rect.y() + (rect.height() - height) / 2, _CARET_W, height)
        )
        p.end()

    def _paint_empty(self) -> None:
        """Three elements and nothing else (6.7): icon, title, hint."""
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.viewport().rect(), qcolor(Color.BG_WINDOW))
        area = self.viewport().rect()
        title_font = qfont(Type.MENU_ITEM)
        hint_font = qfont(Type.BUTTON_SUB)
        title_h = QFontMetricsF(title_font).height()
        hint_h = QFontMetricsF(hint_font).height()
        gap = Space.S2
        icon = _EMPTY_ICON if self._mode.with_labels else 20
        total = icon + gap + title_h + Space.S1 + hint_h
        top = area.top() + (area.height() - total) / 2

        pm = icons.pixmap("app", icon, Color.BORDER_STRONG)
        p.drawPixmap(int(area.center().x() - icon / 2), int(top), pm)

        p.setFont(title_font)
        p.setPen(qcolor(Color.TEXT_SECONDARY))
        p.drawText(
            QRectF(area.left(), top + icon + gap, area.width(), title_h),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            EMPTY_TITLE,
        )
        p.setFont(hint_font)
        p.setPen(qcolor(Color.TEXT_TIMESTAMP))
        p.drawText(
            QRectF(area.left(), top + icon + gap + title_h + Space.S1, area.width(), hint_h),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            EMPTY_HINT,
        )
        p.end()
