"""The answer and the summary as windows beside the overlay (design system 7.9).

`AnswerCard` is an overlay inside the feed, so its height is capped at 160 px
(96 compact) — it may not push the transcript around, and that cap is the whole
reason it exists. It is the right shape for "модель ответила", and the wrong one
for an answer somebody actually reads: three sentences already scroll.

So the same content can live here instead. Two windows, pinned to the sides of
the overlay: the model's answer and the last question on the left, the summary
on the right. They never overlap the overlay or each other, they are exactly as
big as the mode says (600 x 340 or 360 x 200), and they do not close themselves
— a window this size disappearing under you is worse than one that stays.

Chrome is the overlay's, not the settings panel's: `Qt.Tool`, frameless, always
on top, dragged by its own header. Unlike the overlay it may take focus, because
its whole point is that the text can be selected and copied.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui import markup
from ui.theme import icons
from ui.theme.fonts import css, qcolor, qfont
from ui.theme.styles import label_qss, scrollbar_qss, tooltip_qss
from ui.theme.tokens import (
    Border,
    Color,
    PANEL_SIZES,
    PANELS_SIDE_LARGE,
    Radius,
    SIDE_GAP,
    Size,
    Space,
    Type,
)
from ui.widgets.buttons import IconButton, TextButton

_HEADER_ICON = 18
_HEADER_H = 28

LEFT = "left"
RIGHT = "right"


def dock_position(
    anchor: tuple[int, int, int, int],
    size: tuple[int, int],
    side: str,
    area: Optional[tuple[int, int, int, int]] = None,
) -> tuple[int, int]:
    """Top-left corner for a panel docked to `anchor`'s left or right edge.

    Both rectangles are (x, y, w, h). `area` is the screen's usable region: a
    panel that would hang off the edge is pulled back in, because half a window
    beyond the taskbar is worse than one that overlaps a little. Separated from
    the widget so the arithmetic can be checked on a screen bigger than whatever
    the test machine has.
    """
    ax, ay, aw, ah = anchor
    width, height = size
    x = ax - width - SIDE_GAP if side == LEFT else ax + aw + SIDE_GAP
    y = ay
    if area is not None:
        sx, sy, sw, sh = area
        x = min(max(x, sx), max(sx, sx + sw - width))
        y = min(max(y, sy), max(sy, sy + sh - height))
    return int(x), int(y)


class SidePanel(QWidget):
    """One side window. Same public surface as `AnswerCard`, so the overlay can
    hand the LLM's output to either without knowing which it got."""

    closed = pyqtSignal()

    def __init__(
        self,
        title: str,
        icon: str,
        side: str,
        *,
        mode: str = PANELS_SIDE_LARGE,
        anchor: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._side = side
        # The overlay it docks to. Held rather than passed in on every call so
        # `appear()` can re-dock by itself: the window that opens has to land
        # where the overlay is *now*, not where it was when the panel was built.
        self._anchor = anchor
        self._default_title = title
        self._icon_name = icon
        self._streaming = False
        # Set once the user drags this window somewhere: after that the overlay
        # moving no longer drags it back to the side it was born on.
        self._detached = False
        self._drag: Optional[QPoint] = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setWindowTitle(title)
        self.setStyleSheet(tooltip_qss())

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.S3, Space.S2, Space.S3, Space.S3)
        root.setSpacing(Space.S2)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(Space.S2)
        self._icon = QLabel()
        self._icon.setFixedSize(_HEADER_ICON, _HEADER_ICON)
        self._icon.setPixmap(icons.pixmap(icon, _HEADER_ICON, Color.ACCENT_ON))
        self._icon.setStyleSheet(
            f"QLabel {{ background: {Color.ACCENT_SUBTLE};"
            f" border-radius: {Radius.PLATE}px; }}"
        )
        head.addWidget(self._icon)
        self._title = QLabel(title)
        self._title.setFont(qfont(Type.PANEL_TITLE))
        self._title.setStyleSheet(label_qss(Type.PANEL_TITLE))
        head.addWidget(self._title)
        head.addStretch()
        self._time = QLabel("")
        self._time.setFont(qfont(Type.TIMESTAMP))
        self._time.setStyleSheet(label_qss(Type.TIMESTAMP))
        head.addWidget(self._time)
        close = IconButton("close", Size.HIT_MIN, 12, self)
        close.setToolTip("Закрыть (Esc)")
        close.clicked.connect(self.close)
        head.addWidget(close)
        root.addLayout(head)

        self._body = QTextEdit()
        self._body.setReadOnly(True)
        self._body.setFrameStyle(0)
        root.addWidget(self._body, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch()
        self._copy = TextButton("Копировать", self, style=Type.MENU_ITEM)
        self._copy.clicked.connect(self._copy_body)
        footer.addWidget(self._copy)
        root.addLayout(footer)

        self._text = ""
        self._paint_body(Color.TEXT_PRIMARY)
        self.set_panel_mode(mode)

    # -- geometry --------------------------------------------------------

    def set_panel_mode(self, mode: str) -> None:
        """Size comes from the setting, not from the overlay's current layout:
        the user picked "крупные" or "компактные", and a double click on the
        title bar must not silently change that."""
        width, height = PANEL_SIZES.get(mode, PANEL_SIZES[PANELS_SIDE_LARGE])
        self.setFixedSize(width, height)

    def place_beside(
        self, anchor: Optional[QWidget] = None, *, force: bool = False
    ) -> None:
        """Dock to the overlay's left or right edge, clamped to the screen.

        Does nothing once the window has been dragged by hand, unless `force`:
        following the overlay is a convenience, and undoing somebody's deliberate
        placement is not.
        """
        anchor = anchor or self._anchor
        if anchor is None or (self._detached and not force):
            return
        if force:
            self._detached = False
        geo = anchor.frameGeometry()
        screen = QApplication.screenAt(geo.center()) or QApplication.primaryScreen()
        area = None
        if screen is not None:
            usable = screen.availableGeometry()
            area = (usable.x(), usable.y(), usable.width(), usable.height())
        x, y = dock_position(
            (geo.x(), geo.y(), geo.width(), geo.height()),
            (self.width(), self.height()),
            self._side,
            area,
        )
        self.move(x, y)

    # -- content (the AnswerCard surface) --------------------------------

    def begin(self, title: str = "Думаю…") -> None:
        self._streaming = True
        self._title.setText(title)
        self._time.setText(datetime.now().strftime("%H:%M:%S"))
        self._set_body("Собираю ответ…", Color.TEXT_SECONDARY)
        self._copy.setEnabled(False)
        self.appear()

    def set_answer(self, text: str) -> None:
        self._title.setText(self._default_title)
        self._set_body(text, Color.TEXT_PRIMARY, markdown=True)
        self._copy.setEnabled(bool(text.strip()))

    def show_text(self, title: str, text: str) -> None:
        """No model involved — "показать последний вопрос" lands here."""
        self._streaming = False
        self._title.setText(title)
        self._time.setText(datetime.now().strftime("%H:%M:%S"))
        self._set_body(text, Color.TEXT_PRIMARY)
        self._copy.setEnabled(bool(text.strip()))
        self.appear()

    def finish(self) -> None:
        """The stream ended. Nothing closes: this window is not a notification,
        and 30 s is not how long it takes to read half a page."""
        self._streaming = False

    def fail(self, message: str) -> None:
        self._streaming = False
        self._title.setText("Модель не ответила")
        self._set_body(message, Color.STATE_ERROR)
        self._copy.setEnabled(False)
        self.appear()

    def is_streaming(self) -> bool:
        return self._streaming

    def appear(self) -> None:
        self.place_beside()
        self.show()
        self.raise_()

    def fade_out(self) -> None:
        """Named for the card it stands in for; a window just goes away."""
        self.hide()

    def text(self) -> str:
        return self._text

    def _set_body(self, text: str, color: str, *, markdown: bool = False) -> None:
        self._text = text
        self._paint_body(color)
        # Only a model's answer is markdown. Our own strings — a question the
        # speaker asked, an error — stay literal (same rule as the card).
        if markdown:
            self._body.setHtml(markup.to_html(text))
        else:
            self._body.setPlainText(text)
        bar = self._body.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _paint_body(self, color: str) -> None:
        self._body.setStyleSheet(
            f"QTextEdit {{ background: {Color.BG_CARD}; border: {Border.WIDTH}px solid"
            f" {Color.BORDER}; border-radius: {Radius.PANEL}px; padding: 10px 12px;"
            f" {css(Type.ANSWER, with_color=False)} color: {color};"
            f" selection-background-color: {Color.ACCENT_ACTIVE}; }}"
            + scrollbar_qss()
        )

    def _copy_body(self) -> None:
        if self._text.strip():
            QApplication.clipboard().setText(self._text)

    # -- chrome ----------------------------------------------------------

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
            self._detached = True
            self.move(event.globalPosition().toPoint() - self._drag)
            event.accept()

    def mouseReleaseEvent(self, _event) -> None:
        self._drag = None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        event.accept()


def answer_panel(mode: str, anchor: Optional[QWidget] = None) -> SidePanel:
    """Left: what the model answered, and the question it answered."""
    return SidePanel("Ответ модели", "answer", LEFT, mode=mode, anchor=anchor)


def summary_panel(mode: str, anchor: Optional[QWidget] = None) -> SidePanel:
    """Right: the summary. Its history stays where it was (Alt+H)."""
    return SidePanel("Выжимка", "summary", RIGHT, mode=mode, anchor=anchor)
