"""Frameless shell shared by the satellite windows.

Settings, history, onboarding and the fatal dialog all live outside the overlay
but inside the same design system: window radius 10, 1 px border, own title row
(there is no system chrome anywhere in this app), dragged by that row.

Unlike the overlay these windows *do* take focus — they contain fields and the
keyboard focus ring exists for them (section 1.1).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui.theme.fonts import qcolor, qfont
from ui.theme.styles import label_qss, tooltip_qss, window_qss
from ui.theme.tokens import Border, Color, Radius, Size, Space, Type
from ui.widgets.buttons import IconButton


class Panel(QWidget):
    """Frameless, rounded, always above the overlay, focusable."""

    TITLE_H = 36

    def __init__(
        self,
        title: str,
        size: tuple[int, int],
        parent: Optional[QWidget] = None,
        *,
        closable: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Closing a satellite window must never end the session. Qt does not
        # count the overlay itself as a window for quitOnLastWindowClosed (it is
        # a Qt.Tool), so without this the *first* settings window to close is
        # "the last window" and takes the whole app down with it.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setWindowTitle(title)
        self.setFixedSize(*size)
        self.setStyleSheet(window_qss() + tooltip_qss())
        self._drag: Optional[QPoint] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(Space.S3, Space.S2, Space.S3, Space.S3)
        outer.setSpacing(Space.S3)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(Space.S2)
        self._title = QLabel(title)
        self._title.setFont(qfont(Type.WINDOW_TITLE))
        self._title.setStyleSheet(label_qss(Type.WINDOW_TITLE))
        head.addWidget(self._title)
        head.addStretch()
        if closable:
            close = IconButton("close", Size.HIT_MIN, 12, self)
            close.setToolTip("Закрыть (Esc)")
            close.clicked.connect(self.close)
            head.addWidget(close)
        outer.addLayout(head)

        self._body = QWidget(self)
        outer.addWidget(self._body, 1)
        self._outer = outer

    def body(self) -> QWidget:
        return self._body

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
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= self.TITLE_H:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)
            event.accept()

    def mouseReleaseEvent(self, _event) -> None:
        self._drag = None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def center_on(self, other: Optional[QWidget]) -> None:
        """Open next to the window that spawned it, not in a screen corner."""
        if other is None:
            return
        geo = other.frameGeometry()
        self.move(
            geo.center().x() - self.width() // 2,
            max(0, geo.center().y() - self.height() // 2),
        )
