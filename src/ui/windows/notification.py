"""Hotkey notification and the fatal error window (design system 2.6).

The notification exists for exactly one situation: an action was triggered by a
hotkey while the overlay was hidden or behind another window, so the result has
no place to appear. It says what happened and how to open it — with Alt+H, not
Alt+Space, because Alt+Space is show/hide and nothing else (6.4).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ui.theme import icons
from ui.theme.fonts import qcolor, qfont
from ui.theme.styles import label_qss
from ui.theme.tokens import (
    Border,
    Color,
    FATAL_H,
    FATAL_W,
    Motion,
    NOTIFICATION_H,
    NOTIFICATION_MARGIN,
    NOTIFICATION_STRIPE,
    NOTIFICATION_W,
    Radius,
    Size,
    Space,
    Type,
)
from ui.widgets.buttons import IconButton, TextButton


class Notification(QWidget):
    """240x56, bottom-right, 5 s, hover pauses the countdown."""

    activated = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(NOTIFICATION_W, NOTIFICATION_H)

        root = QHBoxLayout(self)
        root.setContentsMargins(NOTIFICATION_STRIPE + Space.S2, Space.S2, Space.S2, Space.S2)
        root.setSpacing(Space.S2)

        self._icon = QLabel()
        self._icon.setFixedSize(18, 18)
        self._icon.setPixmap(icons.pixmap("summary", 18, Color.ACCENT_ON))
        root.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignVCenter)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        self._title = QLabel("")
        self._title.setFont(qfont(Type.BUTTON_LABEL))
        self._title.setStyleSheet(label_qss(Type.BUTTON_LABEL))
        text.addWidget(self._title)
        self._hint = QLabel("")
        self._hint.setFont(qfont(Type.BUTTON_SUB))
        self._hint.setStyleSheet(label_qss(Type.BUTTON_SUB))
        text.addWidget(self._hint)
        root.addLayout(text, 1)

        self._close = IconButton("close", Size.HIT_MIN, 12, self)
        self._close.clicked.connect(self.dismiss)
        root.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(Motion.NOTIFICATION_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.finished.connect(self._after_fade)

        self._life = QTimer(self)
        self._life.setSingleShot(True)
        self._life.timeout.connect(self.dismiss)

    def announce(self, title: str, hint: str, icon: str = "summary") -> None:
        self._title.setText(title)
        self._hint.setText(hint)
        self._icon.setPixmap(icons.pixmap(icon, 18, Color.ACCENT_ON))
        self._place()
        self.show()
        self.raise_()
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._life.start(Motion.NOTIFICATION_LIFETIME_MS)

    def dismiss(self) -> None:
        self._life.stop()
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _after_fade(self) -> None:
        if self._effect.opacity() <= 0.01:
            self.hide()

    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(
            area.right() - self.width() - NOTIFICATION_MARGIN,
            area.bottom() - self.height() - NOTIFICATION_MARGIN,
        )

    # Hover holds the countdown; leaving restarts it from the top.
    def enterEvent(self, event) -> None:
        self._life.stop()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self.isVisible():
            self._life.start(Motion.NOTIFICATION_LIFETIME_MS)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            self.dismiss()
        event.accept()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(qcolor(Color.BORDER_STRONG))
        pen.setWidthF(Border.WIDTH)
        p.setPen(pen)
        p.setBrush(qcolor(Color.BG_CARD))
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.drawRoundedRect(rect, Radius.PANEL, Radius.PANEL)
        # Accent stripe, clipped to the rounded left edge.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(Color.ACCENT))
        p.setClipRect(QRectF(0, 0, NOTIFICATION_STRIPE, self.height()))
        p.drawRoundedRect(rect, Radius.PANEL, Radius.PANEL)
        p.end()


class FatalWindow(QWidget):
    """320x200 modal, one button, and the app is over (section 2.6)."""

    def __init__(self, message: str, on_close, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setFixedSize(FATAL_W, FATAL_H)
        self._on_close = on_close

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.S4, Space.S4, Space.S4, Space.S4)
        root.setSpacing(Space.S2)

        icon = QLabel()
        icon.setFixedSize(32, 32)
        icon.setPixmap(icons.pixmap("warning", 32, Color.STATE_ERROR))
        icon.setStyleSheet(
            f"QLabel {{ background: {Color.ERROR_BG}; border-radius: {Radius.CONTROL}px; }}"
        )
        root.addWidget(icon, 0, Qt.AlignmentFlag.AlignHCenter)

        title = QLabel("Speecher остановлен")
        title.setFont(qfont(Type.WINDOW_TITLE))
        title.setStyleSheet(label_qss(Type.WINDOW_TITLE))
        root.addWidget(title, 0, Qt.AlignmentFlag.AlignHCenter)

        body = QLabel(message)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        body.setFont(qfont(Type.MENU_ITEM))
        body.setStyleSheet(
            f"QLabel {{ background: transparent; color: {Color.TEXT_SECONDARY}; }}"
        )
        root.addWidget(body, 1)

        button = TextButton(
            "Закрыть", self, height=32,
            style=Type.BUTTON_LABEL,
            color=Color.ERROR_TEXT,
            background=Color.ERROR_BG,
            border_color=Color.ERROR_BORDER,
        )
        button.clicked.connect(self._finish)
        root.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)

    def _finish(self) -> None:
        self.close()
        self._on_close()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(qcolor(Color.ERROR_BORDER))
        pen.setWidthF(Border.WIDTH)
        p.setPen(pen)
        p.setBrush(qcolor(Color.BG_WINDOW))
        p.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
            Radius.WINDOW, Radius.WINDOW,
        )
        p.end()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish()
            return
        super().keyPressEvent(event)
