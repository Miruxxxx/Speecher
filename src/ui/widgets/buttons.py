"""Action button and icon button (design system 2.2).

Both paint themselves. QSS would have been shorter, but the spec pins geometry
per state ("высота 40, вторая строка меняет только текст", "слот иконки
фиксирован") and a stylesheet cannot promise that a disabled button occupies
exactly the pixels the enabled one did.

Only the background is animated, 90 ms, no size or border change — anything that
moves in this window steals attention from the conversation.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, pyqtProperty
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QAbstractButton, QWidget

from ui.theme import icons
from ui.theme.fonts import qcolor, qfont
from ui.theme.tokens import Border, Color, Motion, Radius, Size, Space, Type


def _mix(a: str, b: str, t: float) -> QColor:
    """Linear blend of two token colours; t=0 -> a, t=1 -> b."""
    ca, cb = QColor(a), QColor(b)
    return QColor(
        int(ca.red() + (cb.red() - ca.red()) * t),
        int(ca.green() + (cb.green() - ca.green()) * t),
        int(ca.blue() + (cb.blue() - ca.blue()) * t),
    )


class _HoverButton(QAbstractButton):
    """Shared state machine: rest / hover / pressed / disabled / focus."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hover_t = 0.0
        self.setCursor(Qt.CursorShape.ArrowCursor)  # a click here is never a link
        self._anim = QPropertyAnimation(self, b"hoverT", self)
        self._anim.setDuration(Motion.HOVER_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _get_hover_t(self) -> float:
        return self._hover_t

    def _set_hover_t(self, value: float) -> None:
        self._hover_t = value
        self.update()

    hoverT = pyqtProperty(float, fget=_get_hover_t, fset=_set_hover_t)

    def enterEvent(self, event) -> None:
        self._animate(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._animate(0.0)
        super().leaveEvent(event)

    def _animate(self, target: float) -> None:
        if not self.isEnabled():
            return
        self._anim.stop()
        self._anim.setStartValue(self._hover_t)
        self._anim.setEndValue(target)
        self._anim.start()

    def changeEvent(self, event) -> None:
        # A button disabled while the pointer sits on it must not keep the hover
        # tint; nothing else would clear it until the mouse leaves.
        if not self.isEnabled():
            self._anim.stop()
            self._set_hover_t(0.0)
        super().changeEvent(event)

    # -- painting helpers ------------------------------------------------

    def _colors(self) -> tuple[QColor, QColor]:
        """(background, border) for the current state."""
        if not self.isEnabled():
            return qcolor(Color.DISABLED_BG), qcolor(Color.DISABLED_BORDER)
        if self.isDown():
            return qcolor(Color.BG_SURFACE_ACTIVE), qcolor(Color.BORDER_STRONG)
        bg = _mix(Color.BG_SURFACE, Color.BG_SURFACE_HOVER, self._hover_t)
        border = _mix(Color.BORDER, Color.BORDER_STRONG, self._hover_t)
        if self.hasFocus():
            border = qcolor(Color.FOCUS_RING)
        return bg, border

    def _paint_frame(
        self, p: QPainter, radius: int, *, border: bool = True,
        box: Optional[int] = None,
    ) -> None:
        """Paint the chip. `box` shrinks the *visible* rect inside the widget so
        a control can be smaller than the 24 px minimum hit area (the answer
        card's 20 px close button is exactly that case)."""
        bg, border_color = self._colors()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        if box is not None and box < min(self.width(), self.height()):
            dx = (self.width() - box) / 2
            dy = (self.height() - box) / 2
            rect = QRectF(dx, dy, box, box)
        rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(bg)
        if border:
            pen = QPen(border_color)
            pen.setWidthF(Border.WIDTH)
            p.setPen(pen)
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, radius, radius)


class ActionButton(_HoverButton):
    """Two-line action: title (12/500) plus a clarifier (10.5/400).

    The clarifier may be dropped or replaced (with "модель офлайн", for
    instance) — the height stays 40 either way, which is what keeps disabling a
    button from moving the row.
    """

    def __init__(
        self,
        icon_name: str,
        label: str,
        sub: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self._label = label
        self._sub = sub
        self._icon_only = False
        self._icon_size = Size.ICON
        self.setFixedHeight(40)
        self.setMinimumWidth(Size.HIT_MIN)

    def set_sub(self, text: str) -> None:
        if text != self._sub:
            self._sub = text
            self.update()

    def label(self) -> str:
        return self._label

    def sub(self) -> str:
        return self._sub

    def set_layout(self, *, icon_only: bool, height: int, icon: int = Size.ICON) -> None:
        """Switch between the labelled 40 px button and the icon-only variant.

        Compact mode keeps all four actions and drops their captions into the
        tooltip (6.3); the menu button is icon-only in both modes.
        """
        self._icon_only = icon_only
        self._icon_size = icon
        self.setFixedHeight(height)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        self._paint_frame(p, Radius.CONTROL)
        enabled = self.isEnabled()

        if self._icon_only:
            size = self._icon_size
            color = Color.DISABLED_TEXT if not enabled else Color.TEXT_LABEL
            pm = icons.pixmap(self._icon_name, size, color)
            p.drawPixmap(
                int((self.width() - size) / 2), int((self.height() - size) / 2), pm
            )
            p.end()
            return

        pad = Space.S2
        plate = Size.ICON
        plate_y = int((self.height() - plate) / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(Color.ACCENT_SUBTLE if enabled else Color.DISABLED_BORDER))
        p.drawRoundedRect(
            QRectF(pad, plate_y, plate, plate), Radius.PLATE, Radius.PLATE
        )
        p.drawPixmap(
            pad,
            plate_y,
            icons.pixmap(
                self._icon_name,
                plate,
                Color.ACCENT_ON if enabled else Color.DISABLED_TEXT,
            ),
        )

        text_x = pad + plate + Space.S2
        text_w = self.width() - text_x - pad
        if not enabled:
            label_color, sub_color = Color.DISABLED_TEXT, Color.DISABLED_TEXT_SECONDARY
        elif self._hover_t > 0.5 or self.isDown():
            label_color, sub_color = Color.TEXT_LABEL_HOVER, Color.TEXT_SECONDARY_HOVER
        else:
            label_color, sub_color = Color.TEXT_LABEL, Color.TEXT_SECONDARY

        label_h = int(Type.BUTTON_LABEL.line_px)
        sub_h = int(Type.BUTTON_SUB.line_px)
        if self._sub:
            top = int((self.height() - label_h - sub_h) / 2)
        else:
            top = int((self.height() - label_h) / 2)

        p.setFont(qfont(Type.BUTTON_LABEL))
        p.setPen(qcolor(label_color))
        align = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        p.drawText(
            text_x, top, text_w, label_h, align,
            p.fontMetrics().elidedText(self._label, Qt.TextElideMode.ElideRight, text_w),
        )
        if self._sub:
            p.setFont(qfont(Type.BUTTON_SUB))
            p.setPen(qcolor(sub_color))
            p.drawText(
                text_x, top + label_h, text_w, sub_h, align,
                p.fontMetrics().elidedText(self._sub, Qt.TextElideMode.ElideRight, text_w),
            )
        p.end()


class TextButton(_HoverButton):
    """Label-only control on a surface chip: the "Настройки…" affordance inside
    the capture-error plate and the buttons of the satellite windows."""

    def __init__(
        self,
        text: str,
        parent: Optional[QWidget] = None,
        *,
        height: int = Size.HIT_MIN,
        style=Type.BUTTON_LABEL,
        color: str = Color.TEXT_LABEL,
        background: Optional[str] = None,
        border_color: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._label = text
        self._style = style
        self._color = color
        self._bg = background
        self._border = border_color
        self.setFixedHeight(height)
        self.setMinimumWidth(self._natural_width())

    def _natural_width(self) -> int:
        from PyQt6.QtGui import QFontMetrics

        return QFontMetrics(qfont(self._style)).horizontalAdvance(self._label) + 2 * Space.S3

    def set_text(self, text: str) -> None:
        self._label = text
        self.setMinimumWidth(self._natural_width())
        self.update()

    def _colors(self) -> tuple[QColor, QColor]:
        bg, border = super()._colors()
        # A fixed palette (the fatal window's single button) opts out of the
        # hover blend entirely: it is the last thing shown before the app exits.
        if self._bg is not None:
            bg = qcolor(self._bg)
        if self._border is not None:
            border = qcolor(self._border)
        return bg, border

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        self._paint_frame(p, Radius.CONTROL)
        p.setFont(qfont(self._style))
        p.setPen(qcolor(self._color if self.isEnabled() else Color.DISABLED_TEXT))
        p.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            self._label,
        )
        p.end()


class IconButton(_HoverButton):
    """Square icon-only control: 32 normal, 24 in the title bar and compact."""

    def __init__(
        self,
        icon_name: str,
        size: int = 32,
        icon_size: int = Size.ICON,
        parent: Optional[QWidget] = None,
        *, border: bool = True,
        color: str = Color.TEXT_LABEL,
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self._icon_size = icon_size
        self._border = border
        self._color = color
        self._box = size
        self.set_box(size)

    def set_box(self, size: int, icon_size: Optional[int] = None) -> None:
        # The widget is never smaller than hit.min; a smaller `size` only
        # shrinks what is painted, so the click target survives (section 1.3).
        self._box = size
        self.setFixedSize(max(size, Size.HIT_MIN), max(size, Size.HIT_MIN))
        if icon_size is not None:
            self._icon_size = icon_size
        self.update()

    def set_icon(self, icon_name: str) -> None:
        if icon_name != self._icon_name:
            self._icon_name = icon_name
            self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        self._paint_frame(p, Radius.CONTROL, border=self._border, box=self._box)
        color = self._color if self.isEnabled() else Color.DISABLED_TEXT
        pm = icons.pixmap(self._icon_name, self._icon_size, color)
        p.drawPixmap(
            int((self.width() - self._icon_size) / 2),
            int((self.height() - self._icon_size) / 2),
            pm,
        )
        p.end()
