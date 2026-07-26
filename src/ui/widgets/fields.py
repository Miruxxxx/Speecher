"""Settings controls (design system 2.5).

Row geometry is fixed by the spec: caption left, control right, widths 1 : 1.05,
10 px between rows, 20 px between sections. Controls are 28 px high with 6/8
padding and radius 6, and they are the only widgets in the app that show a
keyboard focus ring — the overlay never takes focus.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui.theme.fonts import qcolor, qfont
from ui.theme.styles import label_qss
from ui.theme.tokens import Border, Color, Radius, Space, Type

CONTROL_H = 28
_SEG_PAD = 2


class FieldRow(QWidget):
    """Caption + control on one line."""

    def __init__(self, caption: str, control: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(Space.S3)
        label = QLabel(caption)
        label.setFont(qfont(Type.FIELD_LABEL))
        label.setStyleSheet(label_qss(Type.FIELD_LABEL))
        row.addWidget(label, 100)
        row.addWidget(control, 105)
        self._control = control

    def control(self) -> QWidget:
        return self._control


class Section(QWidget):
    """A titled group of rows."""

    def __init__(self, title: str, rows: Sequence[QWidget], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        head = QLabel(title)
        head.setFont(qfont(Type.PANEL_TITLE))
        head.setStyleSheet(label_qss(Type.PANEL_TITLE))
        box.addWidget(head)
        for row in rows:
            box.addWidget(row)


class SegmentedControl(QWidget):
    """Container 28, padding 2, selected segment filled with the accent."""

    changed = pyqtSignal(int)

    def __init__(self, options: Sequence[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._options: List[str] = list(options)
        self._index = 0
        self.setFixedHeight(CONTROL_H)
        self.setMinimumWidth(len(self._options) * 70)

    def index(self) -> int:
        return self._index

    def value(self) -> str:
        return self._options[self._index]

    def set_index(self, index: int) -> None:
        index = max(0, min(index, len(self._options) - 1))
        if index != self._index:
            self._index = index
            self.update()
            self.changed.emit(index)

    def set_value(self, value: str) -> None:
        if value in self._options:
            self.set_index(self._options.index(value))

    def _segment_width(self) -> float:
        return (self.width() - 2 * _SEG_PAD) / max(1, len(self._options))

    def mousePressEvent(self, event) -> None:
        width = self._segment_width()
        if width > 0:
            self.set_index(int((event.position().x() - _SEG_PAD) // width))
        event.accept()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(qcolor(Color.BORDER))
        pen.setWidthF(Border.WIDTH)
        p.setPen(pen)
        p.setBrush(qcolor(Color.BG_SURFACE))
        p.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), Radius.CONTROL, Radius.CONTROL
        )
        width = self._segment_width()
        for i, option in enumerate(self._options):
            rect = QRectF(
                _SEG_PAD + i * width, _SEG_PAD, width, self.height() - 2 * _SEG_PAD
            )
            if i == self._index:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qcolor(Color.ACCENT))
                p.drawRoundedRect(rect, Radius.SMALL, Radius.SMALL)
                p.setPen(qcolor(Color.ON_ACCENT))
            else:
                p.setPen(qcolor(Color.TEXT_SECONDARY))
            p.setFont(qfont(Type.FIELD_VALUE))
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), option)
        p.end()


class HotkeyEdit(QWidget):
    """Click, press a combination, done. Esc cancels, Backspace clears.

    A combination Windows will not give us is marked here in state.error with
    "занято другим приложением" — the action still works from its button
    (section 5).
    """

    changed = pyqtSignal(str)

    def __init__(self, combo: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._combo = combo
        self._capturing = False
        self._error = ""
        self.setFixedHeight(CONTROL_H)
        self.setMinimumWidth(120)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def combo(self) -> str:
        return self._combo

    def set_combo(self, combo: str) -> None:
        self._combo = combo
        self.update()

    def set_error(self, message: str) -> None:
        self._error = message
        self.setToolTip(message)
        self.update()

    def mousePressEvent(self, event) -> None:
        self._capturing = True
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.update()
        event.accept()

    def focusOutEvent(self, event) -> None:
        self._capturing = False
        self.update()
        super().focusOutEvent(event)

    def keyPressEvent(self, event) -> None:
        if not self._capturing:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in (
            Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta,
        ):
            return  # modifiers alone are not a combination
        if key == Qt.Key.Key_Escape:
            self._capturing = False
            self.update()
            return
        if key == Qt.Key.Key_Backspace:
            self._capturing = False
            self._combo = ""
            self._error = ""
            self.changed.emit("")
            self.update()
            return
        combo = QKeySequence(event.keyCombination()).toString()
        if combo:
            self._combo = combo
            self._error = ""
            self._capturing = False
            self.changed.emit(combo)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        border = Color.BORDER
        if self._error:
            border = Color.STATE_ERROR
        elif self._capturing or self.hasFocus():
            border = Color.FOCUS_RING
        pen = QPen(qcolor(border))
        pen.setWidthF(Border.WIDTH)
        p.setPen(pen)
        p.setBrush(qcolor(Color.BG_SURFACE))
        p.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), Radius.CONTROL, Radius.CONTROL
        )
        if self._capturing:
            text, color, style = "нажмите комбинацию", Color.TEXT_SECONDARY, Type.FIELD_VALUE
        elif self._error:
            text, color, style = self._error, Color.STATE_ERROR, Type.BUTTON_SUB
        elif self._combo:
            text, color, style = self._combo, Color.TEXT_LABEL, Type.HOTKEY
        else:
            text, color, style = "не задан", Color.TEXT_SECONDARY, Type.FIELD_VALUE
        p.setFont(qfont(style))
        p.setPen(qcolor(color))
        rect = self.rect().adjusted(8, 0, -8, 0)
        p.drawText(
            rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, rect.width()),
        )
        p.end()
