"""Title bar: session mark, three indicators, status zone, menu (design system 3).

There is no system chrome, so this row *is* the window's handle: dragging works
anywhere on it except the capture pill (which opens the sound settings) and the
menu button. Double-clicking it switches 520 <-> 360.

No minimise/maximise buttons on purpose (brief section 5.5): maximising would
cover the call, minimising defeats the point of a window that must stay visible.
Show/hide is a hotkey.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QPainter, QPainterPath
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ui.theme import icons
from ui.theme.fonts import qcolor, qfont
from ui.theme.tokens import (
    Color,
    LayoutMode,
    NORMAL,
    Radius,
    STATUS_ZONE_W,
    Size,
    Space,
    Type,
)
from ui.widgets.buttons import IconButton
from ui.widgets.indicators import (
    LLM_STATES,
    REC_FAILED,
    REC_ON,
    REC_PAUSED,
    CaptureIcon,
    DotIcon,
    StatusMessage,
    StatusPill,
    capture_caption,
    disk_caption,
    llm_caption,
)


def _pill_width(captions: tuple[str, ...]) -> int:
    """Width of a pill fixed to its longest caption.

    Otherwise every switch between "Слушаю систему", "Тишина" and "Ошибка
    захвата" would shove the two pills next to it sideways — a shift in the one
    row the user glances at most. Geometry per 2.1: 14 icon slot + 6 gap +
    caption + 6/6 padding.
    """
    metrics = QFontMetrics(qfont(Type.STATUS))
    widest = max(metrics.horizontalAdvance(caption) for caption in captions)
    return Size.INDICATOR_SLOT + 6 + widest + 2 * (Space.S1 + 2)


_CAPTURE_CAPTIONS = tuple(capture_caption(s) for s in ("live", "silent", "error"))
_DISK_CAPTIONS = tuple(disk_caption(s) for s in (REC_ON, REC_PAUSED, REC_FAILED))
_LLM_CAPTIONS = tuple(llm_caption(s) for s in LLM_STATES)


class TitleBar(QWidget):
    menu_requested = pyqtSignal()
    close_requested = pyqtSignal()
    mode_toggled = pyqtSignal()

    def __init__(self, mode: LayoutMode = NORMAL, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mode = mode
        self._drag: Optional[QPointF] = None
        self.setFixedHeight(mode.titlebar_h)

        row = QHBoxLayout(self)
        row.setContentsMargins(mode.padding, 0, mode.padding, 0)
        # Base spacing is the 8 px grid step; the specified 14 px applies
        # *between the indicators* (section 3), added as explicit spacers so the
        # status zone keeps the pixels the rest of the row does not need.
        row.setSpacing(Space.S2)

        self._mark = QLabel()
        self._mark.setFixedSize(Size.ICON, Size.ICON)
        self._mark.setPixmap(icons.pixmap("app", Size.ICON, Color.ACCENT))
        row.addWidget(self._mark, 0, Qt.AlignmentFlag.AlignVCenter)

        self.capture = StatusPill(CaptureIcon(), self)
        self.disk = StatusPill(DotIcon(), self)
        self.llm = StatusPill(DotIcon(), self)
        for pill in (self.capture, self.disk, self.llm):
            row.addWidget(pill, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch()

        self.status = StatusMessage(self)
        row.addWidget(self.status, 0, Qt.AlignmentFlag.AlignVCenter)

        self.menu_button = IconButton("menu", Size.HIT_MIN, Size.ICON, self)
        self.menu_button.setToolTip("Сессия: экспорт, запись, история, настройки")
        self.menu_button.clicked.connect(self.menu_requested.emit)
        row.addWidget(self.menu_button, 0, Qt.AlignmentFlag.AlignVCenter)

        # The brief rules out minimise and maximise, not closing: without a
        # visible way out the only exit is the task manager.
        self.close_button = IconButton("close", Size.HIT_MIN, 12, self)
        self.close_button.setToolTip("Закрыть Speecher")
        self.close_button.clicked.connect(self.close_requested.emit)
        row.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._captions = {"capture": "", "disk": "", "llm": ""}
        self.set_mode(mode, force=True)

    # -- mode ------------------------------------------------------------

    def set_mode(self, mode: LayoutMode, *, force: bool = False) -> None:
        if mode.name == self._mode.name and not force:
            return
        self._mode = mode
        self.setFixedHeight(mode.titlebar_h)
        layout = self.layout()
        layout.setContentsMargins(mode.padding, 0, mode.padding, 0)
        layout.setSpacing(Space.S2)
        layout.invalidate()
        self._mark.setVisible(mode.with_labels)
        for pill in (self.capture, self.disk, self.llm):
            pill.set_labels_visible(mode.with_labels)
        # The 14 px between indicators lives in the following pill's left
        # padding (see StatusPill.set_lead_gap).
        extra = mode.title_gap - Space.S2
        if mode.with_labels:
            self.capture.set_lead_gap(0)
            self.capture.setFixedWidth(_pill_width(_CAPTURE_CAPTIONS))
            for pill, captions in (
                (self.disk, _DISK_CAPTIONS), (self.llm, _LLM_CAPTIONS)
            ):
                pill.set_lead_gap(extra)
                pill.setFixedWidth(_pill_width(captions) + extra)
        else:
            for pill in (self.capture, self.disk, self.llm):
                pill.set_lead_gap(0)
                pill.setFixedWidth(Size.INDICATOR_SLOT + 2 * (Space.S1 + 2))
        self.menu_button.set_box(Size.HIT_MIN, mode.icon)
        self.close_button.set_box(Size.HIT_MIN, mode.icon - 4)
        self._resize_status()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_status()

    def _resize_status(self) -> None:
        """The status zone is always present, so a message shifts nothing.

        Its width is whatever the rest of the row leaves, capped at the
        specified 180: at 520 px with all three captions shown there is less
        than that, and the pills — pinned to their longest captions so they
        never dance — must not be the ones to give way. A message too long for
        the zone elides and keeps its full text in the tooltip (2.6).
        """
        layout = self.layout()
        margins = layout.contentsMargins()
        used = margins.left() + margins.right()
        gaps = 0
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if widget is self.status:
                continue
            if widget is not None:
                if not widget.isVisibleTo(self):
                    continue
                used += widget.width() or widget.sizeHint().width()
            else:
                used += item.sizeHint().width()
            gaps += 1
        used += layout.spacing() * gaps
        self.status.setFixedWidth(max(0, min(STATUS_ZONE_W, self.width() - used)))

    # -- dragging --------------------------------------------------------

    def _in_drag_zone(self, pos) -> bool:
        """Everything drags except the two real buttons.

        The indicators are part of the handle on purpose: they take up most of
        the strip, and a pill that swallows the press would leave the user
        hunting for the empty pixels between them.
        """
        for widget in (self.menu_button, self.close_button):
            if widget.geometry().contains(pos.toPoint()):
                return False
        return True

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._in_drag_zone(event.position()):
            window = self.window()
            self._drag = event.globalPosition() - QPointF(window.frameGeometry().topLeft())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move((event.globalPosition() - self._drag).toPoint())
        event.accept()

    def mouseReleaseEvent(self, _event) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:
        if self._in_drag_zone(event.position()):
            self.mode_toggled.emit()
        event.accept()

    # -- painting --------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        rect = QRectF(self.rect())
        radius = Radius.WINDOW
        path.moveTo(rect.left(), rect.bottom())
        path.lineTo(rect.left(), rect.top() + radius)
        path.quadTo(rect.left(), rect.top(), rect.left() + radius, rect.top())
        path.lineTo(rect.right() - radius, rect.top())
        path.quadTo(rect.right(), rect.top(), rect.right(), rect.top() + radius)
        path.lineTo(rect.right(), rect.bottom())
        path.closeSubpath()
        p.fillPath(path, qcolor(Color.BG_TITLEBAR))
        p.end()
