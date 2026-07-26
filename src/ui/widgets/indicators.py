"""Title-bar state indicators and the status message zone (design system 2.1, 2.6).

Three pills, always present, never banners: a banner shifts content when it
appears, a pill lives in the title bar and costs zero space. The icon slot is a
fixed 14x14 whatever the state draws inside it — that is what keeps the label
from jumping as states change (the no-shift check in section 4).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from ui.theme.fonts import qcolor, qfont
from ui.theme.styles import label_qss
from ui.theme.tokens import (
    Color,
    Motion,
    STATUS_ZONE_H,
    STATUS_ZONE_W,
    Size,
    Space,
    Type,
)

# Capture states, mirroring audio.capture_base.
LIVE = "live"
SILENT = "silent"
ERROR = "error"

# Disk journal states.
REC_ON = "recording"
REC_PAUSED = "paused"
REC_FAILED = "failed"

# Model states. The spec has only online/offline; the three activity states were
# added because a local model can take seconds to answer and the window is the
# only place that can say so.
LLM_OFFLINE = "offline"
LLM_ONLINE = "online"
LLM_THINKING = "thinking"
LLM_GENERATING = "generating"
LLM_ERROR = "error"

# Bar heights (px, inside the 14x14 slot) cycled every Motion.CAPTURE_BAR_MS
# while audio flows. Discrete steps, not a continuous animation: the window
# lives in peripheral vision and a smooth pulse is what draws the eye.
_BARS = (
    (4, 10, 6),
    (8, 4, 11),
    (11, 8, 4),
    (6, 11, 8),
)
_BAR_W = 2
_BAR_GAP = 2
_FLAT_W = 11
_FLAT_H = 2

# "Модель думает": three dots, the accent travelling one step every
# Motion.THINKING_DOT_MS. 3*4 + 2*2 = 16 wide at most, inside the 14 slot once
# the idle dots shrink.
_PULSE_DOTS = 3
_PULSE_ACTIVE = 4.0
_PULSE_IDLE = 2.5
_PULSE_GAP = 1.5


class CaptureIcon(QWidget):
    """Wave = audio flowing, flat line = silence, struck line = error (6.5).

    The inversion in the mock-up is corrected here: the animation belongs to
    live audio, and silence — a normal mode, not a fault — is a grey flat line.
    """

    def __init__(self, slot: int = Size.INDICATOR_SLOT, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._slot = slot
        self._state = SILENT
        self._phase = 0
        self.setFixedSize(slot, slot)
        self._timer = QTimer(self)
        self._timer.setInterval(Motion.CAPTURE_BAR_MS)
        self._timer.timeout.connect(self._advance)

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        if state == LIVE:
            self._timer.start()
        else:
            self._timer.stop()
            self._phase = 0
        self.update()

    def _advance(self) -> None:
        self._phase = (self._phase + 1) % len(_BARS)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        cx, cy = self.width() / 2, self.height() / 2
        if self._state == LIVE:
            p.setBrush(qcolor(Color.STATE_GOOD))
            total = 3 * _BAR_W + 2 * _BAR_GAP
            x = cx - total / 2
            for h in _BARS[self._phase]:
                p.drawRoundedRect(
                    int(x), int(cy - h / 2), _BAR_W, h, _BAR_W / 2, _BAR_W / 2
                )
                x += _BAR_W + _BAR_GAP
            p.end()
            return

        color = Color.STATE_ERROR if self._state == ERROR else Color.STATE_IDLE
        p.setBrush(qcolor(color))
        p.drawRoundedRect(
            int(cx - _FLAT_W / 2), int(cy - _FLAT_H / 2), _FLAT_W, _FLAT_H,
            _FLAT_H / 2, _FLAT_H / 2,
        )
        if self._state == ERROR:
            pen = QPen(qcolor(Color.STATE_ERROR))
            pen.setWidthF(Size.ICON_STROKE)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(int(cx - 4), int(cy + 4), int(cx + 4), int(cy - 4))
        p.end()


class DotIcon(QWidget):
    """8 px status dot in the same fixed slot: filled, hollow, or a pulse.

    The pulse — three small dots with a travelling accent — is what separates
    "модель думает" from "модель онлайн": both are green and a static dot could
    not tell them apart, least of all in compact mode where the caption is gone.
    Same discrete rhythm as the capture indicator, no smooth throb.
    """

    def __init__(self, slot: int = Size.INDICATOR_SLOT, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color = Color.STATE_IDLE
        self._filled = True
        self._pulsing = False
        self._phase = 0
        self.setFixedSize(slot, slot)
        self._timer = QTimer(self)
        self._timer.setInterval(Motion.THINKING_DOT_MS)
        self._timer.timeout.connect(self._advance)

    def set_dot(self, color: str, filled: bool, pulsing: bool = False) -> None:
        if (color, filled, pulsing) == (self._color, self._filled, self._pulsing):
            return
        self._color = color
        self._filled = filled
        self._pulsing = pulsing
        if pulsing:
            self._timer.start()
        else:
            self._timer.stop()
            self._phase = 0
        self.update()

    def _advance(self) -> None:
        self._phase = (self._phase + 1) % _PULSE_DOTS
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        if self._pulsing:
            self._paint_pulse(p, cx, cy)
            p.end()
            return
        r = Size.DOT / 2
        color: QColor = qcolor(self._color)
        if self._filled:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(int(cx - r), int(cy - r), Size.DOT, Size.DOT)
        else:
            pen = QPen(color)
            pen.setWidthF(Size.ICON_STROKE)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            inner = r - Size.ICON_STROKE / 2
            p.drawEllipse(
                int(cx - inner), int(cy - inner), int(inner * 2), int(inner * 2)
            )
        p.end()

    def _paint_pulse(self, p: QPainter, cx: float, cy: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        span = _PULSE_DOTS * _PULSE_ACTIVE + (_PULSE_DOTS - 1) * _PULSE_GAP
        x = cx - span / 2
        for i in range(_PULSE_DOTS):
            active = i == self._phase
            size = _PULSE_ACTIVE if active else _PULSE_IDLE
            p.setBrush(qcolor(self._color if active else Color.STATE_IDLE))
            # The slot keeps its centre line whatever the dot size: the row must
            # not twitch as the accent travels.
            p.drawEllipse(
                QRectF(x + (_PULSE_ACTIVE - size) / 2, cy - size / 2, size, size)
            )
            x += _PULSE_ACTIVE + _PULSE_GAP


class StatusPill(QWidget):
    """Icon slot + optional caption, height 20, padding 4/6, gap 6.

    Draws no backdrop of its own: it sits on the title bar, which already has
    one. Captions disappear in compact mode and the state is read from the
    tooltip instead.

    Nothing here is clickable, and that is deliberate: an indicator reports, it
    does not act. Mouse events are left un-handled so Qt propagates them to the
    title bar — the whole strip drags the window, with only the menu and close
    buttons cut out of it.
    """

    def __init__(self, icon: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._icon = icon
        self.setFixedHeight(20)
        self._lead = 0
        row = QHBoxLayout(self)
        row.setContentsMargins(Space.S1 + 2, Space.S1, Space.S1 + 2, Space.S1)
        row.setSpacing(6)
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)
        self._label = QLabel("")
        self._label.setStyleSheet(label_qss(Type.STATUS))
        self._label.setFont(qfont(Type.STATUS))
        row.addWidget(self._label, 0, Qt.AlignmentFlag.AlignVCenter)

    def icon_widget(self) -> QWidget:
        return self._icon

    def set_caption(self, text: str) -> None:
        self._label.setText(text)
        self._label.setVisible(bool(text))

    def set_labels_visible(self, visible: bool) -> None:
        self._label.setVisible(visible and bool(self._label.text()))

    def set_lead_gap(self, extra: int) -> None:
        """Widen the pill's left padding instead of adding a layout spacer.

        The 14 px specified between indicators is carried inside the pill: a
        QSpacerItem would also collect the layout's own spacing on both sides,
        eating the pixels the status zone needs.
        """
        self._lead = extra
        row = self.layout()
        margins = row.contentsMargins()
        row.setContentsMargins(
            Space.S1 + 2 + extra, margins.top(), Space.S1 + 2, margins.bottom()
        )

    def lead_gap(self) -> int:
        return self._lead


class StatusMessage(QWidget):
    """The reserved 180x20 zone in the title bar (section 2.6).

    The zone exists whether or not there is anything to say, so a message
    appearing moves nothing. Lives 4 s, fades out in 120 ms; a second message
    replaces the first and restarts the clock.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Height is fixed; the width is set by the title bar to whatever the
        # pills leave, capped at STATUS_ZONE_W.
        self.setFixedHeight(STATUS_ZONE_H)
        self.setMaximumWidth(STATUS_ZONE_W)
        self._text = ""
        self._color = Color.TEXT_SECONDARY
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._font = qfont(Type.STATUS)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.clear)
        self._fade = QTimer(self)
        self._fade.setInterval(int(Motion.STATUS_MS / 6))
        self._fade.timeout.connect(self._step_fade)
        self._target = 0.0

    def show_message(
        self,
        text: str,
        *,
        detail: str = "",
        error: bool = False,
        lifetime_ms: Optional[int] = None,
    ) -> None:
        self._text = text
        self._color = Color.STATE_ERROR if error else Color.TEXT_SECONDARY
        self.setToolTip(detail or text)
        self.update()
        self._target = 1.0 if text else 0.0
        self._fade.start()
        self._hide_timer.stop()
        if text:
            self._hide_timer.start(
                Motion.STATUS_LIFETIME_MS if lifetime_ms is None else lifetime_ms
            )

    def clear(self) -> None:
        self._hide_timer.stop()
        self._target = 0.0
        self._fade.start()

    def _step_fade(self) -> None:
        current = self._effect.opacity()
        step = 1.0 / 6
        if abs(current - self._target) <= step:
            self._effect.setOpacity(self._target)
            self._fade.stop()
            if self._target == 0.0:
                self._text = ""
                self.setToolTip("")
            return
        self._effect.setOpacity(current + step * (1 if self._target > current else -1))

    def paintEvent(self, _event) -> None:
        if not self._text:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(self._color))
        p.drawEllipse(0, int(cy - Size.DOT / 2), Size.DOT, Size.DOT)
        p.setFont(self._font)
        p.setPen(qcolor(self._color))
        text_x = Size.DOT + Space.S1 + 2
        metrics = p.fontMetrics()
        elided = metrics.elidedText(
            self._text, Qt.TextElideMode.ElideRight, self.width() - text_x
        )
        p.drawText(
            text_x, 0, self.width() - text_x, self.height(),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
        p.end()


def capture_caption(state: str) -> str:
    return {
        LIVE: "Слушаю систему",
        SILENT: "Тишина",
        ERROR: "Ошибка захвата",
    }[state]


def capture_tooltip(state: str, reason: str = "") -> str:
    base = {
        LIVE: "Слушаю систему · звук идёт",
        SILENT: "Слушаю систему · тишина (в тишине система не отдаёт звук)",
        ERROR: "Ошибка захвата",
    }[state]
    return f"{base}\n{reason}" if reason else base


def disk_caption(state: str) -> str:
    return {
        REC_ON: "Диск: запись",
        REC_PAUSED: "Диск: пауза",
        REC_FAILED: "Диск: ошибка",
    }[state]


def disk_dot(state: str) -> tuple[str, bool]:
    """(colour, filled) per 6.8."""
    return {
        REC_ON: (Color.STATE_GOOD, True),
        REC_PAUSED: (Color.STATE_WARN, False),
        REC_FAILED: (Color.STATE_ERROR, True),
    }[state]


LLM_STATES = (LLM_OFFLINE, LLM_ONLINE, LLM_THINKING, LLM_GENERATING, LLM_ERROR)


def llm_caption(state: str) -> str:
    return {
        LLM_OFFLINE: "LLM: офлайн",
        LLM_ONLINE: "LLM: онлайн",
        LLM_THINKING: "LLM: думает",
        LLM_GENERATING: "LLM: пишет",
        LLM_ERROR: "LLM: ошибка",
    }[state]


def llm_dot(state: str) -> tuple[str, bool, bool]:
    """(colour, filled, pulsing).

    Online keeps the spec's filled green dot; "думает" is the same green with
    the travelling accent, so the two never read as one state.
    """
    return {
        LLM_OFFLINE: (Color.STATE_WARN, False, False),
        LLM_ONLINE: (Color.STATE_GOOD, True, False),
        LLM_THINKING: (Color.STATE_GOOD, True, True),
        LLM_GENERATING: (Color.STATE_WARN, True, False),
        LLM_ERROR: (Color.STATE_ERROR, True, False),
    }[state]


def llm_tooltip(state: str) -> str:
    return {
        LLM_OFFLINE: "LM Studio недоступен — запустите сервер на :1234",
        LLM_ONLINE: "LM Studio отвечает",
        LLM_THINKING: "Запрос отправлен, ответа пока нет",
        LLM_GENERATING: "Модель пишет ответ",
        LLM_ERROR: "Последний запрос не удался",
    }[state]


__all__ = [
    "CaptureIcon",
    "DotIcon",
    "StatusMessage",
    "StatusPill",
    "LIVE",
    "SILENT",
    "ERROR",
    "REC_ON",
    "REC_PAUSED",
    "REC_FAILED",
    "LLM_OFFLINE",
    "LLM_ONLINE",
    "LLM_THINKING",
    "LLM_GENERATING",
    "LLM_ERROR",
    "LLM_STATES",
    "capture_caption",
    "capture_tooltip",
    "disk_caption",
    "disk_dot",
    "llm_caption",
    "llm_dot",
    "llm_tooltip",
]
