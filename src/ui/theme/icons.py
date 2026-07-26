"""SVG icon set.

Inline sources on a 16x16 grid, stroked at Size.ICON_STROKE, monochrome and
recoloured at render time — no emoji anywhere in the interface (brief section 8),
and no dependency on whatever the system font happens to draw for a symbol.

Rendered through QSvgRenderer into a device-pixel-ratio-correct QPixmap and
cached: the same handful of icons is re-requested on every hover repaint.

The capture indicator is *not* here: its three bars animate every 200 ms, so it
paints itself in ui.widgets.indicators instead of re-rasterising an SVG.
"""

from __future__ import annotations

from typing import Dict, Tuple

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from ui.theme.tokens import Size

_S = Size.ICON_STROKE

# `{c}` is the colour placeholder; every path is stroked, never filled, unless
# the shape is a dot.
_SVG: Dict[str, str] = {
    # Session mark: the same five-bar waveform the design uses as its logo.
    # Filled shapes carry stroke="none" — the document sets a 1.5 stroke for the
    # line icons, and letting it apply here would fatten every bar into a blob.
    "app": (
        '<rect x="1" y="6" width="2" height="4" rx="1" fill="{c}" stroke="none"/>'
        '<rect x="4.5" y="3.5" width="2" height="9" rx="1" fill="{c}" stroke="none"/>'
        '<rect x="8" y="2" width="2" height="12" rx="1" fill="{c}" stroke="none"/>'
        '<rect x="11.5" y="4.5" width="2" height="7" rx="1" fill="{c}" stroke="none"/>'
        '<rect x="15" y="6.5" width="1.5" height="3" rx="0.75" fill="{c}" stroke="none"/>'
    ),
    # Title bar / action row
    "menu": (
        '<circle cx="3.5" cy="8" r="1.3" fill="{c}" stroke="none"/>'
        '<circle cx="8" cy="8" r="1.3" fill="{c}" stroke="none"/>'
        '<circle cx="12.5" cy="8" r="1.3" fill="{c}" stroke="none"/>'
    ),
    "close": '<path d="M4 4 L12 12 M12 4 L4 12"/>',
    "chevron_down": '<path d="M4 6.5 L8 10.5 L12 6.5"/>',
    # Actions
    "answer": (  # sparkle — "ответить на последний вопрос" is the model talking
        '<path d="M8 2 L9.4 6.6 L14 8 L9.4 9.4 L8 14 L6.6 9.4 L2 8 L6.6 6.6 Z"/>'
    ),
    "summary": (  # condensed list
        '<path d="M3 4.5 H13 M3 8 H13 M3 11.5 H8.5"/>'
    ),
    "question": (
        '<path d="M5.9 5.9 A2.2 2.2 0 1 1 8 9 V10"/>'
        '<circle cx="8" cy="12.4" r="0.85" fill="{c}" stroke="none"/>'
    ),
    "settings": (
        '<circle cx="8" cy="8" r="2.3"/>'
        '<path d="M8 1.8 V3.4 M8 12.6 V14.2 M1.8 8 H3.4 M12.6 8 H14.2'
        ' M3.6 3.6 L4.8 4.8 M11.2 11.2 L12.4 12.4'
        ' M12.4 3.6 L11.2 4.8 M4.8 11.2 L3.6 12.4"/>'
    ),
    "history": (
        '<circle cx="8" cy="8" r="5.8"/>'
        '<path d="M8 4.6 V8 L10.4 9.6"/>'
    ),
    "folder": (
        '<path d="M2 4.5 A1 1 0 0 1 3 3.5 H6.3 L7.8 5.2 H13 A1 1 0 0 1 14 6.2'
        ' V11.5 A1 1 0 0 1 13 12.5 H3 A1 1 0 0 1 2 11.5 Z"/>'
    ),
    "warning": (
        '<path d="M8 2.6 L14.4 13.4 H1.6 Z"/>'
        '<path d="M8 6.6 V9.6"/>'
        '<circle cx="8" cy="11.6" r="0.8" fill="{c}" stroke="none"/>'
    ),
    "info": (
        '<circle cx="8" cy="8" r="5.8"/>'
        '<path d="M8 7.4 V11"/>'
        '<circle cx="8" cy="5.2" r="0.8" fill="{c}" stroke="none"/>'
    ),
}

_cache: Dict[Tuple[str, int, str, float], QPixmap] = {}


def _document(name: str, color: str, stroke: float) -> bytes:
    body = _SVG[name].replace("{c}", color)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" '
        f'fill="none" stroke="{color}" stroke-width="{stroke}" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    ).encode("utf-8")


def pixmap(name: str, size: int, color: str, *, stroke: float = _S) -> QPixmap:
    """Icon rasterised to `size` logical px in `color`."""
    dpr = 1.0
    from PyQt6.QtGui import QGuiApplication

    screen = QGuiApplication.primaryScreen() if QGuiApplication.instance() else None
    if screen is not None:
        dpr = screen.devicePixelRatio()
    key = (name, size, color, dpr)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    renderer = QSvgRenderer(QByteArray(_document(name, color, stroke)))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    _cache[key] = pm
    return pm


def icon(name: str, size: int, color: str, *, stroke: float = _S) -> QIcon:
    return QIcon(pixmap(name, size, color, stroke=stroke))


def names() -> tuple[str, ...]:
    return tuple(_SVG)
