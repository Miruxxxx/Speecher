"""Font loading and QFont/CSS construction from the typography tokens.

The bundled IBM Plex .ttf files (assets/fonts) are registered with
QFontDatabase at startup; system fonts are never used by design. If the files
are missing the module degrades to Segoe UI / Consolas and says so once, rather
than letting Qt pick something unpredictable per machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication

from ui.theme.tokens import (
    FALLBACK_MONO,
    FALLBACK_SANS,
    FAMILY_MONO,
    FAMILY_SANS,
    TextStyle,
)

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"

# Which files must be present for the design to render as specified.
_BUNDLED = (
    "IBMPlexSans-Regular.ttf",
    "IBMPlexSans-Italic.ttf",
    "IBMPlexSans-Medium.ttf",
    "IBMPlexSans-SemiBold.ttf",
    "IBMPlexMono-Regular.ttf",
)

_loaded_families: set[str] = set()
_loaded = False


def load_app_fonts(directory: Optional[Path] = None) -> bool:
    """Register the bundled fonts. Idempotent; returns True if Plex is usable."""
    global _loaded
    if _loaded:
        return FAMILY_SANS in _loaded_families
    _loaded = True
    d = Path(directory) if directory is not None else FONT_DIR
    missing: list[str] = []
    for name in _BUNDLED:
        path = d / name
        if not path.exists():
            missing.append(name)
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            missing.append(name)
            continue
        _loaded_families.update(QFontDatabase.applicationFontFamilies(font_id))
    if missing:
        logger.warning(
            "fonts: %s missing from %s; falling back to %s/%s",
            ", ".join(missing), d, FALLBACK_SANS, FALLBACK_MONO,
        )
    return FAMILY_SANS in _loaded_families


def family(name: str) -> str:
    """Resolve a token family to something the platform actually has."""
    if name in _loaded_families:
        return name
    return FALLBACK_MONO if name == FAMILY_MONO else FALLBACK_SANS


def _pt(size_px: float) -> float:
    """Token px -> Qt point size.

    Sizes in the spec are px because the window is fixed and Qt counts in px
    anyway, but QFont.setPixelSize is integer-only and two roles are half-pixel
    (10.5 / 12.5). Point size carries the fraction; the conversion uses the
    screen's *logical* DPI, which is what Qt lays widgets out in (physical
    scaling is handled separately by the device pixel ratio).
    """
    dpi = 96.0
    app = QGuiApplication.instance()
    if app is not None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None and screen.logicalDotsPerInch() > 0:
            dpi = screen.logicalDotsPerInch()
    return size_px * 72.0 / dpi


def qfont(style: TextStyle) -> QFont:
    f = QFont(family(style.family))
    f.setPointSizeF(_pt(style.size_px))
    f.setWeight(QFont.Weight(style.weight))
    f.setItalic(style.italic)
    # Kerning/tracking stay at zero everywhere (section 1.2).
    f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100.0)
    return f


def css(style: TextStyle, *, with_color: bool = True) -> str:
    """The same style as a QSS/rich-text declaration block (no braces)."""
    parts = [
        f"font-family: '{family(style.family)}'",
        f"font-size: {_fmt(style.size_px)}px",
        f"font-weight: {style.weight}",
    ]
    if style.italic:
        parts.append("font-style: italic")
    if with_color:
        parts.append(f"color: {style.color}")
    return "; ".join(parts) + ";"


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def qcolor(hex_color: str, alpha: int = 255) -> QColor:
    c = QColor(hex_color)
    if alpha != 255:
        c.setAlpha(alpha)
    return c
