"""Qt stylesheets assembled from the tokens.

Only widgets Qt draws for us (menus, scrollbars, combo boxes, tooltips) are
styled through QSS. Everything with per-state geometry — action buttons, status
pills, cards — paints itself, because QSS cannot express "the icon slot stays
14x14 while the state changes" without pixel drift.
"""

from __future__ import annotations

from ui.theme.fonts import css, family
from ui.theme.tokens import (
    Border,
    Color,
    MENU_ITEM_H,
    MENU_MAX_W,
    Radius,
    TextStyle,
    Type,
)


def scrollbar_qss() -> str:
    """4 px handle, transparent track, no arrows (section 2.3)."""
    return (
        "QScrollBar:vertical { background: transparent; width: 4px; margin: 0; }"
        f"QScrollBar::handle:vertical {{ background: {Color.BORDER_STRONG};"
        " border-radius: 2px; min-height: 24px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
        " height: 0; background: none; border: none; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
        " background: none; }"
        "QScrollBar:horizontal { height: 0; background: transparent; }"
    )


def menu_qss() -> str:
    """Session menu, section 2.5. Hover and keyboard focus share one highlight."""
    return (
        f"QMenu {{ background: {Color.BG_SURFACE};"
        f" border: {Border.WIDTH}px solid {Color.BORDER};"
        f" border-radius: {Radius.PANEL}px; padding: 6px;"
        f" max-width: {MENU_MAX_W}px; {css(Type.MENU_ITEM)} }}"
        f"QMenu::item {{ padding: 7px 10px; border-radius: {Radius.SMALL}px;"
        f" min-height: {MENU_ITEM_H - 14}px; }}"
        f"QMenu::item:selected {{ background: {Color.BG_SURFACE_ACTIVE};"
        f" color: {Color.TEXT_LABEL}; }}"
        f"QMenu::item:disabled {{ color: {Color.DISABLED_TEXT}; }}"
        f"QMenu::separator {{ height: {Border.WIDTH}px;"
        f" background: {Color.BORDER_STRONG}; margin: 5px 10px; }}"
    )


def tooltip_qss() -> str:
    """Section 2.1: 11 px, surface background, 1 px border, radius 6, pad 4/8."""
    return (
        f"QToolTip {{ background: {Color.BG_SURFACE}; color: {Color.TEXT_SECONDARY};"
        f" border: {Border.WIDTH}px solid {Color.BORDER};"
        f" border-radius: {Radius.CONTROL}px; padding: 4px 8px;"
        f" {css(Type.STATUS, with_color=False)} }}"
    )


def field_qss() -> str:
    """Settings controls, section 2.5: height 28, padding 6/8, radius 6."""
    return (
        f"QComboBox, QLineEdit, QSpinBox {{ background: {Color.BG_SURFACE};"
        f" color: {Color.TEXT_LABEL};"
        f" border: {Border.WIDTH}px solid {Color.BORDER};"
        f" border-radius: {Radius.CONTROL}px; padding: 6px 8px;"
        f" min-height: {28 - 2 * 6 - 2 * Border.WIDTH}px;"
        f" {css(Type.FIELD_VALUE, with_color=False)} }}"
        f"QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{"
        f" border-color: {Color.BORDER_STRONG}; }}"
        f"QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{"
        f" border-color: {Color.FOCUS_RING}; }}"
        f"QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled {{"
        f" color: {Color.DISABLED_TEXT}; background: {Color.DISABLED_BG}; }}"
        "QComboBox::drop-down { border: none; width: 18px; }"
        f"QComboBox QAbstractItemView {{ background: {Color.BG_SURFACE};"
        f" color: {Color.TEXT_LABEL};"
        f" border: {Border.WIDTH}px solid {Color.BORDER};"
        f" selection-background-color: {Color.BG_SURFACE_ACTIVE};"
        f" selection-color: {Color.TEXT_LABEL}; outline: none; }}"
        + scrollbar_qss()
    )


def label_qss(style: TextStyle) -> str:
    return f"QLabel {{ background: transparent; {css(style)} }}"


def window_qss() -> str:
    """Base palette for the satellite windows (settings, history, onboarding)."""
    return (
        f"QWidget {{ background: {Color.BG_WINDOW}; color: {Color.TEXT_LABEL};"
        f" font-family: '{family(Type.FIELD_LABEL.family)}'; }}"
        f"QScrollArea {{ background: {Color.BG_WINDOW}; border: none; }}"
        + scrollbar_qss()
    )
