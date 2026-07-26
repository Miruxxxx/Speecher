"""The design system's own rules, checked against the token module.

Section 7's acceptance list ends with "ни одного цвета и отступа вне раздела 1",
so these tests guard both halves: the tokens keep the values the spec fixed, and
the UI modules do not invent their own.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from ui.theme import tokens
from ui.theme.tokens import COMPACT, NORMAL, Border, Color, Radius, Space, Type

SRC = Path(__file__).resolve().parent.parent / "src" / "ui"

# Files allowed to contain literal colours: the token module itself.
_TOKEN_FILE = SRC / "theme" / "tokens.py"
_HEX = re.compile(r"#[0-9A-Fa-f]{3,8}\b")


def _color_values() -> set[str]:
    return {
        value
        for name, value in vars(Color).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def test_every_colour_is_an_opaque_six_digit_hex():
    for name, value in vars(Color).items():
        if name.startswith("_") or not isinstance(value, str):
            continue
        assert _HEX.fullmatch(value), f"{name} = {value!r}"
        assert len(value) == 7, f"{name} carries alpha; the system is opaque"


def test_spacing_is_a_four_pixel_grid():
    for name, value in vars(Space).items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        assert value % 4 == 0, f"Space.{name} = {value} is off the 4 px grid"


def test_border_width_is_always_one():
    assert Border.WIDTH == 1


def test_no_font_smaller_than_the_floor():
    for name, style in vars(Type).items():
        if name.startswith("_") or not isinstance(style, tokens.TextStyle):
            continue
        assert style.size_px >= tokens.MIN_FONT_PX, f"Type.{name}"


def test_layout_modes_carry_the_specified_geometry():
    # Width deviates from the spec's 520 on purpose (see DESIGN_SYSTEM.md §7):
    # the title bar gained a close button and the status zone was starving.
    # Everything else is the spec's, and it stays on the 4 px grid.
    assert (NORMAL.width, NORMAL.height) == (600, 340)
    assert NORMAL.width % 4 == 0
    assert (COMPACT.width, COMPACT.height) == (360, 200)
    assert NORMAL.titlebar_h == 36 and COMPACT.titlebar_h == 28
    assert NORMAL.gutter_w == 56 and COMPACT.gutter_w == 44
    assert NORMAL.action_h == 40 and COMPACT.action_h == 28
    # 6.2: seconds are dropped in compact, the gutter stays.
    assert NORMAL.ts_format == "%H:%M:%S" and COMPACT.ts_format == "%H:%M"
    assert NORMAL.answer_max_h == 160 and COMPACT.answer_max_h == 96


def test_every_layout_mode_field_is_set():
    for mode in (NORMAL, COMPACT):
        for field in fields(mode):
            assert getattr(mode, field.name) is not None


@pytest.mark.parametrize(
    "path",
    sorted(p for p in SRC.rglob("*.py") if p != _TOKEN_FILE),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_ui_modules_do_not_hardcode_colours(path: Path):
    """Hex literals outside tokens.py mean a value escaped the system."""
    text = path.read_text(encoding="utf-8")
    found = {m.group(0) for m in _HEX.finditer(text)}
    assert not found, f"{path.name} hardcodes {sorted(found)}"


def test_radius_scale_matches_the_spec():
    assert (Radius.WINDOW, Radius.PANEL, Radius.CONTROL) == (10, 8, 6)
    assert Radius.PILL == 999
