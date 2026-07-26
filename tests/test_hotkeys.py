"""Hotkey parsing (ui/hotkeys.py). No registration here — that needs Windows."""

from __future__ import annotations

import pytest

from ui.hotkeys import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    HotkeyError,
    describe,
    parse_combo,
)


@pytest.mark.parametrize(
    "combo, mods, key",
    [
        ("Alt+Space", MOD_ALT, 0x20),
        ("alt+space", MOD_ALT, 0x20),
        ("Alt+Enter", MOD_ALT, 0x0D),
        ("Alt+Return", MOD_ALT, 0x0D),
        ("Alt+5", MOD_ALT, ord("5")),
        ("Alt+Q", MOD_ALT, ord("Q")),
        ("Ctrl+Shift+F5", MOD_CONTROL | MOD_SHIFT, 0x74),
    ],
)
def test_parse_known_combinations(combo, mods, key):
    assert parse_combo(combo) == (mods, key)


@pytest.mark.parametrize(
    "combo",
    ["", "Alt", "Space", "Alt+Space+Q", "Alt+Nope", "Alt+F30"],
)
def test_bad_combinations_raise(combo):
    with pytest.raises(HotkeyError):
        parse_combo(combo)


def test_describe_spaces_the_parts_for_the_cheat_sheet():
    assert describe("Alt+Space") == "Alt + Space"
    assert describe("Ctrl+Shift+F5") == "Ctrl + Shift + F5"
