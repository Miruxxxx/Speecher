"""Global hotkeys (design system 5).

Windows' RegisterHotKey with a thread-owned message loop: no third-party
dependency, and the combinations work while the overlay is hidden or another
application has the keyboard — which is the whole reason they exist (a click
costs the user a trip away from the call).

A combination Windows refuses is *not* fatal: the action stays available on its
button, the failure is reported back so the settings window can mark the field
"занято другим приложением", and the app keeps running (section 5).
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from ctypes import wintypes
from typing import Dict, Mapping, Optional

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

_MODIFIERS = {
    "alt": MOD_ALT,
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "meta": MOD_WIN,
}

_NAMED_KEYS = {
    "space": 0x20,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "tab": 0x09,
    "backspace": 0x08,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
}


class HotkeyError(ValueError):
    """The combination cannot be parsed (bad name, no key, modifiers only)."""


def parse_combo(combo: str) -> tuple[int, int]:
    """"Alt+Space" -> (modifier mask, virtual key code)."""
    parts = [p.strip() for p in str(combo).split("+") if p.strip()]
    if not parts:
        raise HotkeyError("пустая комбинация")
    mods = 0
    key: Optional[int] = None
    for part in parts:
        low = part.lower()
        if low in _MODIFIERS:
            mods |= _MODIFIERS[low]
            continue
        if key is not None:
            raise HotkeyError(f"больше одной клавиши: {combo}")
        if low in _NAMED_KEYS:
            key = _NAMED_KEYS[low]
        elif len(part) == 1 and part.isalnum():
            key = ord(part.upper())
        elif low.startswith("f") and low[1:].isdigit() and 1 <= int(low[1:]) <= 24:
            key = 0x70 + int(low[1:]) - 1
        else:
            raise HotkeyError(f"неизвестная клавиша: {part}")
    if key is None:
        raise HotkeyError(f"нет клавиши, только модификаторы: {combo}")
    if mods == 0:
        raise HotkeyError(f"нужен хотя бы один модификатор: {combo}")
    return mods, key


class HotkeyManager(QObject):
    """Registers the bindings and emits `triggered(action)` on the Qt thread."""

    triggered = pyqtSignal(str)
    # action -> reason, after every (re)registration attempt.
    failed = pyqtSignal(dict)

    def __init__(self, bindings: Mapping[str, str], parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._bindings: Dict[str, str] = {k: v for k, v in bindings.items() if v}
        self._failures: Dict[str, str] = {}
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._ready = threading.Event()

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        if not IS_WINDOWS:
            self._failures = {a: "глобальные хоткеи только для Windows" for a in self._bindings}
            self.failed.emit(dict(self._failures))
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        # Registration happens in that thread; waiting briefly means the caller
        # (and the settings window) sees real results instead of an empty dict.
        self._ready.wait(timeout=2.0)

    def stop(self) -> None:
        thread, thread_id = self._thread, self._thread_id
        self._thread = None
        self._thread_id = None
        if thread is None or thread_id is None:
            return
        ctypes.windll.user32.PostThreadMessageW(wintypes.DWORD(thread_id), WM_QUIT, 0, 0)
        thread.join(timeout=2.0)

    def rebind(self, bindings: Mapping[str, str]) -> None:
        """Apply a new set. Restarts the loop — RegisterHotKey is per thread."""
        self.stop()
        self._bindings = {k: v for k, v in bindings.items() if v}
        self.start()

    def failures(self) -> Dict[str, str]:
        return dict(self._failures)

    # -- worker ----------------------------------------------------------

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        failures: Dict[str, str] = {}
        ids: Dict[int, str] = {}
        for index, (action, combo) in enumerate(self._bindings.items(), start=1):
            try:
                mods, key = parse_combo(combo)
            except HotkeyError as exc:
                failures[action] = str(exc)
                continue
            if not user32.RegisterHotKey(None, index, mods | MOD_NOREPEAT, key):
                failures[action] = "занято другим приложением"
                continue
            ids[index] = action
        self._failures = failures
        self.failed.emit(dict(failures))
        if failures:
            logger.warning("hotkeys not registered: %s", failures)
        self._ready.set()

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                action = ids.get(int(msg.wParam))
                if action:
                    # Queued across threads by Qt: the slot runs on the UI thread.
                    self.triggered.emit(action)
        for hotkey_id in ids:
            user32.UnregisterHotKey(None, hotkey_id)
        self._ready.set()


def describe(combo: str) -> str:
    """Human spelling for a cheat sheet row: "Alt+Space" -> "Alt + Space"."""
    parts = [p.strip() for p in str(combo).split("+") if p.strip()]
    return " + ".join(parts)
