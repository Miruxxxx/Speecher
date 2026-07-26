"""First-run onboarding and the hotkey cheat sheet (design system 5, 6.7).

The four hints from the mock-up live here rather than on the empty screen: at
520x340 they do not fit next to the icon and the two lines of the empty state,
and they are a once-in-a-lifetime read while the empty state is shown for
minutes at a time.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app_config import HotkeysConfig
from ui.hotkeys import describe
from ui.theme.fonts import qfont
from ui.theme.styles import label_qss
from ui.theme.tokens import ONBOARDING_H, ONBOARDING_W, Space, Type
from ui.widgets.buttons import TextButton
from ui.windows.base import Panel
from ui.windows.settings import HOTKEY_ROWS


def cheat_sheet(hotkeys: HotkeysConfig) -> List[tuple[str, str]]:
    """(combination, action) rows — the same list the settings window edits."""
    rows = []
    for action, caption in HOTKEY_ROWS:
        combo = getattr(hotkeys, action, "")
        if combo:
            rows.append((describe(combo), caption))
    return rows


class _Row(QWidget):
    """Hotkey in Plex Mono 11, action in 12 text.label, 8 px apart."""

    def __init__(self, combo: str, action: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(Space.S3)
        key = QLabel(combo)
        key.setFont(qfont(Type.HOTKEY))
        key.setStyleSheet(label_qss(Type.HOTKEY))
        key.setFixedWidth(110)
        row.addWidget(key)
        text = QLabel(action)
        text.setFont(qfont(Type.MENU_ITEM))
        text.setStyleSheet(label_qss(Type.MENU_ITEM))
        text.setWordWrap(True)
        row.addWidget(text, 1)


class OnboardingWindow(Panel):
    """420x260: what the window is, and the four keys worth remembering."""

    def __init__(
        self,
        hotkeys: HotkeysConfig,
        parent: Optional[QWidget] = None,
        *,
        intro: bool = True,
    ) -> None:
        super().__init__(
            "О программе" if not intro else "Speecher", (ONBOARDING_W, ONBOARDING_H), parent
        )
        root = QVBoxLayout(self.body())
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(Space.S3)

        lead = QLabel(
            "Окно слушает системный звук — всё, что играет в наушниках, — "
            "и показывает живой транскрипт поверх остальных окон."
        )
        lead.setWordWrap(True)
        lead.setFont(qfont(Type.MENU_ITEM))
        lead.setStyleSheet(label_qss(Type.MENU_ITEM))
        root.addWidget(lead)

        rows: Sequence[tuple[str, str]] = cheat_sheet(hotkeys)[:4] or [
            ("", "Хоткеи не заданы — все действия есть в окне")
        ]
        for combo, action in rows:
            root.addWidget(_Row(combo, action))

        tail = QLabel("Двойной клик по шапке — компактный режим 360 × 200")
        tail.setWordWrap(True)
        tail.setFont(qfont(Type.BUTTON_SUB))
        tail.setStyleSheet(label_qss(Type.BUTTON_SUB))
        root.addWidget(tail)
        root.addStretch()

        footer = QHBoxLayout()
        footer.addStretch()
        ok = TextButton("Понятно", self, style=Type.BUTTON_LABEL)
        ok.clicked.connect(self.close)
        footer.addWidget(ok)
        root.addLayout(footer)


class CheatSheetWindow(Panel):
    """The full list, opened from the session menu ("Хоткеи…")."""

    def __init__(self, hotkeys: HotkeysConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__("Хоткеи", (ONBOARDING_W, ONBOARDING_H + 80), parent)
        root = QVBoxLayout(self.body())
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(Space.S2)
        rows: Iterable[tuple[str, str]] = cheat_sheet(hotkeys)
        for combo, action in rows:
            root.addWidget(_Row(combo, action))
        root.addStretch()
        note = QLabel("Изменить комбинации можно в настройках")
        note.setFont(qfont(Type.BUTTON_SUB))
        note.setStyleSheet(label_qss(Type.BUTTON_SUB))
        note.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(note)
