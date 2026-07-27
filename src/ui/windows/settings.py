"""Settings window (design system 2.5, section 5).

Four sections: where the sound comes from, how it is recognised, how often the
summary runs, and the hotkeys. Everything is written back into config.toml with
the comments intact (app_config.save_overrides).

What applies live and what does not is stated, not guessed: the summary
interval and the hotkeys take effect immediately; the capture device, engine and
language are read when their model loads, so they are marked as needing a
restart rather than silently doing nothing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app_config import AppConfig, save_overrides
from audio.rust_capture import list_devices, resolve_binary
from translate import languages
from ui.theme.fonts import qfont
from ui.theme.styles import field_qss, label_qss
from ui.theme.tokens import (
    Color,
    PANELS_INLINE,
    PANELS_SIDE_COMPACT,
    PANELS_SIDE_LARGE,
    SETTINGS_H,
    SETTINGS_W,
    Space,
    Type,
    panel_mode,
)
from ui.widgets.buttons import TextButton
from ui.widgets.fields import CONTROL_H, FieldRow, HotkeyEdit, Section, SegmentedControl
from ui.widgets.llm_section import LlmSection
from ui.windows.base import Panel

logger = logging.getLogger(__name__)

# Value -> caption. The interval drives both the button caption and Alt+5 (6.6).
INTERVAL_CHOICES: List[tuple[int, str]] = [
    (0, "выкл."),
    (1, "1 минута"),
    (5, "5 минут"),
    (10, "10 минут"),
    (15, "15 минут"),
    (30, "30 минут"),
]

# Where the answer and the summary are shown (§7.9). Captions say what the user
# will see, not what the config key is called.
PANEL_CHOICES = [
    (PANELS_INLINE, "карточкой в ленте"),
    (PANELS_SIDE_LARGE, "окнами по бокам"),
    (PANELS_SIDE_COMPACT, "окнами по бокам, компактными"),
]

LANGUAGE_CHOICES = [
    ("auto", "автоопределение"),
    ("ru", "русский"),
    ("en", "английский"),
    ("de", "немецкий"),
    ("es", "испанский"),
]

HOTKEY_ROWS = [
    ("toggle_window", "Показать / скрыть окно"),
    ("toggle_recording", "Запись на диск"),
    ("pause_recording", "Пауза записи"),
    ("summary", "Выжимка за интервал"),
    ("answer", "Ответить на вопрос"),
    ("last_question", "Показать последний вопрос"),
    ("history", "История выжимок"),
    ("translate", "Перевод на лету"),
]

# Target languages the translator offers, in the table's order.
TRANSLATE_CHOICES = [(lang.code, lang.caption) for lang in languages.LANGUAGES]

_RESTART_HINT = "Устройство, движок и языки применятся после перезапуска"


class SettingsWindow(Panel):
    """Reads an AppConfig, writes the changed keys back, tells the app what
    changed live."""

    applied = pyqtSignal(dict)
    # Device enumeration runs in a worker thread; this carries the names back.
    _devices_loaded = pyqtSignal(list)

    def __init__(
        self,
        cfg: AppConfig,
        config_path: Path,
        parent: Optional[QWidget] = None,
        *,
        hotkey_failures: Optional[Dict[str, str]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__("Настройки", (SETTINGS_W, SETTINGS_H), parent)
        self._cfg = cfg
        self._path = Path(config_path)
        self._on_status = on_status
        self.setStyleSheet(self.styleSheet() + field_qss())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(0)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        body = QVBoxLayout(holder)
        body.setContentsMargins(0, 0, Space.S2, 0)
        body.setSpacing(Space.S5)

        # -- sound ------------------------------------------------------
        self._device = QComboBox()
        self._device.setFixedHeight(CONTROL_H)
        self._device.addItem("по умолчанию", "")
        if cfg.audio.device_hint:
            self._device.addItem(cfg.audio.device_hint, cfg.audio.device_hint)
            self._device.setCurrentIndex(1)
        body.addWidget(
            Section("Источник звука", [FieldRow("Устройство вывода", self._device)])
        )

        # -- recognition ------------------------------------------------
        self._engine = SegmentedControl(["Whisper", "Nemotron"])
        self._engine.set_index(1 if cfg.asr.engine == "nemotron" else 0)
        self._language = QComboBox()
        self._language.setFixedHeight(CONTROL_H)
        for value, caption in LANGUAGE_CHOICES:
            self._language.addItem(caption, value)
        self._select(self._language, self._current_language())
        body.addWidget(
            Section(
                "Распознавание",
                [FieldRow("Движок", self._engine), FieldRow("Язык", self._language)],
            )
        )

        # -- LLM provider -----------------------------------------------
        # Applies live: the engine's client is swapped on save, so a pasted key
        # works on the next button press without a restart.
        self._llm = LlmSection(cfg)
        body.addWidget(self._llm)

        # -- summary ----------------------------------------------------
        self._interval = QComboBox()
        self._interval.setFixedHeight(CONTROL_H)
        for value, caption in INTERVAL_CHOICES:
            self._interval.addItem(caption, value)
        self._select(self._interval, cfg.ui.summary_interval_min)
        self._auto = SegmentedControl(["выкл.", "вкл."])
        self._auto.set_index(1 if cfg.ui.auto_summary else 0)
        body.addWidget(
            Section(
                "Выжимка",
                [
                    FieldRow("Интервал", self._interval),
                    FieldRow("Делать автоматически", self._auto),
                ],
            )
        )

        # -- where the answer and the summary appear --------------------
        # Applies live: the overlay only has to stop using the card and start
        # using the two windows, and nothing in the pipeline knows about either.
        self._panels = QComboBox()
        self._panels.setFixedHeight(CONTROL_H)
        for value, caption in PANEL_CHOICES:
            self._panels.addItem(caption, value)
        self._select(self._panels, panel_mode(cfg.ui.panels))
        panels_row = FieldRow("Ответ и выжимка", self._panels)
        panels_row.setToolTip(
            "Карточка в ленте не выше 160 px; окна по бокам — ответ слева, "
            "выжимка справа, размером с главное окно"
        )
        body.addWidget(Section("Окна", [panels_row]))

        # -- translation ------------------------------------------------
        # The switch applies live (it is the same one Alt+T flips); the target
        # language does not — replies already translated would be left in the
        # previous language, so it is a restart like the ASR language.
        self._translate = SegmentedControl(["выкл.", "вкл."])
        self._translate.set_index(1 if cfg.translate.enabled else 0)
        self._translate_to = QComboBox()
        self._translate_to.setFixedHeight(CONTROL_H)
        for value, caption in TRANSLATE_CHOICES:
            self._translate_to.addItem(caption, value)
        self._select(self._translate_to, cfg.translate.target)
        translate_rows = [
            FieldRow("Перевод на лету", self._translate),
            FieldRow("Язык перевода", self._translate_to),
        ]
        if cfg.translate.backend == "off":
            # Not hidden: a control that does nothing has to say why (4.5).
            for row in translate_rows:
                row.control().setEnabled(False)
                row.setToolTip("Выключено в config.toml: [translate] backend = \"off\"")
        body.addWidget(Section("Перевод", translate_rows))

        # -- hotkeys ----------------------------------------------------
        failures = hotkey_failures or {}
        self._hotkeys: Dict[str, HotkeyEdit] = {}
        rows: List[QWidget] = []
        for action, caption in HOTKEY_ROWS:
            edit = HotkeyEdit(getattr(cfg.hotkeys, action, ""))
            if action in failures:
                edit.set_error(failures[action])
            self._hotkeys[action] = edit
            rows.append(FieldRow(caption, edit))
        body.addWidget(Section("Хоткеи", rows))
        body.addStretch()

        scroll.setWidget(holder)
        root = QVBoxLayout(self.body())
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(Space.S3)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        self._hint = QLabel(_RESTART_HINT)
        self._hint.setFont(qfont(Type.BUTTON_SUB))
        self._hint.setStyleSheet(label_qss(Type.BUTTON_SUB))
        self._hint.setWordWrap(True)
        footer.addWidget(self._hint, 1)
        save = TextButton("Сохранить", self, style=Type.BUTTON_LABEL)
        save.clicked.connect(self._save)
        footer.addWidget(save)
        root.addLayout(footer)

        self._devices_loaded.connect(self._fill_devices)
        self._load_devices_async()

    # -- helpers ---------------------------------------------------------

    def _current_language(self) -> str:
        if self._cfg.asr.engine == "nemotron":
            return self._cfg.asr.nemotron.language
        return self._cfg.asr.language or "auto"

    @staticmethod
    def _select(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _load_devices_async(self) -> None:
        """Enumerating endpoints spawns the capture binary — off the UI thread."""
        binary = resolve_binary(self._cfg.audio.binary)

        def work() -> None:
            try:
                devices = list_devices(binary)
            except Exception as exc:  # binary missing, WASAPI error, timeout
                logger.warning("settings: cannot list devices (%s)", exc)
                return
            names = [d.get("name", "") for d in devices.get("render", []) if d.get("name")]
            try:
                # Back to the UI thread: widgets are not touched from here.
                self._devices_loaded.emit(names)
            except RuntimeError:
                # The window was destroyed while WASAPI was being enumerated.
                pass

        threading.Thread(target=work, name="settings-devices", daemon=True).start()

    def _fill_devices(self, names: List[str]) -> None:
        current = self._device.currentData()
        self._device.blockSignals(True)
        self._device.clear()
        self._device.addItem("по умолчанию", "")
        for name in names:
            self._device.addItem(name, name)
        if current:
            index = self._device.findData(current)
            if index < 0:
                # A hint that matches no endpoint is still the user's setting.
                self._device.addItem(f"{current} (не найдено)", current)
                index = self._device.count() - 1
            self._device.setCurrentIndex(index)
        self._device.blockSignals(False)

    # -- saving ----------------------------------------------------------

    def _save(self) -> None:
        engine = "nemotron" if self._engine.index() == 1 else "whisper"
        language = self._language.currentData()
        interval = int(self._interval.currentData())
        auto = self._auto.index() == 1
        panels = str(self._panels.currentData())
        device = self._device.currentData() or ""

        translate_on = self._translate.index() == 1
        translate_to = self._translate_to.currentData()

        # The key goes to the credential store, never into config.toml — the
        # file is in git. A failure here is reported and blocks the save: a
        # config naming a provider whose key was silently dropped is worse
        # than no change at all.
        key_error = self._llm.commit_key()
        if key_error:
            self._fail(key_error)
            return
        llm = self._llm.values()

        updates: Dict[str, dict] = {
            "audio": {"device_hint": device},
            "asr": {"engine": engine},
            "llm": {"provider": llm["provider"], "model": llm["model"]},
            "llm.models": dict(llm["models"]),
            "ui": {
                "summary_interval_min": interval,
                "auto_summary": auto,
                "panels": panels,
            },
            "translate": {"enabled": translate_on, "target": translate_to},
            "hotkeys": {a: e.combo() for a, e in self._hotkeys.items()},
        }
        if engine == "nemotron":
            updates["asr.nemotron"] = {"language": language}
        else:
            updates["asr"]["language"] = language

        try:
            save_overrides(self._path, updates)
        except OSError as exc:
            self._fail(f"Не удалось записать {self._path.name}: {exc}")
            return

        # Apply what can be applied without a restart, and say so.
        self._cfg.ui.summary_interval_min = interval
        self._cfg.ui.auto_summary = auto
        self._cfg.ui.panels = panels
        self._cfg.llm.provider = str(llm["provider"])
        self._cfg.llm.model = str(llm["model"])
        self._cfg.llm.models = dict(llm["models"])
        for action, edit in self._hotkeys.items():
            setattr(self._cfg.hotkeys, action, edit.combo())
        # Computed while _cfg still holds the previous values — everything below
        # this line overwrites them.
        restart = (
            device != self._cfg.audio.device_hint
            or engine != self._cfg.asr.engine
            or language != self._current_language()
            or translate_to != self._cfg.translate.target
        )
        # These four reach the pipeline only on restart, but _cfg must still
        # mirror what was just written to config.toml. The window is a singleton
        # that recomputes `restart` against them on the next save, so leaving
        # them stale makes every later save claim a restart is needed — and
        # _current_language() reads _cfg.asr.engine, so after switching engines
        # it would compare the new dropdown against the old engine's field.
        # Language first: it is stored per engine, keyed off the new value.
        if engine == "nemotron":
            self._cfg.asr.nemotron.language = language
        else:
            self._cfg.asr.language = language
        self._cfg.audio.device_hint = device
        self._cfg.asr.engine = engine
        self._cfg.translate.target = translate_to
        self._cfg.translate.enabled = translate_on
        self.applied.emit(
            {
                "summary_interval_min": interval,
                "auto_summary": auto,
                "panels": panels,
                "translate_enabled": translate_on,
                "hotkeys": {a: e.combo() for a, e in self._hotkeys.items()},
                # Built here, where the typed key is still in hand: the overlay
                # only has to hand it to the engine.
                "llm_client": self._llm.build_client(),
                "needs_restart": restart,
            }
        )
        if self._on_status is not None:
            self._on_status("Настройки сохранены")
        self.close()

    def _fail(self, message: str) -> None:
        self._hint.setText(message)
        self._hint.setStyleSheet(
            f"QLabel {{ background: transparent; color: {Color.STATE_ERROR}; }}"
        )

    def set_hotkey_failures(self, failures: Dict[str, str]) -> None:
        for action, edit in self._hotkeys.items():
            edit.set_error(failures.get(action, ""))

    def set_translate_enabled(self, on: bool, *, available: bool = True) -> None:
        """Point the switch at the *live* state before the window is shown.

        Without this the window keeps showing `[translate] enabled` from the
        config it was built with, and since saving applies every control, any
        save would silently switch translation back on after Alt+T turned it
        off. The window is a singleton, so this has to happen on every open,
        not only on construction.

        `available=False` means the model failed to load. The switch then keeps
        the config's value and is disabled: reporting "выключен" would make the
        next save overwrite a startup preference the user never changed.
        """
        if not available:
            self._translate.set_index(1 if self._cfg.translate.enabled else 0)
            self._translate.setEnabled(False)
            self._translate.setToolTip("Модель перевода не загрузилась в этой сессии")
            return
        self._translate.setEnabled(True)
        self._translate.setToolTip("")
        self._translate.set_index(1 if on else 0)
