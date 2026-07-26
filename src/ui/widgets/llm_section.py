"""The "Модель ИИ" block of the settings window.

Its own module because it is the only part of the settings that talks to the
network: listing a provider's models and checking a key are HTTP calls, and
both run in worker threads with the result coming back over a signal. The rest
of the settings window is pure widget assembly, and mixing the two made it
unreadable.

The contract with `SettingsWindow` is three methods: `values()` for what to
write into config.toml, `commit_key()` for the secret (which never goes into
config.toml — see llm/credentials.py), and `build_client()` for the live client
handed to the running LLMEngine.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from app_config import AppConfig, LlmConfig
from llm import credentials, providers
from llm.openai_client import OpenAICompatClient, describe_error
from ui.theme.fonts import qfont
from ui.theme.styles import label_qss
from ui.theme.tokens import Color, Space, Type
from ui.widgets.buttons import TextButton
from ui.widgets.fields import CONTROL_H, FieldRow, Section

logger = logging.getLogger(__name__)

_CHECKING = "проверяю…"
_LOADING = "загружаю список моделей…"


def _hint_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setFont(qfont(Type.BUTTON_SUB))
    label.setStyleSheet(label_qss(Type.BUTTON_SUB))
    label.setWordWrap(True)
    return label


class LlmSection(QWidget):
    """Provider + key + model, with a live check."""

    # Worker threads never touch widgets; they emit these instead.
    _models_loaded = pyqtSignal(list, str)
    _checked = pyqtSignal(bool, str)

    def __init__(self, cfg: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        # Working copy of the per-provider model memory: edited as the user
        # switches providers, written out by SettingsWindow on save.
        self._models: Dict[str, str] = dict(cfg.llm.models)
        # Which provider the model field currently belongs to; see
        # _remember_current_model.
        self._active: Optional[str] = None
        self._store = credentials.store()

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(Space.S2)

        self._provider = QComboBox()
        self._provider.setFixedHeight(CONTROL_H)
        for provider in providers.PROVIDERS:
            self._provider.addItem(provider.caption, provider.id)
        self._select(self._provider, providers.resolve(cfg.llm.provider).id)

        self._key = QLineEdit()
        self._key.setFixedHeight(CONTROL_H)
        self._key.setEchoMode(QLineEdit.EchoMode.Password)

        self._model = QComboBox()
        self._model.setFixedHeight(CONTROL_H)
        self._model.setEditable(True)
        # An id we failed to list (a brand-new model, a private deployment) must
        # still be usable — the picker is a convenience, not a whitelist.
        self._model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        self._key_row = FieldRow("API-ключ", self._key)
        box.addWidget(
            Section(
                "Модель ИИ",
                [
                    FieldRow("Провайдер", self._provider),
                    self._key_row,
                    FieldRow("Модель", self._model),
                ],
            )
        )

        self._hint = _hint_label()
        box.addWidget(self._hint)

        footer = QHBoxLayout()
        footer.setSpacing(Space.S3)
        self._status = _hint_label()
        footer.addWidget(self._status, 1)
        self._refresh = TextButton("Список моделей", self, style=Type.BUTTON_LABEL)
        self._refresh.clicked.connect(self._on_refresh_models)
        footer.addWidget(self._refresh)
        self._check = TextButton("Проверить", self, style=Type.BUTTON_LABEL)
        self._check.clicked.connect(self._on_check)
        footer.addWidget(self._check)
        box.addLayout(footer)

        self._provider.currentIndexChanged.connect(self._on_provider_changed)
        self._models_loaded.connect(self._on_models_loaded)
        self._checked.connect(self._on_checked)

        self._apply_provider(initial=True)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _select(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _current(self):
        return providers.resolve(self._provider.currentData())

    def _set_status(self, text: str, color: str = Color.TEXT_SECONDARY) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"QLabel {{ background: transparent; color: {color}; }}")

    # -- provider switching ----------------------------------------------

    def _on_provider_changed(self) -> None:
        self._apply_provider(initial=False)

    def _apply_provider(self, *, initial: bool) -> None:
        """Repoint every control at the selected provider.

        Remembering the model per provider lives here rather than in the config
        loader on purpose: it is a UI convenience, and making it part of the
        load path would silently override a hand-edited `[llm] model`.
        """
        provider = self._current()
        if not initial:
            # Keep whatever was typed for the provider we are *leaving* —
            # currentIndexChanged fires after the index moved, so the combo
            # already reports the new one and cannot be asked.
            self._remember_current_model()

        self._key.clear()
        stored = self._store.get(provider)
        if provider.needs_key:
            self._key_row.setEnabled(True)
            self._key.setPlaceholderText(
                f"сохранён {credentials.mask(stored)}" if stored else "вставьте ключ"
            )
        else:
            self._key_row.setEnabled(False)
            self._key.setPlaceholderText("не нужен")

        model = self._models.get(provider.id, "")
        if initial and self._cfg.llm.model:
            model = self._cfg.llm.model
        self._model.clear()
        for candidate in (model, provider.suggested_model):
            if candidate and self._model.findText(candidate) < 0:
                self._model.addItem(candidate)
        self._model.setCurrentText(model or provider.suggested_model)
        self._model.lineEdit().setPlaceholderText(
            "берётся из сервера" if provider.autoselect_model else "выберите модель"
        )

        self._active = provider.id
        self._hint.setText(self._describe(provider, stored))
        self._set_status("")

    def _describe(self, provider, stored_key: str) -> str:
        parts: List[str] = []
        if provider.note:
            parts.append(provider.note)
        if provider.needs_key:
            from_env = bool(stored_key) and not self._store.has_stored(provider.id)
            if from_env and provider.env_var:
                parts.append(f"Ключ взят из переменной окружения {provider.env_var}.")
            else:
                parts.append(f"Ключ хранится: {self._store.backend_name()}, не в config.toml.")
            if provider.key_url:
                parts.append(f"Получить ключ: {provider.key_url}")
        if provider.id == "custom":
            base = providers.base_url_for(provider, self._cfg.llm.base_url)
            parts.append(f"base_url: {base or 'не задан в config.toml'}")
        if not provider.local:
            # Said plainly and every time the provider is cloud: until now the
            # app never sent a word of the transcript anywhere.
            parts.append("Текст разговора уходит на сервер провайдера.")
        return "\n".join(parts)

    def _remember_current_model(self) -> None:
        provider_id = self._active
        if provider_id is None:
            return
        model = self._model.currentText().strip()
        if model:
            self._models[provider_id] = model
        else:
            self._models.pop(provider_id, None)

    # -- network work ------------------------------------------------------

    def build_client(self, *, key_override: Optional[str] = None) -> OpenAICompatClient:
        """A client for what the UI currently shows, not for the saved config."""
        provider = self._current()
        cfg = replace(
            self._cfg.llm,
            provider=provider.id,
            model=self._model.currentText().strip(),
        )
        key = key_override if key_override is not None else self._typed_or_stored(provider)
        return OpenAICompatClient(cfg, provider=provider, api_key=key)

    def _typed_or_stored(self, provider) -> str:
        typed = self._key.text().strip()
        return typed or self._store.get(provider)

    def _run(self, work) -> None:
        threading.Thread(target=work, name="llm-settings", daemon=True).start()

    def _on_refresh_models(self) -> None:
        self._set_status(_LOADING)
        client = self.build_client()

        def work() -> None:
            try:
                names = client.list_models()
            except Exception as exc:  # noqa: BLE001
                self._emit_models([], describe_error(client.provider, exc))
                return
            self._emit_models(names, "")

        self._run(work)

    def _emit_models(self, names: List[str], error: str) -> None:
        try:
            self._models_loaded.emit(names, error)
        except RuntimeError:
            pass  # the window was closed while the request was in flight

    def _on_models_loaded(self, names: List[str], error: str) -> None:
        if error:
            self._set_status(f"не удалось получить список: {error}", Color.STATE_ERROR)
            return
        current = self._model.currentText().strip()
        self._model.clear()
        self._model.addItems(names)
        if current:
            self._model.setCurrentText(current)
        elif names:
            self._model.setCurrentIndex(0)
        self._set_status(f"моделей доступно: {len(names)}", Color.STATE_GOOD)

    def _on_check(self) -> None:
        self._set_status(_CHECKING)
        client = self.build_client()

        def work() -> None:
            ok, message = client.check(timeout=10.0)
            try:
                self._checked.emit(ok, message)
            except RuntimeError:
                pass

        self._run(work)

    def _on_checked(self, ok: bool, message: str) -> None:
        if ok:
            self._set_status("соединение есть", Color.STATE_GOOD)
        else:
            self._set_status(message or "недоступен", Color.STATE_ERROR)

    # -- what the settings window saves -----------------------------------

    def commit_key(self) -> str:
        """Store the typed key. Returns a message for the window, or "".

        An untouched field means "keep what is stored" — the placeholder shows
        the masked key, so there is nothing to write back.
        """
        provider = self._current()
        typed = self._key.text().strip()
        if not provider.needs_key or not typed:
            return ""
        try:
            where = self._store.set(provider.id, typed)
        except OSError as exc:
            return f"не удалось сохранить ключ: {exc}"
        self._key.clear()
        self._key.setPlaceholderText(f"сохранён {credentials.mask(typed)}")
        logger.info("llm: key for %s stored in %s", provider.id, where)
        return ""

    def values(self) -> Dict[str, object]:
        self._remember_current_model()
        provider = self._current()
        return {
            "provider": provider.id,
            "model": self._model.currentText().strip(),
            "models": dict(self._models),
        }
