"""The settings section that picks a provider.

Headless Qt, no network: `build_client` is never called against a real server
here, only inspected. What these tests actually protect is the boundary — the
key must reach the credential store and must never reach the values written
into config.toml.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app_config import AppConfig  # noqa: E402
from llm import credentials, providers  # noqa: E402
from llm.credentials import CredentialStore  # noqa: E402
from ui.theme.fonts import load_app_fonts  # noqa: E402
from ui.widgets.llm_section import LlmSection  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    load_app_fonts()
    yield app


@pytest.fixture
def section(qapp, tmp_path: Path, monkeypatch):
    store = CredentialStore(path=tmp_path / "credentials.json", use_keyring=False)
    monkeypatch.setattr(credentials, "store", lambda: store)
    cfg = AppConfig()
    widget = LlmSection(cfg)
    widget._test_store = store
    yield widget
    widget.deleteLater()


def _select(widget: LlmSection, provider_id: str) -> None:
    index = widget._provider.findData(provider_id)
    assert index >= 0
    widget._provider.setCurrentIndex(index)


def test_every_provider_can_be_selected(section: LlmSection):
    """A registry row that blows up the settings window is a broken provider."""
    for pid in providers.ids():
        _select(section, pid)
        assert section._current().id == pid


def test_key_field_is_offered_only_where_a_key_is_needed(section: LlmSection):
    _select(section, "lmstudio")
    assert not section._key_row.isEnabled()
    _select(section, "openrouter")
    assert section._key_row.isEnabled()


def test_the_key_never_reaches_the_config_values(section: LlmSection):
    _select(section, "openrouter")
    section._key.setText("sk-or-supersecret")
    section._model.setCurrentText("some/model")
    assert section.commit_key() == ""

    values = section.values()
    assert "sk-or-supersecret" not in repr(values)
    assert values == {
        "provider": "openrouter",
        "model": "some/model",
        "models": {"openrouter": "some/model"},
    }
    # …and it did land in the store.
    assert section._test_store.get(providers.get("openrouter")) == "sk-or-supersecret"


def test_an_untouched_key_field_keeps_the_stored_key(section: LlmSection):
    section._test_store.set("groq", "gsk-existing")
    _select(section, "groq")
    assert section._key.text() == ""
    assert "••••" in section._key.placeholderText()
    assert section.commit_key() == ""
    assert section._test_store.get(providers.get("groq")) == "gsk-existing"


def test_model_is_remembered_per_provider(section: LlmSection):
    _select(section, "groq")
    section._model.setCurrentText("llama-3.3")
    _select(section, "deepseek")
    section._model.setCurrentText("deepseek-reasoner")
    _select(section, "groq")
    assert section._model.currentText() == "llama-3.3"

    values = section.values()
    assert values["models"]["deepseek"] == "deepseek-reasoner"
    assert values["models"]["groq"] == "llama-3.3"


def test_suggested_model_prefills_where_the_id_is_stable(section: LlmSection):
    _select(section, "anthropic")
    assert section._model.currentText() == providers.get("anthropic").suggested_model


def test_client_is_built_from_the_widgets_not_the_saved_config(section: LlmSection):
    _select(section, "deepseek")
    section._model.setCurrentText("deepseek-chat")
    section._key.setText("sk-typed")
    client = section.build_client()
    assert client.provider.id == "deepseek"
    assert client.resolve_model() == "deepseek-chat"
    # The typed key is used before it is even committed, so "Проверить" checks
    # what the user just pasted rather than what was stored last time.
    assert client.has_key()


def test_cloud_provider_warns_that_the_transcript_leaves_the_machine(section: LlmSection):
    _select(section, "openai")
    assert "уходит" in section._hint.text()
    _select(section, "lmstudio")
    assert "уходит" not in section._hint.text()


# ----------------------------------------------------------------------
# The whole window: what a save actually writes
# ----------------------------------------------------------------------


def test_saving_writes_the_provider_and_keeps_the_key_out_of_the_file(
    qapp, tmp_path: Path, monkeypatch
):
    from app_config import load_config
    from ui.windows import settings as settings_module

    # Enumerating endpoints spawns the capture binary; not in a unit test.
    monkeypatch.setattr(settings_module, "list_devices", lambda _b: {"render": []})
    store = CredentialStore(path=tmp_path / "credentials.json", use_keyring=False)
    monkeypatch.setattr(credentials, "store", lambda: store)

    config = tmp_path / "config.toml"
    config.write_text(
        '[llm]\nprovider = "lmstudio"\nmodel = ""\n\n[llm.models]\n', encoding="utf-8"
    )
    cfg = AppConfig()
    window = settings_module.SettingsWindow(cfg, config)
    try:
        index = window._llm._provider.findData("openrouter")
        window._llm._provider.setCurrentIndex(index)
        window._llm._model.setCurrentText("some/model")
        window._llm._key.setText("sk-or-secret")

        seen = []
        window.applied.connect(seen.append)
        window._save()

        text = config.read_text(encoding="utf-8")
        assert "sk-or-secret" not in text
        saved = load_config(config)
        assert saved.llm.provider == "openrouter"
        assert saved.llm.model == "some/model"
        assert store.get(providers.get("openrouter")) == "sk-or-secret"

        # The overlay needs a ready-made client so the change applies without
        # a restart.
        assert seen and seen[0]["llm_client"].provider.id == "openrouter"
    finally:
        window.deleteLater()
