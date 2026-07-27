"""Regressions for the settings window's relationship with live state.

The window is a singleton: built on the first open from the config, then shown
again and again while the session changes underneath it. Every control it owns
is applied on save, so any control that drifts out of sync silently *undoes*
whatever the user did elsewhere. Translation is the one that can drift — Alt+T
and the session menu flip it without touching the config.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app_config import AppConfig, load_config  # noqa: E402
from llm import credentials  # noqa: E402
from llm.credentials import CredentialStore  # noqa: E402
from ui.theme.fonts import load_app_fonts  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    load_app_fonts()
    yield app


@pytest.fixture
def window(qapp, tmp_path: Path, monkeypatch):
    from ui.windows import settings as settings_module

    monkeypatch.setattr(settings_module, "list_devices", lambda _b: {"render": []})
    monkeypatch.setattr(
        credentials, "store", lambda: CredentialStore(path=tmp_path / "c.json", use_keyring=False)
    )
    config = tmp_path / "config.toml"
    # The state that produced the bug: translation on at startup.
    config.write_text('[translate]\nenabled = true\n', encoding="utf-8")
    cfg = AppConfig()
    cfg.translate.enabled = True
    w = settings_module.SettingsWindow(cfg, config)
    w._config = config  # test handle
    yield w
    w.deleteLater()


def _translate_on(window) -> bool:
    return window._translate.index() == 1


def test_switch_follows_the_live_state_not_the_config(window):
    """Alt+T turned translation off; opening the window must show that."""
    assert _translate_on(window)  # built from the config
    window.set_translate_enabled(False)
    assert not _translate_on(window)
    window.set_translate_enabled(True)
    assert _translate_on(window)


def test_saving_does_not_switch_translation_back_on(window):
    """The bug: translation was off, the user changed the summary interval, and
    the save re-enabled translation because the switch still showed the config's
    startup value."""
    window.set_translate_enabled(False)
    seen = []
    window.applied.connect(seen.append)
    window._save()

    assert seen and seen[0]["translate_enabled"] is False
    # …and the startup value follows what the user was actually looking at.
    assert load_config(window._config).translate.enabled is False


def test_a_failed_translation_model_cannot_clobber_the_startup_setting(window):
    """`available=False` means the model did not load this session. Reporting
    "выключен" would overwrite a preference the user never changed."""
    window.set_translate_enabled(False, available=False)
    assert _translate_on(window)  # kept the config's value
    assert not window._translate.isEnabled()

    seen = []
    window.applied.connect(seen.append)
    window._save()
    assert load_config(window._config).translate.enabled is True


def test_a_second_save_does_not_invent_a_restart(window):
    """`needs_restart` is computed against `_cfg`, so `_cfg` has to follow the
    file it just wrote.

    The window is a singleton reused for the whole session. When a save left
    `audio.device_hint` / `asr.engine` / `translate.target` at their old values,
    every later save compared the unchanged controls against stale config and
    kept announcing "часть настроек применится после перезапуска".
    """
    seen = []
    window.applied.connect(seen.append)

    window._translate_to.setCurrentIndex(window._translate_to.findData("de"))
    window._save()
    assert seen[-1]["needs_restart"] is True        # the target really changed
    assert window._cfg.translate.target == "de"     # …and _cfg followed it

    window._save()                                  # nothing touched since
    assert seen[-1]["needs_restart"] is False


def test_the_language_field_follows_the_engine_after_a_save(window):
    """The ASR language is stored per engine. `_current_language()` reads
    `_cfg.asr.engine`, so a stale engine made it read the wrong field on the
    next open — and compare the nemotron dropdown against whisper's language."""
    window._engine.set_index(1)                     # Whisper -> Nemotron
    window._language.setCurrentIndex(window._language.findData("ru"))
    window._save()

    assert window._cfg.asr.engine == "nemotron"
    assert window._cfg.asr.nemotron.language == "ru"
    assert window._current_language() == "ru"
