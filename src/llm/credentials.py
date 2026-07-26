"""Where the API keys live — anywhere except config.toml.

`config/config.toml` is checked into the repository and full of the user's own
comments; a key written there is a key published on the next `git add -A`. So
the config holds the *provider* and the *model*, and the secret is kept out of
the tree entirely:

1. Windows Credential Manager via `keyring` — encrypted by the OS, nothing on
   disk we can leak.
2. `%APPDATA%/Speecher/credentials.json` — the fallback for when keyring is not
   installed or its backend refuses to load (a headless session, a stripped
   Python). Plain text, but outside the repository.
3. The provider's usual environment variable (`OPENAI_API_KEY`, …), read only
   when nothing is stored — so an existing shell keeps working untouched.

Reads never raise: a broken store means "no key", which surfaces in the UI as a
provider that is not configured, not as a crash on startup.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from llm.providers import Provider

logger = logging.getLogger(__name__)

SERVICE = "Speecher"

KEYRING = "Диспетчер учётных данных Windows"
FILE = "файл в %APPDATA%"
NONE = "не сохраняется"


def _default_dir() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "Speecher"
    return Path.home() / ".speecher"


class CredentialStore:
    """Keyring first, a JSON file second, the environment last.

    Instantiated with `use_keyring=False` and an explicit path by the tests —
    the point of the class is that the fallback path is exercisable without a
    Credential Manager.
    """

    def __init__(self, path: Optional[Path] = None, use_keyring: bool = True) -> None:
        self._path = Path(path) if path is not None else _default_dir() / "credentials.json"
        self._keyring = self._load_keyring() if use_keyring else None

    # -- backends ------------------------------------------------------

    @staticmethod
    def _load_keyring():
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError:
            logger.info("credentials: keyring not installed, using the file store")
            return None
        try:
            # get_keyring() itself can raise when no usable backend exists; a
            # fail backend is worse than useless, it swallows writes silently.
            backend = keyring.get_keyring()
            if backend.__class__.__name__.lower().startswith("fail"):
                raise KeyringError("no usable backend")
        except Exception as exc:  # noqa: BLE001 - any backend failure is the same to us
            logger.info("credentials: keyring unusable (%s), using the file store", exc)
            return None
        return keyring

    def backend_name(self) -> str:
        return KEYRING if self._keyring is not None else FILE

    # -- file store ----------------------------------------------------

    def _read_file(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("credentials: cannot read %s (%s)", self._path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _write_file(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    # -- API -----------------------------------------------------------

    def get(self, provider: Provider) -> str:
        """The stored key, or the provider's environment variable, or ""."""
        stored = self._get_stored(provider.id)
        if stored:
            return stored
        if provider.env_var:
            return (os.environ.get(provider.env_var) or "").strip()
        return ""

    def _get_stored(self, provider_id: str) -> str:
        if self._keyring is not None:
            try:
                return (self._keyring.get_password(SERVICE, provider_id) or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("credentials: keyring read failed (%s)", exc)
        return str(self._read_file().get(provider_id, "")).strip()

    def has_stored(self, provider_id: str) -> bool:
        return bool(self._get_stored(provider_id))

    def set(self, provider_id: str, key: str) -> str:
        """Store a key; returns the backend that took it. Empty key deletes.

        Raises OSError only if the file fallback itself cannot be written — the
        caller shows that in the settings window.
        """
        key = (key or "").strip()
        if not key:
            self.delete(provider_id)
            return NONE
        if self._keyring is not None:
            try:
                self._keyring.set_password(SERVICE, provider_id, key)
                # A key that used to live in the file would otherwise shadow
                # nothing but still sit there in plain text.
                self._delete_file_entry(provider_id)
                return KEYRING
            except Exception as exc:  # noqa: BLE001
                logger.warning("credentials: keyring write failed (%s), using the file", exc)
        data = self._read_file()
        data[provider_id] = key
        self._write_file(data)
        return FILE

    def delete(self, provider_id: str) -> None:
        if self._keyring is not None:
            try:
                self._keyring.delete_password(SERVICE, provider_id)
            except Exception:  # noqa: BLE001 - "no such entry" is the common case
                pass
        self._delete_file_entry(provider_id)

    def _delete_file_entry(self, provider_id: str) -> None:
        data = self._read_file()
        if provider_id in data:
            data.pop(provider_id)
            try:
                self._write_file(data)
            except OSError as exc:
                logger.warning("credentials: cannot rewrite %s (%s)", self._path, exc)


_default: Optional[CredentialStore] = None


def store() -> CredentialStore:
    """The process-wide store, built on first use."""
    global _default
    if _default is None:
        _default = CredentialStore()
    return _default


def mask(key: str) -> str:
    """What the settings window shows instead of the key itself."""
    key = (key or "").strip()
    if not key:
        return ""
    return "••••" + key[-4:] if len(key) > 4 else "••••"
