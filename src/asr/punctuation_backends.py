from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class PunctuationBackend(Protocol):
    """A backend restores punctuation in plain text without changing the
    number of whitespace-separated words (the worker enforces this)."""

    name: str

    def load(self) -> None: ...
    def restore(self, text: str) -> str: ...


class SileroBackend:
    """silero-te text-enhancement: punctuation + capitalization, CPU-only,
    ~50 MB, supports ru/en/de/es.

    Loaded directly via torch.package instead of torch.hub.load: the
    silero-models hubconf does `from src.silero import ...`, which collides
    with this project's own `src` package (already in sys.modules when the
    app runs as `python -m src`). The hub path is only a downloader around
    the same .pt package file, so we replicate those few lines."""

    name = "silero"

    # From silero-models models.yml (te_models.latest); languages are fixed
    # for the v2 package.
    _MODEL_URL = "https://models.silero.ai/te_models/v2_4lang_q.pt"
    _LANGUAGES = ("en", "de", "ru", "es")

    # Script -> the silero language to punctuate it with. Latin covers en/de/es
    # and the script cannot tell them apart, so "auto" picks English and a user
    # who speaks German pins it by hand. Guessing wrong within Latin costs some
    # commas; guessing across scripts (Russian model on English) corrupts words,
    # which is what a fixed `language` used to do whenever the speech changed.
    _SCRIPT_LANGUAGE = {"cyrillic": "ru", "latin": "en"}

    def __init__(self, language: str = "auto") -> None:
        self._language = language
        self._model = None
        # Captured by load(); stays None when a test injects a fake model, which
        # is what keeps `restore` runnable on an image with no torch at all.
        self._torch = None

    def load(self) -> None:
        # Validated before torch is touched: a bad language is a config typo and
        # should not cost a multi-second import to report — and it keeps this
        # path testable on the CI image, which deliberately has no torch.
        if self._language != "auto" and self._language not in self._LANGUAGES:
            raise ValueError(
                f"silero-te does not support language {self._language!r} "
                f"(supported: {list(self._LANGUAGES)}, or 'auto')"
            )

        from pathlib import Path

        import torch
        from torch import package as torch_package

        file_name = self._MODEL_URL.rsplit("/", 1)[-1]
        hub_dir = Path(torch.hub.get_dir())
        candidates = [
            # Where torch.hub.load('snakers4/silero-models') would have put it.
            hub_dir / "snakers4_silero-models_master" / "src" / "silero" / "model" / file_name,
            hub_dir / "checkpoints" / file_name,
        ]
        model_path = next((p for p in candidates if p.is_file()), None)
        if model_path is None:
            model_path = candidates[-1]
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(self._MODEL_URL, str(model_path), progress=False)

        importer = torch_package.PackageImporter(str(model_path))
        self._model = importer.load_pickle("te_model", "model")
        self._torch = torch

    def language_for(self, text: str) -> str:
        """Which of silero's four languages to punctuate this window with."""
        if self._language != "auto":
            return self._language
        # Imported here, not at module scope: this file must stay importable
        # without the translate package for the punctuation-only tests.
        from translate.languages import detect_script

        return self._SCRIPT_LANGUAGE.get(detect_script(text), "en")

    def restore(self, text: str) -> str:
        # silero-te expects lowercase input and restores capitalization itself.
        # Feeding already-capitalized text (the worker re-punctuates overlapping
        # windows, so silero sees its own prior output) corrupts the leading
        # letter ("Мы" -> "&Ы") or raises IndexError on short inputs. Lowercasing
        # first is the model's intended usage and makes repeated passes stable.
        with self._one_thread():
            return self._model.enhance_text(text.lower(), self.language_for(text))

    @contextmanager
    def _one_thread(self):
        """Run the model on a single core, then give the setting back.

        Measured on an 80-word window: at torch's default (one thread per core,
        20 here) a call costs ~800 ms of CPU for ~47 ms of wall time — the pool
        burns sixteen times the work for no speedup, because the model is far
        too small to fill it. Pinned to one thread the same call costs ~72 ms of
        CPU and finishes in ~50 ms. Nothing waits on this: it is a background
        rewrite every `interval_sec`, so wall time is not the currency.

        `set_num_threads` is process-global, hence the restore — the translation
        backend picks its own count and must not inherit ours.
        """
        torch = self._torch
        if torch is None:      # a fake model in a test; there is nothing to pin
            yield
            return
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            yield
        finally:
            torch.set_num_threads(previous)


# The "deepmultilingual" backend (fullstop-punctuation-multilang-large via
# deepmultilingualpunctuation) was removed in phase 3 of the Nemotron
# migration: it has no Russian, cost ~2.1 GB, and its package pulls
# transformers below the >=5.13 the nemotron engine needs. Configs still
# naming it fall through to the unknown-backend path (punctuation off).


def create_backend(backend: str, language: str = "auto") -> Optional[PunctuationBackend]:
    """Build the configured backend; None means punctuation is off."""
    kind = (backend or "off").strip().lower()
    if kind == "off":
        return None
    if kind == "silero":
        return SileroBackend(language=language)
    logger.warning("unknown punctuation backend %r; punctuation disabled", backend)
    return None
