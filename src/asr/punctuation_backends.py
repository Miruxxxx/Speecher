from __future__ import annotations

import logging
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

    def __init__(self, language: str = "ru") -> None:
        self._language = language
        self._model = None

    def load(self) -> None:
        from pathlib import Path

        import torch
        from torch import package as torch_package

        if self._language not in self._LANGUAGES:
            raise ValueError(
                f"silero-te does not support language {self._language!r} "
                f"(supported: {list(self._LANGUAGES)})"
            )

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

    def restore(self, text: str) -> str:
        # silero-te expects lowercase input and restores capitalization itself.
        # Feeding already-capitalized text (the worker re-punctuates overlapping
        # windows, so silero sees its own prior output) corrupts the leading
        # letter ("Мы" -> "&Ы") or raises IndexError on short inputs. Lowercasing
        # first is the model's intended usage and makes repeated passes stable.
        return self._model.enhance_text(text.lower(), self._language)


class FullstopBackend:
    """oliverguhr/fullstop-punctuation-multilang-large via
    deepmultilingualpunctuation. EN/DE/FR/IT only (no Russian), ~2.1 GB.
    Forced onto CPU: the stock PunctuationModel grabs the GPU whenever CUDA
    is available and would share VRAM with Whisper."""

    name = "deepmultilingual"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        from transformers import pipeline
        from deepmultilingualpunctuation import PunctuationModel

        # Skip PunctuationModel.__init__ (it hardcodes device=0 on CUDA);
        # its only state is `pipe`, which we build ourselves on CPU.
        model = PunctuationModel.__new__(PunctuationModel)
        model.pipe = pipeline(
            "ner",
            "oliverguhr/fullstop-punctuation-multilang-large",
            aggregation_strategy=None,
            device="cpu",
        )
        self._model = model

    def restore(self, text: str) -> str:
        return self._model.restore_punctuation(text)


def create_backend(backend: str, language: str) -> Optional[PunctuationBackend]:
    """Build the configured backend; None means punctuation is off."""
    kind = (backend or "off").strip().lower()
    if kind == "off":
        return None
    if kind == "silero":
        return SileroBackend(language=language)
    if kind == "deepmultilingual":
        return FullstopBackend()
    logger.warning("unknown punctuation backend %r; punctuation disabled", backend)
    return None
