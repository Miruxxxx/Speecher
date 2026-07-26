"""Translation backends. Same shape as asr/punctuation_backends.py: a protocol,
concrete classes that import their heavy dependencies inside `load()`, and a
factory whose unknown/failed cases turn the feature off instead of raising.

Two of them, because the choice is a real trade-off:

* `nllb` (default) — one NLLB-200-distilled-600M for every direction. The target
  language is a decoder token, so the settings dropdown can offer twelve
  languages without twelve downloads. ~2.5 GB on disk, ~1.3 GB VRAM in fp16.
* `opusmt` — Helsinki-NLP Marian, one checkpoint per *pair* (~300 MB, fast even
  on CPU). It cannot honour a freely chosen target: the pair is fixed at load
  time from `source`/`target`, and "auto" source is not available to it.

Both translate sentence by sentence: these models are sentence-level MT, and a
whole 180-character reply fed in one piece comes back truncated.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Protocol

import heavy_imports
from translate import languages

logger = logging.getLogger(__name__)

KNOWN_BACKENDS = ("nllb", "opusmt", "off")

NLLB_DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# Sentence end followed by a space. Kept deliberately dumb — the input is speech
# with model-restored punctuation, not prose with abbreviations.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


class TranslationBackend(Protocol):
    """Stateless translator: same input, same output, no memory between calls."""

    name: str

    def load(self) -> None: ...

    def translate(self, text: str, source: str, target: str) -> str: ...


def split_sentences(text: str, max_chars: int) -> List[str]:
    """Sentences, then a hard wrap at word boundaries for anything still too long.

    Nothing is dropped and nothing is merged across the cut: the pieces are
    joined back with a space, so a wrong cut costs fluency, never content.
    """
    pieces: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


class NllbBackend:
    """NLLB-200 distilled 600M through transformers.

    The tokenizer carries the *source* language (`src_lang`) and generation is
    forced to start with the *target* language token — that pair is the whole
    reason this model can serve a dropdown instead of a fixed direction.
    """

    name = "nllb"

    def __init__(
        self,
        model: str = "",
        device: str = "cuda",
        dtype: str = "float16",
        max_chars_per_chunk: int = 220,
        max_new_tokens: int = 256,
    ) -> None:
        self._model_id = model or NLLB_DEFAULT_MODEL
        self._device = device
        self._dtype = dtype
        self._max_chars = int(max_chars_per_chunk)
        self._max_new_tokens = int(max_new_tokens)
        self._tokenizer = None
        self._model = None
        self._torch = None

    def load(self) -> None:
        # Guarded: the asr-loader thread reaches into the same lazy transformers
        # submodule at the same moment — see src/heavy_imports.py.
        with heavy_imports.transformers():
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self._device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("translate: CUDA unavailable, loading %s on CPU", self._model_id)
            device = "cpu"
        # fp16 on CPU is not a memory saving, it is a slowdown: there are no
        # half-precision kernels for most CPU ops and torch converts back and
        # forth around every one of them.
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self._dtype, torch.float16)
        if device == "cpu":
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self._model_id, dtype=dtype)
        self._model = model.to(device).eval()
        self._torch = torch
        self._device = device
        logger.info(
            "translate: %s loaded (device=%s, dtype=%s)", self._model_id, device, dtype
        )

    def translate(self, text: str, source: str, target: str) -> str:
        src = languages.get(source)
        tgt = languages.get(target)
        if src is None or tgt is None:
            raise ValueError(f"unsupported direction {source!r} -> {target!r}")
        chunks = split_sentences(text, self._max_chars)
        if not chunks:
            return ""

        self._tokenizer.src_lang = src.flores
        target_id = self._tokenizer.convert_tokens_to_ids(tgt.flores)
        batch = self._tokenizer(chunks, return_tensors="pt", padding=True).to(self._device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **batch,
                forced_bos_token_id=target_id,
                # Greedy: beams would multiply the cost of a call that happens
                # once per reply while the ASR model shares the same GPU.
                num_beams=1,
                do_sample=False,
                max_new_tokens=self._max_new_tokens,
            )
        parts = self._tokenizer.batch_decode(output, skip_special_tokens=True)
        return " ".join(p.strip() for p in parts if p.strip())


class OpusMtBackend:
    """Helsinki-NLP Marian, one checkpoint per direction.

    The pair is resolved at `load()`, so `source` must be a real language —
    "auto" is rejected there rather than failing on the first reply.
    """

    name = "opusmt"

    def __init__(
        self,
        source: str,
        target: str,
        model: str = "",
        device: str = "cpu",
        max_chars_per_chunk: int = 220,
        max_new_tokens: int = 256,
    ) -> None:
        self._source = source
        self._target = target
        self._model_id = model
        self._device = device
        self._max_chars = int(max_chars_per_chunk)
        self._max_new_tokens = int(max_new_tokens)
        self._tokenizer = None
        self._model = None
        self._torch = None

    def load(self) -> None:
        import torch

        try:
            import sentencepiece  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "бэкенду opusmt нужны sentencepiece и sacremoses: "
                ".venv\\Scripts\\python -m pip install sentencepiece sacremoses"
            ) from exc
        with heavy_imports.transformers():
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        src = languages.get(self._source)
        tgt = languages.get(self._target)
        if src is None or tgt is None:
            raise ValueError(
                f"opusmt needs a fixed pair; got {self._source!r} -> {self._target!r}"
            )
        model_id = self._model or f"Helsinki-NLP/opus-mt-{src.code}-{tgt.code}"
        device = self._device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
        self._torch = torch
        self._device = device
        logger.info("translate: %s loaded (device=%s)", model_id, device)

    def translate(self, text: str, source: str, target: str) -> str:
        # The pair is baked into the checkpoint; the arguments are accepted for
        # protocol compatibility and checked, not used.
        if languages.normalize_code(target) != languages.normalize_code(self._target):
            raise ValueError(f"opusmt is pinned to {self._target!r}, not {target!r}")
        chunks = split_sentences(text, self._max_chars)
        if not chunks:
            return ""
        batch = self._tokenizer(chunks, return_tensors="pt", padding=True).to(self._device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **batch, num_beams=1, do_sample=False, max_new_tokens=self._max_new_tokens
            )
        parts = self._tokenizer.batch_decode(output, skip_special_tokens=True)
        return " ".join(p.strip() for p in parts if p.strip())


def create_backend(cfg) -> Optional[TranslationBackend]:
    """Build the configured backend; None means live translation is off."""
    kind = (cfg.backend or "off").strip().lower()
    if kind == "off":
        return None
    if kind == "nllb":
        return NllbBackend(
            model=cfg.model,
            device=cfg.device,
            dtype=cfg.dtype,
            max_chars_per_chunk=cfg.max_chars_per_chunk,
            max_new_tokens=cfg.max_new_tokens,
        )
    if kind == "opusmt":
        return OpusMtBackend(
            source=cfg.source,
            target=cfg.target,
            model=cfg.model,
            device=cfg.device,
            max_chars_per_chunk=cfg.max_chars_per_chunk,
            max_new_tokens=cfg.max_new_tokens,
        )
    logger.warning("unknown translation backend %r; translation disabled", cfg.backend)
    return None
