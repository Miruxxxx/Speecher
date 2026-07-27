"""Deciding whether a model actually fits on the GPU before loading it there.

Until this existed the only question anyone asked was `torch.cuda.is_available()`
— "is there a CUDA device", never "is there any room left on it". That is the
wrong question in this app, because it loads *two* models onto one card: the ASR
engine holds its weights for the whole session, and live translation then asks
for a gigabyte more on top. On a 12 GB card both fit and nobody notices. On a
6 GB one the second load raises `torch.cuda.OutOfMemoryError`, which the
translation worker catches and turns into "feature off" — a silent degradation
whose cause is one line in a log file.

So: ask the driver how much is free, compare it against what the caller says it
needs, and when it does not fit say so in words the user can act on and load on
the CPU instead. Slower is a state you can work in; off with no explanation is
not.

Everything torch is imported inside the functions — see the module-level import
rule in CLAUDE.md.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Headroom over the caller's estimate. Weights are only part of the cost: the
# allocator rounds up, activations and the KV/beam buffers land on top, and a
# load that *just* fits leaves the ASR engine nothing for its own peaks.
SAFETY_MARGIN = 1.25


def free_mib() -> float:
    """Free VRAM on the current device, or -1.0 when there is no CUDA at all."""
    try:
        import torch

        if not torch.cuda.is_available():
            return -1.0
        free, _total = torch.cuda.mem_get_info()
        return free / 2**20
    except Exception:  # noqa: BLE001 - a probe that fails means "we don't know"
        logger.exception("vram: could not read free memory")
        return -1.0


def choose_device(requested: str, needs_mib: float, *, what: str) -> Tuple[str, str]:
    """Pick the device to load on. Returns `(device, reason)`.

    `reason` is empty when the requested device was kept, and a sentence in
    Russian explaining the downgrade when it was not — the caller shows it.

    `requested` may be "auto", which means "cuda if it fits". An explicit
    "cuda" is still checked: a user who wrote `device = "cuda"` wants the GPU,
    but they want a working program more, and an OOM here disables the whole
    feature.
    """
    requested = (requested or "auto").strip().lower()
    if requested == "cpu":
        return "cpu", ""

    free = free_mib()
    if free < 0:
        return "cpu", f"{what}: CUDA недоступна, работаю на CPU"

    required = needs_mib * SAFETY_MARGIN
    if free < required:
        return "cpu", (
            f"{what}: на видеокарте свободно {free:.0f} МБ, "
            f"нужно ~{required:.0f} МБ — работаю на CPU (медленнее)"
        )
    return "cuda", ""
