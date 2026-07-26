"""Serialising the first `from transformers import …` across worker threads.

`transformers` is a lazy package: its `__init__` exposes names through
`_LazyModule.__getattr__`, which imports the real submodule on first access.
Two threads reaching for two names that live in the *same* submodule at the
same time is a race — the loser observes `transformers.models.auto.modeling_auto`
while it is still executing, the attribute it wants is not defined yet, and
Python reports the miss as

    ImportError: cannot import name 'AutoModelForRNNT' from 'transformers'

which reads like a version or install problem and is neither.

This is not theoretical: it is what the app did on every start once live
translation was on. `main.py` starts the asr-loader thread and the translation
thread back to back; the ASR side asks for `AutoModelForRNNT` while the
translation side is importing `AutoModelForSeq2SeqLM` from the same module. The
ASR side lost the race deterministically (its symbol is defined later in the
file), nemotron "failed to load", and the app fell back to whisper with a status
message that scrolls away in a few seconds. The user ran on the wrong engine for
days without a single error in sight.

So the import statement — only the statement, not the model loading that follows
— is taken under one process-wide lock. Whoever gets there first pays the ~3 s
and everyone else waits for a fully initialised module.

Every `from transformers import …` under `src/` must be inside this guard;
`tests/test_heavy_imports.py` walks the AST and fails on one that is not, because
a fourth call site added without it brings the whole thing back.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

# Re-entrant: a guarded import inside another guarded import (a backend that
# grows a second import site) must not deadlock.
_LOCK = threading.RLock()


@contextmanager
def transformers() -> Iterator[None]:
    """Hold the lock across a `from transformers import …` statement.

        with heavy_imports.transformers():
            import torch
            from transformers import AutoModelForRNNT, AutoProcessor

    Keep the block to the imports. `from_pretrained` downloads weights and can
    take minutes; holding this lock across it would serialise model loading for
    no reason — the race is in the module initialisation, not in the weights.
    """
    with _LOCK:
        yield
