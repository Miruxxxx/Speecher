"""Load the translation backend and translate a few phrases, with timings.

The first run downloads the model (~2.5 GB for NLLB-200-distilled-600M) into the
HuggingFace cache — which is the whole point of having this as a probe: the app
does the same download in the background on the first reply, and here you can
watch it happen and see what a translation costs on this machine.

    .venv\\Scripts\\python scripts\\translate_probe.py
    .venv\\Scripts\\python scripts\\translate_probe.py --target de --device cpu
    .venv\\Scripts\\python scripts\\translate_probe.py --text "Your own sentence."

Reads config/config.toml for the defaults, so what it measures is what the app
would run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from app_config import load_config  # noqa: E402
from translate import languages  # noqa: E402
from translate.backends import create_backend  # noqa: E402

SAMPLES = [
    "Let's take a look at the examples and collect some references.",
    "The user should forget the window is open and simply get value out of it.",
    "So what do you think we should do about the latency on the first word?",
]


def main() -> int:
    cfg = load_config(ROOT / "config" / "config.toml").translate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=cfg.backend or "nllb")
    parser.add_argument("--target", default=cfg.target)
    parser.add_argument("--source", default=cfg.source)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--dtype", default=cfg.dtype)
    parser.add_argument("--model", default=cfg.model)
    parser.add_argument("--text", action="append", help="phrase to translate (repeatable)")
    args = parser.parse_args()

    cfg.backend = args.backend
    cfg.target = args.target
    cfg.source = args.source
    cfg.device = args.device
    cfg.dtype = args.dtype
    cfg.model = args.model

    backend = create_backend(cfg)
    if backend is None:
        print(f"backend={cfg.backend!r} — перевод выключен, нечего проверять")
        return 1

    source = languages.normalize_code(cfg.source) or "en"
    target = languages.normalize_code(cfg.target) or "ru"
    print(f"backend={backend.name} {source} -> {target} device={cfg.device} dtype={cfg.dtype}")

    started = time.perf_counter()
    backend.load()
    print(f"загрузка: {time.perf_counter() - started:.1f} с\n")

    for text in args.text or SAMPLES:
        started = time.perf_counter()
        out = backend.translate(text, source, target)
        print(f"[{time.perf_counter() - started:5.2f} с] {text}\n            {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
