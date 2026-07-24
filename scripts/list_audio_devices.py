"""Enumerate audio endpoints and show which one `audio.device_hint` picks.

Asks the capture binary (`audio_capture --list-devices`), i.e. WASAPI itself,
so the names here are exactly the ones the app matches against.

Usage (repo root, project venv):

    .venv\\Scripts\\python scripts\\list_audio_devices.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app_config import load_config  # noqa: E402
from audio.rust_capture import BUILD_HINT, resolve_binary  # noqa: E402


def print_devices(binary: Path, hint: str) -> None:
    out = subprocess.run(
        [str(binary), "--list-devices"], capture_output=True, timeout=30
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace").strip())
    devices = json.loads(out.stdout.decode("utf-8"))

    hint_l = (hint or "").lower().strip()
    for key, title in (
        ("render", 'ВЫВОД — источник для source = "loopback"'),
        ("capture", 'ВХОД — источник для source = "mic"'),
    ):
        print(f"\n{title}:")
        for dev in devices.get(key, []):
            marks = []
            if dev.get("is_default"):
                marks.append("default")
            if hint_l and hint_l in (dev.get("name") or "").lower():
                marks.append(f"совпадает с device_hint='{hint}'")
            suffix = f"   <- {', '.join(marks)}" if marks else ""
            print(
                f"  {dev.get('name')}  |  {dev.get('sample_rate')} Hz, "
                f"{dev.get('channels')}ch {dev.get('format')}{suffix}"
            )


def main() -> int:
    cfg = load_config(ROOT / "config" / "config.toml")
    binary = resolve_binary(cfg.audio.binary)

    if not binary.is_file():
        print(f"бинарь захвата не собран: {binary}\nсобрать: {BUILD_HINT}")
        return 1

    print(f"WASAPI напрямую ({binary.name}):")
    print_devices(binary, cfg.audio.device_hint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
