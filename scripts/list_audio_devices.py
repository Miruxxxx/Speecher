"""Enumerate audio endpoints and show which one `audio.device_hint` picks.

Prefers the native binary (native/audio_capture --list-devices): it asks WASAPI
directly, so the names are exactly the ones the rust backend matches against.
Without a built binary it falls back to PortAudio's view of loopback devices,
which is what the pyaudio backend matches instead -- the two lists differ, and
that is the point of printing which one you are looking at.

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
from audio.capture_backends import resolve_rust_binary  # noqa: E402


def print_native(binary: Path, hint: str) -> None:
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


def print_portaudio(hint: str) -> None:
    import pyaudiowpatch as pyaudio

    from audio.devices import list_loopbacks, pick_loopback_device

    pa = pyaudio.PyAudio()
    try:
        loopbacks = list_loopbacks(pa)
        print("\nLOOPBACK-устройства PortAudio:")
        for idx, name, channels, rate in loopbacks:
            print(f"  [{idx}] {name}  |  {rate} Hz, {channels}ch")
        if loopbacks:
            idx, name, _, _ = pick_loopback_device(pa, hint)
            print(f"\ndevice_hint='{hint}' выбирает: [{idx}] {name}")
    finally:
        pa.terminate()


def main() -> int:
    cfg = load_config(ROOT / "config" / "config.toml")
    hint = cfg.audio.device_hint
    binary = resolve_rust_binary(cfg.audio.rust_binary)

    if binary.is_file():
        print(f"WASAPI напрямую ({binary.name}), backend='{cfg.audio.backend}':")
        try:
            print_native(binary, hint)
            return 0
        except Exception as exc:  # noqa: BLE001 - a probe script, show and fall back
            print(f"нативный список не получился ({exc!r}); падаю на PortAudio")
    else:
        print(
            f"нативный бинарь не собран ({binary}), показываю список PortAudio.\n"
            "Собрать: cargo build --release --manifest-path native/audio_capture/Cargo.toml"
        )

    print_portaudio(hint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
