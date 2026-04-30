from __future__ import annotations

import os


def main() -> None:
    # Manual loopback/system audio level check.
    import numpy as np
    import sounddevice as sd

    device_index = os.environ.get("DEVICE_INDEX")
    sample_rate = int(os.environ.get("SAMPLE_RATE", "48000"))
    channels = int(os.environ.get("CHANNELS", "2"))
    duration_s = float(os.environ.get("DURATION", "5"))

    if device_index is None:
        raise SystemExit("Set DEVICE_INDEX to a loopback/system device index (see scripts/list_audio_devices.py)")

    print("Recording system audio...")
    audio = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=int(device_index),
        blocking=True,
    )

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    print("peak:", peak)
    print("rms :", rms)


if __name__ == "__main__":
    main()

