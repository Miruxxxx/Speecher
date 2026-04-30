from __future__ import annotations

import os


def main() -> None:
    # Simple RMS/peak check for the selected mic.
    import numpy as np
    import sounddevice as sd

    device_index = os.environ.get("DEVICE_INDEX")
    sample_rate = int(os.environ.get("SAMPLE_RATE", "44100"))
    duration_s = float(os.environ.get("DURATION", "3"))

    if device_index is not None:
        sd.default.device = int(device_index)
    sd.default.samplerate = sample_rate
    sd.default.channels = 1

    print("Speak...")
    audio = sd.rec(int(duration_s * sample_rate), dtype="float32")
    sd.wait()

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    print("peak:", peak)
    print("rms :", rms)


if __name__ == "__main__":
    main()

