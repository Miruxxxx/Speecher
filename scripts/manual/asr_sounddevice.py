from __future__ import annotations

import os


def main() -> None:
    # NOTE: This is a manual script (not a unit test).
    # It requires a working `sounddevice` setup and (optionally) CUDA for Whisper.
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    input_sr = int(os.environ.get("INPUT_SR", "48000"))
    target_sr = int(os.environ.get("TARGET_SR", "16000"))
    duration_s = float(os.environ.get("DURATION", "10"))
    device_index = os.environ.get("DEVICE_INDEX")
    language = os.environ.get("LANGUAGE", "en")

    if device_index is not None:
        sd.default.device = int(device_index)
    sd.default.samplerate = input_sr
    sd.default.channels = 2

    print("Loading model...")
    model = WhisperModel(
        os.environ.get("MODEL", "base"),
        device=os.environ.get("WHISPER_DEVICE", "cuda"),
        compute_type=os.environ.get("WHISPER_COMPUTE", "float32"),
    )
    print("Model loaded")

    print(f"Recording {duration_s:.1f}s @ {input_sr}Hz ...")
    audio = sd.rec(int(duration_s * input_sr), dtype=np.float32)
    sd.wait()

    # Use first channel (typical loopback/mix stereo layout).
    audio = audio[:, 0]

    if input_sr != target_sr:
        audio = resample_poly(audio, target_sr, input_sr).astype(np.float32, copy=False)

    audio = np.clip(audio, -1.0, 1.0)

    segments, _info = model.transcribe(audio, language=language, vad_filter=True)

    print("Recognized:")
    for seg in segments:
        print(seg.text)


if __name__ == "__main__":
    main()

