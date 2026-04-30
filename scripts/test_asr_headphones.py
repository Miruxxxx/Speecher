from __future__ import annotations

import os
import sys

import numpy as np
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from audio.devices import list_loopbacks, pick_loopback_device  # noqa: E402


TARGET_SR = int(os.environ.get("TARGET_SR", "16000"))
DURATION = int(os.environ.get("DURATION", "10"))
NAME_HINT = os.environ.get("NAME_HINT", "HyperX")


def record_loopback(loopback_index: int, rate: int, channels: int, duration: int) -> tuple[np.ndarray, int]:
    pa = pyaudio.PyAudio()
    stream = None
    try:
        channels = max(1, min(int(channels), 2))
        rate = int(rate)
        duration = int(duration)

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(loopback_index),
            frames_per_buffer=2048,
        )

        frames: list[bytes] = []
        total_frames = int(rate * duration)
        chunk = 2048
        loops = (total_frames + chunk - 1) // chunk

        for _ in range(loops):
            frames.append(stream.read(chunk, exception_on_overflow=False))

        raw = b"".join(frames)
        audio = np.frombuffer(raw, dtype=np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        return audio, rate
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        pa.terminate()


def main() -> None:
    model_name = os.environ.get("MODEL", "base")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("WHISPER_COMPUTE", "float32")
    language = os.environ.get("LANGUAGE", "en")

    print("Loading model...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print("Model loaded")

    pa = pyaudio.PyAudio()
    try:
        lb_index, lb_name, lb_ch, lb_sr = pick_loopback_device(pa, NAME_HINT)
        print("Loopback devices found:")
        for i, name, ch, sr in list_loopbacks(pa):
            print(f"  {i}: {name} | in={ch} | sr={sr}")
        print(f"\nUsing loopback: {lb_index}: {lb_name} | in={lb_ch} | sr={lb_sr}")
    finally:
        pa.terminate()

    print("Recording...")
    audio, sr = record_loopback(lb_index, lb_sr, lb_ch, DURATION)

    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr).astype(np.float32, copy=False)
    audio = np.clip(audio, -1.0, 1.0)

    segments, _info = model.transcribe(audio, language=language, vad_filter=True)

    print("Recognized:")
    for seg in segments:
        print(seg.text)


if __name__ == "__main__":
    main()

