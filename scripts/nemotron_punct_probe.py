"""Diagnostic: is nemotron's punctuation drift the model's fault or ours?

Runs one live audio clip through two decoders and prints them side by side:

  NATIVE  - the model's own offline decode (full context, `processor.decode`
            with `durations`). This is ground truth: where the MODEL itself
            places '?'/'.' and on which word.
  STREAM  - our production streaming path (NemotronBackend + WordAssembler +
            the splitter's store, incl. the timestamp-based `amend`). This is
            what the overlay actually shows.

If NATIVE puts '?' on the right word but STREAM does not -> our streaming code
is to blame (rewrite the assembler/binding). If NATIVE already drifts the mark
-> it is the model, and punctuation should come from elsewhere (LLM question
detection over TranscriptStore, or the silero punctuation backend).

Usage (from the repo root, project venv):

    .venv\\Scripts\\python scripts\\nemotron_punct_probe.py record --seconds 30 --out probe.wav
    .venv\\Scripts\\python scripts\\nemotron_punct_probe.py decode --wav probe.wav
    .venv\\Scripts\\python scripts\\nemotron_punct_probe.py both  --seconds 30 --out probe.wav

`record` captures WASAPI loopback, so PLAY a Russian dialog with questions
(system audio) while it runs. `decode` needs the project venv (CUDA torch +
transformers); `record` alone needs only the built capture binary
(native/audio_capture).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

# Make the project's `src` packages importable the same way `python -m src` does.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_RATE = 16000


# -- recording -----------------------------------------------------------

class _Collector:
    """Audio sink with the GrowingAudioBuffer's write() contract."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        with self._lock:
            self.chunks.append(np.array(data, copy=True))


def record_loopback(seconds: float, out_path: Path, device_hint: str = "") -> None:
    """Capture `seconds` of WASAPI loopback to a 16 kHz mono PCM16 WAV.

    Runs the app's own capture child, so the file holds exactly what the ASR
    would have been fed - downmix and resampling included.
    """
    from app_config import load_config
    from audio.rust_capture import RustCaptureSupervisor, resolve_binary

    cfg = load_config(ROOT / "config" / "config.toml")
    sink = _Collector()
    events: "queue.Queue" = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    supervisor = RustCaptureSupervisor(
        audio_buffer=sink,
        binary_path=resolve_binary(cfg.audio.binary),
        device_hint=device_hint or cfg.audio.device_hint,
        target_sr=SAMPLE_RATE,
        events=events,
        stop_event=stop_event,
    )

    print(f"[record] PLAY your audio now - capturing {seconds:.0f} s ...")
    supervisor.start()
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.2)
    finally:
        supervisor.stop()
        stop_event.set()

    while True:
        try:
            ev = events.get_nowait()
        except queue.Empty:
            break
        if ev.type == "fatal" or "device=" in (ev.text or ""):
            print(f"[record] {ev.text}")

    mono = np.concatenate(sink.chunks) if sink.chunks else np.empty(0, dtype=np.float32)
    _write_wav(out_path, mono)
    dur = len(mono) / SAMPLE_RATE
    peak = float(np.abs(mono).max()) if len(mono) else 0.0
    print(f"[record] wrote {out_path} - {dur:.1f} s, peak {peak:.3f}")
    if peak < 1e-3:
        print("[record] WARNING: near-silent capture. Was audio actually playing?")


def _write_wav(path: Path, mono: np.ndarray) -> None:
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, "expected mono"
        assert w.getframerate() == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


# -- decoding ------------------------------------------------------------

def _load_backend(lang: str):
    """Load NemotronBackend once; reuse its model/processor for NATIVE too."""
    from app_config import load_config
    from asr.nemotron_backend import NemotronBackend

    cfg = load_config(ROOT / "config" / "config.toml").asr.nemotron
    if lang:
        cfg.language = lang
    frame_sink: "queue.Queue[np.ndarray]" = queue.Queue()
    backend = NemotronBackend(cfg=cfg, frame_sink=frame_sink, on_status=lambda m: print(f"[load] {m}"))
    backend.load()
    return backend, cfg, frame_sink


def native_decode(backend, cfg, audio: np.ndarray) -> None:
    """The model's own offline decode: full context, real durations."""
    import torch

    model, processor = backend._model, backend._processor
    feats = processor(
        audio, sampling_rate=SAMPLE_RATE, is_streaming=False, language=cfg.language
    )
    input_features = feats["input_features"].to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_features=input_features,
            num_lookahead_tokens=cfg.lookahead_tokens,
            prompt_ids=backend._prompt_ids,
            return_dict_in_generate=True,
        )
    durations = getattr(out, "durations", None)
    if durations is None:
        print("[NATIVE] no durations in generate() output; keys:", list(out.keys()))
        print("[NATIVE] text:", processor.decode(out.sequences[0]))
        return

    # `decode` wants the full batch (it zips sequences with durations); batch
    # size is 1 here. Rebuild the text from the token pieces - DecodeStream keeps
    # leading spaces and punctuation, so this is the model's own word grouping.
    _, per_batch = processor.decode(out.sequences, durations=durations)
    tokens = per_batch[0]
    text = "".join(t.get("token", "") for t in tokens).strip()
    print("\n===== NATIVE (model offline decode) =====")
    print(text)
    print("\n--- token timeline around punctuation ---")
    for i, tok in enumerate(tokens):
        piece = tok.get("token", "")
        if any(c in piece for c in ".?!,;:"):
            ctx = "".join(t.get("token", "") for t in tokens[max(0, i - 4): i + 2])
            print(f"  t={tok.get('start'):.2f}s  mark {piece!r}  ...after: {ctx!r}")


def stream_decode(backend, frame_sink: "queue.Queue[np.ndarray]", audio: np.ndarray,
                  speed: float) -> None:
    """Our production streaming path, reconstructed through the real store."""
    from asr.events import ASREvent
    from store.transcript_store import TranscriptStore

    events: "queue.Queue[ASREvent]" = queue.Queue()
    stop = threading.Event()
    store = TranscriptStore()

    runner = threading.Thread(
        target=backend.run, kwargs={"events": events, "stop_event": stop},
        name="nemotron-run", daemon=True,
    )
    runner.start()

    # Feed the clip in 0.1 s chunks, paced ~real time so the silence gaps in the
    # audio produce the same blank-frame / stale-flush timing the live app sees
    # (that timing is exactly what races the late punctuation).
    chunk = int(SAMPLE_RATE * 0.1)
    step = (0.1 / speed) if speed > 0 else 0.0
    for i in range(0, len(audio), chunk):
        frame_sink.put(audio[i:i + chunk].copy())
        if step:
            time.sleep(step)

    # Let the tail drain (flush rules need a beat of silence), then stop.
    time.sleep(2.5)
    stop.set()
    runner.join(timeout=15)

    # Apply events exactly like src/main.py::_run_splitter does.
    applied: list[str] = []
    while True:
        try:
            ev = events.get_nowait()
        except queue.Empty:
            break
        if ev.type == "commit":
            store.append(ev.words)
            applied.append(f"commit  {' '.join(w.text for w in ev.words)!r}")
        elif ev.type == "amend":
            store.attach_punct_at(ev.time_sec, ev.text)
            applied.append(f"amend   {ev.text!r} @ {ev.time_sec:.2f}s")
        elif ev.type in ("fatal", "log"):
            applied.append(f"{ev.type:7} {ev.text}")

    print("\n===== STREAM (our production path) =====")
    print(store.all_text())
    print("\n--- event trace ---")
    for line in applied:
        print("  " + line)


def decode(wav: Path, lang: str, speed: float) -> None:
    audio = _read_wav(wav)
    print(f"[decode] {wav} - {len(audio) / SAMPLE_RATE:.1f} s")
    backend, cfg, frame_sink = _load_backend(lang)
    native_decode(backend, cfg, audio)
    stream_decode(backend, frame_sink, audio, speed)
    print("\n[compare] Same '?'/'.' on the same word in both -> model is fine, "
          "our streaming drifts it.\n          Both drift -> it's the model; "
          "detect questions via LLM/silero instead.")


# -- cli -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="capture loopback to WAV")
    r.add_argument("--seconds", type=float, default=30.0)
    r.add_argument("--out", type=Path, default=Path("probe.wav"))
    r.add_argument("--device", default="", help="loopback device name hint")

    d = sub.add_parser("decode", help="decode a WAV two ways and compare")
    d.add_argument("--wav", type=Path, required=True)
    d.add_argument("--lang", default="", help="override asr.nemotron.language (e.g. ru, auto)")
    d.add_argument("--speed", type=float, default=1.0, help="feed pace vs real time (1.0=realtime)")

    b = sub.add_parser("both", help="record then decode")
    b.add_argument("--seconds", type=float, default=30.0)
    b.add_argument("--out", type=Path, default=Path("probe.wav"))
    b.add_argument("--device", default="")
    b.add_argument("--lang", default="")
    b.add_argument("--speed", type=float, default=1.0)

    args = ap.parse_args()
    if args.cmd == "record":
        record_loopback(args.seconds, args.out, args.device)
    elif args.cmd == "decode":
        decode(args.wav, args.lang, args.speed)
    elif args.cmd == "both":
        record_loopback(args.seconds, args.out, args.device)
        decode(args.out, args.lang, args.speed)


if __name__ == "__main__":
    main()
