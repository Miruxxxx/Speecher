from __future__ import annotations

import queue
import threading

from faster_whisper import WhisperModel

from asr.events import ASREvent
from asr.streaming_engine import StreamingASREngine
from asr.whisper_adapter import WhisperAdapter
from audio.buffer import GrowingAudioBuffer
from audio.capture_stream import LoopbackStreamCapture
from ui.output_sink import OutputSink
from utils.latency import LatencyTracker


TARGET_SR = 16000


def main() -> None:
    stop_event = threading.Event()
    events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=400)
    latency = LatencyTracker(window=120, keys=("capture_convert", "whisper"))

    audio_buffer = GrowingAudioBuffer(sample_rate=TARGET_SR, max_seconds=60.0)

    capture = LoopbackStreamCapture(
        audio_buffer=audio_buffer,
        name_hint="HyperX",
        target_sr=TARGET_SR,
        events=events,
        stop_event=stop_event,
        latency=latency,
    )

    model = WhisperModel("base", device="cuda", compute_type="float16")
    transcribe_fn = WhisperAdapter(
        model,
        language="en",  # autodetect; pass "en" / "ru" / etc. to lock it
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,
    )

    engine = StreamingASREngine(
        audio_buffer=audio_buffer,
        transcribe_fn=transcribe_fn,
        events=events,
        stop_event=stop_event,
        chunk_interval_sec=1.0,
        min_audio_sec=1.0,
        commit_safety_margin_sec=0.1,
        max_buffer_sec=25.0,
        force_commit_keep_tail_words=6,
        latency=latency,
    )

    sink = OutputSink(
        events=events,
        stop_event=stop_event,
        verbose_logs=False,
    )

    sink_thread = threading.Thread(target=sink.run, name="output", daemon=True)
    engine_thread = threading.Thread(target=engine.run, name="engine", daemon=True)

    sink_thread.start()
    capture.start()
    engine_thread.start()

    try:
        input("Press Enter to stop...\n")
    finally:
        stop_event.set()
        capture.stop()
        engine_thread.join(timeout=3.0)
        sink_thread.join(timeout=2.0)
        print(latency.report())


if __name__ == "__main__":
    main()
