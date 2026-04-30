from __future__ import annotations

import queue
import threading

from faster_whisper import WhisperModel

from asr.events import ASREvent
from asr.streaming_worker import FastASRWorker, RefinerASRWorker
from audio.buffer import RingBuffer
from audio.capture_stream import LoopbackStreamCapture
from ui.output_sink import OutputSink
from utils.latency import LatencyTracker

TARGET_SR = 16000


def main() -> None:
    stop_event = threading.Event()
    events: queue.Queue[ASREvent] = queue.Queue(maxsize=200)
    latency = LatencyTracker(window=120)

    output_sink = OutputSink(
        events=events,
        stop_event=stop_event,
        latency=latency,
        final_stability_n=2,
        final_stability_ratio=0.82,
        final_stability_window_words=40,
        min_final_overlap_words=3,
        fuzzy_overlap_min_ratio=0.78,
        max_backtrack_words=48,
        tail_anchor_words=140,
        live_indicator="⚡",
        enable_colors=True,
    )

    output_thread = threading.Thread(
        target=output_sink.run,
        name="output_sink",
        daemon=True,
    )
    output_thread.start()

    ring = RingBuffer(max_seconds=12, sample_rate=TARGET_SR)

    capture = LoopbackStreamCapture(
        ring_buffer=ring,
        name_hint="HyperX",
        target_sr=TARGET_SR,
        latency=latency,
        events=events,
        stop_event=stop_event,
    )
    capture.start()

    fast_model = WhisperModel("base", device="cuda", compute_type="float16")
    refine_model = fast_model

    fast_worker = FastASRWorker(
        ring_buffer=ring,
        model=fast_model,
        events=events,
        stop_event=stop_event,
        input_sr=TARGET_SR,
        target_sr=TARGET_SR,
        window_sec=3.0,
        step_sec=0.25,
        latency=latency,
        language="none",
    )

    refiner = RefinerASRWorker(
        ring_buffer=ring,
        model=refine_model,
        events=events,
        stop_event=stop_event,
        input_sr=TARGET_SR,
        target_sr=TARGET_SR,
        window_sec=8.0,
        every_sec=1.0,
        latency=latency,
        language="none",
    )

    threading.Thread(target=fast_worker.run, daemon=True).start()
    threading.Thread(target=refiner.run, daemon=True).start()

    try:
        input("Press Enter to stop...\n")
    finally:
        stop_event.set()
        capture.stop()
        fast_worker.stop()
        refiner.stop()
        output_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
