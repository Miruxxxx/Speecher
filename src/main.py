from __future__ import annotations

import queue
import sys
import threading

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from faster_whisper import WhisperModel

from asr.events import ASREvent
from asr.streaming_engine import StreamingASREngine
from asr.whisper_adapter import WhisperAdapter
from audio.buffer import GrowingAudioBuffer
from audio.capture_stream import LoopbackStreamCapture
from llm.engine import LLMEngine
from store.transcript_store import TranscriptStore
from ui.output_sink import OutputSink
from ui.overlay import Overlay
from utils.latency import LatencyTracker


TARGET_SR = 16000
LLM_MODEL_PATH = "models/qwen2.5-7b-instruct-q4_k_m.gguf"


def _run_splitter(
    source: "queue.Queue[ASREvent]",
    sinks: "list[queue.Queue[ASREvent]]",
    stop_event: threading.Event,
) -> None:
    """Fan-out: reads one source queue and copies each event to all sinks."""
    while not stop_event.is_set():
        try:
            ev = source.get(timeout=0.1)
        except queue.Empty:
            continue
        for q in sinks:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        source.task_done()


def main() -> None:
    app = QApplication(sys.argv)

    stop_event = threading.Event()
    latency = LatencyTracker(window=120, keys=("capture_convert", "whisper"))

    # Engine writes here; splitter fans out to per-consumer queues.
    raw_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=800)
    sink_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=400)
    overlay_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=400)

    audio_buffer = GrowingAudioBuffer(sample_rate=TARGET_SR, max_seconds=60.0)

    capture = LoopbackStreamCapture(
        audio_buffer=audio_buffer,
        name_hint="HyperX",
        target_sr=TARGET_SR,
        events=raw_events,
        stop_event=stop_event,
        latency=latency,
    )

    model = WhisperModel("base", device="cuda", compute_type="float16")
    transcribe_fn = WhisperAdapter(
        model,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,
    )

    engine = StreamingASREngine(
        audio_buffer=audio_buffer,
        transcribe_fn=transcribe_fn,
        events=raw_events,
        stop_event=stop_event,
        chunk_interval_sec=1.0,
        min_audio_sec=1.0,
        commit_safety_margin_sec=0.1,
        max_buffer_sec=25.0,
        force_commit_keep_tail_words=6,
        latency=latency,
    )

    sink = OutputSink(
        events=sink_events,
        stop_event=stop_event,
        verbose_logs=False,
    )

    store = TranscriptStore()
    llm = LLMEngine(model_path=LLM_MODEL_PATH)
    overlay = Overlay(events_queue=overlay_events, store=store, llm=llm)

    # When the overlay (last window) closes, stop everything.
    app.aboutToQuit.connect(stop_event.set)

    # Forward a fatal stop_event (set by engine/capture) into Qt's event loop.
    shutdown_check = QTimer()
    shutdown_check.timeout.connect(lambda: app.quit() if stop_event.is_set() else None)
    shutdown_check.start(500)

    splitter_thread = threading.Thread(
        target=_run_splitter,
        args=(raw_events, [sink_events, overlay_events], stop_event),
        name="splitter",
        daemon=True,
    )
    sink_thread = threading.Thread(target=sink.run, name="output", daemon=True)
    engine_thread = threading.Thread(target=engine.run, name="engine", daemon=True)

    llm.start()
    splitter_thread.start()
    sink_thread.start()
    capture.start()
    engine_thread.start()

    overlay.show()
    ret = app.exec()

    stop_event.set()
    capture.stop()
    engine_thread.join(timeout=3.0)
    sink_thread.join(timeout=2.0)
    splitter_thread.join(timeout=2.0)
    llm.stop()
    print(latency.report())

    sys.exit(ret)


if __name__ == "__main__":
    main()
