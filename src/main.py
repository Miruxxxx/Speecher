from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app_config import AppConfig, load_config
from asr.backends import create_asr_backend
from asr.events import ASREvent
from asr.punctuation_backends import create_backend
from asr.punctuation_worker import PunctuationWorker
from audio.buffer import GrowingAudioBuffer
from audio.capture_backends import create_capture_supervisor
from llm.engine import LLMEngine
from llm.lmstudio_client import LMStudioClient
from store.transcript_store import TranscriptStore
from ui.output_sink import OutputSink
from ui.overlay import Overlay
from utils.latency import LatencyTracker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"


def _run_splitter(
    source: "queue.Queue[ASREvent]",
    sinks: "list[queue.Queue[ASREvent]]",
    store: TranscriptStore,
    stop_event: threading.Event,
) -> None:
    """Fan-out: reads one source queue and copies each event to all sinks.

    Commits are also appended to the TranscriptStore *here*, so the store
    (the source of truth for summaries/questions) never loses words to a
    full UI queue — sinks only render. Amend events bind nemotron's late
    punctuation to the word it belongs to (by timestamp) in the same in-order
    pass.
    """
    while not stop_event.is_set():
        try:
            ev = source.get(timeout=0.1)
        except queue.Empty:
            continue
        if ev.type == "commit":
            store.append(ev.words)
        elif ev.type == "amend":
            # Late punctuation for a word already committed. Applied here (in
            # queue order, so the target word is in the store first) rather than
            # in the overlay, keeping the store the single source of truth. Binds
            # by timestamp, not to the last word: nemotron emits the mark ~1.5 s
            # late, after 1-2 following words have committed.
            store.attach_punct_at(ev.time_sec, ev.text)
        for q in sinks:
            try:
                q.put_nowait(ev)
            except queue.Full:
                # Sinks are render-only; a dropped event costs at most a
                # slightly stale view until the next one arrives.
                pass
        source.task_done()


class _Pipeline:
    """Holds pipeline pieces created by the background loader thread."""

    def __init__(self) -> None:
        self.engine_thread: threading.Thread | None = None


def _load_asr_pipeline(
    cfg: AppConfig,
    audio_buffer: GrowingAudioBuffer,
    raw_events: "queue.Queue[ASREvent]",
    stop_event: threading.Event,
    latency: LatencyTracker,
    overlay: Overlay,
    pipeline: _Pipeline,
    frame_sink: "queue.Queue | None" = None,
) -> None:
    """Loads the ASR backend and starts its loop. Runs in a background thread
    so the overlay window is interactive from the first second."""
    try:
        backend = create_asr_backend(
            cfg,
            audio_buffer=audio_buffer,
            latency=latency,
            on_status=overlay.post_status,
            frame_sink=frame_sink,
        )
        try:
            backend.load()
        except Exception as exc:
            if backend.name == "whisper":
                raise
            # A second engine that won't load is a config/environment problem,
            # not a reason to leave the user without transcription.
            logger.exception("%s backend failed to load; falling back to whisper", backend.name)
            overlay.post_status(f"{backend.name} не загрузился ({exc}); включаю Whisper…")
            backend = create_asr_backend(
                cfg,
                audio_buffer=audio_buffer,
                latency=latency,
                on_status=overlay.post_status,
                engine="whisper",
            )
            backend.load()
        if stop_event.is_set():  # app closed while the model was loading
            return
        pipeline.engine_thread = threading.Thread(
            target=backend.run,
            kwargs={"events": raw_events, "stop_event": stop_event},
            name="engine",
            daemon=True,
        )
        pipeline.engine_thread.start()
        overlay.post_status("")
        model = cfg.asr.nemotron.model if backend.name == "nemotron" else cfg.asr.model
        logger.info("ASR engine started (backend=%s, model=%s)", backend.name, model)
    except Exception as exc:
        logger.exception("ASR pipeline failed to load")
        overlay.post_status(f"Ошибка загрузки ASR: {exc}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(CONFIG_PATH)

    app = QApplication(sys.argv)

    stop_event = threading.Event()
    latency = LatencyTracker(window=120, keys=("capture_convert", "whisper"))

    # Engine writes here; splitter fans out to per-consumer queues.
    raw_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=800)
    sink_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=400)
    overlay_events: "queue.Queue[ASREvent]" = queue.Queue(maxsize=400)

    audio_buffer = GrowingAudioBuffer(
        sample_rate=cfg.audio.target_sample_rate, max_seconds=cfg.asr.buffer_max_seconds
    )
    store = TranscriptStore()

    # Push-style backends read frames as they arrive; the pull-style whisper
    # path takes snapshots of audio_buffer and wants no queue at all.
    frame_sink: "queue.Queue | None" = (
        queue.Queue(maxsize=256) if cfg.asr.engine == "nemotron" else None
    )

    capture = create_capture_supervisor(
        cfg,
        audio_buffer=audio_buffer,
        events=raw_events,
        stop_event=stop_event,
        latency=latency,
        frame_sink=frame_sink,
    )

    sink = OutputSink(
        events=sink_events,
        stop_event=stop_event,
        verbose_logs=cfg.ui.console_verbose_logs,
    )

    llm = LLMEngine(LMStudioClient(cfg.llm))
    overlay = Overlay(
        events_queue=overlay_events,
        store=store,
        llm=llm,
        max_recent_chars=cfg.ui.max_recent_chars,
        max_summary_chars=cfg.llm.max_summary_chars,
    )

    # When the overlay (last window) closes, stop everything.
    app.aboutToQuit.connect(stop_event.set)

    # Forward a fatal stop_event (set by engine/capture) into Qt's event loop.
    # If the overlay is surfacing the fatal reason in a dialog, defer to it so
    # the process doesn't quit underneath the message. One grace tick covers
    # the gap between stop_event being set and the overlay polling the fatal
    # event; if no dialog appears (the fatal event was dropped), we still quit.
    shutdown_grace = [1]

    def _check_shutdown() -> None:
        if not stop_event.is_set():
            return
        if overlay.fatal_active():
            return  # the overlay owns the quit once its dialog is dismissed
        if shutdown_grace[0] > 0:
            shutdown_grace[0] -= 1
            return
        app.quit()

    shutdown_check = QTimer()
    shutdown_check.timeout.connect(_check_shutdown)
    shutdown_check.start(500)

    splitter_thread = threading.Thread(
        target=_run_splitter,
        args=(raw_events, [sink_events, overlay_events], store, stop_event),
        name="splitter",
        daemon=True,
    )
    sink_thread = threading.Thread(target=sink.run, name="output", daemon=True)

    # Window first, heavy models later: capture starts immediately (buffer
    # caps itself), Whisper loads in the background and reports progress
    # into the overlay's status line.
    llm.start()
    splitter_thread.start()
    sink_thread.start()
    capture.start()

    pipeline = _Pipeline()
    loader_thread = threading.Thread(
        target=_load_asr_pipeline,
        args=(cfg, audio_buffer, raw_events, stop_event, latency, overlay, pipeline, frame_sink),
        name="asr-loader",
        daemon=True,
    )
    loader_thread.start()

    punct_thread: threading.Thread | None = None
    punct_backend = None
    if cfg.asr.engine == "nemotron":
        # Nemotron punctuates and cases its own output; re-punctuating it would
        # only fight the model. Logged so it doesn't read as a broken feature.
        logger.info("punctuation worker disabled: engine=nemotron punctuates natively")
    else:
        punct_backend = create_backend(cfg.punctuation.backend, cfg.punctuation.language)
    if punct_backend is not None:
        worker = PunctuationWorker(
            store=store,
            backend=punct_backend,
            stop_event=stop_event,
            interval_sec=cfg.punctuation.interval_sec,
            context_words=cfg.punctuation.context_words,
        )
        punct_thread = threading.Thread(target=worker.run, name="punctuation", daemon=True)
        punct_thread.start()

    overlay.show()
    ret = app.exec()

    stop_event.set()
    capture.stop()
    loader_thread.join(timeout=2.0)
    if pipeline.engine_thread is not None:
        pipeline.engine_thread.join(timeout=3.0)
    sink_thread.join(timeout=2.0)
    splitter_thread.join(timeout=2.0)
    if punct_thread is not None:
        punct_thread.join(timeout=2.0)
    llm.stop()
    print(latency.report())

    sys.exit(ret)


if __name__ == "__main__":
    main()
