from __future__ import annotations

import logging
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app_config import AppConfig, load_config
from asr.backends import EngineProfile, create_asr_backend
from asr.events import ASREvent
from asr.punctuation_backends import create_backend
from asr.punctuation_worker import PunctuationWorker
from audio.buffer import GrowingAudioBuffer
from audio.rust_capture import RustCaptureSupervisor, resolve_binary
import logging_setup
from llm.engine import LLMEngine
from llm.openai_client import OpenAICompatClient
from llm.summaries import SummaryHistory
from store.transcript_store import TranscriptStore
from store.transcript_writer import TranscriptWriter
from translate.backends import create_backend as create_translation_backend
from translate.store import TranslationStore
from translate.worker import TranslationWorker
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
    capture=None,
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
            # Nobody drains the frame queue once whisper is the engine, so stop
            # filling it — otherwise the supervisor counts a dropped chunk eight
            # times a second and says so in the log for the whole session.
            if capture is not None:
                capture.set_frame_sink(None)
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
        # The profile of what actually loaded, which after a fallback is not the
        # configured one. The session header was written with the *configured*
        # engine before the model had tried to load; correct it now that the
        # truth is known — the export and the journal must not name an engine
        # that produced nothing.
        actual = EngineProfile.for_name(cfg, backend.name)
        overlay.set_engine(actual.name, actual.model, requested=cfg.asr.engine)
        logger.info("ASR engine started (backend=%s, model=%s)", actual.name, actual.model)
    except Exception as exc:
        logger.exception("ASR pipeline failed to load")
        overlay.post_status(f"Ошибка загрузки ASR: {exc}")


def main() -> None:
    # Read the config before logging is set up: it says where the log goes.
    # Anything that fails in load_config reports itself through the root
    # logger's default handler, which is enough for a file that cannot parse.
    cfg = load_config(CONFIG_PATH)
    log_path = logging_setup.setup(cfg.logging)
    logging_setup.banner(log_path, CONFIG_PATH)

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

    # Everything that follows from `asr.engine`, decided once. Five separate
    # `== "nemotron"` checks used to live in this function.
    engine = EngineProfile.resolve(cfg)

    # Persistence hangs off the store, so words committed by the splitter and
    # punctuation rewritten by nemotron/PunctuationWorker all reach the journal.
    session_dir = Path(cfg.storage.dir)
    if not session_dir.is_absolute():
        session_dir = PROJECT_ROOT / session_dir
    session_meta = {
        # Computed here, not inside the writer, so the journal's meta line and
        # the export header carry the same start time.
        "started": datetime.now().isoformat(timespec="seconds"),
        "engine": engine.name,
        "model": engine.model,
        "language": engine.language,
        "sample_rate": cfg.audio.target_sample_rate,
    }
    if cfg.translate.backend != "off":
        session_meta["translate_target"] = cfg.translate.target
    writer = TranscriptWriter(session_dir, meta=session_meta)
    store = TranscriptStore(journal=writer)
    # Same journal: a session file carries its translations, so resuming one
    # does not re-run the model over text it already translated.
    translations = TranslationStore(target=cfg.translate.target, journal=writer)
    # Opened before the overlay is built so its ⏺ indicator renders the real
    # state on the first frame rather than flipping a moment later.
    if cfg.storage.enabled:
        writer.start()
    else:
        logger.info("transcript journal off at startup: [storage] enabled = false")

    # Push-style backends read frames as they arrive; the pull-style whisper
    # path takes snapshots of audio_buffer and wants no queue at all.
    frame_sink: "queue.Queue | None" = (
        queue.Queue(maxsize=256) if engine.push_style else None
    )

    capture = RustCaptureSupervisor(
        audio_buffer=audio_buffer,
        binary_path=resolve_binary(cfg.audio.binary),
        device_hint=cfg.audio.device_hint,
        target_sr=cfg.audio.target_sample_rate,
        events=raw_events,
        stop_event=stop_event,
        source=cfg.audio.source,
        restart_backoff_sec=cfg.audio.restart_backoff_sec,
        latency=latency,
        frame_sink=frame_sink,
        # The overlay's capture indicator reads "звук идёт" off this threshold,
        # so it matches what the ASR silence gate would have decoded.
        sound_rms_threshold=cfg.asr.filters.silence_rms_threshold,
    )

    sink = OutputSink(
        events=sink_events,
        stop_event=stop_event,
        verbose_logs=cfg.ui.console_verbose_logs,
    )

    llm = LLMEngine(OpenAICompatClient(cfg.llm))
    summaries = SummaryHistory()
    overlay = Overlay(
        overlay_events,
        store,
        llm,
        cfg,
        CONFIG_PATH,
        writer=writer,
        session_dir=session_dir,
        capture=capture,
        summaries=summaries,
        translations=translations,
    )
    overlay.set_session_meta(session_meta)
    # A dying journal must surface in the UI, not only in the log: the ⏺ button
    # is the user's only signal that recording is happening at all.
    writer.set_error_handler(overlay.post_journal_error)

    # Shutdown is the overlay's alone: it closes -> app.quit() -> stop_event.
    # Qt's implicit "last window closed" rule cannot be trusted here, because it
    # does not count the overlay (a Qt.Tool) as a window at all — leaving it on
    # means the first settings/about window to close ends the session.
    app.setQuitOnLastWindowClosed(False)
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
        args=(
            cfg, audio_buffer, raw_events, stop_event, latency, overlay, pipeline,
            frame_sink, capture,
        ),
        name="asr-loader",
        daemon=True,
    )
    loader_thread.start()

    # Live translation. Its own thread and its own model: it must not share the
    # LLM queue the answer/summary buttons use, and the model load is slow, so
    # the worker loads it itself (a failure disables translation, nothing else).
    translate_thread: threading.Thread | None = None
    translate_backend = create_translation_backend(cfg.translate)
    if translate_backend is not None:
        translator = TranslationWorker(
            store=store,
            translations=translations,
            backend=translate_backend,
            stop_event=stop_event,
            target=cfg.translate.target,
            source=cfg.translate.source,
            asr_language=engine.language,
            close_after_sec=cfg.translate.close_after_sec,
            pause_sec=cfg.translate.pause_sec,
            max_words_per_segment=cfg.translate.max_words_per_segment,
            max_pending=cfg.translate.max_pending,
            poll_sec=cfg.translate.poll_sec,
            skip_same_script=cfg.translate.skip_same_script,
            enabled=cfg.translate.enabled,
            on_status=overlay.post_status,
            # A CPU fallback is a lasting change to how the feature performs,
            # not a passing status line — it gets a panel the user dismisses.
            on_notice=overlay.post_advice,
        )
        overlay.set_translator(translator)
        translate_thread = threading.Thread(
            target=translator.run, name="translate", daemon=True
        )
        translate_thread.start()
    else:
        logger.info("live translation off: [translate] backend = %r", cfg.translate.backend)

    punct_thread: threading.Thread | None = None
    punct_backend = None
    if engine.punctuates_itself and not cfg.punctuation.over_native:
        # Nemotron punctuates and cases its own output, and on Russian it needs
        # no second opinion. Logged so it doesn't read as a broken feature —
        # `[punctuation] over_native = true` turns it on for the case where the
        # model emits almost nothing (run-on speech; see the config comment).
        logger.info(
            "punctuation worker disabled: engine=%s punctuates natively "
            "([punctuation] over_native = false)", engine.name
        )
    else:
        punct_backend = create_backend(cfg.punctuation.backend, cfg.punctuation.language)
        if punct_backend is not None and engine.punctuates_itself:
            logger.info(
                "punctuation worker runs on top of %s ([punctuation] over_native = true)",
                engine.name,
            )
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
    if translate_thread is not None:
        # Longer than the others on purpose: a translation already inside the
        # model cannot be interrupted, and its result still has to be journaled.
        translate_thread.join(timeout=5.0)
    llm.stop()
    # After the splitter and punctuation threads are joined, so nothing is still
    # writing when the file closes.
    writer.close()
    if writer.path is not None and writer.path.exists():
        print(f"Транскрипт сохранён: {writer.path}")
    print(latency.report())

    sys.exit(ret)


if __name__ == "__main__":
    main()
