# CLAUDE.md

Speecher — Windows-only real-time transcription of **system audio** (WASAPI loopback) with a PyQt6 always-on-top overlay and LM Studio (local LLM) integration. Python 3.11, global interpreter (no venv). User's language: Russian.

## Run / test

```powershell
python -m src            # run the app (needs NVIDIA GPU; opens overlay window)
python -m pytest tests/  # BROKEN: tests target the legacy engine (see below)
python scripts/list_audio_devices.py  # enumerate WASAPI loopback devices
```

No CI, no venv, no pyproject — requirements.txt only.

## Architecture (data flow)

```
[capture thread] LoopbackStreamCapture (pyaudiowpatch, WASAPI loopback "HyperX",
                 stereo→mono, resample_poly →16kHz)
      └─ writes → GrowingAudioBuffer (append-only, head-trim by global seconds)
[engine thread]  StreamingASREngine: every 1s snapshot buffer → WhisperAdapter
                 (faster-whisper "base", CUDA fp16, word timestamps) →
                 LocalAgreement-2: words identical in two consecutive decodes
                 (normalize_word) are committed; buffer trimmed to last commit;
                 force-commit when buffer > 25s
      └─ puts ASREvent(commit|partial|log|fatal) → raw_events queue
[splitter thread] fans raw_events out to sink_events + overlay_events (put_nowait,
                 silently drops when a sink queue is full — including commits!)
[output thread]  OutputSink → console ANSI (partials only when stdout is a tty)
[Qt main thread] Overlay polls overlay_events via QTimer(50ms); commits go into
                 TranscriptStore; buttons build prompts → LLMEngine
[llm thread]     LLMEngine queue → LMStudioClient (OpenAI-compatible REST,
                 localhost:1234, streaming; token callbacks re-enter Qt via signals)
[punctuation thread] PunctuationWorker: every 12s re-punctuates last 80 words
                 in TranscriptStore (deepmultilingualpunctuation, xlm-roberta,
                 loads onto GPU when CUDA available; model knows EN/DE/FR/IT — no RU)
```

Shutdown: single shared `stop_event`; overlay close → `app.quit()` → stop_event; a QTimer(500ms) forwards a fatal stop_event back into Qt.

## Invariants / gotchas

- **Word timestamps are global stream seconds**: WhisperAdapter returns local-to-snapshot times; engine adds `offset_sec` from the buffer. Buffer trimming relies on these; breaking them makes Whisper re-decode the same audio forever.
- TranscriptStore is **append-only**; PunctuationWorker rewrites word *texts* in place by index — safe only because indices never shift.
- TranscriptStore is populated **only** from the overlay's queue consumer (`_handle_asr` on "commit"), not by the engine.
- `language=None` in main.py → Whisper re-detects language every ~1s cycle; unstable on noise (commits wrong-alphabet garbage). Pin the language when quality matters.
- LMStudioClient hardcodes model id "qwen3.5-9b" and ignores its `timeout` field; `is_available()` runs on the UI thread (freezes UI for seconds when LM Studio is down — openai client retries 2x).
- All heavy init (Whisper load, HF hub checks, punctuation model download ~2.1GB on cold cache) happens **before** the window shows.
- **Known blocker**: native crash 0xc0000005 (heap corruption, ntdll.dll) after ~2-8 min. Bisected 2026-07-15: **PyAudioWPatch 0.2.12.7 loopback stream crashes the process on its own** (capture-only run died; whisper-only decode loop was clean over 5438 decodes). Correlates with idle loopback (no system audio rendering). Fix direction: capture in a subprocess / replace pyaudiowpatch. Details: docs/CODE_REVIEW_2026-07-15.md §B1.

## Legacy (do not extend)

- `src/transcript_engine.py`, `src/asr/streaming_worker.py` — pre-rewrite pipeline, imported by nothing in the app, reference a `utils.text.common_prefix_len` that no longer exists.
- `tests/` import that legacy module → whole pytest run fails at collection. New core (engine/buffer/store) has no tests yet.

## Conventions

- Threads communicate via `queue.Queue[ASREvent]` only; UI updates cross into Qt via pyqtSignal.
- Every worker takes `stop_event: threading.Event` and is a daemon thread with join-on-exit in main.py.
- Comments in code are English; user-facing docs Russian.
