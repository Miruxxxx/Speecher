# CLAUDE.md

Speecher — Windows-only real-time transcription of **system audio** (WASAPI loopback) with a PyQt6 always-on-top overlay and LM Studio (local LLM) integration. Python 3.11, project venv in `.venv` (gitignored). User's language: Russian.

## Run / test

```powershell
.venv\Scripts\python -m src            # run the app (overlay window; models load in background)
.venv\Scripts\python -m pytest tests/  # unit tests, no GPU/audio needed (80 tests)
.venv\Scripts\python scripts\list_audio_devices.py  # enumerate WASAPI loopback devices
```

The venv exists because the nemotron backend needs a CUDA torch build (cu128) while the global interpreter is on `torch 2.8.0+cpu` — see docs/MIGRATION_NEMOTRON.md. The global interpreter is no longer the project's runtime; anything installed there is unrelated.

All tunables live in `config/config.toml` (loaded by `src/app_config.py` over dataclass defaults; missing file/keys → defaults). No CI, no pyproject.

## Architecture (data flow)

```
[capture CHILD PROCESS] audio/capture_process.py: pyaudiowpatch WASAPI loopback
      raw float32 frames → mp.Queue     (PortAudio corrupts its host process's
                                         heap on idle loopback streams — §B1 —
                                         so it lives in a disposable process)
[capture-supervisor thread] audio/capture_supervisor.py: pumps the queue,
      stereo→mono, stateful soxr resample →16kHz → GrowingAudioBuffer
      (+ optional frame_sink queue for a push-style backend, drop-on-full);
      restarts dead children (backoff; 3 fast fails in a row → fatal)
[engine thread]  the ASRBackend picked by asr.engine runs its own loop here.
                 whisper (default): WhisperBackend → StreamingASREngine, every
                 1s snapshot buffer → WhisperAdapter (faster-whisper
                 large-v3-turbo CUDA fp16) → LocalAgreement-2 commit/partial;
                 silence: RMS gate skips decodes, N empty → trim to short tail
                 nemotron: NemotronBackend pulls frames from frame_sink, cuts
                 fixed-size mel chunks, feeds one long-lived generate() and
                 emits commit-only words from its streamer (no partials)
[splitter thread] fans raw_events out to sink_events + overlay_events;
                 appends commit words to TranscriptStore (store is the source
                 of truth — a dropped UI event can't lose transcript data)
[output thread]  OutputSink → console ANSI (partials only when stdout is a tty)
[Qt main thread] Overlay polls overlay_events via QTimer(50ms); repaint timer
                 (3s) picks up punctuation edits; status line for loader
                 progress; LM health checks run in worker threads, never UI
[llm thread]     LLMEngine queue → LMStudioClient (no retries, short health
                 timeout, model id resolved from LM Studio when cfg empty)
[punctuation thread] PunctuationWorker: lazy-loads backend (silero-te default,
                 CPU, ru/en/de/es; load failure = feature off, not crash),
                 every 12s re-punctuates last 80 words in place
[asr-loader thread] loads WhisperModel in background; window shows instantly
```

Shutdown: single shared `stop_event`; overlay close → `app.quit()` → stop_event; a QTimer(500ms) forwards a fatal stop_event back into Qt. Capture child is terminated by the supervisor (its read() blocks in silence and can't see events).

## Invariants / gotchas

- **Word timestamps are global stream seconds**: WhisperAdapter returns local-to-snapshot times; engine adds `offset_sec` from the buffer. Buffer trimming relies on these; breaking them makes Whisper re-decode the same audio forever.
- TranscriptStore is **append-only**; PunctuationWorker rewrites word *texts* in place by index — safe only because indices never shift. The worker's backend must preserve whitespace word count (checked, mismatch = skip).
- Store is populated by the **splitter**, not the overlay; overlay only renders.
- `language = "auto"` (default) = sticky: WhisperAdapter pins the language after the first decode with words and detection probability ≥ threshold. Re-detection per cycle (`language = ""`) commits wrong-alphabet garbage on noise.
- WASAPI loopback delivers frames **only while something renders audio**; in silence the child blocks in read() and the buffer stays static.
- Everything network-ish (LM Studio health, LLM calls) stays off the UI thread; `LMStudioClient` has max_retries=0 and a 2s health timeout for exactly that reason.
- multiprocessing uses **spawn**: capture child target must stay a top-level importable function; `python -m src` main module is skipped on child bootstrap (safe), but scratch scripts that spawn must guard `if __name__ == "__main__"`.

## Known issue (mitigated, root cause external)

PyAudioWPatch 0.2.12.7 corrupts the heap of its host process after ~2-8 min of an idle loopback stream (0xc0000005 in ntdll; bisected 2026-07-15: capture-only run crashed, whisper-only loop clean over 5438 decodes). Mitigation: process isolation + supervisor restart (see above). History and crash matrix: docs/CODE_REVIEW_2026-07-15.md §B1.

## Second ASR backend (Nemotron) — done, off by default

`asr.engine = "whisper" | "nemotron"`; the migration and every measurement live in **docs/MIGRATION_NEMOTRON.md** — read it before touching `nemotron_backend.py`, `streaming_engine.py`, `whisper_adapter.py` or `capture_supervisor.py`. All four phases are done (2026-07-23); `src/asr/nemotron_backend.py` streams live in `.venv` (~0.7 s to a word vs 2.5-3.5 s for whisper).

**Whisper stays the default on purpose**: both engines were only ever measured on synthesized speech (silero TTS, SAPI), and nemotron's open limits — `input_ids` growing linearly inside the one long-lived `generate()`, encoder context surviving a minute of silence, no RMS silence gate — show up on long live sessions, not on 12-second files. Flipping the default needs a real-speech run, not a code change. A backend that fails to load falls back to whisper with a status message.

How the nemotron path differs from whisper (details in the doc):

- Push, not pull: `CaptureSupervisor(frame_sink=...)` feeds a queue; `main.py` creates that queue only for this engine.
- Streaming `generate()` is **one long-lived call** that pulls mel chunks from a generator and blocks in silence. Live output comes only from `streamer=`, so `durations` never arrive — `_WordStreamer` reproduces the encoder-frame counter (blank advances a frame) to get global timestamps.
- Output is append-only: no `partial` events, no LocalAgreement-2, no `GrowingAudioBuffer`, and `PunctuationWorker` is not started (punctuation is native — `main.py` logs that).
- Words are closed by the *next* word's leading space, plus two flush rules (`IDLE_FLUSH_SEC` in stream frames, `STALE_FLUSH_SEC` in wall clock). Sentence-final punctuation lands ~1.5 s late, so closing a word earlier would split "встреча?" in two.

## Conventions

- Threads communicate via `queue.Queue[ASREvent]` only; UI updates cross into Qt via pyqtSignal (`post_status` is the thread-safe status entry).
- Every worker takes `stop_event: threading.Event`, is a daemon thread, and is joined with a timeout in main.py.
- Comments in code are English; user-facing docs Russian.
- Config keys map 1:1 to dataclasses in `src/app_config.py`; add new tunables there first, then to `config/config.toml` with a comment.
