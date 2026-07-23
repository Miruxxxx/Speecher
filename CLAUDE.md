# CLAUDE.md

Speecher — Windows-only real-time transcription of **system audio** (WASAPI loopback) with a PyQt6 always-on-top overlay and LM Studio (local LLM) integration. Python 3.11, global interpreter (no venv). User's language: Russian.

## Run / test

```powershell
python -m src            # run the app (overlay window; models load in background)
python -m pytest tests/  # unit tests, no GPU/audio needed (58 tests)
python scripts/list_audio_devices.py  # enumerate WASAPI loopback devices
```

All tunables live in `config/config.toml` (loaded by `src/app_config.py` over dataclass defaults; missing file/keys → defaults). No CI, no venv, no pyproject.

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

## Planned: second ASR backend (Nemotron)

Migration to a switchable ASR engine (`asr.engine = "whisper" | "nemotron"`) is specified in **docs/MIGRATION_NEMOTRON.md** — read it before touching `streaming_engine.py`, `whisper_adapter.py`, or `capture_supervisor.py`. Status: **phases 0 and 1 done, phase 2 unblocked**. The `ASRBackend` protocol (`src/asr/backends.py`), `WhisperBackend`, `asr.engine` config key and the supervisor's `frame_sink` exist; `"whisper"` is the only known engine and anything else falls back to it with a warning.

Phase-0 probe (2026-07-23) settled the open questions: `nemo_toolkit` is **not** needed (`transformers` ≥5.13 with `AutoModelForRNNT`/`AutoProcessor`, plus the undocumented-but-required `librosa` and `accelerate`); word timestamps **are** available via `processor.decode(sequences, durations=out.durations)` at an 80 ms encoder frame, and stay global across streaming chunks (so no `offset_sec` bookkeeping); output is append-only, so the Nemotron path needs no `partial` events and no LocalAgreement-2. Caveats: `Word.end` is an emission point (+80 ms), not the acoustic end; fp16 halves VRAM but decodes ~2x slower than fp32; Russian output omits "?", which `find_last_question` depends on.

Environment gotcha for phase 2: the global interpreter has `torch 2.8.0+cpu`, but Nemotron needs a CUDA build (cu128 on this Blackwell GPU). Deciding between a project venv and a global cu128 install is a project-level call — do not make it silently inside phase 2.

## Conventions

- Threads communicate via `queue.Queue[ASREvent]` only; UI updates cross into Qt via pyqtSignal (`post_status` is the thread-safe status entry).
- Every worker takes `stop_event: threading.Event`, is a daemon thread, and is joined with a timeout in main.py.
- Comments in code are English; user-facing docs Russian.
- Config keys map 1:1 to dataclasses in `src/app_config.py`; add new tunables there first, then to `config/config.toml` with a comment.
