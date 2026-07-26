# CLAUDE.md

Speecher — Windows-only real-time transcription of **system audio** (WASAPI loopback) with a PyQt6 always-on-top overlay and LM Studio (local LLM) integration. Python 3.11, project venv in `.venv` (gitignored). User's language: Russian.

## Run / test

```powershell
.venv\Scripts\python -m src            # run the app (overlay window; models load in background)
.venv\Scripts\python -m pytest tests/  # unit tests, no GPU/audio needed (298 tests)
.venv\Scripts\python scripts\list_audio_devices.py  # enumerate audio endpoints
.venv\Scripts\python scripts\capture_soak.py --backend rust --seconds 900  # capture soak test
.venv\Scripts\python scripts\translate_probe.py  # load the MT model, time a few phrases
cargo build --release --manifest-path native\audio_capture\Cargo.toml  # native capture binary
cargo test --manifest-path native\audio_capture\Cargo.toml             # its 14 unit tests
```

The venv exists because the nemotron backend needs a CUDA torch build (cu128) while the global interpreter is on `torch 2.8.0+cpu` — see docs/MIGRATION_NEMOTRON.md. The global interpreter is no longer the project's runtime; anything installed there is unrelated.

All tunables live in `config/config.toml` (loaded by `src/app_config.py` over dataclass defaults; missing file/keys → defaults). No CI, no pyproject.

## Architecture (data flow)

```
[capture CHILD PROCESS] native/audio_capture (Rust, WASAPI directly): mono
      float32 @16kHz on stdout, JSON lines on stderr, stdin close = stop.
      Downmix and resampling happen there; no Python and no PortAudio in that
      process (that library was the cause of §B1 and is gone)
[capture-supervisor thread] audio/capture_base.py: restart policy (backoff;
      3 fast fails in a row → fatal) + the write into GrowingAudioBuffer
      (+ optional frame_sink queue for a push-style backend, drop-on-full);
      audio/rust_capture.py runs the child and slices its bytes into arrays
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
[journal]        TranscriptWriter hangs off the *store*, not the splitter, so
                 commits, nemotron's late amend and PunctuationWorker rewrites
                 all reach storage/transcripts/<stamp>.jsonl. Writes happen
                 outside the store's lock; an I/O failure disables the journal
                 and surfaces in the overlay, never in the pipeline
[output thread]  OutputSink → console ANSI (partials only when stdout is a tty)
[Qt main thread] Overlay polls overlay_events via QTimer(50ms); repaint timer
                 (3s) picks up punctuation edits; capture-state poll (250ms)
                 drives the title bar indicator; status zone for loader
                 progress; LM health checks run in worker threads, never UI.
                 Words become timestamped replies in ui/transcript_model.py
[hotkey thread]  ui/hotkeys.py: RegisterHotKey + its own Windows message loop
                 (they must work while the window is hidden); a combination
                 Windows refuses is reported, not fatal
[llm thread]     LLMEngine queue → OpenAICompatClient (no retries, short health
                 timeout). One client for every provider in llm/providers.py:
                 local (LM Studio, Ollama) and 12 cloud ones, all
                 OpenAI-compatible. The key never touches config.toml —
                 llm/credentials.py keeps it in the Windows Credential Manager
[translate thread] TranslationWorker: local MT (NLLB-600M by default), never
                 LM Studio — the answer/summary queue must stay free. Walks the
                 store forward by word index, cuts a segment (sentence / pause /
                 word cap), translates it and appends a *final* line to
                 TranslationStore, which journals it. The overlay then shows
                 those lines instead of the transcript, and the original keeps
                 running in the source window. Off by default, Alt+T toggles
[punctuation thread] PunctuationWorker: lazy-loads backend (silero-te default,
                 CPU, ru/en/de/es; load failure = feature off, not crash),
                 every 12s re-punctuates last 80 words in place
[asr-loader thread] loads WhisperModel in background; window shows instantly
```

Shutdown: single shared `stop_event`; overlay close → `app.quit()` → stop_event; a QTimer(500ms) forwards a fatal stop_event back into Qt. Capture child is terminated by the supervisor (its read() blocks in silence and can't see events).

## Invariants / gotchas

- **Word timestamps are global stream seconds**: WhisperAdapter returns local-to-snapshot times; engine adds `offset_sec` from the buffer. Buffer trimming relies on these; breaking them makes Whisper re-decode the same audio forever.
- TranscriptStore is **append-only**; PunctuationWorker rewrites word *texts* in place by index — safe only because indices never shift. The worker's backend must preserve whitespace word count (checked, mismatch = skip).
- The JSONL journal records **outcomes, not intents**: a rewrite is written as the `(index, final_text)` the store actually applied, so a replay can't diverge. This is why index stability above is load-bearing for persistence too, not just for the UI. `update_word_texts` journals only the words that actually changed — the worker resubmits its whole 80-word window every 12 s.
- Store is populated by the **splitter**, not the overlay; overlay only renders.
- The UI has one rule that outranks taste: **nothing may shift the transcript**. The answer card and the capture-error plate are overlays inside the feed area (not layout children), the status message has a permanently reserved zone, the indicator icon slot is a fixed 14×14, and a disabled action button keeps its height and only changes its second line. `tests/test_overlay_layout.py` is that table, not a nicety.
- **The settings window is a singleton built once from the config**, and every control it owns is applied on save — so a control that drifts out of sync silently undoes what the user did elsewhere. Translation is the one that drifts (Alt+T and the session menu are session-scoped and never touch the config), so `_open_settings` re-syncs it through `set_translate_enabled(live, available=…)` on every open. Any future control with a session-scoped twin needs the same treatment; `tests/test_settings_window.py` is that regression.
- A model's answer goes through `ui/markup.py` before it reaches the card — chat models emit `**bold**` and bullet lists whether or not the system prompt asks them not to. It is a deliberate subset, not CommonMark, and it leaves an *unclosed* marker literal so a streamed answer does not jump on every token. Our own strings (errors, "Собираю ответ…", the user's own question) bypass it.
- **No colour or spacing literal outside `src/ui/theme/tokens.py`** — `tests/test_design_tokens.py` greps for hex literals in `src/ui/` and fails on any.
- `language = "auto"` (default) = sticky: WhisperAdapter pins the language after the first decode with words and detection probability ≥ threshold. Re-detection per cycle (`language = ""`) commits wrong-alphabet garbage on noise.
- WASAPI loopback delivers frames **only while something renders audio**; in silence the child blocks in read() and the buffer stays static.
- Everything network-ish (LLM health, LLM calls) stays off the UI thread; `OpenAICompatClient` has max_retries=0 and a 2s health timeout for exactly that reason. A cloud provider's health result is additionally cached for 5 min — the overlay re-probes every 20 s and that would otherwise be 180 requests an hour.
- **An API key must never reach `config/config.toml`** — that file is in git and full of the user's comments. `LlmConfig` holds `provider`/`model` only; the secret lives in `llm/credentials.py` (Credential Manager → `%APPDATA%` file → the provider's env var, in that order). `tests/test_llm_providers.py` asserts the config has no secret-shaped field.
- Providers disagree about optional request fields (`reasoning_effort`, `max_tokens` vs `max_completion_tokens`, a non-default `temperature`). There is deliberately **no per-provider quirk table**: `OpenAICompatClient._request` reads the 400, drops the field the server named, retries once and remembers for the session. A table encoding this would rot with the next model family.
- A cloud provider's model is **never auto-resolved**. `resolve_model()` picks the loaded model from `/models` only for a local server; for a catalogue of hundreds, "the first one" would answer with something arbitrary and bill for it, so an unset model is an error naming the settings window.
- **Every `from transformers import …` under `src/` goes inside `with heavy_imports.transformers():`.** `transformers` resolves its names by lazily importing the submodule that defines them, so two worker threads reaching for two names out of the same submodule race — the loser sees a half-initialised module and Python reports the miss as `ImportError: cannot import name 'AutoModelForRNNT' from 'transformers'`, which looks like a bad install and is not. This happened on every start once translation was enabled (asr-loader thread vs translation thread, both hitting `models/auto/modeling_auto.py`): nemotron "failed to load", the app fell back to whisper, and the only trace was a status line that scrolls away. Reproduced 6/6 before the guard, 0/6 after. `tests/test_heavy_imports.py` walks the AST of `src/` and fails on an unguarded call site — the lock only works if everyone takes it. Keep the guarded block to the import statement: holding it across `from_pretrained` would serialise the model downloads for nothing.
- multiprocessing uses **spawn**: capture child target must stay a top-level importable function; `python -m src` main module is skipped on child bootstrap (safe), but scratch scripts that spawn must guard `if __name__ == "__main__"`.

## Former blocker (root cause removed 2026-07-24)

PyAudioWPatch 0.2.12.7 corrupted the heap of its host process after ~2-8 min of an idle loopback stream (0xc0000005 in ntdll; bisected 2026-07-15: capture-only run crashed, whisper-only loop clean over 5438 decodes). It was mitigated by process isolation, then removed outright: capture is native now and the library is no longer a dependency. History and crash matrix: docs/CODE_REVIEW_2026-07-15.md §B1 — keep it in mind before reintroducing any PortAudio-based capture.

## Capture is a native binary (Rust)

Design, protocol and measurements live in **docs/NATIVE_CAPTURE_RUST.md** — read it before touching `native/audio_capture/`, `rust_capture.py` or `capture_base.py`.

- The binary is built locally (`cargo build --release …`) and gitignored. There is no second capture path: a missing binary is a fatal event naming the build command, not a fallback.
- It delivers **mono float32 already at target_sample_rate**: downmix (channel average) and resampling (`rubato::Fft`, one long-lived state) happen in Rust, so `numpy`/`soxr` are not in that path at all.
- Silence policy is identical to the PortAudio path *on purpose*: an idle loopback endpoint yields no packets and none are invented, so word timestamps stay tied to real captured audio. `AUDCLNT_BUFFERFLAGS_SILENT` packets are forwarded as zeros — they are real time on the device timeline.
- `device_hint` matches the **output endpoint's** name (WASAPI opens loopback on a render endpoint), not the "… [Loopback]" input PortAudio used to invent.
- Tests need no audio hardware: `tests/fake_capture.py` speaks the same protocol (including a sample split across two reads and a `fatal` line) and `tests/test_rust_capture.py` runs the real supervisor against it.

## Second ASR backend (Nemotron) — done, off by default

`asr.engine = "whisper" | "nemotron"`; the migration and every measurement live in **docs/MIGRATION_NEMOTRON.md** — read it before touching `nemotron_backend.py`, `streaming_engine.py`, `whisper_adapter.py` or the capture path. All four phases are done (2026-07-23); `src/asr/nemotron_backend.py` streams live in `.venv` (~0.7 s to a word vs 2.5-3.5 s for whisper).

**Whisper stays the default on purpose**: both engines were only ever measured on synthesized speech (silero TTS, SAPI), and nemotron's open limits — `input_ids` growing linearly inside the one long-lived `generate()`, encoder context surviving a minute of silence, no RMS silence gate — show up on long live sessions, not on 12-second files. Flipping the default needs a real-speech run, not a code change. A backend that fails to load falls back to whisper with a status message.

How the nemotron path differs from whisper (details in the doc):

- Push, not pull: `CaptureSupervisor(frame_sink=...)` feeds a queue; `main.py` creates that queue only for this engine.
- Streaming `generate()` is **one long-lived call** that pulls mel chunks from a generator and blocks in silence. Live output comes only from `streamer=`, so `durations` never arrive — `_WordStreamer` reproduces the encoder-frame counter (blank advances a frame) to get global timestamps.
- Output is append-only: no `partial` events, no LocalAgreement-2, no `GrowingAudioBuffer`, and `PunctuationWorker` is not started (punctuation is native — `main.py` logs that).
- Words are closed by the *next* word's leading space, plus two flush rules (`IDLE_FLUSH_SEC` in stream frames, `STALE_FLUSH_SEC` in wall clock). Sentence-final punctuation lands ~1.5 s late, so closing a word earlier would split "встреча?" in two.
- That ~1.5 s lateness still races the flush rules: when the word is committed before its `.`/`?` arrives, the punctuation would be lost. `WordAssembler(on_late_punct=...)` catches it (with the mark's emission time) and the backend emits an **`amend`** event carrying `time_sec`; `_run_splitter` binds it to the word whose `end` is closest via `TranscriptStore.attach_punct_at` — not the last word, because by the time the mark arrives 1–2 later words are already committed. Same append-only, index-stable edit the whisper `PunctuationWorker` uses. Without this the mark landed on the wrong word (or none at all) on live speech.

## Live translation (src/translate/) — done, off by default

`[translate] backend = "nllb" | "opusmt" | "off"`, `enabled` = on at startup,
`Alt+T` toggles for the session (the config key is untouched, same contract as
the ⏺ recording toggle). Rationale and the roadmap entry: docs/ROADMAP.md §3.

- **The feed is append-only while translating, and that is the whole design.**
  A line enters the main feed already final — translated, or the original when
  the segment was skipped (already the target language) or the model failed on
  it — and is never touched again. The first version mutated transcript lines in
  place (original → dimmed → translation) and was unreadable; see
  docs/DESIGN_SYSTEM.md §7.8 before changing any of this back.
- **Two windows.** Main feed = finished lines. The original runs alongside in
  `ui/windows/source.py`, a 360 × 168 `Qt.Tool` docked under the overlay — an
  ordinary feed over the same store, keeping its lines and its scrollback (an
  earlier version showed only the untranslated tail and blanked itself every
  few seconds). The caret moves there with it.
- **The unit is a segment, keyed by word index** (`translate/segments.py`):
  sentence end, pause, or `max_words_per_segment`, whichever comes first. The
  index is an absolute position in the append-only store — the same handle
  `patch` records use. *Not* a reply timestamp: replies are regrouped from a
  sliding window on every repaint, two readers with different windows cut them
  differently, and the same translation landed on two lines.
- The worker walks forward by index and never revisits. That is why nemotron's
  late `amend` and whisper's PunctuationWorker cost nothing here — they can only
  touch words it has not reached yet.
- Falling behind **merges**, never drops (`max_pending`): consecutive waiting
  segments become one longer line, so the feed stays in order and complete.
- **Not LM Studio.** `LLMEngine` is one thread and one queue serving the
  "Ответить"/"Выжимка" buttons; background translations in it would add seconds
  to a button press. Translation is a local seq2seq model in its own thread.
- `skip_same_script` is trusted **only when nobody pinned the source language** —
  the script test separates Cyrillic from Latin, not German from English.
- The journal gains one record type, `{"t":"tr","i":<index>,"n":<words>,...}` —
  the only one that is not a store mutation. Older readers skip it, so
  `JOURNAL_VERSION` stayed 1. Timestamps are not in it: a replay recovers them
  from the words.
- `scripts/translate_probe.py` loads the backend and times a few phrases; the
  first run downloads ~2.5 GB.

## UI is a design system, not a stylesheet

Tokens, components, both layout modes, the screen states and the eight resolved
questions live in **docs/DESIGN_SYSTEM.md** — read it before touching anything
under `src/ui/`. The window is 520×340 (compact 360×200), IBM Plex ships in
`assets/fonts/` and is loaded with `QFontDatabase.addApplicationFont`; system
fonts and emoji are not used anywhere.

```
src/ui/theme/     tokens.py (colours/type/space/motion, Qt-free), fonts, icons (inline SVG), styles (QSS)
src/ui/widgets/   indicators, buttons, transcript feed, cards, titlebar, settings fields
src/ui/windows/   settings, history, notification + fatal, onboarding + cheat sheet, source (live original)
src/ui/overlay.py the window itself; transcript_model.py groups words into replies; markup.py renders a model's markdown; hotkeys.py registers Alt+…
```

Section 7 of that doc lists the five places the implementation deviates from the
spec and why (button captions that do not fit IBM Plex metrics, the status zone
being the header's remainder rather than a fixed 180 px, …). Extend that list
rather than silently drifting.

## Conventions

- Threads communicate via `queue.Queue[ASREvent]` only; UI updates cross into Qt via pyqtSignal (`post_status` is the thread-safe status entry).
- Every worker takes `stop_event: threading.Event`, is a daemon thread, and is joined with a timeout in main.py.
- Comments in code are English; user-facing docs Russian.
- Config keys map 1:1 to dataclasses in `src/app_config.py`; add new tunables there first, then to `config/config.toml` with a comment. The settings window writes back through `save_overrides`, which is line-based on purpose — it must not delete the Russian comments explaining each value.
- Every colour, size, spacing and duration comes from `ui/theme/tokens.py`; user-visible strings are Russian and human ("Ответить", not "Last ?").
