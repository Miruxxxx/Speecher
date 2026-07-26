from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from translate import languages
from translate.backends import KNOWN_BACKENDS as KNOWN_TRANSLATE_BACKENDS

logger = logging.getLogger(__name__)

# Backends `create_asr_backend` knows how to build. Anything else falls back
# to "whisper" with a warning instead of crashing at startup.
KNOWN_ASR_ENGINES = ("whisper", "nemotron")

# What the capture binary can record: system output or a recording endpoint.
KNOWN_CAPTURE_SOURCES = ("loopback", "mic")

# Right attention contexts the Nemotron checkpoint was trained with, in
# subsampled encoder frames. Latency is (n + 1) * 80 ms.
NEMOTRON_LOOKAHEAD_TOKENS = (0, 3, 6, 13)
NEMOTRON_DTYPES = ("float32", "float16", "bfloat16")



@dataclass(slots=True)
class AudioConfig:
    # Substring of the *output* endpoint's name (the capture binary opens it in
    # loopback mode); empty picks the default endpoint.
    device_hint: str = "HyperX"
    target_sample_rate: int = 16000
    restart_backoff_sec: float = 2.0
    # "mic" is groundwork for the two-track mode (docs/ROADMAP.md stage 4),
    # not a finished feature.
    source: str = "loopback"
    # "" = native/audio_capture/target/release/audio_capture.exe in the tree.
    binary: str = ""


@dataclass(slots=True)
class AsrFiltersConfig:
    # Segments with no_speech_prob above / avg_logprob below these are dropped.
    no_speech_prob_max: float = 0.9
    avg_logprob_min: float = -1.5
    # Collapse runs of the same normalized word longer than this (hallucination loops).
    max_word_repeats: int = 6
    # Snapshot RMS below this counts as silence: Whisper is not called at all.
    silence_rms_threshold: float = 1e-4
    # After this many consecutive empty decodes the buffer is trimmed to the tail.
    empty_decodes_before_trim: int = 3
    silence_keep_tail_sec: float = 2.0


@dataclass(slots=True)
class NemotronConfig:
    """Settings for the `nemotron` engine (src/asr/nemotron_backend.py)."""

    model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    device: str = "cuda"            # cuda | cpu
    # float32 by default on purpose: fp16 halves VRAM but decodes ~2x slower
    # (greedy RNN-T is a loop over frames, conversion overhead dominates).
    dtype: str = "float32"          # float32 | float16 | bfloat16
    # Right attention context; latency is (n + 1) * 80 ms.
    lookahead_tokens: int = 6
    # "auto" lets the model detect the language per stream; a locale ("ru",
    # "en-US", ...) pins the prompt. Unknown values raise at load().
    language: str = "auto"


@dataclass(slots=True)
class AsrConfig:
    # Which ASR backend runs the recognition loop (see src/asr/backends.py).
    engine: str = "whisper"
    model: str = "large-v3-turbo"
    device: str = "cuda"            # cuda | cpu
    compute_type: str = "float16"   # float16 | int8 | auto
    # "auto" = sticky: detect once from the first confident decode, then pin.
    # "" = re-detect every cycle (old behaviour). "ru"/"en"/... = fixed.
    language: str = "auto"
    sticky_min_probability: float = 0.7
    beam_size: int = 5
    vad_min_silence_ms: int = 500
    chunk_interval_sec: float = 1.0
    min_audio_sec: float = 1.0
    commit_safety_margin_sec: float = 0.1
    max_buffer_sec: float = 25.0
    force_commit_keep_tail_words: int = 6
    buffer_max_seconds: float = 60.0
    filters: AsrFiltersConfig = field(default_factory=AsrFiltersConfig)
    nemotron: NemotronConfig = field(default_factory=NemotronConfig)


@dataclass(slots=True)
class PunctuationConfig:
    backend: str = "silero"   # silero | off (unknown → off with a warning)
    language: str = "ru"      # silero supports ru/en/de/es
    interval_sec: float = 12.0
    context_words: int = 80


@dataclass(slots=True)
class LlmConfig:
    # Which entry of src/llm/providers.py to talk to. Local servers need no
    # key; a cloud provider's key lives outside this file entirely (it is in
    # git) — see src/llm/credentials.py.
    provider: str = "lmstudio"
    # "" = the provider's own URL from the registry. Only `custom` requires it,
    # but any provider can be pointed at a proxy this way.
    base_url: str = ""
    # The model to use. "" is resolved from /models for a *local* server only:
    # picking one of a cloud provider's hundreds at random would answer with
    # something arbitrary and bill for it.
    model: str = ""
    # Per-provider memory, written by the settings window so switching provider
    # and back does not lose the choice. Consulted only when `model` is empty,
    # so a hand-edited `model` always wins.
    models: dict = field(default_factory=dict)
    temperature: float = 0.6
    max_tokens: int = 2048
    request_timeout_sec: float = 120.0
    health_timeout_sec: float = 2.0
    # Cap on transcript chars sent in a summary prompt: past this the oldest
    # text is dropped so we never silently overflow a small local context.
    max_summary_chars: int = 6000
    # How much chain-of-thought the model may spend before answering, sent as
    # the OpenAI-compatible `reasoning_effort`. "none" disables thinking
    # entirely; "" omits the field (for servers/models that reject it).
    # Measured against LM Studio + qwen3.5-9b (2026-07-24): only "none" works —
    # "minimal", `chat_template_kwargs.enable_thinking=false` and the old
    # `/no_think` prompt prefix all still produced pure reasoning, which ate
    # the whole max_tokens budget and returned an EMPTY answer.
    reasoning_effort: str = "none"


@dataclass(slots=True)
class TranslateConfig:
    """Live translation (src/translate). Off by default: it loads a second
    model onto the same GPU the ASR engine is using."""

    # nllb (любой язык одной моделью) | opusmt (одна пара, быстрее) | off
    backend: str = "nllb"
    # Whether translation is on when the app starts; Alt+T flips it for the
    # session without touching this file (same contract as the ⏺ toggle).
    enabled: bool = False
    # Target language code from src/translate/languages.py.
    target: str = "ru"
    # Show the original transcript in its own small window while translating.
    # The main feed deliberately holds only finished lines, so without this
    # there is a couple of seconds where nothing visibly happens.
    show_source: bool = True
    # "auto" = take the ASR's language, or guess from the script.
    source: str = "auto"
    # "" = the backend's own default checkpoint.
    model: str = ""
    device: str = "cuda"            # cuda | cpu
    dtype: str = "float16"          # float16 | bfloat16 | float32 (cpu forces float32)
    # The trailing words are cut into a line this long after the last one;
    # until then they may still grow and the line would have to change.
    close_after_sec: float = 1.5
    # Silence between two words that ends a segment even mid-sentence.
    pause_sec: float = 1.2
    # Hard cap so speech without punctuation or pauses still reaches the screen.
    max_words_per_segment: int = 40
    # Segments allowed to wait before the worker merges them into one longer
    # line to catch up (never dropped: the feed must stay in order).
    max_pending: int = 4
    poll_sec: float = 0.4
    max_chars_per_chunk: int = 220
    max_new_tokens: int = 256
    # Skip a reply already written in the target language's script. Only used
    # when the source language is unknown — see TranslationWorker._same_language.
    skip_same_script: bool = True


@dataclass(slots=True)
class UiConfig:
    max_recent_chars: int = 700
    console_verbose_logs: bool = False
    # Which of the two layout modes the window opens in (design system 3).
    compact: bool = False
    # Interval the "Выжимка" button and Alt+5 summarise, in minutes. The button
    # caption is generated from this value, never written by hand (6.6);
    # 0 means "не задан" and the button says so.
    summary_interval_min: int = 5
    # Run that summary automatically every interval and announce it.
    auto_summary: bool = False
    # Flipped to true after the onboarding window has been shown once (6.7).
    onboarding_shown: bool = False


@dataclass(slots=True)
class HotkeysConfig:
    """Global hotkeys (design system 5). Empty value = not registered.

    Alt+Space is show/hide and nothing else; the ready-summary notification
    uses Alt+H instead, which is how the mock-up's collision is resolved (6.4).
    """

    toggle_window: str = "Alt+Space"
    toggle_recording: str = "Alt+R"
    pause_recording: str = "Alt+P"
    summary: str = "Alt+5"
    answer: str = "Alt+Enter"
    last_question: str = "Alt+Q"
    history: str = "Alt+H"
    translate: str = "Alt+T"


@dataclass(slots=True)
class StorageConfig:
    """Transcript persistence (src/store/transcript_writer.py)."""

    # Whether recording is on at startup. The overlay's ⏺ toggle flips it for
    # the running session without touching this file.
    enabled: bool = True
    # Relative paths resolve against the project root, not the CWD.
    dir: str = "storage/transcripts"


@dataclass(slots=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    punctuation: PunctuationConfig = field(default_factory=PunctuationConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)


def _apply_section(obj: Any, data: dict, path: str) -> None:
    """Overlay a TOML table onto a config dataclass, coercing to field types."""
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            logger.warning("config: unknown key %s.%s ignored", path, key)
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if isinstance(value, dict):
                _apply_section(current, value, f"{path}.{key}")
            else:
                logger.warning("config: %s.%s must be a table; ignored", path, key)
            continue
        try:
            setattr(obj, key, type(current)(value))
        except (TypeError, ValueError):
            logger.warning(
                "config: %s.%s=%r has wrong type (expected %s); keeping default %r",
                path, key, value, type(current).__name__, current,
            )


def load_config(path: str | Path) -> AppConfig:
    """Load config/config.toml over built-in defaults. Missing file → defaults."""
    cfg = AppConfig()
    p = Path(path)
    if not p.exists():
        logger.info("config: %s not found, using defaults", p)
        return cfg
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.error("config: failed to read %s (%s); using defaults", p, exc)
        return cfg
    _apply_section(cfg, data, "config")
    _normalize(cfg)
    return cfg


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def save_overrides(path: str | Path, updates: dict[str, dict[str, Any]]) -> None:
    """Write `{"section": {"key": value}}` back into config.toml in place.

    Line-based on purpose: the file is the user's, full of Russian comments that
    explain why each value is what it is, and a round-trip through a TOML
    serialiser would delete every one of them. Only the value part of a matched
    key is replaced; an unknown key is appended to its section, an unknown
    section to the end of the file.

    Raises OSError if the file cannot be written — the caller reports it.
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    remaining = {sec: dict(keys) for sec, keys in updates.items() if keys}

    out: list[str] = []
    section = ""
    # Where each section ends, so a new key lands inside it rather than under
    # the next header.
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            _flush_section(out, section, remaining)
            section = stripped[1:-1].strip()
            out.append(raw)
            continue
        keys = remaining.get(section)
        if keys and "=" in raw and not stripped.startswith("#"):
            name = raw.split("=", 1)[0].strip()
            if name in keys:
                comment = ""
                after = raw.split("=", 1)[1]
                # Keep a trailing comment; '#' inside a quoted value is not one.
                hash_pos = _comment_pos(after)
                if hash_pos >= 0:
                    comment = "  " + after[hash_pos:].strip()
                out.append(f"{name} = {_format_value(keys.pop(name))}{comment}")
                continue
        out.append(raw)
    _flush_section(out, section, remaining)

    for section, keys in remaining.items():
        if not keys:
            continue
        if out and out[-1].strip():
            out.append("")
        out.append(f"[{section}]")
        for name, value in keys.items():
            out.append(f"{name} = {_format_value(value)}")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def _flush_section(out: list[str], section: str, remaining: dict[str, dict]) -> None:
    """Append whatever is still unwritten for `section` before leaving it."""
    keys = remaining.get(section)
    if not keys:
        return
    while out and not out[-1].strip():
        out.pop()
    for name, value in keys.items():
        out.append(f"{name} = {_format_value(value)}")
    out.append("")
    keys.clear()


def _comment_pos(text: str) -> int:
    in_string = False
    for i, ch in enumerate(text):
        if ch == '"':
            in_string = not in_string
        elif ch == "#" and not in_string:
            return i
    return -1


def _normalize(cfg: AppConfig) -> None:
    """Fix up values that are the right type but not a valid choice."""
    source = cfg.audio.source.strip().lower()
    if source not in KNOWN_CAPTURE_SOURCES:
        logger.warning(
            "config: audio.source=%r is unknown (known: %s); using 'loopback'",
            cfg.audio.source, ", ".join(KNOWN_CAPTURE_SOURCES),
        )
        source = "loopback"
    cfg.audio.source = source

    engine = cfg.asr.engine.strip().lower()
    if engine not in KNOWN_ASR_ENGINES:
        logger.warning(
            "config: asr.engine=%r is unknown (known: %s); using 'whisper'",
            cfg.asr.engine, ", ".join(KNOWN_ASR_ENGINES),
        )
        engine = "whisper"
    cfg.asr.engine = engine

    nem = cfg.asr.nemotron
    if nem.lookahead_tokens not in NEMOTRON_LOOKAHEAD_TOKENS:
        logger.warning(
            "config: asr.nemotron.lookahead_tokens=%r is not supported (allowed: %s); using 6",
            nem.lookahead_tokens, ", ".join(str(v) for v in NEMOTRON_LOOKAHEAD_TOKENS),
        )
        nem.lookahead_tokens = 6
    dtype = nem.dtype.strip().lower()
    if dtype not in NEMOTRON_DTYPES:
        logger.warning(
            "config: asr.nemotron.dtype=%r is unknown (allowed: %s); using 'float32'",
            nem.dtype, ", ".join(NEMOTRON_DTYPES),
        )
        dtype = "float32"
    nem.dtype = dtype

    _normalize_llm(cfg.llm)
    _normalize_translate(cfg.translate)


def _normalize_llm(llm: "LlmConfig") -> None:
    # Imported here: providers.py must stay importable without the config, and
    # the config must not drag the openai client in at load time.
    from llm import providers

    provider = llm.provider.strip().lower()
    if providers.get(provider) is None:
        logger.warning(
            "config: llm.provider=%r is unknown (known: %s); using %r",
            llm.provider, ", ".join(providers.ids()), providers.DEFAULT_PROVIDER,
        )
        provider = providers.DEFAULT_PROVIDER
    llm.provider = provider

    llm.models = {
        str(k): str(v).strip() for k, v in (llm.models or {}).items() if str(v).strip()
    }
    if not llm.model.strip():
        llm.model = llm.models.get(provider, "")


def _normalize_translate(tr: "TranslateConfig") -> None:
    """A translation setting that makes no sense turns the feature off or falls
    back — never raises. It is an extra, and the transcript must run without it."""
    backend = tr.backend.strip().lower()
    if backend not in KNOWN_TRANSLATE_BACKENDS:
        logger.warning(
            "config: translate.backend=%r is unknown (known: %s); translation off",
            tr.backend, ", ".join(KNOWN_TRANSLATE_BACKENDS),
        )
        backend = "off"
    tr.backend = backend

    target = languages.normalize_code(tr.target)
    if not target:
        logger.warning(
            "config: translate.target=%r is not a supported language (known: %s); using 'ru'",
            tr.target, ", ".join(languages.codes()),
        )
        target = "ru"
    tr.target = target

    # "auto" is a valid source and normalizes to "": the worker then asks the
    # ASR language, and finally the script of the text itself.
    source = tr.source.strip().lower()
    if source not in ("", "auto"):
        normalized = languages.normalize_code(source)
        if not normalized:
            logger.warning(
                "config: translate.source=%r is not a supported language; using 'auto'",
                tr.source,
            )
        source = normalized or "auto"
    tr.source = source or "auto"

    dtype = tr.dtype.strip().lower()
    if dtype not in NEMOTRON_DTYPES:
        logger.warning(
            "config: translate.dtype=%r is unknown (allowed: %s); using 'float16'",
            tr.dtype, ", ".join(NEMOTRON_DTYPES),
        )
        dtype = "float16"
    tr.dtype = dtype

    if tr.backend == "opusmt" and tr.source == "auto":
        # A Marian checkpoint *is* a direction; there is nothing to load without
        # both ends of it.
        logger.warning(
            "config: translate.backend='opusmt' needs a fixed translate.source "
            "(a checkpoint exists per pair); translation off"
        )
        tr.backend = "off"
