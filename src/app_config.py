from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Backends `create_asr_backend` knows how to build. Anything else falls back
# to "whisper" with a warning instead of crashing at startup.
KNOWN_ASR_ENGINES = ("whisper", "nemotron")

# Right attention contexts the Nemotron checkpoint was trained with, in
# subsampled encoder frames. Latency is (n + 1) * 80 ms.
NEMOTRON_LOOKAHEAD_TOKENS = (0, 3, 6, 13)
NEMOTRON_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(slots=True)
class AudioConfig:
    device_hint: str = "HyperX"
    target_sample_rate: int = 16000
    frames_per_buffer: int = 2048
    max_channels: int = 2
    restart_backoff_sec: float = 2.0


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
    base_url: str = "http://localhost:1234/v1"
    # "" = use whatever model is loaded in LM Studio (first from /models).
    model: str = ""
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
class UiConfig:
    max_recent_chars: int = 700
    console_verbose_logs: bool = False


@dataclass(slots=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    punctuation: PunctuationConfig = field(default_factory=PunctuationConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    ui: UiConfig = field(default_factory=UiConfig)


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


def _normalize(cfg: AppConfig) -> None:
    """Fix up values that are the right type but not a valid choice."""
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
