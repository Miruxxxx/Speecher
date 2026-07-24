from pathlib import Path

from app_config import AppConfig, load_config


def test_missing_file_returns_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.asr.model == AppConfig().asr.model
    assert cfg.punctuation.backend == "silero"


def test_partial_override_keeps_other_defaults(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        """
[asr]
model = "small"
language = "ru"

[asr.filters]
max_word_repeats = 3

[llm]
base_url = "http://127.0.0.1:9999/v1"
""",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.asr.model == "small"
    assert cfg.asr.language == "ru"
    assert cfg.asr.filters.max_word_repeats == 3
    # untouched values keep defaults
    assert cfg.asr.beam_size == 5
    assert cfg.asr.filters.no_speech_prob_max == AppConfig().asr.filters.no_speech_prob_max
    assert cfg.llm.base_url == "http://127.0.0.1:9999/v1"
    assert cfg.audio.device_hint == "HyperX"


def test_wrong_type_keeps_default_and_unknown_keys_ignored(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        """
[asr]
beam_size = "many"
totally_unknown = 1

[nonexistent_section]
x = 2
""",
        encoding="utf-8",
    )
    cfg = load_config(p)  # must not raise
    assert cfg.asr.beam_size == 5


def test_numeric_coercion_int_to_float(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[asr]\nchunk_interval_sec = 2\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.asr.chunk_interval_sec == 2.0
    assert isinstance(cfg.asr.chunk_interval_sec, float)


def test_capture_backend_is_normalized(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[audio]\nbackend = "RUST"\nsource = "MIC"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.audio.backend == "rust"
    assert cfg.audio.source == "mic"


def test_unknown_capture_backend_falls_back(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[audio]\nbackend = "cpal"\nsource = "speakers"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.audio.backend == "pyaudio"
    assert cfg.audio.source == "loopback"


def test_broken_toml_falls_back_to_defaults(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[asr\nmodel=", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.asr.model == AppConfig().asr.model
