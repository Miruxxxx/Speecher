from pathlib import Path

from app_config import AppConfig, load_config, save_overrides


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


def test_capture_source_is_normalized(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[audio]\nsource = "MIC"\n', encoding="utf-8")
    assert load_config(p).audio.source == "mic"


def test_unknown_capture_source_falls_back(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[audio]\nsource = "speakers"\n', encoding="utf-8")
    assert load_config(p).audio.source == "loopback"


def test_broken_toml_falls_back_to_defaults(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[asr\nmodel=", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.asr.model == AppConfig().asr.model


# --- save_overrides: the settings window writes back into the user's file ---


def test_save_overrides_keeps_comments_and_layout(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        """# заголовок файла
[ui]
max_recent_chars = 700
summary_interval_min = 5      # интервал выжимки

[storage]
enabled = true
""",
        encoding="utf-8",
    )
    save_overrides(p, {"ui": {"summary_interval_min": 10, "auto_summary": True}})
    text = p.read_text(encoding="utf-8")

    assert "# заголовок файла" in text
    assert "summary_interval_min = 10" in text
    assert "# интервал выжимки" in text
    assert "max_recent_chars = 700" in text
    cfg = load_config(p)
    assert cfg.ui.summary_interval_min == 10
    assert cfg.ui.auto_summary is True
    assert cfg.storage.enabled is True


def test_save_overrides_adds_a_missing_section(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[ui]\nmax_recent_chars = 700\n", encoding="utf-8")
    save_overrides(p, {"hotkeys": {"summary": "Alt+7"}})
    assert load_config(p).hotkeys.summary == "Alt+7"
    assert load_config(p).ui.max_recent_chars == 700


def test_save_overrides_writes_nested_tables(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[asr.nemotron]\nlanguage = "auto"\n', encoding="utf-8")
    save_overrides(p, {"asr.nemotron": {"language": "ru"}})
    assert load_config(p).asr.nemotron.language == "ru"


def test_save_overrides_quotes_and_escapes_strings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[audio]\ndevice_hint = \"\"\n", encoding="utf-8")
    save_overrides(p, {"audio": {"device_hint": 'Наушники "HyperX"'}})
    assert load_config(p).audio.device_hint == 'Наушники "HyperX"'


def test_save_overrides_creates_the_file(tmp_path: Path):
    p = tmp_path / "sub" / "config.toml"
    save_overrides(p, {"ui": {"compact": True}})
    assert load_config(p).ui.compact is True


def test_translate_defaults_are_off_but_valid():
    cfg = AppConfig()
    assert cfg.translate.enabled is False
    assert cfg.translate.target == "ru"
    assert cfg.translate.show_source is True
    assert cfg.hotkeys.translate == "Alt+T"


def test_unknown_translate_values_fall_back(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[translate]\nbackend = "deepl"\ntarget = "klingon"\n'
        'source = "elvish"\nmode = "beside"\ndtype = "int4"\n',
        encoding="utf-8",
    )
    tr = load_config(p).translate
    assert tr.backend == "off"
    assert tr.target == "ru"
    assert tr.source == "auto"
    assert tr.dtype == "float16"


def test_opusmt_without_a_source_language_turns_translation_off(tmp_path: Path):
    """A Marian checkpoint is a direction; "auto" cannot pick one."""
    p = tmp_path / "config.toml"
    p.write_text('[translate]\nbackend = "opusmt"\nsource = "auto"\n', encoding="utf-8")
    assert load_config(p).translate.backend == "off"

    p.write_text(
        '[translate]\nbackend = "opusmt"\nsource = "en"\ntarget = "ru"\n', encoding="utf-8"
    )
    assert load_config(p).translate.backend == "opusmt"


def test_asr_locales_are_accepted_as_translate_languages(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[translate]\ntarget = "en-US"\nsource = "RU"\n', encoding="utf-8")
    tr = load_config(p).translate
    assert tr.target == "en"
    assert tr.source == "ru"
