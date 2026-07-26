"""The languages the translator offers, and the script heuristic it leans on.

Twelve, not two hundred: the settings window shows a list a human reads, and
NLLB's full FLORES-200 table would turn a field into a phone book. Adding one is
a single row here — the config, the settings dropdown and the backends all read
this table.

Qt-free and torch-free on purpose: `app_config` validates against it at load
time, and the tests use it without importing either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Unicode ranges wide enough to tell the alphabets in the table apart, and
# nothing finer: this decides "is that text already in the target language",
# not "which language is it".
_SCRIPT_RANGES: Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...] = (
    ("cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("han", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF))),
    ("latin", ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))),
)


@dataclass(frozen=True, slots=True)
class Language:
    code: str       # what the config and the journal store
    caption: str    # what the settings window shows (Russian, lowercase)
    flores: str     # NLLB's FLORES-200 code
    script: str


LANGUAGES: Tuple[Language, ...] = (
    Language("ru", "русский", "rus_Cyrl", "cyrillic"),
    Language("en", "английский", "eng_Latn", "latin"),
    Language("de", "немецкий", "deu_Latn", "latin"),
    Language("es", "испанский", "spa_Latn", "latin"),
    Language("fr", "французский", "fra_Latn", "latin"),
    Language("it", "итальянский", "ita_Latn", "latin"),
    Language("pt", "португальский", "por_Latn", "latin"),
    Language("cs", "чешский", "ces_Latn", "latin"),
    Language("pl", "польский", "pol_Latn", "latin"),
    Language("uk", "украинский", "ukr_Cyrl", "cyrillic"),
    Language("zh", "китайский", "zho_Hans", "han"),
    Language("ja", "японский", "jpn_Jpan", "kana"),
)

_BY_CODE: Dict[str, Language] = {lang.code: lang for lang in LANGUAGES}

# Codes accepted in config but not in the table above (ASR speaks locales).
_ALIASES = {"en-us": "en", "en-gb": "en", "pt-br": "pt", "zh-cn": "zh", "zh-hans": "zh"}


def normalize_code(code: str) -> str:
    """"en-US" -> "en"; unknown values come back as "" so callers can warn."""
    raw = (code or "").strip().lower().replace("_", "-")
    if raw in _BY_CODE:
        return raw
    if raw in _ALIASES:
        return _ALIASES[raw]
    head = raw.split("-", 1)[0]
    return head if head in _BY_CODE else ""


def get(code: str) -> Optional[Language]:
    return _BY_CODE.get(normalize_code(code))


def caption(code: str) -> str:
    lang = get(code)
    return lang.caption if lang else code


def codes() -> List[str]:
    return [lang.code for lang in LANGUAGES]


def detect_script(text: str) -> str:
    """The script most of `text` is written in, or "" if it is all digits/punctuation."""
    counts: Dict[str, int] = {}
    for ch in text:
        point = ord(ch)
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= point <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def looks_like(text: str, code: str) -> bool:
    """True when `text` is already written in `code`'s script.

    A heuristic and named as one: it separates Cyrillic from Latin, which is the
    case that matters (do not spend a GPU pass translating Russian into
    Russian). It cannot tell German from English, which is why the worker only
    trusts it when the source language is "auto".
    """
    lang = get(code)
    if lang is None:
        return False
    script = detect_script(text)
    return bool(script) and script == lang.script
