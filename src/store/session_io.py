from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from asr.events import Word

logger = logging.getLogger(__name__)

# A gap this long between two words starts a new paragraph in the exports.
# Speech pauses shorter than this are within a sentence far more often than
# between topics, and 2 s is what the ASR silence gate already treats as a break.
PARAGRAPH_GAP_SEC = 2.0


class SessionEntry:
    """One replayed word plus the wall clock at which it was first committed."""

    __slots__ = ("word", "received_at")

    def __init__(self, word: Word, received_at: float) -> None:
        self.word = word
        self.received_at = received_at


def read_session(path: Path | str) -> Tuple[dict, List[SessionEntry]]:
    """Replay a JSONL journal into (meta, entries).

    Patches are applied by index, exactly as the store applied them live, so
    the result carries the punctuation and casing the session ended with.
    Malformed or unknown lines are skipped with a warning rather than aborting
    the read: a journal truncated by a crash mid-line is still worth loading.
    """
    p = Path(path)
    meta: dict = {}
    entries: List[SessionEntry] = []
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("session %s: line %d is not valid JSON, skipped", p.name, lineno)
                continue
            kind = rec.get("t")
            if kind == "w":
                entries.append(
                    SessionEntry(
                        Word(
                            text=str(rec.get("text", "")),
                            start=float(rec.get("s", 0.0)),
                            end=float(rec.get("e", 0.0)),
                        ),
                        float(rec.get("at", 0.0)),
                    )
                )
            elif kind == "patch":
                i = rec.get("i")
                if isinstance(i, int) and 0 <= i < len(entries):
                    old = entries[i].word
                    entries[i].word = Word(
                        text=str(rec.get("text", old.text)), start=old.start, end=old.end
                    )
                else:
                    logger.warning(
                        "session %s: patch index %r out of range (%d words)",
                        p.name, i, len(entries),
                    )
            elif kind == "meta":
                meta = rec
    return meta, entries


def read_translations(path: Path | str) -> dict[int, tuple[int, str]]:
    """Every `tr` record of a session: `{first word index: (word count, text)}`.

    A separate pass rather than a third return value from `read_session`:
    translations are optional, a session written before the feature existed has
    none, and callers that only want words should not have to unpack an empty
    dict. Timestamps are not in the record — they are recovered from the words
    the same replay produces, which is why the index is the key.
    """
    p = Path(path)
    out: dict[int, tuple[int, str]] = {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or '"tr"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("t") != "tr":
                    continue
                index, n_words = rec.get("i"), rec.get("n")
                if not isinstance(index, int) or not isinstance(n_words, int):
                    continue
                text = str(rec.get("text", "")).strip()
                if text and n_words > 0:
                    out[index] = (n_words, text)
    except OSError as exc:
        logger.warning("session %s: translations not read (%s)", p.name, exc)
    return out


def latest_session(directory: Path | str) -> Path | None:
    """Newest *.jsonl in `directory`, or None if there is nothing to resume.

    Sorted by name, not mtime: file names are timestamps, and the current
    session's own file is touched constantly, which would win on mtime.
    """
    d = Path(directory)
    if not d.is_dir():
        return None
    files = sorted(d.glob("*.jsonl"))
    return files[-1] if files else None


def list_sessions(directory: Path | str) -> List[Path]:
    """All session files, newest first."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.jsonl"), reverse=True)


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def _blocks(words: List[Word]) -> List[Tuple[float, str, int, int]]:
    """Paragraphs as (start_sec, text, first index, word count).

    The indices are positions in `words` — the same absolute indices the store
    and the `tr` records use — so a translation can be matched to the paragraph
    it belongs to without guessing from timestamps.
    """
    blocks: List[Tuple[float, List[str], int]] = []
    prev_end: float | None = None
    for i, w in enumerate(words):
        if not w.text.strip():
            continue
        if prev_end is None or w.start - prev_end >= PARAGRAPH_GAP_SEC:
            blocks.append((w.start, [w.text], i))
        else:
            blocks[-1][1].append(w.text)
        prev_end = w.end
    return [
        (start, " ".join(parts).strip(), index, len(parts))
        for start, parts, index in blocks
    ]


def _paragraphs(words: List[Word]) -> List[Tuple[float, str]]:
    """Group words into (start_sec, text) blocks, breaking on speech pauses."""
    return [(start, text) for start, text, _i, _n in _blocks(words)]


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def export_text(words: List[Word], path: Path | str) -> Path:
    """Plain text: paragraphs only, no timestamps and no header."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(text for _, text in _paragraphs(words))
    p.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return p


def _translations_per_paragraph(
    blocks: List[Tuple[float, str, int, int]], translations: dict[int, str]
) -> List[str]:
    """Fold index-keyed translations onto paragraphs.

    The two groupings do not coincide — a paragraph breaks on a 2 s pause, a
    translated segment on a sentence end — so a paragraph collects every
    translation whose first word falls inside it, in order.
    """
    keys = sorted(translations)
    out: List[str] = []
    for _start, _text, index, count in blocks:
        parts = [translations[k] for k in keys if index <= k < index + count]
        out.append(" ".join(parts).strip())
    return out


def export_markdown(
    words: List[Word],
    path: Path | str,
    meta: dict | None = None,
    translations: dict[int, str] | None = None,
) -> Path:
    """Markdown: a header from the journal's meta, then `[mm:ss]` paragraphs.

    A translated session writes the translation as a blockquote under its
    paragraph — the original stays the document, the translation is a gloss.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    # The live overlay knows the engine but not when the journal opened (that
    # stamp is written by the writer), so fall back to now rather than emitting
    # a bare "# Транскрипт".
    started = str(meta.get("started") or datetime.now().isoformat(timespec="seconds"))
    started = started.replace("T", " ")

    lines = [f"# Транскрипт {started}".rstrip(), ""]
    details = []
    if meta.get("engine"):
        details.append(f"движок: `{meta['engine']}`")
    if meta.get("model"):
        details.append(f"модель: `{meta['model']}`")
    if meta.get("language"):
        details.append(f"язык: `{meta['language']}`")
    if meta.get("resumed_from"):
        details.append(f"продолжение: `{meta['resumed_from']}`")
    if translations and meta.get("translate_target"):
        details.append(f"перевод: `{meta['translate_target']}`")
    details.append(f"слов: {sum(1 for w in words if w.text.strip())}")
    lines += [" · ".join(details), "", "---", ""]

    blocks = _blocks(words)
    glosses = _translations_per_paragraph(blocks, translations or {})
    for (start, text, _i, _n), gloss in zip(blocks, glosses):
        lines.append(f"**[{_clock(start)}]** {text}")
        if gloss:
            lines.append("")
            lines.append(f"> {gloss}")
        lines.append("")

    lines.append(f"<!-- экспортировано Speecher {datetime.now().isoformat(timespec='seconds')} -->")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def suggest_export_name(meta: dict | None, source: Path | None, suffix: str) -> str:
    """Default file name for an export dialog: the session stamp plus `suffix`."""
    if source is not None:
        return f"{source.stem}.{suffix}"
    started = str((meta or {}).get("started", "")) or datetime.now().isoformat()
    stem = started.replace(":", "").replace("-", "").replace("T", "_")[:15]
    return f"{stem}.{suffix}"
