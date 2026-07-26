"""Markdown as models actually emit it, rendered into the card's rich text.

Every chat model writes `**жирным**`, ``` `кодом` ``` and `- списками` whether
asked to or not — it is how they were trained, and telling them not to in the
system prompt works about half the time. So the asterisks are dealt with here,
where the result is deterministic.

Deliberately *not* a CommonMark implementation. This covers the subset that
shows up in an answer to a spoken question — emphasis, inline code, fenced
blocks, bullet and numbered lists, headings, quotes, links, rules — and leaves
anything else as the literal text the model wrote. Tables, footnotes, nested
lists and reference links are not worth the parser in a 520 px card.

Two rules follow from the card being *streamed*:

- An unclosed marker stays literal. `**жир` renders as `**жир` until its closing
  `**` arrives, and only then flips to bold. Guessing at half a marker would
  make the text jump on every token.
- The output is re-rendered from the whole accumulated string each time, so the
  function must be pure and cheap. It is: one pass over the lines, a handful of
  regexes per line.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ui.theme.fonts import family
from ui.theme.tokens import FAMILY_MONO, Color, Radius, Space

# -- block-level shapes -------------------------------------------------

_FENCE = re.compile(r"^\s*(?:```|~~~)\s*[\w+-]*\s*$")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.+)$")
_ORDERED = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.+)$")
_RULE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")

# -- inline shapes ------------------------------------------------------
#
# Order of application matters and is enforced by _inline: code spans are
# pulled out first (so `**` inside them stays literal), then the two-character
# markers, then the one-character ones. `(?=\S)…(?<=\S)` is what keeps
# "5 * 3 * 2" and snake_case_names from turning into emphasis.

_CODE = re.compile(r"`([^`\n]+)`")
_BOLD_ITALIC = re.compile(r"(\*\*\*|___)(?=\S)(.+?)(?<=\S)\1", re.S)
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.S)
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.S)
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_ITALIC_UNDER = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_LINK = re.compile(r"\[([^\]\n]+)\]\(\s*([^)\s]+)[^)]*\)")

_INDENT_PX = Space.S4
_ITEM_GAP_PX = Space.S1
_PARA_GAP_PX = Space.S2

# A single placeholder character (U+0000 is impossible in model output) keeps
# code spans out of reach of the emphasis regexes without changing offsets in a
# way that could split a marker.
_SENTINEL = "\x00"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _code_span(text: str) -> str:
    return (
        f'<span style="font-family: \'{family(FAMILY_MONO)}\';'
        f" color: {Color.ACCENT_ON}; background: {Color.ACCENT_SUBTLE};\">"
        f"&nbsp;{text}&nbsp;</span>"
    )


def _inline(text: str) -> str:
    """Emphasis, code and links inside one line of text."""
    spans: List[str] = []

    def stash(match: re.Match) -> str:
        spans.append(_code_span(_esc(match.group(1))))
        return _SENTINEL

    # Code first: backticks win over asterisks, so `a**b**c` stays literal.
    text = _CODE.sub(stash, text)
    text = _esc(text)
    text = _BOLD_ITALIC.sub(r"<b><i>\2</i></b>", text)
    text = _BOLD.sub(r"<b>\2</b>", text)
    text = _STRIKE.sub(r"<s>\1</s>", text)
    text = _ITALIC_STAR.sub(r"<i>\1</i>", text)
    text = _ITALIC_UNDER.sub(r"<i>\1</i>", text)
    # The URL is dropped, not linkified: the card is not a browser and a click
    # in it would do nothing. The link text is what the reader needs.
    text = _LINK.sub(r"\1", text)

    for span in spans:
        text = text.replace(_SENTINEL, span, 1)
    return text


def _block(html: str, *, margin_left: int = 0, margin_top: int = 0) -> str:
    style = f"margin: {margin_top}px 0 0 {margin_left}px;"
    return f'<div style="{style}">{html}</div>'


def to_html(text: str) -> str:
    """Render a model's answer. Returns rich text for a QTextEdit.

    Headings come out as bold text in the body's own type scale rather than
    markdown's default sizes: a card 340 px tall has no room for an `<h1>`, and
    the widget's style is set by the caller anyway.
    """
    if not text.strip():
        return ""

    out: List[str] = []
    para: List[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def flush() -> None:
        if para:
            out.append(_block("<br>".join(para), margin_top=_PARA_GAP_PX if out else 0))
            para.clear()

    i = 0
    first = True
    while i < len(lines):
        line = lines[i]

        if _FENCE.match(line):
            flush()
            body, i = _read_fence(lines, i + 1)
            out.append(_code_block(body, top=0 if first else _PARA_GAP_PX))
            first = False
            continue

        if not line.strip():
            flush()
            i += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            out.append(
                _block(
                    f"<b>{_inline(heading.group(2))}</b>",
                    margin_top=0 if first else _PARA_GAP_PX,
                )
            )
            first = False
            i += 1
            continue

        if _RULE.match(line):
            flush()
            out.append(
                f'<div style="margin: {_PARA_GAP_PX}px 0 0 0;'
                f' border-top: 1px solid {Color.BORDER_STRONG};"></div>'
            )
            first = False
            i += 1
            continue

        quote = _QUOTE.match(line)
        if quote:
            flush()
            out.append(
                _block(
                    f'<span style="color: {Color.TEXT_SECONDARY};">'
                    f"{_inline(quote.group(1))}</span>",
                    margin_left=_INDENT_PX,
                    margin_top=_ITEM_GAP_PX,
                )
            )
            first = False
            i += 1
            continue

        item = _list_item(line)
        if item is not None:
            flush()
            out.append(_block(item, margin_left=_INDENT_PX, margin_top=_ITEM_GAP_PX))
            first = False
            i += 1
            continue

        para.append(_inline(line))
        first = False
        i += 1

    flush()
    return "".join(out)


def _list_item(line: str) -> Optional[str]:
    """A bullet or a number, rendered with a real "•" and a hanging indent."""
    bullet = _BULLET.match(line)
    if bullet:
        marker = "•"
        body = bullet.group(2)
    else:
        ordered = _ORDERED.match(line)
        if not ordered:
            return None
        marker = f"{ordered.group(2)}."
        body = ordered.group(3)
    return (
        f'<span style="color: {Color.TEXT_SECONDARY};">{marker}</span>&nbsp;'
        f"{_inline(body)}"
    )


def _read_fence(lines: List[str], start: int) -> Tuple[List[str], int]:
    """Collect a fenced block. An unterminated fence runs to the end — while the
    answer is still streaming, that is exactly where it is."""
    body: List[str] = []
    i = start
    while i < len(lines) and not _FENCE.match(lines[i]):
        body.append(lines[i])
        i += 1
    return body, min(i + 1, len(lines))


def _code_block(body: List[str], *, top: int) -> str:
    inner = "<br>".join(_esc(line).replace(" ", "&nbsp;") or "&nbsp;" for line in body)
    return (
        f'<div style="margin: {top}px 0 0 0; background: {Color.ACCENT_SUBTLE};'
        f" border-radius: {Radius.SMALL}px;\">"
        f'<span style="font-family: \'{family(FAMILY_MONO)}\'; color: {Color.ACCENT_ON};">'
        f"&nbsp;{inner}&nbsp;</span></div>"
    )


def to_plain(text: str) -> str:
    """The same subset stripped rather than rendered — for the console and the
    clipboard, where the markers are noise and there is nothing to style."""
    out: List[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if _FENCE.match(line) or _RULE.match(line):
            continue
        heading = _HEADING.match(line)
        if heading:
            out.append(_strip_inline(heading.group(2)))
            continue
        bullet = _BULLET.match(line)
        if bullet:
            out.append(f"• {_strip_inline(bullet.group(2))}")
            continue
        ordered = _ORDERED.match(line)
        if ordered:
            out.append(f"{ordered.group(2)}. {_strip_inline(ordered.group(3))}")
            continue
        out.append(_strip_inline(line))
    return "\n".join(out)


def _strip_inline(text: str) -> str:
    text = _CODE.sub(r"\1", text)
    text = _BOLD_ITALIC.sub(r"\2", text)
    text = _BOLD.sub(r"\2", text)
    text = _STRIKE.sub(r"\1", text)
    text = _ITALIC_STAR.sub(r"\1", text)
    text = _ITALIC_UNDER.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    return text
