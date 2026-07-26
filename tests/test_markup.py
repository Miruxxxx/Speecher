"""Markdown that models actually emit, and the text that must survive untouched.

Half of these are negative: an asterisk in speech, a snake_case identifier and a
multiplication sign are not emphasis, and a card that decides otherwise mangles
the answer. The other half is the streaming contract — a half-typed `**` stays
literal until its pair arrives.
"""

from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui import markup  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    # markup resolves the mono family through the font database.
    app = QApplication.instance() or QApplication([])
    yield app


def text_of(html: str) -> str:
    """The rendered text with the tags removed — what the reader sees."""
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ").replace("&amp;", "&")


# ----------------------------------------------------------------------
# Emphasis
# ----------------------------------------------------------------------


def test_bold_becomes_bold_and_loses_its_asterisks():
    html = markup.to_html("Это **важное** слово")
    assert "<b>важное</b>" in html
    assert "**" not in html
    assert text_of(html) == "Это важное слово"


def test_italic_underscore_and_star():
    assert "<i>раз</i>" in markup.to_html("*раз* и всё")
    assert "<i>два</i>" in markup.to_html("_два_ и всё")


def test_bold_italic_triple():
    html = markup.to_html("***очень*** важно")
    assert "<b><i>очень</i></b>" in html


def test_strikethrough():
    assert "<s>нет</s>" in markup.to_html("~~нет~~ да")


@pytest.mark.parametrize(
    "text",
    [
        "5 * 3 * 2 = 30",
        "snake_case_name и another_one",
        "звёздочка * одна",
        "3*4",
        "путь/к_файлу_тут",
    ],
)
def test_text_that_only_looks_like_emphasis_is_left_alone(text: str):
    """Speech transcripts are full of these; turning them into italics is worse
    than showing an asterisk."""
    html = markup.to_html(text)
    assert "<i>" not in html
    assert "<b>" not in html
    assert text_of(html) == text


def test_an_unclosed_marker_stays_literal_while_streaming():
    """The card re-renders on every token: guessing at half a marker would make
    the text jump."""
    assert "<b>" not in markup.to_html("Это **важ")
    assert "**важ" in text_of(markup.to_html("Это **важ"))
    # …and flips the moment the pair arrives.
    assert "<b>важно</b>" in markup.to_html("Это **важно**")


# ----------------------------------------------------------------------
# Code
# ----------------------------------------------------------------------


def test_inline_code_is_monospaced_and_keeps_its_content():
    html = markup.to_html("Вызови `store.add(x)` потом")
    assert "store.add(x)" in text_of(html)
    assert "`" not in text_of(html)


def test_asterisks_inside_code_are_not_emphasis():
    html = markup.to_html("`a ** b`")
    assert "<b>" not in html
    assert "a ** b" in text_of(html)


def test_fenced_block_keeps_its_lines():
    html = markup.to_html("Пример:\n```python\nx = 1\ny = 2\n```\nвсё")
    body = text_of(html)
    assert "x = 1" in body and "y = 2" in body
    assert "```" not in body


def test_unterminated_fence_still_renders_while_streaming():
    html = markup.to_html("```\nx = 1")
    assert "x = 1" in text_of(html)


# ----------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------


def test_bullets_get_a_real_bullet_character():
    html = markup.to_html("- раз\n- два")
    body = text_of(html)
    assert body.count("•") == 2
    assert "раз" in body and "два" in body
    assert not re.search(r"^\s*-\s", body, re.M)


def test_numbered_list_keeps_its_numbers():
    body = text_of(markup.to_html("1. раз\n2. два"))
    assert "1." in body and "2." in body


def test_headings_lose_their_hashes_and_stay_in_the_body_type_scale():
    html = markup.to_html("## Итого\nтекст")
    assert "<b>Итого</b>" in html
    assert "#" not in text_of(html)
    # No <h1>…<h6>: a 340 px card has no room for markdown's default sizes.
    assert not re.search(r"<h[1-6]", html)


def test_quote_is_rendered_without_its_marker():
    body = text_of(markup.to_html("> цитата"))
    assert body.strip() == "цитата"


def test_links_keep_the_text_and_drop_the_url():
    body = text_of(markup.to_html("см. [документацию](https://example.com/a)"))
    assert "документацию" in body
    assert "example.com" not in body


# ----------------------------------------------------------------------
# Safety and edges
# ----------------------------------------------------------------------


def test_html_in_the_answer_is_escaped_not_executed():
    html = markup.to_html("Тег <script>alert(1)</script> и 5 < 6")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_empty_and_blank_render_to_nothing():
    assert markup.to_html("") == ""
    assert markup.to_html("   \n\n ") == ""


def test_plain_paragraphs_survive_a_round_trip():
    text = "Первая строка.\nВторая строка.\n\nНовый абзац."
    body = text_of(markup.to_html(text))
    for line in ("Первая строка.", "Вторая строка.", "Новый абзац."):
        assert line in body


# ----------------------------------------------------------------------
# to_plain — the same subset, stripped
# ----------------------------------------------------------------------


def test_to_plain_strips_markers_without_adding_tags():
    out = markup.to_plain("## Итого\n- **раз**\n- `код`\n\nобычный текст")
    assert "<" not in out
    assert "**" not in out and "`" not in out and "#" not in out
    assert "• раз" in out
    assert "обычный текст" in out
