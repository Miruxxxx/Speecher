"""The design system's acceptance checks that need a live Qt widget tree.

Section 4 ends with a "проверка на сдвиг" table: a list of events after which the
transcript, the indicator row and the button row must sit on exactly the same
pixels. Those are the tests here — they are the criterion the whole layout was
built around, so they are worth more than any screenshot.

Runs headless (QT_QPA_PLATFORM=offscreen); no audio, no GPU, no hotkeys — the
Overlay is not constructed, because building it would register global hotkeys on
the machine running the tests.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from asr.events import Word  # noqa: E402
from ui.overlay import minutes_caption  # noqa: E402
from ui.theme.fonts import load_app_fonts  # noqa: E402
from ui.theme.tokens import COMPACT, NORMAL, SOURCE_H, SOURCE_W  # noqa: E402
from ui.transcript_model import group_entries  # noqa: E402
from ui.widgets.cards import AnswerCard  # noqa: E402
from ui.widgets.indicators import (  # noqa: E402
    ERROR,
    LIVE,
    LLM_ONLINE,
    LLM_STATES,
    LLM_THINKING,
    SILENT,
    capture_caption,
    llm_caption,
    llm_dot,
    llm_tooltip,
)
from ui.widgets.titlebar import TitleBar  # noqa: E402
from ui.widgets.transcript import TranscriptView, wrap_chunks  # noqa: E402
from ui.windows.base import Panel  # noqa: E402
from ui.windows.source import SourceWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    load_app_fonts()
    yield app


@pytest.fixture
def replies():
    text = (
        "Пользователь должен забыть, что окно открыто, и просто получать пользу. "
        "Давай тогда посмотрим на примеры и соберём референсы."
    )
    entries = []
    start = 0.0
    for word in text.split():
        entries.append((Word(text=word, start=start, end=start + 0.3), 1_700_000_000.0))
        start += 0.35
    return group_entries(entries)


def _feed(qapp, mode=NORMAL):
    area = QWidget()
    area.resize(mode.width - 2 * mode.padding, 228)
    layout = QVBoxLayout(area)
    layout.setContentsMargins(0, 0, 0, 0)
    view = TranscriptView(mode, area)
    layout.addWidget(view)
    card = AnswerCard(mode, area)
    area.show()
    qapp.processEvents()
    return area, view, card


def _feed_geometry(view: TranscriptView) -> tuple:
    cursor = view.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    rect = view.cursorRect(cursor)
    return (
        round(view.document().size().height()),
        view.viewport().geometry().getRect(),
        (rect.x(), rect.y()),
    )


# -- no-shift table ---------------------------------------------------------


def test_answer_card_does_not_move_the_feed(qapp, replies):
    area, view, card = _feed(qapp)
    view.set_content(replies, "")
    qapp.processEvents()
    before = _feed_geometry(view)

    card.show_text("Ответ модели", "Минимализм помогает сосредоточиться на главном.")
    qapp.processEvents()
    assert _feed_geometry(view) == before

    card.hide()
    qapp.processEvents()
    assert _feed_geometry(view) == before
    area.close()


def test_confirming_a_partial_rewrites_in_place(qapp, replies):
    """The hypothesis lives at the end of the last reply, not on its own line."""
    area, view, _ = _feed(qapp)
    view.set_content(replies, "и, возможно")
    qapp.processEvents()
    with_partial = round(view.document().size().height())

    view.set_content(replies, "")
    qapp.processEvents()
    assert round(view.document().size().height()) <= with_partial
    area.close()


def test_indicator_states_keep_the_row_in_place(qapp):
    bar = TitleBar(NORMAL)
    bar.resize(NORMAL.width, NORMAL.titlebar_h)
    bar.show()
    qapp.processEvents()

    seen = set()
    for state in (LIVE, SILENT, ERROR, LIVE):
        bar.capture.icon_widget().set_state(state)
        bar.capture.set_caption(capture_caption(state))
        qapp.processEvents()
        seen.add(
            (
                bar.capture.width(),
                bar.disk.x(),
                bar.llm.x(),
                bar.menu_button.x(),
                bar.capture.icon_widget().width(),
            )
        )
    assert len(seen) == 1, "the capture pill's width follows its caption"
    bar.close()


def test_status_message_shifts_nothing(qapp):
    bar = TitleBar(NORMAL)
    bar.resize(NORMAL.width, NORMAL.titlebar_h)
    bar.show()
    qapp.processEvents()
    before = (bar.capture.x(), bar.disk.x(), bar.llm.x(), bar.menu_button.x())

    bar.status.show_message("Экспорт", detail="Экспортировано в session.md")
    qapp.processEvents()
    assert (bar.capture.x(), bar.disk.x(), bar.llm.x(), bar.menu_button.x()) == before

    bar.status.clear()
    qapp.processEvents()
    assert (bar.capture.x(), bar.disk.x(), bar.llm.x(), bar.menu_button.x()) == before
    bar.close()


def test_a_translated_feed_never_rewrites_a_line(qapp):
    """The rule the second design exists for: lines are appended, not edited.

    A line reaches the feed already translated, so re-rendering with a newer
    line appended must leave every earlier line byte-identical — no re-wrap, no
    style change, nothing that moves what is being read.
    """
    from translate.segments import Segment  # noqa: PLC0415 — Qt-free, test-local
    from translate.store import TRANSLATED, TranslationStore

    store = TranslationStore()
    for i, text in enumerate(("Первая строка.", "Вторая строка.")):
        store.add(
            Segment(index=i, n_words=2, at=1_700_000_000.0 + i,
                    start_sec=i * 2.0, end_sec=i * 2.0 + 1.0, text="src"),
            text, TRANSLATED,
        )

    area, view, _ = _feed(qapp)
    view.set_content(store.replies(), "")
    qapp.processEvents()
    before = view.toPlainText()

    store.add(
        Segment(index=2, n_words=2, at=1_700_000_002.0, start_sec=4.0, end_sec=5.0,
                text="src"),
        "Третья строка.", TRANSLATED,
    )
    view.set_content(store.replies(), "")
    qapp.processEvents()
    after = view.toPlainText()

    assert after.startswith(before.rstrip("\n"))
    assert "Третья строка." in after
    area.close()


def test_source_window_carries_the_live_original(qapp):
    """The tail the main feed deliberately does not show has to be somewhere."""
    window = SourceWindow()
    entries = [
        (Word(text=w, start=i * 0.4, end=i * 0.4 + 0.3), 1_700_000_000.0)
        for i, w in enumerate("still being spoken".split())
    ]
    window.set_content(group_entries(entries), "")
    window.show()
    qapp.processEvents()

    assert "still being spoken" in window._view.toPlainText()
    assert (window.width(), window.height()) == (SOURCE_W, SOURCE_H)
    window.close()


# -- wrapping ---------------------------------------------------------------


def test_wrap_chunks_caps_lines_per_reply(qapp):
    text = " ".join(["слово"] * 120)
    chunks = wrap_chunks(text, NORMAL.transcript, 300.0, 4)
    assert len(chunks) > 1
    # Nothing is lost, and nothing is truncated with an ellipsis (6.1).
    assert " ".join(chunks) == text
    assert "…" not in "".join(chunks)
    for chunk in chunks:
        assert len(wrap_chunks(chunk, NORMAL.transcript, 300.0, 4)) == 1


def test_wrap_chunks_passes_short_text_through(qapp):
    assert wrap_chunks("короткая строка", NORMAL.transcript, 300.0, 4) == [
        "короткая строка"
    ]
    assert wrap_chunks("", NORMAL.transcript, 300.0, 4) == []


# -- modes ------------------------------------------------------------------


def test_compact_mode_keeps_the_gutter_and_the_buttons(qapp):
    """6.2 and 6.3: neither the timestamp gutter nor the action row is dropped."""
    assert COMPACT.gutter_w > 0
    assert COMPACT.action_h > 0
    assert COMPACT.transcript.size_px >= 12  # "не опускается ниже 12 px"


def test_title_bar_switches_to_icons_only(qapp):
    bar = TitleBar(NORMAL)
    bar.resize(NORMAL.width, NORMAL.titlebar_h)
    bar.show()
    qapp.processEvents()
    wide = bar.capture.width()

    bar.set_mode(COMPACT)
    bar.resize(COMPACT.width, COMPACT.titlebar_h)
    qapp.processEvents()
    assert bar.capture.width() < wide
    assert bar.height() == COMPACT.titlebar_h
    bar.close()


# -- generated captions -----------------------------------------------------


# -- the window must survive its own dialogs --------------------------------


def test_closing_a_satellite_window_does_not_quit_the_app(qapp):
    """Regression: settings / hotkeys / about used to take the app down.

    The overlay is a Qt.Tool, which Qt does not count for
    quitOnLastWindowClosed — so the first *counted* window to close was also the
    last one, and closing settings ended the session.
    """
    overlay = QWidget()
    overlay.setWindowFlags(
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.Tool
    )
    overlay.resize(NORMAL.width, NORMAL.height)
    overlay.show()
    panel = Panel("Настройки", (300, 200))
    panel.show()
    qapp.processEvents()

    survived: list[bool] = []

    def still_running() -> None:
        # Sampled before quit(): once the loop is unwinding Qt reports every
        # widget as hidden, so asking afterwards proves nothing.
        survived.append(overlay.isVisible())
        qapp.quit()

    QTimer.singleShot(50, panel.close)
    QTimer.singleShot(350, still_running)  # only reached if the app is alive
    qapp.exec()

    assert survived == [True]
    panel.deleteLater()
    overlay.close()


def test_indicator_pills_do_not_swallow_the_drag(qapp):
    """The whole title strip drags the window; only the buttons are cut out."""
    bar = TitleBar(NORMAL)
    bar.resize(NORMAL.width, NORMAL.titlebar_h)
    bar.show()
    qapp.processEvents()

    for pill in (bar.capture, bar.disk, bar.llm):
        assert bar._in_drag_zone(pill.geometry().center().toPointF())
    for button in (bar.menu_button, bar.close_button):
        assert not bar._in_drag_zone(button.geometry().center().toPointF())
    bar.close()


def test_title_bar_has_a_close_button(qapp):
    bar = TitleBar(NORMAL)
    seen: list[str] = []
    bar.close_requested.connect(lambda: seen.append("close"))
    bar.close_button.click()
    assert seen == ["close"]
    bar.close()


# -- model states -----------------------------------------------------------


def test_every_llm_state_has_a_caption_a_dot_and_a_tooltip():
    for state in LLM_STATES:
        assert llm_caption(state)
        assert llm_tooltip(state)
        colour, filled, pulsing = llm_dot(state)
        assert colour.startswith("#") and isinstance(filled, bool)
        assert isinstance(pulsing, bool)


def test_thinking_is_distinguishable_from_online_without_reading():
    """Both are green: only the travelling accent tells them apart, and that is
    the whole point in compact mode where the caption is gone."""
    online = llm_dot(LLM_ONLINE)
    thinking = llm_dot(LLM_THINKING)
    assert online[0] == thinking[0]
    assert thinking[2] and not online[2]


@pytest.mark.parametrize(
    "minutes, expected",
    [
        (1, "за 1 минуту"),
        (2, "за 2 минуты"),
        (5, "за 5 минут"),
        (10, "за 10 минут"),
        (11, "за 11 минут"),
        (21, "за 21 минуту"),
        (0, "интервал не задан"),
    ],
)
def test_summary_caption_is_generated(minutes, expected):
    assert minutes_caption(minutes) == expected
