"""The answer/summary side windows (design system 7.9).

Two things are worth pinning down here, and they are the two the feature is
actually about: the windows are exactly as big as the mode says (the complaint
that started this was "они мизерные"), and they sit on opposite sides of the
overlay without overlapping it or each other. Everything else — which window a
task's output lands in — is routing, and routing is where a mode switch
silently stops working, so it is tested through the real Overlay.

Headless (QT_QPA_PLATFORM=offscreen); no audio, no model, no hotkeys.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import queue  # noqa: E402

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from app_config import AppConfig  # noqa: E402
from asr.events import Word  # noqa: E402
from store.transcript_store import TranscriptStore  # noqa: E402
from translate.store import TranslationStore  # noqa: E402
from ui.theme.fonts import load_app_fonts  # noqa: E402
from ui.theme.tokens import (  # noqa: E402
    COMPACT,
    NORMAL,
    PANELS_INLINE,
    PANELS_SIDE_COMPACT,
    PANELS_SIDE_LARGE,
    SIDE_GAP,
    panel_mode,
)
from ui.windows.side import (  # noqa: E402
    LEFT,
    RIGHT,
    answer_panel,
    dock_position,
    summary_panel,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    load_app_fonts()
    yield app


class FakeEngine:
    """An LLMEngine that keeps the task instead of talking to a server."""

    def __init__(self) -> None:
        self.tasks: list = []

    def is_available(self) -> bool:
        return True

    def submit(self, task) -> None:
        self.tasks.append(task)


# -- the windows themselves -------------------------------------------------


def test_an_unknown_mode_reads_as_the_card():
    """A typo in config.toml may not stop the window from opening."""
    assert panel_mode("side_large") == PANELS_SIDE_LARGE
    assert panel_mode("огромные") == PANELS_INLINE
    assert panel_mode("") == PANELS_INLINE


@pytest.mark.parametrize(
    "mode,size",
    [
        (PANELS_SIDE_LARGE, (NORMAL.width, NORMAL.height)),
        (PANELS_SIDE_COMPACT, (COMPACT.width, COMPACT.height)),
    ],
)
def test_a_panel_is_exactly_as_big_as_the_main_window(qapp, mode, size):
    panel = answer_panel(mode)
    assert (panel.width(), panel.height()) == size
    panel.deleteLater()


def test_the_two_panels_flank_the_overlay_without_touching_it():
    """The arithmetic, on a screen big enough for it — the headless platform
    gives an 800x600 one, where everything would be clamped and the test would
    only be checking the clamp."""
    screen = (0, 0, 2560, 1440)
    overlay = (900, 400, NORMAL.width, NORMAL.height)
    size = (NORMAL.width, NORMAL.height)

    lx, ly = dock_position(overlay, size, LEFT, screen)
    rx, ry = dock_position(overlay, size, RIGHT, screen)

    assert lx + size[0] + SIDE_GAP == overlay[0]              # left of it
    assert rx == overlay[0] + overlay[2] + SIDE_GAP           # right of it
    assert ly == ry == overlay[1]                             # tops aligned
    # The whole point: neither of them is over the transcript, or over the other.
    assert lx + size[0] < overlay[0] < rx
    assert lx + size[0] < rx


def test_a_panel_that_would_hang_off_the_screen_is_pulled_back():
    screen = (0, 0, 1280, 800)
    overlay = (60, 40, NORMAL.width, NORMAL.height)
    x, y = dock_position(overlay, (NORMAL.width, NORMAL.height), LEFT, screen)
    assert x == 0 and y == 40


def test_a_panel_dragged_by_hand_stops_following_the_overlay(qapp):
    anchor = QWidget()
    anchor.resize(NORMAL.width, NORMAL.height)
    anchor.move(700, 300)
    panel = answer_panel(PANELS_SIDE_LARGE)
    panel.place_beside(anchor)

    panel._detached = True          # what a drag of its header sets
    panel.move(40, 40)
    panel.place_beside(anchor)
    assert (panel.x(), panel.y()) == (40, 40)

    # A mode change is deliberate and does re-dock it.
    panel.place_beside(anchor, force=True)
    assert panel.x() + panel.width() + SIDE_GAP == anchor.frameGeometry().left()
    panel.deleteLater()
    anchor.deleteLater()


def test_only_a_model_answer_is_rendered_as_markdown(qapp):
    """Same rule as the card: a question containing an asterisk is not emphasis."""
    panel = answer_panel(PANELS_SIDE_LARGE)
    panel.show_text("Последний вопрос", "Сколько будет 2 * 2?")
    assert "2 * 2" in panel._body.toPlainText()

    panel.set_answer("**Ответ**")
    assert panel._body.toPlainText().strip() == "Ответ"
    panel.deleteLater()


def test_a_finished_answer_does_not_close_itself(qapp):
    panel = answer_panel(PANELS_SIDE_LARGE)
    panel.begin("Думаю…")
    panel.set_answer("Готово")
    panel.finish()
    assert panel.isVisible()
    assert not panel.is_streaming()
    panel.deleteLater()


# -- routing through the real overlay --------------------------------------


@pytest.fixture
def overlay(qapp, tmp_path, monkeypatch):
    from ui import overlay as overlay_module

    class _Sig:
        def connect(self, _slot):
            pass

    class NoHotkeys:
        def __init__(self, *_a, **_kw):
            self.triggered = _Sig()
            self.failed = _Sig()

        def start(self):
            pass

        def stop(self):
            pass

        def failures(self):
            return {}

    monkeypatch.setattr(overlay_module, "HotkeyManager", NoHotkeys)

    cfg = AppConfig()
    cfg.ui.onboarding_shown = True
    cfg.ui.panels = PANELS_SIDE_LARGE
    store = TranscriptStore()
    store.append([Word(text="Что?", start=0.0, end=0.2)])
    w = overlay_module.Overlay(
        queue.Queue(), store, FakeEngine(), cfg, tmp_path / "config.toml",
        session_dir=tmp_path, translations=TranslationStore(target="ru"),
    )
    # Shown: several paths ask whether anyone can see the window and send the
    # result to a notification instead when nobody can.
    w.show()
    w._llm_ui._on_health(True)
    yield w
    w._llm_ui.stop()
    w.deleteLater()


def test_an_answer_goes_left_and_a_summary_goes_right(overlay):
    overlay._llm_ui.answer()
    overlay._llm.tasks[-1].on_token("Ответ")
    overlay._llm.tasks[-1].on_done()
    assert "Ответ" in overlay._answer_panel._body.toPlainText()
    assert overlay._summary_panel is None
    # The in-feed card is not used at all in this mode.
    assert not overlay._answer.isVisible()

    overlay._llm_ui.summarize()
    overlay._llm.tasks[-1].on_token("Выжимка")
    overlay._llm.tasks[-1].on_done()
    assert "Выжимка" in overlay._summary_panel._body.toPlainText()
    # The answer is still there: the two windows do not replace each other.
    assert "Ответ" in overlay._answer_panel._body.toPlainText()


def test_the_last_question_shares_the_answer_window(overlay):
    """"AI + ?" is one conversation and one window (the user's words)."""
    overlay._on_question_ready("Сколько это стоит?")
    assert "Сколько это стоит?" in overlay._answer_panel._body.toPlainText()
    assert overlay._summary_panel is None


def test_switching_back_to_the_card_hides_the_windows(overlay):
    overlay._llm_ui.answer()
    overlay._llm.tasks[-1].on_token("Ответ")
    overlay._llm.tasks[-1].on_done()
    assert overlay._answer_panel.isVisible()

    overlay.set_panels_mode(PANELS_INLINE)
    assert not overlay._answer_panel.isVisible()

    overlay._llm_ui.answer()
    overlay._llm.tasks[-1].on_token("Второй")
    overlay._llm.tasks[-1].on_done()
    assert "Второй" in overlay._answer._body.toPlainText()
    assert not overlay._answer_panel.isVisible()


def test_a_mode_change_resizes_and_redocks_an_open_window(overlay):
    from PyQt6.QtWidgets import QApplication

    overlay._llm_ui.answer()
    overlay._llm.tasks[-1].on_done()
    overlay.set_panels_mode(PANELS_SIDE_COMPACT)
    panel = overlay._answer_panel
    assert (panel.width(), panel.height()) == (COMPACT.width, COMPACT.height)

    geo = overlay.frameGeometry()
    usable = QApplication.primaryScreen().availableGeometry()
    assert (panel.x(), panel.y()) == dock_position(
        (geo.x(), geo.y(), geo.width(), geo.height()),
        (COMPACT.width, COMPACT.height),
        LEFT,
        (usable.x(), usable.y(), usable.width(), usable.height()),
    )


def test_hiding_the_overlay_takes_the_windows_with_it(overlay):
    overlay.show()
    overlay._llm_ui.answer()
    overlay._llm.tasks[-1].on_done()
    assert overlay._answer_panel.isVisible()

    overlay.toggle_visibility()
    # The panels go at once; the overlay fades, and only `hide()` at the end of
    # that fade makes the next toggle a "show" rather than a second "hide".
    assert not overlay._answer_panel.isVisible()
    anim = overlay._window_anim
    anim.setCurrentTime(anim.duration())

    overlay.toggle_visibility()
    assert overlay._answer_panel.isVisible()
