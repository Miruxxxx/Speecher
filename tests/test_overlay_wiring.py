"""That the overlay and the two collaborators it was split into stay connected.

`Overlay` used to own the window, the LLM state machine and everything the
session menu does. Those are now `ui/llm_presenter.py` and `ui/session.py`, which
is only an improvement if the seams hold — and the layout tests deliberately
never build an Overlay (constructing one registers global hotkeys on whatever
machine runs them).

So: build one with the hotkey manager stubbed out, and check the wiring in both
directions. The presenter's job is tested without a window at all, which is the
main thing the split bought.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import queue  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app_config import AppConfig  # noqa: E402
from asr.events import Word  # noqa: E402
from llm.engine import LLMFailure  # noqa: E402
from llm.summaries import SummaryHistory  # noqa: E402
from store.transcript_store import TranscriptStore  # noqa: E402
from store.transcript_writer import TranscriptWriter  # noqa: E402
from translate.store import TranslationStore  # noqa: E402
from ui.theme.fonts import load_app_fonts  # noqa: E402
from ui.widgets.indicators import (  # noqa: E402
    LLM_ERROR,
    LLM_GENERATING,
    LLM_OFFLINE,
    LLM_ONLINE,
    LLM_THINKING,
    REC_ON,
    REC_PAUSED,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    load_app_fonts()
    yield app


class FakeEngine:
    """An LLMEngine that hands the task back instead of talking to a server."""

    def __init__(self, available: bool = True) -> None:
        self.tasks: list = []
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def submit(self, task) -> None:
        self.tasks.append(task)


def _presenter(qapp, store=None, *, visible=True, engine=None):
    from ui.llm_presenter import LlmPresenter
    from ui.overlay import minutes_caption

    return LlmPresenter(
        llm=engine or FakeEngine(),
        store=store or TranscriptStore(),
        cfg=AppConfig(),
        summaries=SummaryHistory(),
        is_visible=lambda: visible,
        minutes_caption=minutes_caption,
    )


# -- the presenter, with no window at all ----------------------------------


def test_a_streaming_answer_walks_thinking_generating_online(qapp):
    store = TranscriptStore()
    store.append([Word(text="Что", start=0.0, end=0.2),
                  Word(text="думаешь?", start=0.3, end=0.6)])
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)

    states: list[str] = []
    texts: list[str] = []
    p.state_changed.connect(states.append)
    p.answer_updated.connect(texts.append)

    p.answer()
    assert p.busy and states[-1] == LLM_THINKING
    task = engine.tasks[-1]

    task.on_token("Пол")
    task.on_token("овина")
    assert states[-1] == LLM_GENERATING
    # Every update carries the whole text, so a dropped signal cannot
    # desynchronise the card.
    assert texts == ["Пол", "Половина"]

    task.on_done()
    assert not p.busy and states[-1] == LLM_ONLINE


def test_a_failure_says_whether_the_provider_was_reachable(qapp):
    store = TranscriptStore()
    store.append([Word(text="Что?", start=0.0, end=0.2)])
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    states: list[str] = []
    p.state_changed.connect(states.append)

    p.answer()
    engine.tasks[-1].on_error(LLMFailure("неверный API-ключ", offline=False))
    assert states[-1] == LLM_ERROR      # the server answered, it refused us

    p.answer()
    engine.tasks[-1].on_error(
        LLMFailure("сервер не отвечает — запустите его", offline=True)
    )
    assert states[-1] == LLM_OFFLINE


def test_a_half_written_answer_survives_a_mid_stream_failure(qapp):
    store = TranscriptStore()
    store.append([Word(text="Что?", start=0.0, end=0.2)])
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    failures: list[tuple[str, bool]] = []
    p.answer_failed.connect(lambda text, empty: failures.append((text, empty)))

    p.answer()
    engine.tasks[-1].on_token("Начало ответа")
    engine.tasks[-1].on_error(LLMFailure("сеть отвалилась"))

    text, empty = failures[-1]
    assert empty is False                  # something had streamed
    assert "Начало ответа" in text and "сеть отвалилась" in text


def test_an_empty_transcript_never_reaches_the_model(qapp):
    engine = FakeEngine()
    p = _presenter(qapp, TranscriptStore(), engine=engine)
    said: list[tuple[str, str]] = []
    p.status.connect(lambda a, b: said.append((a, b)))

    p.answer()
    assert engine.tasks == []
    assert said and said[-1][0] == "Пусто"


# -- finding the question when the engine never punctuated it ---------------


def _unpunctuated(store: TranscriptStore) -> TranscriptStore:
    """What nemotron produces on run-on speech: no '?' anywhere."""
    for i, w in enumerate("how often have you had that dream in your life".split()):
        store.append([Word(text=w, start=i * 0.3, end=i * 0.3 + 0.25)])
    return store


def test_a_missing_question_mark_is_not_a_missing_question(qapp):
    """The '?' is the ASR's opinion, and on run-on English it never has one.

    This used to answer "вопрос не найден" to a transcript full of questions —
    the button was dead for exactly the material it was most wanted on.
    """
    store = _unpunctuated(TranscriptStore())
    assert store.find_last_question() is None      # the cheap path finds nothing

    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    p.answer()

    assert len(engine.tasks) == 1                  # …but the model is still asked
    prompt = engine.tasks[0].prompt
    assert "punctuates unreliably" in prompt
    assert "how often have you had that dream" in prompt


def test_a_found_question_mark_still_takes_the_cheap_path(qapp):
    store = TranscriptStore()
    store.append([Word(text="Так", start=0.0, end=0.2),
                  Word(text="что?", start=0.3, end=0.6)])
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    p.answer()

    prompt = engine.tasks[0].prompt
    assert "Question: Так что?" in prompt           # the known-question prompt
    assert "punctuates unreliably" not in prompt


def test_alt_q_is_instant_when_the_transcript_has_a_mark(qapp):
    """The '?' path must not cost a request — it works with no provider at all."""
    store = TranscriptStore()
    store.append([Word(text="Кто", start=0.0, end=0.2),
                  Word(text="здесь?", start=0.3, end=0.6)])
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    found: list[str] = []
    p.question_ready.connect(found.append)

    p.last_question()
    assert found == ["Кто здесь?"]
    assert engine.tasks == []


def test_alt_q_falls_back_to_the_model_and_reports_the_quote(qapp):
    store = _unpunctuated(TranscriptStore())
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    found: list[str] = []
    started: list[str] = []
    p.question_ready.connect(found.append)
    p.answer_started.connect(started.append)

    p.last_question()
    assert len(engine.tasks) == 1
    # No card: this task's whole output *is* the question, and a card that says
    # "Думаю…" and then turns into a quote reads as an answer.
    assert started == []

    engine.tasks[0].on_token("How often have you had that dream in your life?")
    engine.tasks[0].on_done()
    assert found == ["How often have you had that dream in your life?"]


def test_the_model_saying_nobody_asked_does_not_reach_the_card(qapp):
    """NO_QUESTION is a protocol answer, not something to show the user."""
    from ui.llm_presenter import NO_QUESTION, is_no_question

    assert is_no_question(NO_QUESTION)
    assert is_no_question(' "no_question." ')       # models decorate it
    assert not is_no_question("No question is more important than this one")

    store = _unpunctuated(TranscriptStore())
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    found, said = [], []
    p.question_ready.connect(found.append)
    p.status.connect(lambda a, b: said.append((a, b)))

    p.last_question()
    engine.tasks[0].on_token(NO_QUESTION)
    engine.tasks[0].on_done()

    assert found == []
    assert said[-1][0] == "Нет вопроса"


def test_an_answer_task_that_finds_no_question_clears_the_card(qapp):
    store = _unpunctuated(TranscriptStore())
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)
    dismissed, said = [], []
    p.answer_dismissed.connect(lambda: dismissed.append(1))
    p.status.connect(lambda a, b: said.append((a, b)))

    p.answer()
    engine.tasks[0].on_token("NO_QUESTION")
    engine.tasks[0].on_done()

    assert dismissed == [1]
    assert said[-1][0] == "Нет вопроса"


def test_a_second_task_is_refused_while_one_is_running(qapp):
    store = TranscriptStore()
    store.append([Word(text="Что?", start=0.0, end=0.2)])
    engine = FakeEngine()
    p = _presenter(qapp, store, engine=engine)

    p.answer()
    p.answer()
    p.summarize()
    assert len(engine.tasks) == 1


def test_a_hidden_window_gets_a_notification_instead_of_a_card(qapp):
    store = TranscriptStore()
    store.append([Word(text="Что?", start=0.0, end=0.2)])
    engine = FakeEngine()
    p = _presenter(qapp, store, visible=False, engine=engine)
    notes: list[tuple[str, str, str]] = []
    p.notify.connect(lambda *a: notes.append(a))

    p.answer()
    engine.tasks[-1].on_token("ответ")
    engine.tasks[-1].on_done()

    assert notes and notes[-1][0] == "Ответ готов"
    # The presenter names the *action*, not a key combination — which key opens
    # it is the window's business.
    assert notes[-1][1] == "toggle_window"


# -- the overlay, wired to both --------------------------------------------


@pytest.fixture
def overlay(qapp, tmp_path, monkeypatch):
    from ui import overlay as overlay_module

    class NoHotkeys:
        """RegisterHotKey would grab real combinations on the test machine."""

        def __init__(self, *_a, **_kw):
            self.triggered = _Sig()
            self.failed = _Sig()

        def start(self):
            pass

        def stop(self):
            pass

        def failures(self):
            return {}

    class _Sig:
        def connect(self, _slot):
            pass

    monkeypatch.setattr(overlay_module, "HotkeyManager", NoHotkeys)

    cfg = AppConfig()
    cfg.ui.onboarding_shown = True          # no popup on construction
    writer = TranscriptWriter(tmp_path)
    writer.start()
    store = TranscriptStore(journal=writer)
    w = overlay_module.Overlay(
        queue.Queue(), store, FakeEngine(), cfg, tmp_path / "config.toml",
        writer=writer, session_dir=tmp_path,
        translations=TranslationStore(target="ru"),
    )
    yield w
    w._llm_ui.stop()
    writer.close()
    w.deleteLater()


def test_the_overlay_builds_and_reads_its_collaborators(overlay):
    """The construction order matters: the presenter and the session controller
    are built before `_refresh_indicators`/`_refresh_actions` read them."""
    assert overlay._llm_ui.state == LLM_OFFLINE
    assert overlay._session.journal_state() == REC_ON
    # Buttons follow "is there a transcript" and "is the model reachable".
    assert not overlay._btn_question.isEnabled()      # nothing said yet


def test_presenter_state_reaches_the_indicator_and_the_buttons(overlay):
    overlay._store.append([Word(text="Что?", start=0.0, end=0.2)])
    overlay._refresh_actions()
    assert overlay._btn_question.isEnabled()          # needs no model
    assert not overlay._btn_answer.isEnabled()        # model still offline

    overlay._llm_ui._on_health(True)
    assert overlay._llm_ui.state == LLM_ONLINE
    assert overlay._btn_answer.isEnabled()


def test_streamed_tokens_reach_the_answer_card(overlay):
    overlay._store.append([Word(text="Что?", start=0.0, end=0.2)])
    overlay._llm_ui._on_health(True)
    overlay._llm_ui.answer()

    task = overlay._llm.tasks[-1]
    task.on_token("Ответ")
    task.on_done()
    assert "Ответ" in overlay._answer._body.toPlainText()


def test_the_engine_that_actually_ran_reaches_the_header_and_the_journal(overlay):
    """After a fallback the session must name whisper, not the configured
    nemotron — in the export header and in the journal alike."""
    overlay.set_engine("whisper", "large-v3-turbo", requested="nemotron")

    assert overlay._session.meta["engine"] == "whisper"
    assert overlay._session.meta["engine_requested"] == "nemotron"
    assert "вместо nemotron" in overlay._session.engine_caption()


def test_recording_toggle_moves_the_journal_and_the_indicator(overlay):
    assert overlay._session.journal_state() == REC_ON
    overlay.toggle_recording()
    assert overlay._session.journal_state() == REC_PAUSED
    overlay.toggle_recording()
    assert overlay._session.journal_state() == REC_ON


def test_a_resume_re_renders_the_feed(overlay, tmp_path):
    """SessionController.changed is what tells the window the store moved
    underneath it — without that signal a resume loaded silently."""
    seen = []
    overlay._session.changed.connect(lambda: seen.append(1))

    other = tmp_path / "2020-01-01_000000.jsonl"
    other.write_text(
        '{"t":"meta","version":1}\n'
        '{"t":"w","text":"Привет","s":0.0,"e":0.3,"at":1700000000.0}\n',
        encoding="utf-8",
    )
    overlay._session.resume(other)

    assert seen == [1]
    assert overlay._store.size() == 1
