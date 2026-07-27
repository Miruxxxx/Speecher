"""The LLM half of the overlay: tasks, streaming, health, and the auto-summary.

Split out of `Overlay` for the same reason `ui/session.py` was — that class had
become the place where everything met. This part is a small state machine with
three inputs (a button, a hotkey, a timer), one worker thread it must never
block on, and a handful of widgets it has to keep truthful. None of that needs
to know how the window is laid out.

The boundary is deliberately one-way: the presenter never touches a widget. It
owns the *state* (is a task running, what has streamed so far, is the provider
reachable) and reports transitions as signals; the overlay owns the answer card,
the indicator and the notification, and decides how each transition looks. That
is what makes it possible to reason about "what is the LLM doing" without
reading paint code, and it is why the health probe could be reduced to one
thread here without touching the view at all.

Everything network-ish stays off the Qt thread: `LLMEngine` has its own worker
for requests, and `_health_loop` is the single long-lived prober behind the 20 s
timer. Callbacks from both cross back through signals.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from app_config import AppConfig
from llm.engine import LLMEngine, LLMTask
from llm.summaries import SummaryHistory
from store.transcript_store import TranscriptStore
from ui.widgets.indicators import (
    LLM_ERROR,
    LLM_GENERATING,
    LLM_OFFLINE,
    LLM_ONLINE,
    LLM_THINKING,
)

logger = logging.getLogger(__name__)

# How much transcript a question's prompt carries with it.
_ANSWER_CONTEXT_CHARS = 1500

# How much of the tail to hand the model when it has to find the question
# itself. Longer than the context above: with no '?' to anchor on, the model is
# reading for meaning and needs the run-up to the question, not just its words.
_QUESTION_TAIL_CHARS = 2000

# How often the provider is re-probed, so the indicator tracks a server going up
# or down rather than showing its state at startup forever.
_HEALTH_POLL_MS = 20000

# What the model says when it read the transcript and nobody asked anything.
# Checked case-insensitively on a stripped prefix, because models like to add
# a period or wrap it in quotes however firmly the prompt says not to.
NO_QUESTION = "NO_QUESTION"

# Why the '?' cannot be trusted, said once so both prompts stay in sync. The
# nemotron engine emits sentence-final punctuation only after hearing a pause,
# so a speaker who runs sentences together — a lecturer, anyone on a call —
# produces long unpunctuated stretches with real questions buried in them
# (measured: 0.7-6.7% of English words end a sentence, against 4.7-8.8% on
# Russian). See docs/NEMOTRON_PUNCTUATION_TODO.md.
_UNRELIABLE = (
    "This is the tail of a live speech transcript. It was produced by streaming "
    "speech recognition that punctuates unreliably: question marks and full "
    "stops are often missing entirely, and sentence boundaries may be wrong. "
    "Read it for meaning, not for punctuation."
)


def find_question_prompt(tail: str) -> str:
    """Ask for the last question, and nothing else.

    The reply is shown to the user verbatim as "the last question", so the model
    is told to quote rather than paraphrase — a reworded question sends the user
    looking for words nobody said.
    """
    return (
        f"{_UNRELIABLE}\n\n"
        "Find the LAST question the speaker asked. Reply with that question "
        "alone, on a single line, in the language it was asked in, quoting the "
        "speaker's own words and adding only the punctuation it needs. Do not "
        "answer it, translate it, or add anything else. If nobody asked a "
        f"question, reply with exactly {NO_QUESTION} and nothing else.\n\n"
        f"Transcript:\n{tail}"
    )


def answer_found_question_prompt(question: str, context: str) -> str:
    """The cheap path: the transcript had a '?', so the question is known."""
    return (
        "You are given a speech transcript excerpt. "
        "Answer the question that appears at the end concisely and helpfully. "
        "Reply in the same language as the question.\n\n"
        f"Transcript:\n{context}\n\nQuestion: {question}"
    )


def answer_last_question_prompt(tail: str) -> str:
    """Find the question and answer it in one call.

    One request, not two: a separate detection round trip would double the
    latency of a button press and, on a metered provider, its cost. The question
    comes back on the first line so the user can see which one was picked —
    without that, a model answering the wrong question is indistinguishable from
    a model answering badly.
    """
    return (
        f"{_UNRELIABLE}\n\n"
        "Find the LAST question the speaker asked and answer it concisely and "
        "helpfully. Reply in the language the question was asked in. Start your "
        "reply with the question itself in bold on its own line, then the "
        f"answer. If nobody asked a question, reply with exactly {NO_QUESTION} "
        "and nothing else.\n\n"
        f"Transcript:\n{tail}"
    )


def is_no_question(text: str) -> bool:
    """True when the model reported that nothing was asked."""
    return text.strip().strip("\"'*`.").upper().startswith(NO_QUESTION)


class LlmPresenter(QObject):
    """Runs answer/summary tasks and keeps the provider's state honest."""

    # State of the provider/task, one of ui.widgets.indicators' LLM_* constants.
    state_changed = pyqtSignal(str)
    # A task began; the argument is the placeholder to show.
    answer_started = pyqtSignal(str)
    # Everything streamed so far — the card re-renders from the whole text, so a
    # dropped signal cannot desynchronise it.
    answer_updated = pyqtSignal(str)
    answer_finished = pyqtSignal()
    # The text the card should show for a failure. Carries any partial answer,
    # because a mid-stream failure must not wipe what was already written.
    answer_failed = pyqtSignal(str, bool)   # (text, nothing streamed yet)
    # A summary reached SummaryHistory.
    summary_added = pyqtSignal()
    # The last question, ready to show. Emitted instantly when the transcript
    # had a '?', and after a model round trip when it did not.
    question_ready = pyqtSignal(str)
    # Take the card away: a task finished with nothing worth showing.
    answer_dismissed = pyqtSignal()
    # (title, hint, icon) for the hidden-window notification.
    notify = pyqtSignal(str, str, str)
    # Short label + full sentence, exactly what Overlay.post_status takes.
    status = pyqtSignal(str, str)

    # Worker thread -> Qt thread. Private; the overlay binds the public ones.
    _token_sig = pyqtSignal(str)
    _done_sig = pyqtSignal()
    _error_sig = pyqtSignal(str, bool)
    _health_sig = pyqtSignal(bool)

    def __init__(
        self,
        *,
        llm: LLMEngine,
        store: TranscriptStore,
        cfg: AppConfig,
        summaries: SummaryHistory,
        is_visible: Callable[[], bool],
        minutes_caption: Callable[[int], str],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._llm = llm
        self._store = store
        self._cfg = cfg
        self._summaries = summaries
        # The window decides how a finished task announces itself, and that
        # depends on whether anyone can see it.
        self._is_visible = is_visible
        self._minutes_caption = minutes_caption

        self._state = LLM_OFFLINE
        self._busy = False
        self._output = ""
        self._kind = ""
        self._minutes = 0.0
        self._auto = False

        self._token_sig.connect(self._on_token)
        self._done_sig.connect(self._on_done)
        self._error_sig.connect(self._on_error)
        self._health_sig.connect(self._on_health)

        # One long-lived prober woken by the timer, rather than a fresh thread
        # per tick — that was 180 threads an hour to compute a single bool.
        self._health_wake = threading.Event()
        self._health_stop = threading.Event()
        self._health_thread = threading.Thread(
            target=self._health_loop, name="lm-health", daemon=True
        )
        self._health_thread.start()
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self.probe)
        self._health_timer.start(_HEALTH_POLL_MS)
        QTimer.singleShot(500, self.probe)

        self._auto_summary_timer = QTimer(self)
        self._auto_summary_timer.timeout.connect(lambda: self.summarize(auto=True))
        self.apply_auto_summary()

    # -- state -----------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def kind(self) -> str:
        """What the running (or last) task was: answer / summary / question.

        The window needs it to decide *where* the output goes — the answer and
        the summary have their own windows in the side-panel modes — and that is
        a view decision, so the presenter reports the fact instead of routing.
        """
        return self._kind

    @property
    def online(self) -> bool:
        return self._state != LLM_OFFLINE

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self.state_changed.emit(state)

    # -- tasks -----------------------------------------------------------

    def answer(self) -> None:
        """Answer the last question in the transcript.

        Two paths, and the second is the point. `find_last_question` looks for a
        '?', which the nemotron engine only emits when the speaker paused long
        enough for it to hear the sentence end. On run-on speech there is no '?'
        anywhere, and this used to answer "вопрос не найден" to a transcript
        full of questions. So a missing mark now means "the model has to read
        for it", not "nobody asked" — and it costs the same single request the
        button already made.
        """
        if self._busy:
            return
        question = self._store.find_last_question()
        if question:
            ctx = self._store.context_for_question(
                question, max_chars=_ANSWER_CONTEXT_CHARS
            )
            self._start(answer_found_question_prompt(question, ctx), kind="answer")
            return
        tail = self._store.recent_text(_QUESTION_TAIL_CHARS)
        if not tail:
            self.status.emit("Пусто", "В транскрипте пока ничего нет")
            return
        self._start(answer_last_question_prompt(tail), kind="answer")

    def last_question(self) -> None:
        """Show the last question without answering it (Alt+Q).

        The '?' path stays instant and works with no provider at all; only a
        transcript without one pays for a request.
        """
        found = self._store.find_last_question()
        if found:
            self.question_ready.emit(found)
            return
        if self._busy:
            return
        tail = self._store.recent_text(_QUESTION_TAIL_CHARS)
        if not tail:
            self.status.emit("Нет вопроса", "Вопрос в транскрипте не найден")
            return
        # No card for this one: the result replaces it wholesale, and a card
        # that says "Думаю…" and then turns into a quote reads as an answer.
        self.status.emit("Вопрос", "Ищу последний вопрос в транскрипте…")
        self._start(find_question_prompt(tail), kind="question", show_card=False)

    def summarize(self, *, auto: bool = False) -> None:
        if self._busy:
            return
        minutes = float(self._cfg.ui.summary_interval_min)
        words = (
            self._store.words_since_minutes(minutes) if minutes > 0
            else self._store.snapshot()
        )
        text = " ".join(w.text for w in words).strip()
        if not text:
            if not auto:
                self.status.emit("Пусто", "За этот интервал ничего не сказано")
            return
        # Keep the most recent text within budget so a long window can't
        # silently overflow the local model's context (LM Studio would drop
        # the middle without telling us).
        truncated = len(text) > self._cfg.llm.max_summary_chars
        if truncated:
            text = text[-self._cfg.llm.max_summary_chars:]
        span = self._minutes_caption(int(minutes)) if minutes > 0 else "за всю сессию"
        prompt = (
            f"Summarize the following speech transcript ({span}). "
            "Focus on key points, decisions, and important information. "
            "Reply in the same language as the transcript.\n\n"
            + ("[earlier text omitted]\n" if truncated else "")
            + text
        )
        self._start(prompt, kind="summary", minutes=minutes, auto=auto)

    def apply_auto_summary(self) -> None:
        interval = self._cfg.ui.summary_interval_min
        if self._cfg.ui.auto_summary and interval > 0:
            self._auto_summary_timer.start(interval * 60 * 1000)
        else:
            self._auto_summary_timer.stop()

    def _start(
        self, prompt: str, *, kind: str, minutes: float = 0.0, auto: bool = False,
        show_card: bool = True,
    ) -> None:
        # No availability pre-check here: it would block the UI thread. The LLM
        # thread checks and reports back through on_error.
        self._busy = True
        self._output = ""
        self._kind = kind
        self._minutes = minutes
        self._auto = auto
        self._set_state(LLM_THINKING)
        # An automatic summary on a hidden window must not pop the card open.
        if show_card and (not auto or self._is_visible()):
            self.answer_started.emit("Думаю…")
        self._llm.submit(
            LLMTask(
                type="answer" if kind == "answer" else "summarize",
                prompt=prompt,
                on_token=self._token_sig.emit,
                on_done=self._done_sig.emit,
                on_error=lambda f: self._error_sig.emit(f.message, f.offline),
            )
        )

    # -- task callbacks (Qt thread) --------------------------------------

    def _on_token(self, token: str) -> None:
        # Tokens flowing => the server is alive and past thinking.
        self._set_state(LLM_GENERATING)
        self._output += token
        self.answer_updated.emit(self._output)

    def _on_done(self) -> None:
        self._busy = False
        self._set_state(LLM_ONLINE)
        if self._kind == "question":
            # This task never opened a card: its whole output *is* the question,
            # and it goes wherever the '?' path's answer would have gone.
            text = self._output.strip()
            if not text or is_no_question(text):
                self.status.emit("Нет вопроса", "Вопрос в транскрипте не найден")
            else:
                self.question_ready.emit(text)
            return
        self.answer_finished.emit()
        if self._kind == "answer" and is_no_question(self._output):
            # The model read the tail and found nothing to answer. Say so in the
            # status line rather than leaving NO_QUESTION on the card.
            self.status.emit("Нет вопроса", "Вопрос в транскрипте не найден")
            self.answer_dismissed.emit()
            return
        if self._kind == "summary" and self._output.strip():
            self._summaries.add(self._output, self._minutes)
            self.summary_added.emit()
            if not self._is_visible() or self._auto:
                self.notify.emit("Выжимка готова", "history", "summary")
        elif self._kind == "answer" and not self._is_visible():
            self.notify.emit("Ответ готов", "toggle_window", "answer")

    def _on_error(self, msg: str, offline: bool) -> None:
        self._busy = False
        # "Офлайн" and "запрос не удался" are different things to know: the
        # first says start the server, the second says try again. The engine
        # decides which it was (llm.engine.LLMFailure) — this used to be guessed
        # by searching the message for "недоступен", which is absent from every
        # failure that has a specific cause.
        self._set_state(LLM_OFFLINE if offline else LLM_ERROR)
        # Keep any tokens already streamed: a mid-stream failure must not wipe a
        # half-written answer, so the error is appended rather than replacing it.
        if self._output.strip():
            self.answer_failed.emit(f"{self._output}\n\n[ошибка] {msg}", False)
        else:
            self.answer_failed.emit(msg, True)

    # -- health ----------------------------------------------------------

    def probe(self) -> None:
        """Ask for a fresh availability check. Cheap and coalescing."""
        # A concurrent health probe mid-generation could flip the indicator on a
        # transient miss while tokens are clearly flowing; skip it while busy.
        if self._busy:
            return
        # A set() on an already-set event is a no-op, so a tick landing while the
        # previous probe is still running is coalesced rather than queued.
        self._health_wake.set()

    def _health_loop(self) -> None:
        """The one thread that ever calls `is_available()`.

        `models.list()` is a network call and must not run on the Qt thread. It
        used to get a freshly spawned thread per tick.
        """
        while not self._health_stop.is_set():
            self._health_wake.wait()
            if self._health_stop.is_set():
                break
            self._health_wake.clear()
            try:
                ok = self._llm.is_available()
            except Exception:  # noqa: BLE001 - any failure means "not available"
                logger.exception("lm health probe failed")
                ok = False
            self._health_sig.emit(ok)

    def _on_health(self, ok: bool) -> None:
        # The probe only decides between offline and idle-online; it must not
        # overwrite an in-flight request's state (it is skipped while busy).
        self._set_state(LLM_ONLINE if ok else LLM_OFFLINE)

    def reset_provider(self) -> None:
        """The settings window swapped the client; re-probe the new one."""
        self._set_state(LLM_OFFLINE)
        QTimer.singleShot(0, self.probe)

    def stop(self) -> None:
        """Release the prober before the window it reports to goes away."""
        self._health_stop.set()
        self._health_wake.set()
