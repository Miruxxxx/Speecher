from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from llm.openai_client import OpenAICompatClient, describe_error, is_offline


LLMTaskType = Literal["answer", "summarize"]


@dataclass(frozen=True, slots=True)
class LLMFailure:
    """Why a task failed, plus the one bit the message text cannot carry.

    `offline` means the provider was not reachable — the indicator says "модель
    офлайн" and the answer buttons grey out. Anything else is a request that
    failed while the server was there, which is a different instruction to the
    user ("попробуйте ещё раз", not "запустите сервер"). The overlay used to
    infer this by searching the Russian message for "недоступен", a word that is
    absent from every failure with a known cause — a bad key or a dead local
    server both reported the wrong state.
    """

    message: str
    offline: bool = False

# The queue is bounded so `submit`'s "queue full" path is reachable at all — it
# used to guard an unbounded Queue and could never fire. In practice the overlay
# already refuses to start a second task while one is running, so this is the
# backstop for a caller that does not (a hotkey held down, a future auto-task),
# not a limit anyone should hit.
_MAX_PENDING_TASKS = 8


@dataclass
class LLMTask:
    type: LLMTaskType
    prompt: str
    on_token: Callable[[str], None]
    on_done: Callable[[], None]
    on_error: Callable[[LLMFailure], None]


class LLMEngine:
    """
    The configured LLM provider, running tasks in a dedicated daemon thread.

    Tasks arrive via submit(); token callbacks run in the LLM thread —
    wire them to Qt signals so updates safely land on the UI thread.
    """

    def __init__(self, client: Optional[OpenAICompatClient] = None) -> None:
        self._client = client or OpenAICompatClient()
        self._queue: queue.Queue[Optional[LLMTask]] = queue.Queue(
            maxsize=_MAX_PENDING_TASKS
        )
        self._thread = threading.Thread(target=self._run, name="llm", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            # Bounded queue: the sentinel needs a slot. If none frees up the
            # worker is wedged inside a request, and it is a daemon thread —
            # shutdown must not block on it.
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            return
        self._thread.join(timeout=10.0)

    def submit(self, task: LLMTask) -> None:
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            task.on_error(LLMFailure("очередь задач переполнена — попробуйте ещё раз"))

    def is_available(self) -> bool:
        return self._client.is_available()

    def set_client(self, client: OpenAICompatClient) -> None:
        """Swap the provider without restarting the app.

        The settings window calls this after the user changes provider, key or
        model. Safe because the reference is only read at the top of _process:
        a task already streaming keeps the client it started with, and the next
        one picks up the new one.
        """
        self._client = client

    def client(self) -> OpenAICompatClient:
        return self._client

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            self._process(task)

    def _process(self, task: LLMTask) -> None:
        client = self._client
        # Availability is checked here, in the LLM thread — never block the UI
        # thread on network calls (the client has short health timeouts and
        # no retries, so a dead server fails fast). For a cloud provider the
        # result is cached, so this is not a request per button press.
        if not client.is_available():
            # The pre-flight probe failed: nothing was sent, the provider is
            # simply not there.
            task.on_error(LLMFailure(client.unavailable_message(), offline=True))
            return
        try:
            messages = [{"role": "user", "content": task.prompt}]
            for token in client.stream_completion(messages):
                task.on_token(token)
            task.on_done()
        except Exception as exc:
            # A failure mid-request is "the request failed", except when the
            # connection itself dropped — then the server went away under us and
            # the indicator should say so.
            task.on_error(
                LLMFailure(describe_error(client.provider, exc), offline=is_offline(exc))
            )
