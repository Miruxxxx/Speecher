from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from llm.openai_client import OpenAICompatClient, describe_error


LLMTaskType = Literal["answer", "summarize"]


@dataclass
class LLMTask:
    type: LLMTaskType
    prompt: str
    on_token: Callable[[str], None]
    on_done: Callable[[], None]
    on_error: Callable[[str], None]


class LLMEngine:
    """
    The configured LLM provider, running tasks in a dedicated daemon thread.

    Tasks arrive via submit(); token callbacks run in the LLM thread —
    wire them to Qt signals so updates safely land on the UI thread.
    """

    def __init__(self, client: Optional[OpenAICompatClient] = None) -> None:
        self._client = client or OpenAICompatClient()
        self._queue: queue.Queue[Optional[LLMTask]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="llm", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)

    def submit(self, task: LLMTask) -> None:
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            task.on_error("LLM task queue full — try again")

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
            task.on_error(client.unavailable_message())
            return
        try:
            messages = [{"role": "user", "content": task.prompt}]
            for token in client.stream_completion(messages):
                task.on_token(token)
            task.on_done()
        except Exception as exc:
            task.on_error(describe_error(client.provider, exc))
