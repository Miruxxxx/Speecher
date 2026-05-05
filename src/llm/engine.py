from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from llm.lmstudio_client import LMStudioClient


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
    LM Studio backend running tasks in a dedicated daemon thread.

    Tasks arrive via submit(); token callbacks run in the LLM thread —
    wire them to Qt signals so updates safely land on the UI thread.
    """

    def __init__(self) -> None:
        self._client = LMStudioClient()
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

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            self._process(task)

    def _process(self, task: LLMTask) -> None:
        if not self._client.is_available():
            task.on_error("LM Studio недоступен — запустите сервер на :1234")
            return
        try:
            messages = [{"role": "user", "content": task.prompt}]
            for token in self._client.stream_completion(messages):
                task.on_token(token)
            task.on_done()
        except Exception as exc:
            task.on_error(str(exc))
