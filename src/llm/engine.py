from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Literal, Optional


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
    llama-cpp-python wrapper running in a dedicated daemon thread.

    Tasks arrive via submit(); token callbacks run in the LLM thread —
    wire them to Qt signals so updates safely land on the UI thread.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
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

    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
            llm = Llama(
                model_path=self._model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False,
            )
        except Exception as exc:
            self._drain_with_error(f"LLM failed to load: {exc}")
            return

        while True:
            task = self._queue.get()
            if task is None:
                break
            self._process(llm, task)

    def _process(self, llm, task: LLMTask) -> None:
        try:
            stream = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Be concise and clear.",
                    },
                    {"role": "user", "content": task.prompt},
                ],
                stream=True,
                max_tokens=512,
                temperature=0.3,
            )
            for chunk in stream:
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    task.on_token(content)
            task.on_done()
        except Exception as exc:
            task.on_error(str(exc))

    def _drain_with_error(self, msg: str) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            task.on_error(msg)
