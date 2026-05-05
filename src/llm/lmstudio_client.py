from __future__ import annotations

import threading
from typing import Callable, Generator, Iterator, List, Optional

import openai


class LMStudioClient:
    """
    Тонкая обёртка над LM Studio OpenAI-compatible API.

    Использование:
        client = LMStudioClient()

        # Проверить что сервер живой
        if not client.is_available():
            print("LM Studio не запущен")

        # Стриминг
        for chunk in client.stream_completion(messages=[...]):
            print(chunk, end="", flush=True)

        # Одним куском
        text = client.complete(messages=[...])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen3.5-9b",
        temperature: float = 0.6,
        max_tokens: int = 2048,
        timeout: float = 10.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        # api_key обязателен для openai клиента, но LM Studio его игнорирует
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key="lm-studio",
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Быстрая проверка что LM Studio сервер запущен и модель загружена."""
        try:
            models = self._client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    def loaded_model_id(self) -> Optional[str]:
        """Вернуть ID первой загруженной модели, или None."""
        try:
            models = self._client.models.list()
            if models.data:
                return models.data[0].id
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Системный промпт
    # ------------------------------------------------------------------

    @staticmethod
    def _no_think_system() -> dict:
        return {
            "role": "system",
            "content": "You are a helpful, concise assistant.",
        }

    @staticmethod
    def _inject_no_think(messages: list) -> list:
        """Prepend /no_think to the first user message so Qwen3 skips chain-of-thought."""
        out = []
        done = False
        for msg in messages:
            if not done and msg.get("role") == "user":
                out.append({**msg, "content": "/no_think " + msg["content"]})
                done = True
            else:
                out.append(msg)
        return out

    # ------------------------------------------------------------------
    # Стриминг
    # ------------------------------------------------------------------

    def stream_completion(
        self,
        messages: List[dict],
        *,
        include_system: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Стримит токены по одному. Используй в отдельном потоке.

        Пример:
            for token in client.stream_completion([
                {"role": "user", "content": "Summarize: ..."}
            ]):
                print(token, end="", flush=True)
        """
        all_messages = []
        if include_system:
            all_messages.append(self._no_think_system())
        all_messages.extend(self._inject_no_think(messages))

        stream: Iterator = self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Одним куском (для перевода — короткие ответы)
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[dict],
        *,
        include_system: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Блокирующий вызов, возвращает полный текст ответа."""
        all_messages = []
        if include_system:
            all_messages.append(self._no_think_system())
        all_messages.extend(self._inject_no_think(messages))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Хелперы для конкретных задач
    # ------------------------------------------------------------------

    def translate(self, text: str, target_language: str = "English") -> str:
        """Перевод одного committed-чанка. Неблокирующий вывод не нужен — чанки короткие."""
        return self.complete(
            messages=[{
                "role": "user",
                "content": f"Translate to {target_language}. Output only the translation, nothing else:\n{text}",
            }],
            max_tokens=256,
            temperature=0.3,
        )

    def summarize(self, text: str, minutes: Optional[float] = None) -> Generator[str, None, None]:
        """Саммари с стримингом. minutes=None -> всего текста."""
        label = f"the last {minutes} minutes of" if minutes else "the following"
        return self.stream_completion(
            messages=[{
                "role": "user",
                "content": (
                    f"Write a concise summary of {label} this meeting transcript. "
                    f"Use bullet points. Focus on decisions, action items, and key points:\n\n{text}"
                ),
            }],
            max_tokens=512,
        )

    def answer_question(self, question: str, context: str) -> Generator[str, None, None]:
        """Ответ на последний вопрос из контекста разговора."""
        return self.stream_completion(
            messages=[{
                "role": "user",
                "content": (
                    f"Based on this meeting transcript context:\n\n{context}\n\n"
                    f"Answer this question concisely: {question}"
                ),
            }],
            max_tokens=300,
        )
