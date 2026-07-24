from __future__ import annotations

import logging
from typing import Generator, Iterator, List, Optional

import openai

from app_config import LlmConfig

logger = logging.getLogger(__name__)


class LMStudioClient:
    """
    Тонкая обёртка над LM Studio OpenAI-compatible API.

    Всё сетевое настроено на «локальный сервер, который может быть выключен»:
    без ретраев, health-check с коротким таймаутом. Вызовы блокирующие —
    использовать только из LLM-потока (LLMEngine), не из UI.
    """

    def __init__(self, cfg: Optional[LlmConfig] = None) -> None:
        self._cfg = cfg or LlmConfig()
        # api_key обязателен для openai-клиента, но LM Studio его игнорирует.
        self._client = openai.OpenAI(
            base_url=self._cfg.base_url,
            api_key="lm-studio",
            timeout=self._cfg.request_timeout_sec,
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # Health / model resolution
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Быстрая проверка, что сервер запущен и модель загружена."""
        try:
            models = self._client.with_options(
                timeout=self._cfg.health_timeout_sec
            ).models.list()
            return len(models.data) > 0
        except Exception:
            return False

    def loaded_model_id(self) -> Optional[str]:
        """ID первой загруженной модели, или None."""
        try:
            models = self._client.with_options(
                timeout=self._cfg.health_timeout_sec
            ).models.list()
            if models.data:
                return models.data[0].id
        except Exception:
            pass
        return None

    def resolve_model(self) -> Optional[str]:
        """Модель из конфига, иначе — первая загруженная в LM Studio."""
        if self._cfg.model:
            return self._cfg.model
        return self.loaded_model_id()

    # ------------------------------------------------------------------
    # Промпт-обвязка
    # ------------------------------------------------------------------

    @staticmethod
    def _system_message() -> dict:
        return {
            "role": "system",
            "content": "You are a helpful, concise assistant.",
        }

    def _prepare(self, messages: List[dict], include_system: bool) -> List[dict]:
        all_messages: List[dict] = []
        if include_system:
            all_messages.append(self._system_message())
        all_messages.extend(messages)
        return all_messages

    def _extra(self) -> dict:
        """Request fields beyond the standard ones.

        `reasoning_effort` is how a reasoning model is told not to think. It
        matters more than it looks: with thinking on, LM Studio streams the
        chain-of-thought in `delta.reasoning_content` (never in `content`), and
        on a long prompt it exhausts `max_tokens` before answering — the stream
        ends with finish_reason="length" and the caller gets an empty string.
        """
        effort = (self._cfg.reasoning_effort or "").strip()
        return {"reasoning_effort": effort} if effort else {}

    # ------------------------------------------------------------------
    # Вызовы
    # ------------------------------------------------------------------

    def stream_completion(
        self,
        messages: List[dict],
        *,
        include_system: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Стримит токены по одному. Блокирует — использовать в LLM-потоке."""
        model = self.resolve_model()
        if model is None:
            raise RuntimeError("LM Studio: нет загруженной модели")
        stream: Iterator = self._client.chat.completions.create(
            model=model,
            messages=self._prepare(messages, include_system),
            temperature=temperature if temperature is not None else self._cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._cfg.max_tokens,
            stream=True,
            **self._extra(),
        )
        produced = False
        finish: Optional[str] = None
        for chunk in stream:
            # Some OpenAI-compatible servers send a final chunk with an empty
            # choices list (usage-only); indexing it blindly would raise.
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            # Only `content` is the answer. A reasoning model's chain-of-thought
            # arrives in `delta.reasoning_content` and is deliberately dropped.
            delta = choice.delta.content
            if delta:
                produced = True
                yield delta
        if not produced:
            raise RuntimeError(self._empty_answer_reason(finish))

    def complete(
        self,
        messages: List[dict],
        *,
        include_system: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Блокирующий вызов, возвращает полный текст ответа."""
        model = self.resolve_model()
        if model is None:
            raise RuntimeError("LM Studio: нет загруженной модели")
        response = self._client.chat.completions.create(
            model=model,
            messages=self._prepare(messages, include_system),
            temperature=temperature if temperature is not None else self._cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._cfg.max_tokens,
            stream=False,
            **self._extra(),
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise RuntimeError(self._empty_answer_reason(choice.finish_reason))
        return text

    @staticmethod
    def _empty_answer_reason(finish_reason: Optional[str]) -> str:
        """Turn a silently-empty answer into something diagnosable."""
        if finish_reason == "length":
            return (
                "модель израсходовала max_tokens на размышления и не дала ответ — "
                'поставьте llm.reasoning_effort = "none" в config/config.toml '
                "или поднимите llm.max_tokens"
            )
        return f"модель вернула пустой ответ (finish_reason={finish_reason!r})"
