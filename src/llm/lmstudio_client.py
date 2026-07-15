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

    def _prepare(self, messages: List[dict], include_system: bool) -> List[dict]:
        all_messages: List[dict] = []
        if include_system:
            all_messages.append(self._system_message())
        if self._cfg.qwen_no_think:
            all_messages.extend(self._inject_no_think(messages))
        else:
            all_messages.extend(messages)
        return all_messages

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
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

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
        )
        return (response.choices[0].message.content or "").strip()
