from __future__ import annotations

import logging
import time
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple

import openai

from app_config import LlmConfig
from llm import credentials, providers
from llm.providers import Provider

logger = logging.getLogger(__name__)

# A cloud endpoint does not go up and down the way a local server does, and the
# overlay re-probes every 20 s — without a TTL that is 180 pointless requests an
# hour. A local server is a localhost round trip; probe it every time.
_CLOUD_HEALTH_TTL_SEC = 300.0

# Optional request fields providers disagree about. Dropped one at a time, in
# this order, when the server says it does not know them.
_OPTIONAL_FIELDS = ("reasoning_effort", "temperature")


class OpenAICompatClient:
    """One client for every provider in `llm/providers.py`.

    They all speak OpenAI's `/chat/completions`, so there is no per-provider
    subclass — only a `Provider` row and a key. What is genuinely
    provider-specific lives in two places: `_request`, which learns which
    optional fields this server rejects, and `describe_error`, which turns an
    HTTP status into something a user can act on.

    Calls block — use them from the LLM thread (`LLMEngine`), never from Qt.
    """

    def __init__(
        self,
        cfg: Optional[LlmConfig] = None,
        *,
        provider: Optional[Provider] = None,
        api_key: Optional[str] = None,
        client: Optional[object] = None,
    ) -> None:
        self._cfg = cfg or LlmConfig()
        self._provider = provider or providers.resolve(self._cfg.provider)
        self._base_url = providers.base_url_for(self._provider, self._cfg.base_url)
        if api_key is None:
            api_key = credentials.store().get(self._provider)
        self._api_key = (api_key or "").strip()
        # Fields this server has already rejected once; never sent again this
        # session. See _request.
        self._quirks: Set[str] = set()
        self._health: Optional[bool] = None
        self._health_at = 0.0
        self._health_message = ""

        if client is not None:
            self._client = client          # injected by the tests
        elif not self._base_url:
            # `custom` without a base_url: there is nothing to point openai at,
            # and constructing it would raise on the first call instead of here.
            self._client = None
            self._health, self._health_message = False, "не задан base_url в config.toml"
        else:
            self._client = openai.OpenAI(
                base_url=self._base_url,
                # Required by the SDK; local servers ignore whatever is sent.
                api_key=self._api_key or "no-key",
                timeout=self._cfg.request_timeout_sec,
                max_retries=0,
                default_headers=dict(self._provider.extra_headers) or None,
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def base_url(self) -> str:
        return self._base_url

    def has_key(self) -> bool:
        return bool(self._api_key)

    def is_configured(self) -> bool:
        """Everything needed to make a call is present (nothing was tried yet)."""
        if self._client is None:
            return False
        if self._provider.needs_key and not self._api_key and self._provider.id != "custom":
            return False
        return self._provider.autoselect_model or bool(self.resolve_model())

    # ------------------------------------------------------------------
    # Health / model resolution
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Cheap enough for a 20 s poll: cloud results are cached, local are not."""
        if not self.is_configured():
            return False
        ttl = 0.0 if self._provider.local else _CLOUD_HEALTH_TTL_SEC
        if self._health is not None and (time.monotonic() - self._health_at) < ttl:
            return self._health
        return self.check()[0]

    def check(self, *, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """Really talk to the server. Returns (ok, message in Russian)."""
        if self._client is None:
            return False, self._health_message or "клиент не настроен"
        if self._provider.needs_key and not self._api_key and self._provider.id != "custom":
            return self._remember(False, "не задан API-ключ")
        try:
            models = self._client.with_options(
                timeout=timeout if timeout is not None else self._cfg.health_timeout_sec
            ).models.list()
        except Exception as exc:  # noqa: BLE001 - every failure is "not available"
            return self._remember(False, describe_error(self._provider, exc))
        if self._provider.autoselect_model and not models.data:
            return self._remember(False, "сервер запущен, но модель не загружена")
        return self._remember(True, "")

    def _remember(self, ok: bool, message: str) -> Tuple[bool, str]:
        self._health, self._health_at, self._health_message = ok, time.monotonic(), message
        return ok, message

    def unavailable_message(self) -> str:
        """What the answer card says when a task cannot even be started."""
        return f"{self._provider.caption}: {self._health_message or 'недоступен'}"

    def list_models(self, *, timeout: float = 10.0) -> List[str]:
        """Model ids for the settings picker. Raises — the caller shows why."""
        if self._client is None:
            raise RuntimeError(self._health_message or "клиент не настроен")
        models = self._client.with_options(timeout=timeout).models.list()
        return sorted(m.id for m in models.data if getattr(m, "id", ""))

    def loaded_model_id(self) -> Optional[str]:
        try:
            models = self._client.with_options(
                timeout=self._cfg.health_timeout_sec
            ).models.list()
            if models.data:
                return models.data[0].id
        except Exception:  # noqa: BLE001
            pass
        return None

    def resolve_model(self) -> Optional[str]:
        """Configured model, or — for a local server only — the loaded one.

        A cloud catalogue is deliberately not auto-resolved: picking "the first
        of 300 models" would answer with something arbitrary and bill for it.
        """
        model = (self._cfg.model or "").strip()
        if model:
            return model
        if self._provider.autoselect_model:
            return self.loaded_model_id()
        return self._provider.suggested_model or None

    def _require_model(self) -> str:
        model = self.resolve_model()
        if model:
            return model
        if self._provider.autoselect_model:
            raise RuntimeError(f"{self._provider.caption}: нет загруженной модели")
        raise RuntimeError(f"{self._provider.caption}: не выбрана модель — откройте настройки")

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

    def _payload(
        self,
        messages: List[dict],
        include_system: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Dict:
        """The request as we would like to send it, before any quirk is applied.

        `reasoning_effort` is how a reasoning model is told not to think. It
        matters more than it looks: with thinking on, the server streams the
        chain-of-thought in `delta.reasoning_content` (never in `content`), and
        on a long prompt it exhausts `max_tokens` before answering — the stream
        ends with finish_reason="length" and the caller gets an empty string.
        """
        payload: Dict = {
            "model": self._require_model(),
            "messages": self._prepare(messages, include_system),
            "temperature": temperature if temperature is not None else self._cfg.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._cfg.max_tokens,
            "stream": stream,
        }
        effort = (self._cfg.reasoning_effort or "").strip()
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    # ------------------------------------------------------------------
    # Request, with the compatibility retry
    # ------------------------------------------------------------------

    def _apply_quirks(self, payload: Dict) -> Dict:
        payload = dict(payload)
        for quirk in self._quirks:
            if quirk == "max_completion_tokens":
                if "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
            else:
                payload.pop(quirk, None)
        return payload

    @staticmethod
    def _quirk_from_error(message: str, payload: Dict) -> Optional[str]:
        """Which field to stop sending, judging by what the server complained about.

        Deliberately driven by the error text rather than a per-provider table:
        the set of accepted fields changes with every new model family, and a
        table encoding it would be wrong within a month. The server is the only
        authority on what it takes, so ask it once and remember.
        """
        text = message.lower()
        if "max_completion_tokens" in text and "max_tokens" in payload:
            return "max_completion_tokens"
        for name in _OPTIONAL_FIELDS:
            if name in payload and name in text:
                return name
        return None

    def _request(self, payload: Dict):
        """Send the request, retrying without whatever field the server rejects.

        Bounded by the number of fields that can be dropped, and each drop is
        remembered for the session — a steady state is reached after the first
        call to a given provider.
        """
        for _ in range(len(_OPTIONAL_FIELDS) + 2):
            attempt = self._apply_quirks(payload)
            try:
                return self._client.chat.completions.create(**attempt)
            except openai.BadRequestError as exc:
                quirk = self._quirk_from_error(str(exc), attempt)
                if quirk is None or quirk in self._quirks:
                    raise
                logger.info(
                    "llm: %s rejected a field, retrying with %s",
                    self._provider.id, quirk,
                )
                self._quirks.add(quirk)
        raise RuntimeError(f"{self._provider.caption}: не удалось согласовать параметры запроса")

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
        stream: Iterator = self._request(
            self._payload(messages, include_system, temperature, max_tokens, True)
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
        response = self._request(
            self._payload(messages, include_system, temperature, max_tokens, False)
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


def is_offline(exc: Exception) -> bool:
    """The provider could not be reached at all, as opposed to refusing us.

    This is the distinction the title-bar indicator draws — "модель офлайн"
    (start the server / check the network) against "ошибка" (the request failed,
    try again). It cannot be recovered from the message text: `describe_error`
    returns a specific sentence for every case it recognises, and the word the
    overlay used to search for ("недоступен") appears in none of them.
    """
    return isinstance(exc, openai.APIConnectionError)


def describe_error(provider: Provider, exc: Exception) -> str:
    """An HTTP failure as an instruction, not a stack trace.

    The same status means different things per provider: a refused connection
    to localhost means "the server is not running", to api.openai.com it means
    "no internet".
    """
    if isinstance(exc, openai.AuthenticationError):
        return "неверный API-ключ"
    if isinstance(exc, openai.PermissionDeniedError):
        return "ключ не даёт доступа — проверьте права или оплату"
    if isinstance(exc, openai.NotFoundError):
        return "модель не найдена у этого провайдера"
    if isinstance(exc, openai.RateLimitError):
        return "лимит запросов исчерпан — попробуйте позже"
    if isinstance(exc, openai.APIConnectionError):
        if provider.local:
            return "сервер не отвечает — запустите его"
        return "нет соединения с сервером провайдера"
    if isinstance(exc, openai.APIStatusError):
        return f"сервер ответил {exc.status_code}"
    return str(exc)
