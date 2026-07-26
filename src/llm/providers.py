"""The list of LLM providers, as data.

Every entry here speaks the OpenAI-compatible `/chat/completions` protocol, so
adding a provider is a row in `PROVIDERS`, not a new client. That is the whole
design: `openai.OpenAI(base_url=...)` already works against all of them, and
what actually differs between them fits in this dataclass — the URL, where the
user gets a key, whether one model is implied (a local server) or has to be
picked (a cloud catalogue), and which optional request fields the server will
accept.

Fields the providers *disagree* about at request time (`reasoning_effort`,
`max_tokens` vs `max_completion_tokens`, a non-default `temperature`) are
deliberately NOT enumerated here: see `OpenAICompatClient._request`, which asks
the server once and remembers the answer. A per-provider quirk table would go
stale the week a provider ships a new model family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    caption: str
    base_url: str
    # Local servers need no key and load exactly one model, so the model can be
    # resolved from /models. A cloud catalogue cannot: "the first of 300" is a
    # random model, and silently answering with it is worse than saying nothing.
    local: bool = False
    key_url: str = ""
    # Prefilled in the model picker before /models answers. Left empty unless
    # the id is stable — a wrong guess here is a 404 the user has to decode.
    suggested_model: str = ""
    # Some gateways want an identifying header; harmless everywhere else.
    extra_headers: Dict[str, str] = field(default_factory=dict)
    # Read when no key is stored for this provider, so an existing shell/CI
    # environment keeps working without retyping anything.
    env_var: str = ""
    note: str = ""

    @property
    def needs_key(self) -> bool:
        return not self.local

    @property
    def autoselect_model(self) -> bool:
        return self.local


_OPENROUTER_HEADERS = {
    # OpenRouter attributes traffic by these; both are optional and public.
    "HTTP-Referer": "https://github.com/mirux/speecher",
    "X-Title": "Speecher",
}


PROVIDERS: Tuple[Provider, ...] = (
    Provider(
        id="lmstudio",
        caption="LM Studio (локально)",
        base_url="http://localhost:1234/v1",
        local=True,
        note="Модель берётся та, что загружена в LM Studio.",
    ),
    Provider(
        id="ollama",
        caption="Ollama (локально)",
        base_url="http://localhost:11434/v1",
        local=True,
        note="Нужен запущенный `ollama serve`.",
    ),
    Provider(
        id="openrouter",
        caption="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        key_url="https://openrouter.ai/keys",
        extra_headers=_OPENROUTER_HEADERS,
        env_var="OPENROUTER_API_KEY",
        note="Один ключ на модели почти всех провайдеров сразу.",
    ),
    Provider(
        id="openai",
        caption="OpenAI",
        base_url="https://api.openai.com/v1",
        key_url="https://platform.openai.com/api-keys",
        env_var="OPENAI_API_KEY",
    ),
    Provider(
        id="anthropic",
        caption="Anthropic (Claude)",
        base_url="https://api.anthropic.com/v1",
        key_url="https://console.anthropic.com/settings/keys",
        suggested_model="claude-opus-5",
        env_var="ANTHROPIC_API_KEY",
        note="OpenAI-совместимый слой поверх Messages API.",
    ),
    Provider(
        id="google",
        caption="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_url="https://aistudio.google.com/apikey",
        env_var="GEMINI_API_KEY",
    ),
    Provider(
        id="deepseek",
        caption="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        key_url="https://platform.deepseek.com/api_keys",
        suggested_model="deepseek-chat",
        env_var="DEEPSEEK_API_KEY",
    ),
    Provider(
        id="groq",
        caption="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_url="https://console.groq.com/keys",
        env_var="GROQ_API_KEY",
        note="Самые низкие задержки из облачных.",
    ),
    Provider(
        id="mistral",
        caption="Mistral",
        base_url="https://api.mistral.ai/v1",
        key_url="https://console.mistral.ai/api-keys",
        env_var="MISTRAL_API_KEY",
    ),
    Provider(
        id="xai",
        caption="xAI (Grok)",
        base_url="https://api.x.ai/v1",
        key_url="https://console.x.ai",
        env_var="XAI_API_KEY",
    ),
    Provider(
        id="together",
        caption="Together AI",
        base_url="https://api.together.xyz/v1",
        key_url="https://api.together.ai/settings/api-keys",
        env_var="TOGETHER_API_KEY",
    ),
    Provider(
        id="fireworks",
        caption="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        key_url="https://fireworks.ai/account/api-keys",
        env_var="FIREWORKS_API_KEY",
    ),
    Provider(
        id="cerebras",
        caption="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_url="https://cloud.cerebras.ai",
        env_var="CEREBRAS_API_KEY",
    ),
    Provider(
        id="deepinfra",
        caption="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai",
        key_url="https://deepinfra.com/dash/api_keys",
        env_var="DEEPINFRA_API_KEY",
    ),
    Provider(
        id="custom",
        caption="Свой сервер",
        # Nothing to default to: `llm.base_url` in config.toml *is* the setting.
        base_url="",
        key_url="",
        note="Задайте base_url в config.toml; ключ — если сервер его требует.",
    ),
)

_BY_ID = {p.id: p for p in PROVIDERS}

DEFAULT_PROVIDER = "lmstudio"

# `custom` has no key requirement of its own — a self-hosted vLLM usually needs
# none — but the key field must still be offered, so it is not `local`.
CUSTOM = _BY_ID["custom"]


def ids() -> Tuple[str, ...]:
    return tuple(p.id for p in PROVIDERS)


def get(provider_id: str) -> Optional[Provider]:
    return _BY_ID.get((provider_id or "").strip().lower())


def resolve(provider_id: str) -> Provider:
    """Provider by id, falling back to the default rather than raising.

    Config values reach here unvalidated; an unknown provider must degrade to
    the local server with a warning, exactly like an unknown asr.engine does.
    """
    return get(provider_id) or _BY_ID[DEFAULT_PROVIDER]


def base_url_for(provider: Provider, override: str = "") -> str:
    """The URL to talk to: an explicit config override wins over the registry."""
    return (override or "").strip() or provider.base_url
