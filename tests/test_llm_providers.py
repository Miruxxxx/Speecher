"""Provider registry, credential store and the request-compatibility retry.

No network and no Credential Manager: the store is driven through its file
backend, and the client is given a fake `openai` client that records what it
was actually sent.
"""

from __future__ import annotations

from pathlib import Path

import openai
import pytest

from app_config import AppConfig, LlmConfig, load_config, save_overrides
from llm import providers
from llm.credentials import CredentialStore, mask
from llm.openai_client import OpenAICompatClient, describe_error


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_ids_are_unique_and_resolvable():
    assert len(providers.ids()) == len(set(providers.ids()))
    for pid in providers.ids():
        assert providers.resolve(pid).id == pid


def test_unknown_provider_falls_back_to_the_default():
    assert providers.resolve("nope").id == providers.DEFAULT_PROVIDER
    assert providers.resolve("").id == providers.DEFAULT_PROVIDER


def test_every_cloud_provider_can_be_signed_up_for():
    """A key field with nowhere to get the key is a dead end for the user."""
    for p in providers.PROVIDERS:
        if p.local or p.id == "custom":
            continue
        assert p.key_url, p.id
        assert p.base_url.startswith("https://"), p.id
        assert not p.base_url.endswith("/"), p.id


def test_only_local_providers_autoselect_a_model():
    """Picking one of a cloud catalogue's hundreds at random must never happen."""
    for p in providers.PROVIDERS:
        assert p.autoselect_model == p.local, p.id


def test_base_url_override_wins():
    custom = providers.get("custom")
    assert providers.base_url_for(custom, "http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    assert providers.base_url_for(custom, "") == ""
    lm = providers.get("lmstudio")
    assert providers.base_url_for(lm, "") == lm.base_url


# ----------------------------------------------------------------------
# Credentials
# ----------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> CredentialStore:
    return CredentialStore(path=tmp_path / "credentials.json", use_keyring=False)


def test_file_store_round_trip(store: CredentialStore):
    assert store.get(providers.get("groq")) == ""
    store.set("groq", "gsk-secret")
    assert store.has_stored("groq")
    assert store.get(providers.get("groq")) == "gsk-secret"
    store.delete("groq")
    assert not store.has_stored("groq")


def test_empty_key_deletes(store: CredentialStore):
    store.set("groq", "gsk-secret")
    store.set("groq", "   ")
    assert not store.has_stored("groq")


def test_environment_is_the_last_resort(store: CredentialStore, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert store.get(providers.get("groq")) == "from-env"
    # An explicitly stored key outranks the environment.
    store.set("groq", "stored")
    assert store.get(providers.get("groq")) == "stored"


def test_unreadable_store_reads_as_no_key(tmp_path: Path):
    path = tmp_path / "credentials.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = CredentialStore(path=path, use_keyring=False)
    assert store.get(providers.get("groq")) == ""


def test_mask_never_shows_a_short_key():
    assert mask("sk-abcdefgh") == "••••efgh"
    assert mask("abc") == "••••"
    assert mask("") == ""


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


def test_unknown_provider_in_config_falls_back(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[llm]\nprovider = "skynet"\n', encoding="utf-8")
    assert load_config(p).llm.provider == providers.DEFAULT_PROVIDER


def test_remembered_model_is_used_only_when_model_is_empty(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[llm]\nprovider = "groq"\nmodel = ""\n\n[llm.models]\ngroq = "llama-3.3"\n',
        encoding="utf-8",
    )
    assert load_config(p).llm.model == "llama-3.3"

    p.write_text(
        '[llm]\nprovider = "groq"\nmodel = "hand-edited"\n\n'
        '[llm.models]\ngroq = "llama-3.3"\n',
        encoding="utf-8",
    )
    # A hand-edited `model` always wins — the memory must never override it.
    assert load_config(p).llm.model == "hand-edited"


def test_settings_write_round_trips_through_config(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[llm]\nprovider = "lmstudio"  # комментарий\nmodel = ""\n\n[llm.models]\n',
        encoding="utf-8",
    )
    save_overrides(
        p,
        {
            "llm": {"provider": "openrouter", "model": "some/model"},
            "llm.models": {"openrouter": "some/model", "lmstudio": "local"},
        },
    )
    cfg = load_config(p)
    assert cfg.llm.provider == "openrouter"
    assert cfg.llm.model == "some/model"
    assert cfg.llm.models["lmstudio"] == "local"
    # The Russian comments the file exists for must survive the write.
    assert "# комментарий" in p.read_text(encoding="utf-8")


def test_config_carries_no_key_field():
    """The one invariant of this feature: config.toml is in git.

    Guards against the obvious future convenience — "just add llm.api_key" —
    which would publish the key on the next commit.
    """
    secretish = ("api_key", "apikey", "secret", "password", "token=", "credential")
    for name in LlmConfig.__dataclass_fields__:
        assert not any(s in name for s in secretish), name
    text = Path("config/config.toml").read_text(encoding="utf-8")
    assert "api_key" not in text


# ----------------------------------------------------------------------
# Client: the compatibility retry
# ----------------------------------------------------------------------


class _Response:
    def __init__(self, text: str = "ok"):
        message = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": message, "finish_reason": "stop"})()]


class _FakeCompletions:
    """Rejects the named fields the way a real server does, once each."""

    def __init__(self, rejects=()):
        self.rejects = set(rejects)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        for name in self.rejects:
            if name in kwargs:
                raise openai.BadRequestError(
                    f"Unsupported parameter: '{name}' is not supported with this model. "
                    f"Use 'max_completion_tokens' instead."
                    if name == "max_tokens"
                    else f"Unrecognized request argument supplied: {name}",
                    response=_httpx_response(),
                    body=None,
                )
        return _Response()


def _httpx_response():
    import httpx

    return httpx.Response(400, request=httpx.Request("POST", "http://x/v1/chat/completions"))


class _FakeClient:
    def __init__(self, rejects=(), models=("m1", "m2")):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(rejects)})()
        self._models = models

    def with_options(self, **_kwargs):
        return self

    @property
    def models(self):
        data = [type("M", (), {"id": name})() for name in self._models]
        return type("L", (), {"list": lambda _self=None: type("R", (), {"data": data})()})()


def _client(rejects=(), **cfg_kwargs) -> OpenAICompatClient:
    cfg = LlmConfig(provider="groq", model="m1", reasoning_effort="none", **cfg_kwargs)
    fake = _FakeClient(rejects)
    client = OpenAICompatClient(cfg, api_key="k", client=fake)
    client._fake = fake  # test handle
    return client


def test_request_passes_through_when_nothing_is_rejected():
    c = _client()
    assert c.complete([{"role": "user", "content": "hi"}]) == "ok"
    sent = c._fake.chat.completions.calls
    assert len(sent) == 1
    assert sent[0]["reasoning_effort"] == "none"
    assert sent[0]["max_tokens"]


def test_rejected_reasoning_effort_is_dropped_and_remembered():
    c = _client(rejects=("reasoning_effort",))
    assert c.complete([{"role": "user", "content": "hi"}]) == "ok"
    calls = c._fake.chat.completions.calls
    assert len(calls) == 2
    assert "reasoning_effort" not in calls[1]

    # The second task must not repeat the failed attempt.
    c.complete([{"role": "user", "content": "again"}])
    assert len(calls) == 3
    assert "reasoning_effort" not in calls[2]


def test_max_tokens_is_renamed_when_the_server_asks_for_it():
    c = _client(rejects=("max_tokens",))
    assert c.complete([{"role": "user", "content": "hi"}]) == "ok"
    last = c._fake.chat.completions.calls[-1]
    assert "max_tokens" not in last
    assert last["max_completion_tokens"] == LlmConfig().max_tokens


def test_an_unrelated_bad_request_is_not_retried():
    class _Always:
        calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise openai.BadRequestError(
                "context length exceeded", response=_httpx_response(), body=None
            )

    cfg = LlmConfig(provider="groq", model="m1")
    fake = _FakeClient()
    fake.chat = type("Chat", (), {"completions": _Always()})()
    c = OpenAICompatClient(cfg, api_key="k", client=fake)
    with pytest.raises(openai.BadRequestError):
        c.complete([{"role": "user", "content": "hi"}])
    assert len(fake.chat.completions.calls) == 1


# ----------------------------------------------------------------------
# Client: model resolution and health
# ----------------------------------------------------------------------


def test_cloud_provider_without_a_model_refuses_to_guess():
    cfg = LlmConfig(provider="groq", model="")
    c = OpenAICompatClient(cfg, api_key="k", client=_FakeClient())
    assert c.resolve_model() is None
    with pytest.raises(RuntimeError, match="не выбрана модель"):
        c.complete([{"role": "user", "content": "hi"}])


def test_local_provider_takes_the_loaded_model():
    cfg = LlmConfig(provider="lmstudio", model="")
    c = OpenAICompatClient(cfg, api_key="", client=_FakeClient(models=("loaded",)))
    assert c.resolve_model() == "loaded"


def test_cloud_provider_without_a_key_is_unavailable():
    cfg = LlmConfig(provider="groq", model="m1")
    c = OpenAICompatClient(cfg, api_key="", client=_FakeClient())
    assert not c.is_configured()
    assert not c.is_available()
    assert "Groq" in c.unavailable_message()


def test_cloud_health_is_cached_but_local_is_not():
    cfg = LlmConfig(provider="groq", model="m1")
    fake = _FakeClient()
    seen = []
    original = fake.with_options
    fake.with_options = lambda **kw: (seen.append(kw), original(**kw))[1]
    c = OpenAICompatClient(cfg, api_key="k", client=fake)
    assert c.is_available()
    assert c.is_available()
    assert len(seen) == 1  # second call served from the cache

    local = OpenAICompatClient(
        LlmConfig(provider="lmstudio"), api_key="", client=_FakeClient()
    )
    seen_local = []
    orig = local._client.with_options
    local._client.with_options = lambda **kw: (seen_local.append(kw), orig(**kw))[1]
    local.is_available()
    local.is_available()
    assert len(seen_local) == 2


def test_custom_provider_without_base_url_says_so():
    c = OpenAICompatClient(LlmConfig(provider="custom", base_url=""), api_key="")
    assert not c.is_available()
    assert "base_url" in c.unavailable_message()


# ----------------------------------------------------------------------
# Error messages
# ----------------------------------------------------------------------


def test_connection_error_reads_differently_local_vs_cloud():
    exc = openai.APIConnectionError(request=_httpx_response().request)
    local = describe_error(providers.get("lmstudio"), exc)
    cloud = describe_error(providers.get("groq"), exc)
    assert local != cloud
    assert "запустите" in local


def test_auth_error_names_the_key():
    exc = openai.AuthenticationError(
        "bad key",
        response=__import__("httpx").Response(
            401, request=__import__("httpx").Request("GET", "http://x/v1/models")
        ),
        body=None,
    )
    assert "ключ" in describe_error(providers.get("groq"), exc)


# -- failure classification -------------------------------------------------


def test_offline_is_a_flag_not_a_word_in_the_message():
    """The indicator must not be driven by searching Russian prose.

    `unavailable_message()` only contains "недоступен" when the client has no
    idea why. Every failure with a known cause reads differently — a dead local
    server says "сервер не отвечает — запустите его", a bad key says "неверный
    API-ключ" — so the substring test the overlay used to do reported those as
    "ошибка" instead of "офлайн".
    """
    import openai

    from llm.engine import LLMEngine, LLMFailure, LLMTask
    from llm.openai_client import describe_error, is_offline

    provider = providers.get("lmstudio")
    exc = openai.APIConnectionError(request=_httpx_response().request)
    message = describe_error(provider, exc)

    assert is_offline(exc) is True
    assert "недоступен" not in message      # the old test would have missed it

    seen: list[LLMFailure] = []

    class DeadClient:
        provider = providers.get("lmstudio")

        def is_available(self) -> bool:
            return False

        def unavailable_message(self) -> str:
            return "LM Studio: сервер не отвечает — запустите его"

    engine = LLMEngine.__new__(LLMEngine)
    engine._client = DeadClient()
    engine._process(
        LLMTask(type="answer", prompt="x", on_token=lambda _t: None,
                on_done=lambda: None, on_error=seen.append)
    )

    assert len(seen) == 1
    assert seen[0].offline is True
    assert "недоступен" not in seen[0].message
