from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from hybrid_rag.config import LLMSettings
from hybrid_rag.generation.llm import AnthropicClient, LLMError, make_llm_client


@dataclass
class Block:
    type: str
    text: str = ""


@dataclass
class Response:
    content: list[Block]


@dataclass
class FakeMessages:
    response: Response | Exception
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Response:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@dataclass
class FakeAnthropic:
    messages: FakeMessages


def client_for(
    response: Response | Exception, **kwargs: Any
) -> tuple[AnthropicClient, FakeMessages]:
    messages = FakeMessages(response)
    return AnthropicClient(cast(Any, FakeAnthropic(messages)), model="m", **kwargs), messages


def test_text_blocks_are_joined_and_other_blocks_ignored() -> None:
    client, messages = client_for(
        Response([Block("text", "Hello "), Block("tool_use"), Block("text", "world")])
    )

    assert client.complete(system="sys", prompt="question", max_tokens=50) == "Hello world"
    assert messages.calls == [
        {
            "model": "m",
            "max_tokens": 50,
            "system": "sys",
            "messages": [{"role": "user", "content": "question"}],
        }
    ]


def test_temperature_is_only_sent_when_configured() -> None:
    client, messages = client_for(Response([Block("text", "x")]), temperature=0.0)
    client.complete(system="s", prompt="p", max_tokens=5)
    assert messages.calls[0]["temperature"] == 0.0


def test_empty_completion_is_an_error() -> None:
    client, _ = client_for(Response([Block("tool_use")]))
    with pytest.raises(LLMError, match="empty"):
        client.complete(system="s", prompt="p", max_tokens=5)


def test_sdk_errors_are_wrapped() -> None:
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.invalid"))
    client, _ = client_for(error)
    with pytest.raises(LLMError):
        client.complete(system="s", prompt="p", max_tokens=5)


def test_factory_returns_none_when_disabled_or_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_llm_client(LLMSettings(provider="none", api_key=SecretStr("k"))) is None
    assert make_llm_client(LLMSettings(provider="anthropic")) is None


def test_factory_builds_a_client_from_settings_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_llm_client(LLMSettings(api_key=SecretStr("k"))), AnthropicClient)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert isinstance(make_llm_client(LLMSettings()), AnthropicClient)
