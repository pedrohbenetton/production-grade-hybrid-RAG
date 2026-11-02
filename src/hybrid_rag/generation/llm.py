from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol, cast

import anthropic

from hybrid_rag.config import LLMSettings


class LLMError(RuntimeError):
    """The LLM provider failed or returned nothing usable."""


class LLMClient(Protocol):
    def complete(self, *, system: str, prompt: str, max_tokens: int) -> str: ...


class _MessageResponse(Protocol):
    @property
    def content(self) -> Sequence[Any]: ...


class _Messages(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        **extra: float,
    ) -> _MessageResponse: ...


class AnthropicLike(Protocol):
    """The slice of ``anthropic.Anthropic`` used here (fakeable in tests)."""

    @property
    def messages(self) -> _Messages: ...


class AnthropicClient:
    def __init__(
        self, client: AnthropicLike, model: str, *, temperature: float | None = None
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def complete(self, *, system: str, prompt: str, max_tokens: int) -> str:
        extra = {} if self._temperature is None else {"temperature": self._temperature}
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        text = "".join(
            str(block.text) for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise LLMError("empty completion from the model")
        return text


def make_llm_client(settings: LLMSettings) -> LLMClient | None:
    """``None`` when generation is disabled or no API key is available."""
    if settings.provider == "none":
        return None
    key = settings.api_key.get_secret_value() if settings.api_key else None
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    sdk = cast(AnthropicLike, anthropic.Anthropic(api_key=key))
    return AnthropicClient(sdk, settings.model, temperature=settings.temperature)
