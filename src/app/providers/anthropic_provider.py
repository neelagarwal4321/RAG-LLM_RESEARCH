from __future__ import annotations

from typing import List

import anthropic

from ..core.config import settings
from .base import ChatMessage, LLMProvider, ProviderResponse, registry


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        super().__init__()
        self._client: anthropic.Anthropic | None = None

    def is_available(self) -> bool:
        return bool(settings.anthropic_api_key)

    def list_models(self) -> List[str]:
        return ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]

    def _client_or_raise(self) -> anthropic.Anthropic:
        if not self.is_available():
            raise ValueError("Anthropic provider not configured (missing ANTHROPIC_API_KEY)")
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def _generate(self, messages: List[ChatMessage], model: str, temperature: float) -> ProviderResponse:
        client = self._client_or_raise()
        system_messages = [m.content for m in messages if m.role == "system"]
        user_messages = [m.content for m in messages if m.role == "user"]
        message = "\n\n".join(user_messages)
        response = client.messages.create(
            model=model,
            system="\n\n".join(system_messages) if system_messages else None,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": message}],
        )
        content = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return ProviderResponse(
            text=content,
            usage={
                "prompt_tokens": float(getattr(usage, "input_tokens", 0)),
                "completion_tokens": float(getattr(usage, "output_tokens", 0)),
                "total_tokens": float(getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)),
            },
        )


registry.register(AnthropicProvider())
