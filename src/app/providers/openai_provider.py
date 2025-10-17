from __future__ import annotations

from typing import List

from openai import OpenAI

from ..core.config import settings
from .base import ChatMessage, LLMProvider, ProviderResponse, registry


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        super().__init__()
        self._client: OpenAI | None = None

    def is_available(self) -> bool:
        return bool(settings.openai_api_key)

    def list_models(self) -> List[str]:
        return ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]

    def _client_or_raise(self) -> OpenAI:
        if not self.is_available():
            raise ValueError("OpenAI provider not configured (missing OPENAI_API_KEY)")
        if self._client is None:
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def _generate(self, messages: List[ChatMessage], model: str, temperature: float) -> ProviderResponse:
        client = self._client_or_raise()
        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        response = client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
        )
        choice = response.choices[0].message
        usage = response.usage or {}
        return ProviderResponse(
            text=choice.content or "",
            usage={
                "prompt_tokens": float(usage.get("prompt_tokens", 0)),
                "completion_tokens": float(usage.get("completion_tokens", 0)),
                "total_tokens": float(usage.get("total_tokens", 0)),
            },
        )


registry.register(OpenAIProvider())
