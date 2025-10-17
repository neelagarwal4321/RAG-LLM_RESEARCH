from __future__ import annotations

from typing import List

from azure.ai.openai import AzureOpenAI

from ..core.config import settings
from .base import ChatMessage, LLMProvider, ProviderResponse, registry


class AzureOpenAIProvider(LLMProvider):
    name = "azure_openai"

    def __init__(self) -> None:
        super().__init__()
        self._client: AzureOpenAI | None = None

    def is_available(self) -> bool:
        return bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)

    def list_models(self) -> List[str]:
        if settings.azure_openai_deployment:
            return [settings.azure_openai_deployment]
        return []

    def _client_or_raise(self) -> AzureOpenAI:
        if not self.is_available():
            raise ValueError("Azure OpenAI provider not configured")
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version="2024-02-01",
            )
        return self._client

    def _generate(self, messages: List[ChatMessage], model: str, temperature: float) -> ProviderResponse:
        client = self._client_or_raise()
        payload = [{"role": message.role, "content": message.content} for message in messages]
        deployment = settings.azure_openai_deployment or model
        response = client.chat.completions.create(
            model=deployment,
            messages=payload,
            temperature=temperature,
        )
        choice = response.choices[0].message
        usage = response.usage
        return ProviderResponse(
            text=choice.content or "",
            usage={
                "prompt_tokens": float(getattr(usage, "prompt_tokens", 0)),
                "completion_tokens": float(getattr(usage, "completion_tokens", 0)),
                "total_tokens": float(getattr(usage, "total_tokens", 0)),
            },
        )


registry.register(AzureOpenAIProvider())
