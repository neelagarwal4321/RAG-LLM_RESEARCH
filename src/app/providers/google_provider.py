from __future__ import annotations

from typing import List

import google.generativeai as genai

from ..core.config import settings
from .base import ChatMessage, LLMProvider, ProviderResponse, registry


class GoogleProvider(LLMProvider):
    name = "google"

    def __init__(self) -> None:
        super().__init__()
        self._configured = False

    def is_available(self) -> bool:
        return bool(settings.google_api_key)

    def _ensure_configured(self) -> None:
        if not self._configured and self.is_available():
            genai.configure(api_key=settings.google_api_key)
            self._configured = True

    def list_models(self) -> List[str]:
        return ["gemini-1.5-flash", "gemini-1.5-pro"]

    def _generate(self, messages: List[ChatMessage], model: str, temperature: float) -> ProviderResponse:
        self._ensure_configured()
        if not self.is_available():
            raise ValueError("GOOGLE_API_KEY not configured")
        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        user_prompt = "\n".join(m.content for m in messages if m.role == "user")
        generative_model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt or None)
        response = generative_model.generate_content(
            [user_prompt],
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        text = "".join(c.text for c in response.candidates[0].content.parts)
        usage_metadata = response.usage_metadata
        return ProviderResponse(
            text=text,
            usage={
                "prompt_tokens": float(getattr(usage_metadata, "prompt_token_count", 0)),
                "completion_tokens": float(getattr(usage_metadata, "candidates_token_count", 0)),
                "total_tokens": float(getattr(usage_metadata, "total_token_count", 0)),
            },
        )


registry.register(GoogleProvider())
