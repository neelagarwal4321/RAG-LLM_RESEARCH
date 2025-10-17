from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from diskcache import Cache  # type: ignore

from ..core.config import settings
from ..core.utils import dict_hash


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ProviderResponse:
    text: str
    usage: Dict[str, float]


class LLMProvider(ABC):
    name: str

    def __init__(self) -> None:
        self.cache = Cache(settings.cache_dir / "llm")

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def list_models(self) -> List[str]:
        ...

    @abstractmethod
    def _generate(self, messages: List[ChatMessage], model: str, temperature: float) -> ProviderResponse:
        ...

    def generate(
        self, messages: List[ChatMessage], model: Optional[str] = None, temperature: float = 0.0
    ) -> ProviderResponse:
        model_name = model or self.default_model()
        cache_key = dict_hash(
            {
                "provider": self.name,
                "model": model_name,
                "messages": [message.__dict__ for message in messages],
                "temperature": temperature,
            }
        )
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return ProviderResponse(text=cached["text"], usage=cached["usage"])
        response = self._generate(messages, model_name, temperature)
        self.cache[cache_key] = {"text": response.text, "usage": response.usage}
        return response

    def default_model(self) -> str:
        available = self.list_models()
        if not available:
            raise ValueError(f"No models available for provider {self.name}")
        return available[0]


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            raise KeyError(f"Unknown provider '{name}'")
        provider = self._providers[name]
        if not provider.is_available():
            raise ValueError(f"Provider {name} is not available (missing credentials?)")
        return provider

    def providers(self) -> Dict[str, LLMProvider]:
        return dict(self._providers)

    def list_models(self) -> Dict[str, List[str]]:
        return {
            name: provider.list_models()
            for name, provider in self._providers.items()
            if provider.is_available()
        }


registry = ProviderRegistry()


class LocalProvider(LLMProvider):
    name = "local"

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["stub-finance-001"]

    def _generate(self, messages: List[ChatMessage], model: str, temperature: float) -> ProviderResponse:
        user_prompts = [message.content for message in messages if message.role == "user"]
        joined = "\n".join(user_prompts)
        response = f"[LOCAL MODEL RESPONSE]\n{joined}"
        return ProviderResponse(text=response, usage={"total_tokens": len(joined.split())})


registry.register(LocalProvider())
