from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from diskcache import Cache  # type: ignore

from ..core.config import settings
from ..core.logging import get_logger
from ..core.utils import dict_hash

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from openai import OpenAI


class EmbeddingService:
    def __init__(self) -> None:
        self.cache = Cache(settings.cache_dir / "embeddings")
        self._local_model = None
        self._openai_client: OpenAI | None = None

    def embed_documents(self, texts: Iterable[str], provider: str | None = None) -> List[List[float]]:
        provider_name = provider or settings.embeddings_provider
        return self._embed(list(texts), provider_name)

    def embed_query(self, text: str, provider: str | None = None) -> List[float]:
        provider_name = provider or settings.embeddings_provider
        embeddings = self._embed([text], provider_name)
        return embeddings[0]

    def _embed(self, texts: List[str], provider: str) -> List[List[float]]:
        cache_key = dict_hash({"provider": provider, "texts": texts})
        if cache_key in self.cache:
            return self.cache[cache_key]
        if provider == "local":
            embeddings = self._local_embeddings(texts)
        elif provider == "openai":
            embeddings = self._openai_embeddings(texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        self.cache[cache_key] = embeddings
        return embeddings

    def _local_embeddings(self, texts: List[str]) -> List[List[float]]:
        if SentenceTransformer is None:
            logger.warning("sentence_transformers_missing", fallback="hash")
            return [self._hash_embedding(text) for text in texts]
        if self._local_model is None:
            try:
                logger.info("loading_local_embedder", model=settings.local_embedder_model)
                self._local_model = SentenceTransformer(settings.local_embedder_model)
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning("local_embedder_load_failed", error=str(exc), fallback="hash")
                return [self._hash_embedding(text) for text in texts]
        vectors = self._local_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return [vector.tolist() for vector in vectors]

    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured for remote embeddings")
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        response = self._openai_client.embeddings.create(
            model=settings.remote_embedder_model,
            input=texts,
        )
        return [list(data.embedding) for data in response.data]

    def _hash_embedding(self, text: str, dimensions: int = 384) -> List[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for idx in range(dimensions):
            byte = digest[idx % len(digest)]
            values.append(((byte / 255.0) * 2) - 1)  # normalize to [-1, 1]
        return values


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
