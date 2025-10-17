from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.config import settings
from ..core.logging import get_logger
from .types import RetrievalResult

logger = get_logger(__name__)

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore


@dataclass
class RerankResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None

    def _load(self) -> None:
        if self._model is not None or CrossEncoder is None:
            return
        logger.info("reranker_loading", model=self.model_name)
        self._model = CrossEncoder(self.model_name)

    @retry(
        retry=retry_if_exception_type(RuntimeError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> List[RerankResult]:
        if not candidates:
            return []
        try:
            self._load()
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("reranker_unavailable", error=str(exc))
            return [
                RerankResult(
                    chunk_id=result.chunk.chunk_id,
                    score=result.score,
                    text=result.chunk.text,
                    metadata=result.chunk.metadata,
                )
                for result in candidates[:top_k]
            ]

        assert self._model is not None  # for mypy
        pairs = [(query, result.chunk.text) for result in candidates]
        scores = self._model.predict(pairs)
        ranked: List[Tuple[float, RetrievalResult]] = sorted(
            zip(scores, candidates), key=lambda item: float(item[0]), reverse=True
        )
        reranked: List[RerankResult] = []
        for score, result in ranked[:top_k]:
            reranked.append(
                RerankResult(
                    chunk_id=result.chunk.chunk_id,
                    score=float(score),
                    text=result.chunk.text,
                    metadata=result.chunk.metadata,
                )
            )
        return reranked
