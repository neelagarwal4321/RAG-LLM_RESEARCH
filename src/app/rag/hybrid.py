from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .bm25 import BM25Index
from .index import VectorStore
from .types import DocumentChunk, RetrievalResult


@dataclass
class HybridRetriever:
    vector_store: Optional[VectorStore]
    bm25_index: BM25Index
    rrf_k: int = 60
    weight_dense: float = 0.5
    weight_bm25: float = 0.5

    def retrieve(
        self,
        dataset: str,
        query: str,
        query_embedding: List[float],
        top_k: int = 6,
        top_k_bm25: int = 12,
        top_k_dense: int = 12,
    ) -> List[RetrievalResult]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not configured for hybrid retriever")

        bm25_results = self.bm25_index.search(dataset, query, top_k_bm25)
        dense_results = self.vector_store.similarity_search(dataset, query_embedding, top_k_dense)

        scores: Dict[str, float] = {}
        lookup: Dict[str, DocumentChunk] = {}

        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            fused = self.weight_bm25 * self._rrf(rank)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + fused
            lookup[chunk.chunk_id] = chunk

        for rank, (chunk, distance) in enumerate(dense_results, start=1):
            similarity = 1.0 / (1.0 + distance)
            fused = self.weight_dense * self._rrf(rank) * similarity
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + fused
            lookup[chunk.chunk_id] = chunk

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results: List[RetrievalResult] = []
        for chunk_id, score in ranked[:top_k]:
            chunk = lookup[chunk_id]
            metadata = dict(chunk.metadata)
            metadata["hybrid_score"] = score
            results.append(RetrievalResult(chunk=chunk, score=score, source=chunk_id))
        return results

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)
