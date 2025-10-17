from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.logging import get_logger
from .bm25 import BM25Index
from .citations import ensure_evidence_block
from .embeddings import get_embedding_service
from .generator import generate_answer
from .hybrid import HybridRetriever
from .reranker import CrossEncoderReranker, RerankResult
from .types import DocumentChunk, RetrievalResult

logger = get_logger(__name__)


def _merge_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    merged: Dict[str, RetrievalResult] = {}
    for result in results:
        existing = merged.get(result.chunk.chunk_id)
        if not existing or result.score > existing.score:
            merged[result.chunk.chunk_id] = result
    ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return ranked


def _from_rerank(results: List[RerankResult], original_lookup: Dict[str, RetrievalResult]) -> List[RetrievalResult]:
    reranked: List[RetrievalResult] = []
    for rerank in results:
        original = original_lookup.get(rerank.chunk_id)
        if original:
            chunk = DocumentChunk(
                chunk_id=rerank.chunk_id,
                text=rerank.text,
                metadata=original.chunk.metadata,
            )
            reranked.append(RetrievalResult(chunk=chunk, score=rerank.score, source=rerank.chunk_id))
    return reranked


@dataclass
class AgentOutput:
    answer: str
    citations: List[str]
    chunks: List[RetrievalResult]
    tool_traces: List[Dict[str, object]] = field(default_factory=list)
    usage: Dict[str, float] = field(default_factory=dict)


class AgenticRAG:
    def __init__(self, retriever: HybridRetriever, reranker: Optional[CrossEncoderReranker] = None) -> None:
        self.embedding_service = get_embedding_service()
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()

    def run(
        self,
        dataset: str,
        query: str,
        top_k: int = 6,
        use_reranker: bool = True,
        agentic: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        embedding_provider: Optional[str] = None,
    ) -> AgentOutput:
        plan_queries = [query]
        if agentic:
            plan_queries.append(f"{query} financial context and drivers")
        tool_traces: List[Dict[str, object]] = []
        aggregated: List[RetrievalResult] = []
        original_lookup: Dict[str, RetrievalResult] = {}

        for sub_query in plan_queries:
            embedding = self.embedding_service.embed_query(sub_query, provider=embedding_provider)
            results = self.retriever.retrieve(dataset, sub_query, embedding, top_k=top_k * 2)
            for result in results:
                original_lookup[result.chunk.chunk_id] = result
            aggregated.extend(results)

        merged = _merge_results(aggregated)
        if use_reranker and merged:
            reranked = self.reranker.rerank(query, merged, top_k=top_k)
            merged = _from_rerank(reranked, original_lookup)
        top_chunks = merged[:top_k]

        response = generate_answer(
            query=query,
            retrieval_results=top_chunks,
            provider_name=provider,
            model=model,
            temperature=temperature,
        )
        answer, citations = ensure_evidence_block(response.text, top_chunks)
        logger.info("agent_answer_ready", citations=citations, total_chunks=len(top_chunks))
        return AgentOutput(
            answer=answer,
            citations=citations,
            chunks=top_chunks,
            tool_traces=tool_traces,
            usage=response.usage,
        )


bm25_index = BM25Index()
hybrid_retriever = HybridRetriever(vector_store=None, bm25_index=bm25_index)  # type: ignore[arg-type]
# The vector store will be injected by set_vector_store once ready.


def set_vector_store(vector_store) -> None:
    hybrid_retriever.vector_store = vector_store


agent = AgenticRAG(retriever=hybrid_retriever)
