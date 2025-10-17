from __future__ import annotations

import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException, status

from ..core.config import settings
from ..core.logging import get_logger
from ..core.schemas import QueryRequest, QueryResponse, RetrievalChunk, ToolCall
from ..rag.agent import agent
from ..rag.datasets import dataset_registry
from ..rag.embeddings import get_embedding_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not dataset_registry.get(request.dataset):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{request.dataset}' not found. Ingest documents first.",
        )
    start = time.perf_counter()
    embeddings_provider = settings.embeddings_provider
    try:
        output = agent.run(
            dataset=request.dataset,
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            agentic=request.agentic,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            embedding_provider=embeddings_provider,
        )
    except Exception as exc:  # pragma: no cover - to surface errors gracefully
        logger.error("query_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    retrieval_chunks: List[RetrievalChunk] = []
    for result in output.chunks:
        retrieval_chunks.append(
            RetrievalChunk(
                chunk_id=result.chunk.chunk_id,
                text=result.chunk.text,
                score=result.score,
                metadata=result.chunk.metadata,
            )
        )

    timings: Dict[str, float] = {"total_seconds": time.perf_counter() - start}

    return QueryResponse(
        answer=output.answer,
        citations=output.citations,
        chunks=retrieval_chunks,
        tool_traces=[ToolCall(**trace) for trace in output.tool_traces],
        metrics={"usage": output.usage, "embedding_provider": embeddings_provider},
        timings=timings,
    )
