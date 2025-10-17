from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..core.logging import get_logger
from ..core.schemas import IngestRequest, IngestSummary
from ..rag.ingestion import ingest_dataset

router = APIRouter()
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestSummary, status_code=status.HTTP_201_CREATED)
async def ingest(request: IngestRequest) -> IngestSummary:
    try:
        documents, chunks, duration = ingest_dataset(
            dataset=request.dataset_name, paths=request.paths, embedder=request.embedder
        )
    except ValueError as exc:
        logger.error("ingest_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return IngestSummary(
        dataset_name=request.dataset_name,
        documents=documents,
        chunks=chunks,
        embedding_model=request.embedder,
        duration_seconds=duration,
    )
