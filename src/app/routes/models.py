from __future__ import annotations

from fastapi import APIRouter

from ..core.config import settings

from ..core.schemas import DatasetInfo, ModelsResponse
from ..rag.datasets import dataset_registry
from ..providers.base import registry

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    provider_models = registry.list_models()
    embeddings = {
        "local": [settings.local_embedder_model],
        "openai": [settings.remote_embedder_model],
    }
    return ModelsResponse(
        providers=provider_models,
        embeddings=embeddings,
        rerankers=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
    )


@router.get("/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    datasets = []
    for metadata in dataset_registry.list():
        last_ingested = None
        try:
            from datetime import datetime

            last_ingested = datetime.fromisoformat(metadata.last_ingested_at)
        except Exception:
            last_ingested = None
        datasets.append(
            DatasetInfo(
                name=metadata.name,
                document_count=metadata.document_count,
                chunk_count=metadata.chunk_count,
                last_ingested_at=last_ingested,
            )
        )
    return datasets
