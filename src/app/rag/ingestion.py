from __future__ import annotations

import time
from typing import Iterable, List, Tuple

from ..core.logging import get_logger
from ..core.utils import resolve_paths
from .agent import bm25_index
from .datasets import dataset_registry
from .embeddings import get_embedding_service
from .index import vector_store
from .loaders import load_documents
from .splitter import SemanticTextSplitter, chunk_document
from .types import DocumentChunk, LoadedDocument

logger = get_logger(__name__)


def ingest_dataset(dataset: str, paths: Iterable[str], embedder: str) -> Tuple[int, int, float]:
    start = time.perf_counter()
    resolved_paths = resolve_paths(paths)
    documents = load_documents(dataset, resolved_paths)
    splitter = SemanticTextSplitter()
    chunks: List[DocumentChunk] = []
    for document in documents:
        chunks.extend(_chunk_document(document, splitter))

    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_documents([chunk.text for chunk in chunks], provider=embedder)
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = [float(value) for value in embedding]

    vector_store.add(dataset, chunks)
    bm25_index.add(dataset, chunks)

    dataset_registry.update(dataset, document_count=len(documents), chunk_count=len(chunks))
    duration = time.perf_counter() - start
    logger.info(
        "ingest_completed",
        dataset=dataset,
        documents=len(documents),
        chunks=len(chunks),
        duration=duration,
    )
    return len(documents), len(chunks), duration


def _chunk_document(document: LoadedDocument, splitter: SemanticTextSplitter) -> List[DocumentChunk]:
    return chunk_document(document, splitter)
