from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple

import chromadb
from chromadb import Collection

from ..core.config import settings
from .agent import set_vector_store
from .types import DocumentChunk


class VectorStore(Protocol):
    def add(self, dataset: str, chunks: Sequence[DocumentChunk]) -> None:
        ...

    def similarity_search(
        self, dataset: str, embedding: List[float], top_k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        ...


def _collection_name(dataset: str) -> str:
    return dataset.replace(":", "_")


@dataclass
class ChromaVectorStore:
    client: chromadb.PersistentClient

    @classmethod
    def from_settings(cls) -> "ChromaVectorStore":
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_path))
        return cls(client=client)

    def _collection(self, dataset: str) -> Collection:
        name = _collection_name(dataset)
        try:
            return self.client.get_collection(name)
        except chromadb.errors.InvalidCollectionException:
            return self.client.create_collection(name)

    def add(self, dataset: str, chunks: Sequence[DocumentChunk]) -> None:
        collection = self._collection(dataset)
        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, object]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("Chunk missing embedding for vector index")
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)
        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    def similarity_search(
        self, dataset: str, embedding: List[float], top_k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        collection = self._collection(dataset)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["embeddings", "metadatas", "documents", "distances", "ids"],
        )
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        chunks: List[Tuple[DocumentChunk, float]] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata or {},
                embedding=None,
            )
            chunks.append((chunk, float(distance)))
        return chunks


vector_store: VectorStore = ChromaVectorStore.from_settings()
set_vector_store(vector_store)
