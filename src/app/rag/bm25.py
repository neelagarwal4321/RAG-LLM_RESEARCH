from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .types import DocumentChunk


class BM25Index:
    def __init__(self) -> None:
        self._tokenizer = _default_tokenizer
        self._bm25: Dict[str, BM25Okapi] = {}
        self._corpus_tokens: Dict[str, List[List[str]]] = defaultdict(list)
        self._chunks: Dict[str, List[DocumentChunk]] = defaultdict(list)
        self._chunk_lookup: Dict[str, DocumentChunk] = {}

    def add(self, dataset: str, chunks: List[DocumentChunk]) -> None:
        for chunk in chunks:
            tokens = self._tokenizer(chunk.text)
            self._corpus_tokens[dataset].append(tokens)
            self._chunks[dataset].append(chunk)
            self._chunk_lookup[chunk.chunk_id] = chunk
        self._bm25[dataset] = BM25Okapi(self._corpus_tokens[dataset])

    def search(self, dataset: str, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if dataset not in self._bm25:
            return []
        tokens = self._tokenizer(query)
        scores = self._bm25[dataset].get_scores(tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda item: item[1], reverse=True)
        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in indexed[:top_k]:
            chunk = self._chunks[dataset][idx]
            results.append((chunk, float(score)))
        return results


def _default_tokenizer(text: str) -> List[str]:
    return [token.lower() for token in text.split()]
