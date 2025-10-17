from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

import tiktoken

from .types import DocumentChunk, DocumentSection, LoadedDocument


@dataclass
class SplitterConfig:
    max_tokens: int = 400
    min_tokens: int = 40
    overlap_tokens: int = 40


class SemanticTextSplitter:
    def __init__(self, config: SplitterConfig | None = None) -> None:
        self.config = config or SplitterConfig()
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def split(self, sections: Iterable[DocumentSection]) -> List[str]:
        chunks: List[str] = []
        for section in sections:
            paragraphs = re.split(r"\n{2,}", section.text.strip())
            for paragraph in paragraphs:
                sentences = self._sentence_split(paragraph)
                chunks.extend(self._sliding_window(sentences))
        return chunks

    def _sentence_split(self, text: str) -> List[str]:
        cleaned = text.replace("\n", " ").strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return [s for s in sentences if s]

    def _encode(self, text: str) -> List[int]:
        return self.encoder.encode(text)

    def _token_count(self, text: str) -> int:
        return len(self._encode(text))

    def _sliding_window(self, sentences: List[str]) -> List[str]:
        window: List[str] = []
        chunks: List[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = self._token_count(sentence)
            if sentence_tokens >= self.config.max_tokens:
                chunks.append(sentence)
                window = []
                current_tokens = 0
                continue
            if current_tokens + sentence_tokens > self.config.max_tokens and window:
                chunks.append(" ".join(window).strip())
                window = self._apply_overlap(window)
                current_tokens = sum(self._token_count(s) for s in window)
            window.append(sentence)
            current_tokens += sentence_tokens
        if window:
            chunks.append(" ".join(window).strip())
        return [chunk for chunk in chunks if chunk]

    def _apply_overlap(self, sentences: List[str]) -> List[str]:
        overlap_tokens = 0
        overlapped: List[str] = []
        for sentence in reversed(sentences):
            overlapped.insert(0, sentence)
            overlap_tokens += self._token_count(sentence)
            if overlap_tokens >= self.config.overlap_tokens:
                break
        return overlapped


def chunk_document(document: LoadedDocument, splitter: SemanticTextSplitter | None = None) -> List[DocumentChunk]:
    splitter = splitter or SemanticTextSplitter()
    text_chunks = splitter.split(document.sections)
    chunks: List[DocumentChunk] = []
    for idx, chunk_text in enumerate(text_chunks):
        metadata = {
            **document.metadata,
            "source_id": document.doc_id,
            "chunk_index": idx,
        }
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.doc_id}:{idx}",
                text=chunk_text,
                metadata=metadata,
            )
        )
    return chunks
