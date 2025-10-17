from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentSection:
    text: str
    metadata: Dict[str, Any]


@dataclass
class LoadedDocument:
    path: Path
    dataset: str
    doc_id: str
    sections: List[DocumentSection]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    source: str
