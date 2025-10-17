from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    dataset_name: str = Field(..., description="Target dataset identifier.")
    paths: List[str] = Field(..., description="List of file paths or glob patterns to ingest.")
    embedder: Literal["local", "openai"] = Field(
        default="local", description="Embedding provider to use for the ingest run."
    )


class IngestSummary(BaseModel):
    dataset_name: str
    documents: int
    chunks: int
    embedding_model: str
    duration_seconds: float


class DatasetInfo(BaseModel):
    name: str
    document_count: int
    chunk_count: int
    last_ingested_at: Optional[datetime] = None


class ModelsResponse(BaseModel):
    providers: Dict[str, List[str]]
    embeddings: Dict[str, List[str]]
    rerankers: List[str]


class RetrievalChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class ToolCall(BaseModel):
    tool: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    duration_seconds: float


class QueryRequest(BaseModel):
    dataset: str
    query: str
    top_k: int = Field(default=6, ge=1, le=20)
    use_reranker: bool = Field(default=True)
    agentic: bool = Field(default=True)
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    chunks: List[RetrievalChunk]
    tool_traces: List[ToolCall]
    metrics: Dict[str, Any]
    timings: Dict[str, float]


class EvaluateRequest(BaseModel):
    suite_path: str
    provider: Optional[str] = None
    model: Optional[str] = None
    agentic: Optional[bool] = None
    output_dir: Optional[str] = None


class EvaluateResponse(BaseModel):
    suite_name: str
    tasks_run: int
    metrics: Dict[str, Any]
    reports: Dict[str, str]
    duration_seconds: float
