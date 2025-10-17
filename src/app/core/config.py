from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Finance RAG Bench"
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Storage paths
    data_dir: Path = Field(default=BASE_DIR / "data", alias="DATA_DIR")
    cache_dir: Path = Field(default=BASE_DIR / "data" / "cache", alias="CACHE_DIR")
    chroma_persist_path: Path = Field(default=BASE_DIR / "data" / "chroma", alias="CHROMA_PERSIST_PATH")
    reports_dir: Path = Field(default=BASE_DIR / "reports", alias="EVAL_REPORT_DIR")

    # Embeddings
    embeddings_provider: str = Field(default="local", alias="EMBEDDINGS_PROVIDER")
    local_embedder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="LOCAL_EMBEDDER_MODEL"
    )
    remote_embedder_model: str = Field(
        default="text-embedding-3-large", alias="REMOTE_EMBEDDER_MODEL"
    )

    # Vector store
    vector_db: str = Field(default="chroma", alias="VECTOR_DB")

    # Reranker
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # Providers and defaults
    default_provider: str = Field(default="openai", alias="DEFAULT_PROVIDER")
    default_model: str = Field(default="gpt-4o-mini", alias="DEFAULT_MODEL")
    agentic_default: bool = Field(default=True, alias="AGENTIC")

    # Provider keys / endpoints
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    azure_openai_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: Optional[str] = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")

    # Evaluation settings
    judge_runs: int = Field(default=3, alias="JUDGE_RUNS")

    # Feature toggles
    enabled_providers: Optional[List[str]] = Field(default=None, alias="ENABLED_PROVIDERS")

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.cache_dir, self.chroma_persist_path, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
