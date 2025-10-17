from __future__ import annotations

from fastapi import FastAPI

from .core.config import settings
from .core.logging import configure_logging
from .routes import evaluate, ingest, models, query

# Import provider modules to ensure registry is populated.
from .providers import (  # noqa: F401
    anthropic_provider,
    azure_openai_provider,
    google_provider,
    openai_provider,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    app = FastAPI(
        title="Finance RAG Bench",
        version="0.1.0",
        description="Finance-focused RAG benchmarking service with ingestion, retrieval, and evaluation.",
    )

    app.include_router(models.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(evaluate.router)

    return app


app = create_app()
