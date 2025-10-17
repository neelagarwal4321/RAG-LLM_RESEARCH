from __future__ import annotations

import json
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from ..app.core.config import settings
from ..app.core.logging import configure_logging
from ..app.rag.agent import agent
from ..app.rag.datasets import dataset_registry

app = typer.Typer(add_completion=False, help="Query the Finance RAG benchmark datasets.")
console = Console()


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Target dataset name."),
    question: str = typer.Option(..., "--q", "--question", help="User question to ask."),
    top_k: int = typer.Option(6, "--top-k", help="Number of retrieval chunks."),
    reranker: bool = typer.Option(True, "--reranker/--no-reranker", help="Toggle cross-encoder reranking."),
    agentic: bool = typer.Option(True, "--agentic/--no-agentic", help="Enable the agentic loop."),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider to use."),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model ID."),
    temperature: float = typer.Option(0.0, "--temperature", min=0.0, max=1.0),
) -> None:
    """Execute a query against the API pipeline without starting the server."""
    configure_logging()

    if not dataset_registry.get(dataset):
        raise typer.BadParameter(f"Dataset '{dataset}' not found. Run ingest first.")

    output = agent.run(
        dataset=dataset,
        query=question,
        top_k=top_k,
        use_reranker=reranker,
        agentic=agentic,
        provider=provider or settings.default_provider,
        model=model or settings.default_model,
        temperature=temperature,
        embedding_provider=settings.embeddings_provider,
    )

    console.print(Panel.fit(output.answer, title="Answer", subtitle="Finance RAG"))
    console.print(f"[bold]Citations:[/bold] {', '.join(output.citations)}")
    console.print("[bold]Chunks:[/bold]")
    for idx, chunk in enumerate(output.chunks, start=1):
        console.print(
            Panel.fit(
                chunk.chunk.text,
                title=f"{idx}. {chunk.chunk.chunk_id}",
                subtitle=str(chunk.chunk.metadata.get("source_path", "")),
            )
        )
    console.print("[bold]Usage:[/bold]")
    console.print(json.dumps(output.usage, indent=2))


if __name__ == "__main__":
    app()
