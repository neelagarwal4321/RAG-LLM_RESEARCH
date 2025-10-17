from __future__ import annotations

from typing import List

import typer
from rich import print
from rich.table import Table

from ..app.core.logging import configure_logging
from ..app.rag.ingestion import ingest_dataset

app = typer.Typer(add_completion=False, help="Ingest documents into the Finance RAG benchmark store.")


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name to ingest into."),
    embedder: str = typer.Option(
        "local",
        "--embedder",
        "-e",
        help="Embedding provider to use (local|openai).",
    ),
    paths: List[str] = typer.Argument(..., help="Document paths or glob patterns."),
) -> None:
    """CLI entrypoint for dataset ingestion."""
    configure_logging()
    documents, chunks, duration = ingest_dataset(dataset=dataset, paths=paths, embedder=embedder)
    table = Table(title="Ingest Summary")
    table.add_column("Dataset")
    table.add_column("# Documents", style="cyan")
    table.add_column("# Chunks", style="magenta")
    table.add_column("Embedding Provider")
    table.add_column("Duration (s)")
    table.add_row(dataset, str(documents), str(chunks), embedder, f"{duration:0.2f}")
    print(table)


if __name__ == "__main__":
    app()
