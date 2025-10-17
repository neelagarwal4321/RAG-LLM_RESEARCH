from __future__ import annotations

import textwrap
from typing import List, Optional

from ..core.config import settings
from ..core.logging import get_logger
from ..providers.base import ChatMessage, ProviderResponse, registry
from .types import RetrievalResult

logger = get_logger(__name__)


def _format_context(results: List[RetrievalResult]) -> str:
    sections = []
    for idx, result in enumerate(results, start=1):
        section = textwrap.dedent(
            f"""
            === Context {idx} ===
            Chunk ID: {result.chunk.chunk_id}
            Source: {result.chunk.metadata.get('source_path', 'unknown')}
            Content:
            {result.chunk.text}
            """
        ).strip()
        sections.append(section)
    return "\n\n".join(sections)


def generate_answer(
    query: str,
    retrieval_results: List[RetrievalResult],
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> ProviderResponse:
    provider_key = provider_name or settings.default_provider
    provider = registry.get(provider_key)
    model_name = model or settings.default_model

    citations = [result.chunk.chunk_id for result in retrieval_results]
    context = _format_context(retrieval_results)

    system_prompt = textwrap.dedent(
        """
        You are a finance intelligence assistant that must answer with grounded, concise responses.
        Requirements:
        - Cite every factual statement with inline references like [CITE:chunk_id].
        - Provide an evidence block at the end listing the chunk IDs used.
        - Respect numeric tolerances by reporting numbers as they appear in context.
        - If the answer is uncertain, state the uncertainty and request more data.
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        User Question:
        {query}

        Retrieved Context (ranked):
        {context}

        Respond with a well-structured answer that references the context.
        """
    ).strip()

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    logger.info(
        "generation_request",
        provider=provider_key,
        model=model_name,
        temperature=temperature,
        citations=citations,
    )
    response = provider.generate(messages=messages, model=model_name, temperature=temperature)
    return response
