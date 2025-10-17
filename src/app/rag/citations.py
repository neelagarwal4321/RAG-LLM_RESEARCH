from __future__ import annotations

import re
from typing import List, Tuple

from .types import RetrievalResult


def extract_citations(text: str) -> List[str]:
    pattern = re.compile(r"\[CITE:([^\]]+)\]")
    return list(dict.fromkeys(pattern.findall(text)))


def ensure_evidence_block(text: str, retrieval_results: List[RetrievalResult]) -> Tuple[str, List[str]]:
    citations = extract_citations(text)
    if not citations:
        citations = [result.chunk.chunk_id for result in retrieval_results]
        evidence_lines = "\n".join(f"- {citation}" for citation in citations)
        text = f"{text}\n\nEvidence:\n{evidence_lines}"
    elif "Evidence:" not in text:
        evidence_lines = "\n".join(f"- {citation}" for citation in citations)
        text = f"{text}\n\nEvidence:\n{evidence_lines}"
    return text, citations
