from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rouge_score import rouge_scorer

from ..core.logging import get_logger
from ..rag.citations import extract_citations
from ..rag.embeddings import get_embedding_service

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\w+")
_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:[kKmMbB])?")


@dataclass
class NumericTolerance:
    mode: str
    value: float


def parse_tolerance(config: Optional[str]) -> Optional[NumericTolerance]:
    if not config:
        return None
    try:
        mode, value = config.split(":", 1)
        if mode != "numeric":
            return None
        value = value.strip()
        if value.endswith("%"):
            return NumericTolerance(mode="percentage", value=float(value[:-1]) / 100.0)
        return NumericTolerance(mode="absolute", value=float(value))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("tolerance_parse_failed", value=config, error=str(exc))
        return None


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = _WORD_RE.findall(normalize_text(prediction))
    ref_tokens = _WORD_RE.findall(normalize_text(reference))
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)["rougeL"]
    return float(score.fmeasure)


def semantic_similarity(prediction: str, reference: str) -> float:
    embedder = get_embedding_service()
    vectors = embedder.embed_documents([prediction, reference])
    if len(vectors) < 2:
        return 0.0
    return cosine_similarity(vectors[0], vectors[1])


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def numeric_within_tolerance(prediction: str, reference: str, tolerance: NumericTolerance) -> bool:
    pred_numbers = extract_numbers(prediction)
    ref_numbers = extract_numbers(reference)
    if not pred_numbers or not ref_numbers:
        return False
    for ref in ref_numbers:
        for pred in pred_numbers:
            if tolerance.mode == "percentage":
                if ref == 0:
                    continue
                if abs(pred - ref) / abs(ref) <= tolerance.value:
                    return True
            else:
                if abs(pred - ref) <= tolerance.value:
                    return True
    return False


def extract_numbers(text: str) -> List[float]:
    numbers: List[float] = []
    for match in _NUMERIC_RE.findall(text):
        value = match.lower()
        multiplier = 1.0
        if value.endswith("k"):
            multiplier = 1_000.0
            value = value[:-1]
        elif value.endswith("m"):
            multiplier = 1_000_000.0
            value = value[:-1]
        elif value.endswith("b"):
            multiplier = 1_000_000_000.0
            value = value[:-1]
        try:
            numbers.append(float(value) * multiplier)
        except ValueError:
            continue
    return numbers


def pointwise_coverage(prediction: str, expected_points: Iterable[str]) -> Dict[str, Any]:
    prediction_norm = normalize_text(prediction)
    supported = []
    missing = []
    for point in expected_points:
        normalized_point = normalize_text(point)
        if normalized_point in prediction_norm:
            supported.append(point)
        else:
            missing.append(point)
    total = len(supported) + len(missing)
    score = len(supported) / total if total else 0.0
    return {"score": score, "supported": supported, "missing": missing}


def faithfulness(
    answer: str, retrieval_map: Dict[str, str], citations: List[str]
) -> Dict[str, Any]:
    if not citations:
        return {"score": 0.0, "unsupported": [], "total": 0}
    unsupported: List[str] = []
    overlaps: Dict[str, float] = {}
    for citation in citations:
        context = retrieval_map.get(citation, "")
        if not context:
            unsupported.append(citation)
            continue
        overlap = lexical_overlap(answer, context)
        overlaps[citation] = overlap
        if overlap < 0.1:
            unsupported.append(citation)
    supported = len(citations) - len(unsupported)
    score = supported / len(citations)
    return {"score": score, "unsupported": unsupported, "overlap": overlaps, "total": len(citations)}


def lexical_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(_WORD_RE.findall(normalize_text(text_a)))
    tokens_b = set(_WORD_RE.findall(normalize_text(text_b)))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def evaluate_metrics(
    answer: str,
    expected: Optional[str],
    expected_points: Optional[List[str]],
    retrieval_map: Dict[str, str],
    metrics: List[str],
    tolerance: Optional[NumericTolerance],
    must_cite: bool,
    citations: List[str],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    if must_cite:
        results["must_cite"] = float(bool(citations))
    if expected is None and expected_points is None:
        return results

    answer_norm = answer.strip()
    expected_text = expected or ""

    for metric in metrics:
        if metric == "exact_match" and expected is not None:
            results["exact_match"] = exact_match(answer_norm, expected_text)
        elif metric == "f1" and expected is not None:
            results["f1"] = f1_score(answer_norm, expected_text)
        elif metric in {"rougeL", "rouge_l"} and expected is not None:
            results["rougeL"] = rouge_l(answer_norm, expected_text)
        elif metric in {"sem_sim", "semantic_similarity"} and expected is not None:
            results["sem_sim"] = semantic_similarity(answer_norm, expected_text)
        elif metric == "pointwise_coverage" and expected_points is not None:
            coverage = pointwise_coverage(answer_norm, expected_points)
            results["pointwise_coverage"] = coverage["score"]
            results["coverage_supported"] = coverage["supported"]
            results["coverage_missing"] = coverage["missing"]
        elif metric == "faithfulness":
            faith = faithfulness(answer_norm, retrieval_map, citations)
            results["faithfulness"] = faith["score"]
            results["faithfulness_details"] = faith

    if tolerance and expected is not None:
        results["numeric_within_tolerance"] = float(numeric_within_tolerance(answer_norm, expected_text, tolerance))

    return results
