from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..core.config import settings
from ..core.logging import get_logger
from ..rag.agent import agent
from ..rag.datasets import dataset_registry
from ..rag.citations import extract_citations
from ..rag.types import RetrievalResult
from .judge import JudgeConfig, LLMJudge
from .metrics import NumericTolerance, evaluate_metrics, parse_tolerance

logger = get_logger(__name__)


@dataclass
class TaskSpec:
    task_id: str
    task_type: str
    query: str
    expected: Optional[str]
    expected_points: Optional[List[str]]
    grading_metrics: List[str]
    tolerance: Optional[NumericTolerance]
    must_cite: bool


@dataclass
class SuiteSpec:
    name: str
    dataset: str
    tasks: List[TaskSpec]


def load_suite(path: Path) -> SuiteSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks: List[TaskSpec] = []
    for task in data.get("tasks", []):
        grading = task.get("grading", {})
        tasks.append(
            TaskSpec(
                task_id=task.get("id"),
                task_type=task.get("type", "factual_qa"),
                query=task["query"],
                expected=task.get("expected"),
                expected_points=task.get("expected_points"),
                grading_metrics=list(grading.get("metric", [])),
                tolerance=parse_tolerance(grading.get("tolerance")),
                must_cite=grading.get("must_cite", False),
            )
        )
    return SuiteSpec(name=data.get("suite_name", "suite"), dataset=data["dataset"], tasks=tasks)


def _build_output_dir(base: Path, suite_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = base / timestamp / suite_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _retrieval_map(results: List[RetrievalResult]) -> Dict[str, str]:
    return {result.chunk.chunk_id: result.chunk.text for result in results}


def _retrieval_context(results: List[RetrievalResult]) -> List[str]:
    return [result.chunk.text for result in results]


def default_judge_configs(provider: Optional[str], model: Optional[str]) -> List[JudgeConfig]:
    configs: List[JudgeConfig] = []
    if provider:
        configs.append(JudgeConfig(provider=provider, model=model or settings.default_model))
        return configs
    for name, models in registry_models().items():
        if models:
            configs.append(JudgeConfig(provider=name, model=models[0]))
    return configs


def registry_models() -> Dict[str, List[str]]:
    from ..providers.base import registry

    return registry.list_models()


def run_suite(
    suite_path: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    agentic: Optional[bool] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    suite = load_suite(Path(suite_path))
    if not dataset_registry.get(suite.dataset):
        raise ValueError(f"Dataset '{suite.dataset}' not found. Please ingest before evaluation.")
    output_directory = Path(output_dir) if output_dir else _build_output_dir(settings.reports_dir, suite.name)
    results_path = output_directory / "results.jsonl"
    summary_path = output_directory / "summary.csv"
    meta_path = output_directory / "summary.json"

    judge_configs = default_judge_configs(provider, model)
    judge = LLMJudge(
        configs=judge_configs,
        rubric_path=Path(__file__).resolve().parent / "rubrics" / "default_rubric.yaml",
    )

    rows: List[Dict[str, Any]] = []
    metrics_cumulative: Dict[str, float] = {}
    metrics_counts: Dict[str, int] = {}

    with results_path.open("w", encoding="utf-8") as jsonl:
        for task in suite.tasks:
            logger.info("evaluation_task_start", task=task.task_id, query=task.query)
            run_result = agent.run(
                dataset=suite.dataset,
                query=task.query,
                top_k=8,
                use_reranker=True,
                agentic=settings.agentic_default if agentic is None else agentic,
                provider=provider or settings.default_provider,
                model=model or settings.default_model,
                temperature=0.0,
                embedding_provider=settings.embeddings_provider,
            )
            retrieval_results = run_result.chunks
            retrieval_map = _retrieval_map(retrieval_results)
            citations = extract_citations(run_result.answer)
            metric_scores = evaluate_metrics(
                answer=run_result.answer,
                expected=task.expected,
                expected_points=task.expected_points,
                retrieval_map=retrieval_map,
                metrics=task.grading_metrics,
                tolerance=task.tolerance,
                must_cite=task.must_cite,
                citations=citations,
            )
            judge_result = {}
            if "judge" in task.grading_metrics:
                judge_result = judge.evaluate(
                    question=task.query,
                    answer=run_result.answer,
                    expected=task.expected,
                    citations=citations,
                    retrieval_context=_retrieval_context(retrieval_results),
                )
                metric_scores["judge_scores"] = judge_result.get("scores", {})

            record = {
                "task_id": task.task_id,
                "query": task.query,
                "expected": task.expected,
                "answer": run_result.answer,
                "citations": citations,
                "metrics": metric_scores,
                "judge": judge_result,
                "retrieval": [result.chunk.metadata for result in retrieval_results],
            }
            jsonl.write(json.dumps(record) + "\n")
            rows.append(record)

            for key, value in metric_scores.items():
                if isinstance(value, (int, float)):
                    metrics_cumulative[key] = metrics_cumulative.get(key, 0.0) + float(value)
                    metrics_counts[key] = metrics_counts.get(key, 0) + 1

    with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "average"])
        for key, total in metrics_cumulative.items():
            average = total / metrics_counts[key]
            writer.writerow([key, f"{average:.4f}"])

    summary_payload = {
        "suite": suite.name,
        "dataset": suite.dataset,
        "tasks": len(suite.tasks),
        "metrics": {
            key: metrics_cumulative[key] / metrics_counts[key]
            for key in metrics_cumulative
        },
        "results_path": str(results_path),
        "summary_csv": str(summary_path),
    }
    meta_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    logger.info("evaluation_suite_complete", suite=suite.name, output=str(output_directory))
    return summary_payload
