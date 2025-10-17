from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..core.config import settings
from ..core.logging import get_logger
from ..providers.base import ChatMessage, ProviderResponse, registry

logger = get_logger(__name__)


@dataclass
class JudgeConfig:
    provider: str
    model: str


def load_rubric(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class LLMJudge:
    def __init__(self, configs: List[JudgeConfig], rubric_path: Path) -> None:
        self.configs = [config for config in configs if _provider_available(config.provider)]
        self.rubric = load_rubric(rubric_path)
        if not self.configs:
            logger.warning("llm_judge_disabled", reason="no providers configured")

    def evaluate(
        self,
        question: str,
        answer: str,
        expected: Optional[str],
        citations: List[str],
        retrieval_context: List[str],
    ) -> Dict[str, Any]:
        if not self.configs:
            return self._heuristic_judge(question, answer, expected, citations)

        judgments: List[Dict[str, Any]] = []
        for config in self.configs:
            provider = registry.get(config.provider)
            messages = self._build_messages(question, answer, expected, citations, retrieval_context)
            all_runs: List[Dict[str, Any]] = []
            for _ in range(settings.judge_runs):
                try:
                    response = provider.generate(
                        messages=messages,
                        model=config.model,
                        temperature=0.0,
                    )
                    parsed = self._parse_response(response)
                    all_runs.append(parsed)
                except Exception as exc:
                    logger.warning(
                        "judge_call_failed",
                        provider=config.provider,
                        model=config.model,
                        error=str(exc),
                    )
            if all_runs:
                judgments.append(self._aggregate_runs(config, all_runs))
        if not judgments:
            return self._heuristic_judge(question, answer, expected, citations)
        return self._aggregate_judgments(judgments)

    def _build_messages(
        self,
        question: str,
        answer: str,
        expected: Optional[str],
        citations: List[str],
        retrieval_context: List[str],
    ) -> List[ChatMessage]:
        rubric_text = json.dumps(self.rubric, indent=2)
        context_blob = "\n\n".join(retrieval_context)
        system_prompt = (
            "You are a meticulous finance QA judge. Score model answers against the rubric."
            " Always respond with strict JSON."
        )
        user_prompt = f"""
Rubric (JSON):
{rubric_text}

Question:
{question}

Model Answer:
{answer}

Expected Answer (optional):
{expected or 'N/A'}

Citations Provided:
{citations}

Retrieved Context:
{context_blob}

Respond with JSON: {{
  "scores": {{
    "relevance": 0-5,
    "faithfulness": 0-5,
    "completeness": 0-5,
    "citation_quality": 0-5,
    "numerical_correctness": 0-5
  }},
  "rationale": "Concise reasoning",
  "issues": ["List problematic claims"]
}}
"""
        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

    def _parse_response(self, response: ProviderResponse) -> Dict[str, Any]:
        text = response.text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("judge_invalid_json", text=text[:200])
            return {"scores": {}, "rationale": text, "issues": []}

    def _aggregate_runs(self, config: JudgeConfig, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregated_scores: Dict[str, float] = {}
        for run in runs:
            for key, value in run.get("scores", {}).items():
                aggregated_scores.setdefault(key, 0.0)
                aggregated_scores[key] += float(value)
        for key in aggregated_scores:
            aggregated_scores[key] /= len(runs)
        return {
            "provider": config.provider,
            "model": config.model,
            "scores": aggregated_scores,
            "runs": runs,
        }

    def _aggregate_judgments(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        macro_scores: Dict[str, float] = {}
        for judgment in judgments:
            for key, value in judgment["scores"].items():
                macro_scores.setdefault(key, 0.0)
                macro_scores[key] += float(value)
        for key in macro_scores:
            macro_scores[key] /= len(judgments)
        return {"scores": macro_scores, "judgments": judgments}

    def _heuristic_judge(
        self, question: str, answer: str, expected: Optional[str], citations: List[str]
    ) -> Dict[str, Any]:
        # Fallback heuristic: reward citations and basic alignment.
        relevance = 5.0 if question.lower().split()[0] in answer.lower() else 3.0
        faithfulness = 5.0 if citations else 2.5
        completeness = 4.0 if expected and expected.lower() in answer.lower() else 3.0
        citation_quality = 5.0 if citations else 1.0
        numerical = 4.0 if any(char.isdigit() for char in answer) else 2.0
        return {
            "scores": {
                "relevance": relevance,
                "faithfulness": faithfulness,
                "completeness": completeness,
                "citation_quality": citation_quality,
                "numerical_correctness": numerical,
            },
            "judgments": [],
            "heuristic": True,
        }


def _provider_available(name: str) -> bool:
    try:
        provider = registry.providers().get(name)
        return bool(provider and provider.is_available())
    except Exception:
        return False
