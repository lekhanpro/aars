"""RAGAS evaluation wrapper for AARS benchmarking."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


async def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, Any] | None:
    """Run RAGAS evaluation on a batch of question-answer pairs.

    Parameters
    ----------
    questions:
        User queries.
    answers:
        Generated answers.
    contexts:
        Retrieved document texts per question.
    ground_truths:
        Optional reference answers for context_recall.

    Returns
    -------
    dict or None
        RAGAS evaluation results, or None if ragas is not installed.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError:
        logger.warning("ragas_not_installed", hint="pip install ragas datasets")
        return None

    data: dict[str, Any] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths:
        metrics.append(context_recall)

    dataset = Dataset.from_dict(data)

    logger.info("ragas_evaluation_starting", num_samples=len(questions))

    try:
        result = await asyncio.to_thread(evaluate, dataset=dataset, metrics=metrics)
    except Exception as exc:
        logger.error("ragas_evaluation_failed", error=str(exc))
        return {"error": str(exc)}

    scores: dict[str, float] = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric_name in result:
            scores[metric_name] = float(result[metric_name])

    logger.info("ragas_evaluation_complete", scores=scores)

    return {
        "scores": scores,
        "per_sample": result.to_pandas().to_dict(orient="records") if hasattr(result, "to_pandas") else [],
    }


def ragas_available() -> bool:
    """Check if RAGAS library is importable."""
    try:
        import ragas  # noqa: F401
        return True
    except ImportError:
        return False
