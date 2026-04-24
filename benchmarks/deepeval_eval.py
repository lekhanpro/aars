"""DeepEval evaluation wrapper for AARS regression testing."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def run_deepeval_suite(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, Any] | None:
    """Run DeepEval metrics on a batch of question-answer pairs.

    Parameters
    ----------
    questions:
        User queries.
    answers:
        Generated answers.
    contexts:
        Retrieved document texts per question.
    ground_truths:
        Optional reference answers.

    Returns
    -------
    dict or None
        DeepEval results, or None if deepeval is not installed.
    """
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        logger.warning("deepeval_not_installed", hint="pip install deepeval")
        return None

    logger.info("deepeval_suite_starting", num_samples=len(questions))

    test_cases: list[LLMTestCase] = []
    for i in range(len(questions)):
        tc = LLMTestCase(
            input=questions[i],
            actual_output=answers[i],
            retrieval_context=contexts[i] if i < len(contexts) else [],
            expected_output=ground_truths[i] if ground_truths and i < len(ground_truths) else None,
        )
        test_cases.append(tc)

    metrics_to_run = [
        ("faithfulness", FaithfulnessMetric, {"threshold": 0.5}),
        ("answer_relevancy", AnswerRelevancyMetric, {"threshold": 0.5}),
        ("contextual_precision", ContextualPrecisionMetric, {"threshold": 0.5}),
    ]
    if ground_truths:
        metrics_to_run.append(
            ("contextual_recall", ContextualRecallMetric, {"threshold": 0.5})
        )

    results: dict[str, Any] = {"per_metric": {}, "per_sample": []}

    for metric_name, metric_cls, kwargs in metrics_to_run:
        try:
            metric = metric_cls(**kwargs)
            scores: list[float] = []
            for tc in test_cases:
                metric.measure(tc)
                scores.append(metric.score)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            results["per_metric"][metric_name] = {
                "average": avg_score,
                "scores": scores,
                "passed": sum(1 for s in scores if s >= kwargs.get("threshold", 0.5)),
                "total": len(scores),
            }
        except Exception as exc:
            logger.error("deepeval_metric_failed", metric=metric_name, error=str(exc))
            results["per_metric"][metric_name] = {"error": str(exc)}

    logger.info(
        "deepeval_suite_complete",
        metrics=list(results["per_metric"].keys()),
    )
    return results


def deepeval_available() -> bool:
    """Check if DeepEval library is importable."""
    try:
        import deepeval  # noqa: F401
        return True
    except ImportError:
        return False
