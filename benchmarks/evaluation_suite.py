"""Unified evaluation suite combining custom metrics, RAGAS, and DeepEval."""

from __future__ import annotations

from typing import Any

import structlog

from benchmarks.ragas_eval import ragas_available, run_ragas_evaluation
from benchmarks.deepeval_eval import deepeval_available, run_deepeval_suite

logger = structlog.get_logger(__name__)


async def run_full_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Run a combined evaluation across available frameworks.

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
    metrics:
        Which evaluation suites to run. Options: "ragas", "deepeval".
        If None, runs all available frameworks.

    Returns
    -------
    dict
        Combined results from all enabled evaluation frameworks.
    """
    enabled = set(metrics) if metrics else {"ragas", "deepeval"}
    results: dict[str, Any] = {"available_frameworks": []}

    if "ragas" in enabled:
        if ragas_available():
            results["available_frameworks"].append("ragas")
            ragas_results = await run_ragas_evaluation(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths,
            )
            results["ragas"] = ragas_results
        else:
            results["ragas"] = {"status": "not_installed"}
            logger.info("ragas_skipped", reason="not installed")

    if "deepeval" in enabled:
        if deepeval_available():
            results["available_frameworks"].append("deepeval")
            deepeval_results = run_deepeval_suite(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths,
            )
            results["deepeval"] = deepeval_results
        else:
            results["deepeval"] = {"status": "not_installed"}
            logger.info("deepeval_skipped", reason="not installed")

    logger.info(
        "full_evaluation_complete",
        frameworks=results["available_frameworks"],
    )
    return results
