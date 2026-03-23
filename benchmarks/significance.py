"""Statistical significance testing via paired bootstrap resampling."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap resampling for statistical significance testing.

    Tests whether system A is significantly better than system B.

    Args:
        scores_a: Per-sample scores for system A.
        scores_b: Per-sample scores for system B.
        n_iterations: Number of bootstrap iterations.
        confidence: Confidence level for interval.

    Returns:
        Dict with p_value, significant, mean_diff, ci_lower, ci_upper.
    """
    assert len(scores_a) == len(scores_b), "Score lists must be same length"
    n = len(scores_a)
    if n == 0:
        return {
            "p_value": 1.0,
            "significant": False,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    a = np.array(scores_a)
    b = np.array(scores_b)
    observed_diff = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(42)
    diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        indices = rng.integers(0, n, size=n)
        sample_a = a[indices]
        sample_b = b[indices]
        diffs[i] = np.mean(sample_a) - np.mean(sample_b)

    # Two-sided p-value: proportion of bootstrap diffs with opposite sign or larger magnitude
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0))
    else:
        p_value = float(np.mean(diffs >= 0))

    # Confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return {
        "p_value": p_value,
        "significant": p_value < (1 - confidence),
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compare_systems(
    results: dict[str, list[float]],
    baseline_name: str = "naive_rag",
    n_iterations: int = 1000,
) -> dict[str, dict]:
    """Compare all systems against a baseline.

    Args:
        results: Dict mapping system name → list of per-sample scores.
        baseline_name: Name of the baseline system to compare against.
        n_iterations: Number of bootstrap iterations.

    Returns:
        Dict mapping system name → significance test result.
    """
    if baseline_name not in results:
        raise ValueError(f"Baseline '{baseline_name}' not found in results")

    baseline_scores = results[baseline_name]
    comparisons = {}

    for name, scores in results.items():
        if name == baseline_name:
            continue
        comparisons[name] = paired_bootstrap(
            scores, baseline_scores, n_iterations=n_iterations
        )

    return comparisons
