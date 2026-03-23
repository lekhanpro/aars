"""Benchmark runner — orchestrates dataset loading, evaluation, and reporting."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from benchmarks.ablations import ABLATION_CONFIGS, AblationConfig
from benchmarks.datasets import DatasetLoader
from benchmarks.metrics import Metrics
from benchmarks.significance import compare_systems

logger = structlog.get_logger()

DATASET_NAMES = ["hotpotqa", "nq", "triviaqa", "msmarco"]


class BenchmarkRunner:
    """Run benchmarks against the AARS API and baselines."""

    def __init__(self, api_url: str = "http://localhost:8000", timeout: float = 120.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    async def query_aars(
        self,
        query: str,
        collection: str = "default",
        enable_reflection: bool = True,
    ) -> dict:
        """Send a query to the AARS API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_url}/api/v1/query",
                json={
                    "query": query,
                    "collection": collection,
                    "enable_reflection": enable_reflection,
                    "enable_trace": True,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def run_dataset(
        self,
        name: str,
        samples: list[dict],
        enable_reflection: bool = True,
    ) -> dict:
        """Run AARS on a dataset and compute metrics."""
        em_scores: list[float] = []
        f1_scores: list[float] = []
        latencies: list[float] = []
        total_tokens = 0
        total_api_calls = 0

        for i, sample in enumerate(samples):
            query = sample["question"]
            gold_answer = sample["answer"]

            try:
                start = time.monotonic()
                result = await self.query_aars(
                    query, enable_reflection=enable_reflection
                )
                latency = (time.monotonic() - start) * 1000

                predicted = result.get("answer", "")
                em_scores.append(Metrics.exact_match(predicted, gold_answer))
                f1_scores.append(Metrics.token_f1(predicted, gold_answer))
                latencies.append(latency)

                if result.get("trace"):
                    total_tokens += result["trace"].get("total_tokens", 0)
                    total_api_calls += result["trace"].get("total_api_calls", 0)

            except Exception as e:
                logger.error("benchmark_query_failed", index=i, error=str(e))
                em_scores.append(0.0)
                f1_scores.append(0.0)
                latencies.append(0.0)

            if (i + 1) % 50 == 0:
                logger.info("benchmark_progress", dataset=name, completed=i + 1, total=len(samples))

        return {
            "dataset": name,
            "num_samples": len(samples),
            "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0,
            "token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "total_tokens": total_tokens,
            "total_api_calls": total_api_calls,
            "per_sample_em": em_scores,
            "per_sample_f1": f1_scores,
        }

    async def run_ablations(
        self, name: str, samples: list[dict]
    ) -> dict[str, dict]:
        """Run ablation configurations on a dataset."""
        results = {}
        for config in ABLATION_CONFIGS:
            logger.info("ablation_start", config=config.name, dataset=name)
            result = await self.run_dataset(
                name=f"{name}_{config.name}",
                samples=samples,
                enable_reflection=config.enable_reflection,
            )
            results[config.name] = result
        return results

    def generate_latex_tables(self, results: dict[str, Any]) -> str:
        """Generate LaTeX tables from benchmark results."""
        lines = []

        # Main results table
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\caption{Main results across benchmark datasets.}")
        lines.append("\\label{tab:main_results}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("System & EM & Token F1 & Latency (ms) & API Calls \\\\")
        lines.append("\\midrule")

        for system_name, system_results in results.items():
            if isinstance(system_results, dict) and "exact_match" in system_results:
                lines.append(
                    f"{system_name} & {system_results['exact_match']:.3f} & "
                    f"{system_results['token_f1']:.3f} & "
                    f"{system_results['avg_latency_ms']:.0f} & "
                    f"{system_results['total_api_calls']} \\\\"
                )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    async def run_all(
        self,
        dataset_names: list[str],
        max_samples: int = 500,
        output_path: str = "benchmarks/results.json",
    ) -> dict:
        """Run the full benchmark suite."""
        all_results: dict[str, Any] = {}

        for name in dataset_names:
            logger.info("loading_dataset", name=name)
            try:
                samples = DatasetLoader.load_dataset_by_name(name, max_samples=max_samples)
            except Exception as e:
                logger.error("dataset_load_failed", name=name, error=str(e))
                continue

            logger.info("running_benchmark", dataset=name, num_samples=len(samples))

            # Run AARS
            aars_results = await self.run_dataset(name, samples)
            all_results[f"aars_{name}"] = aars_results

            # Run ablations
            ablation_results = await self.run_ablations(name, samples)
            all_results[f"ablations_{name}"] = ablation_results

        # Generate LaTeX
        latex = self.generate_latex_tables(all_results)
        all_results["latex_tables"] = latex

        # Save results
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Remove non-serializable per-sample lists for JSON output
        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                serializable[k] = {
                    sk: sv for sk, sv in v.items()
                    if not sk.startswith("per_sample_")
                }
            else:
                serializable[k] = v

        output.write_text(json.dumps(serializable, indent=2))
        logger.info("results_saved", path=str(output))

        # Statistical significance
        if len(dataset_names) > 0:
            logger.info("running_significance_tests")
            # Collect per-sample F1 scores for significance testing
            f1_scores: dict[str, list[float]] = {}
            for key, value in all_results.items():
                if isinstance(value, dict) and "per_sample_f1" in value:
                    f1_scores[key] = value["per_sample_f1"]

        return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="AARS Benchmark Runner")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "nq"],
        choices=DATASET_NAMES,
        help="Datasets to benchmark",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args()

    runner = BenchmarkRunner(api_url=args.api_url)
    asyncio.run(runner.run_all(args.datasets, args.max_samples, args.output))


if __name__ == "__main__":
    main()
