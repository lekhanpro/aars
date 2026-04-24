"""Benchmark dashboard tab with results visualization."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "results_local.json"


def render() -> None:
    """Render the benchmark dashboard tab."""
    st.header("Benchmark Results")

    results = _load_results()
    if results is None:
        st.warning("No benchmark results found. Run the benchmark first:")
        st.code("python benchmarks/runner.py --output benchmarks/results_local.json")
        return

    # Summary metrics
    systems = results.get("systems", {})
    if not systems:
        st.info("No system results in the benchmark file.")
        return

    # System comparison table
    st.subheader("System Comparison")

    rows: list[dict] = []
    for system_name, system_data in systems.items():
        metrics = system_data.get("aggregate", system_data.get("metrics", {}))
        row = {"System": system_name}
        row.update({k: f"{v:.3f}" if isinstance(v, float) else str(v) for k, v in metrics.items()})
        rows.append(row)

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Metrics comparison chart
    st.subheader("Metrics Comparison")

    metric_names = set()
    for system_data in systems.values():
        metrics = system_data.get("aggregate", system_data.get("metrics", {}))
        metric_names.update(k for k, v in metrics.items() if isinstance(v, (int, float)))

    if metric_names:
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            sorted(metric_names),
            default=sorted(metric_names)[:4],
        )

        if selected_metrics:
            import pandas as pd

            chart_data: dict[str, list[float]] = {m: [] for m in selected_metrics}
            system_names: list[str] = []

            for system_name, system_data in systems.items():
                system_names.append(system_name)
                metrics = system_data.get("aggregate", system_data.get("metrics", {}))
                for m in selected_metrics:
                    val = metrics.get(m, 0.0)
                    chart_data[m].append(float(val) if isinstance(val, (int, float)) else 0.0)

            chart_df = pd.DataFrame(chart_data, index=system_names)
            st.bar_chart(chart_df)

    # Per-question results
    st.subheader("Per-Question Results")
    system_names_list = list(systems.keys())
    selected_system = st.selectbox("Select system", system_names_list)

    if selected_system:
        system_data = systems[selected_system]
        per_question = system_data.get("per_question", system_data.get("questions", []))
        if per_question:
            import pandas as pd

            st.dataframe(pd.DataFrame(per_question), use_container_width=True, hide_index=True)
        else:
            st.info("No per-question data available for this system.")

    # RAGAS / DeepEval results
    if results.get("ragas"):
        st.subheader("RAGAS Evaluation")
        ragas = results["ragas"]
        if "scores" in ragas:
            import pandas as pd
            ragas_df = pd.DataFrame([ragas["scores"]])
            st.dataframe(ragas_df, use_container_width=True, hide_index=True)
        elif "error" in ragas:
            st.error(f"RAGAS error: {ragas['error']}")

    if results.get("deepeval"):
        st.subheader("DeepEval Evaluation")
        deepeval = results["deepeval"]
        if "per_metric" in deepeval:
            import pandas as pd
            rows = []
            for metric_name, metric_data in deepeval["per_metric"].items():
                if isinstance(metric_data, dict) and "average" in metric_data:
                    rows.append({
                        "Metric": metric_name,
                        "Average": f"{metric_data['average']:.3f}",
                        "Passed": f"{metric_data.get('passed', 0)}/{metric_data.get('total', 0)}",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Raw JSON viewer
    with st.expander("Raw Results JSON"):
        st.json(results)


def _load_results() -> dict | None:
    """Load benchmark results from the checked-in results file."""
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
