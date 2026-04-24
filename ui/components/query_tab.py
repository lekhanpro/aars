"""Query interface tab with agent trace visualization."""

from __future__ import annotations

import json
import time

import httpx
import streamlit as st


def render(api_url: str, collection: str, top_k: int, settings: dict) -> None:
    """Render the query interface tab."""
    st.header("Ask a Question")

    query = st.text_area(
        "Enter your query",
        placeholder="e.g., What is the relationship between transformers and attention mechanisms?",
        height=100,
        key="query_input",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        enable_reflection = st.checkbox("Reflection", value=settings.get("reflection", True))
    with col2:
        enable_reranker = st.checkbox("Cross-Encoder Reranker", value=settings.get("reranker", True))
    with col3:
        enable_hallucination = st.checkbox("Hallucination Check", value=settings.get("hallucination", True))

    col4, col5 = st.columns(2)
    with col4:
        enable_grading = st.checkbox("Relevance Grading", value=settings.get("grading", True))
    with col5:
        use_graph_pipeline = st.checkbox("Use LangGraph Pipeline", value=False)

    if st.button("Submit Query", type="primary", disabled=not query):
        endpoint = "/api/v1/query/graph" if use_graph_pipeline else "/api/v1/query"

        with st.spinner("Processing query through AARS pipeline..."):
            start = time.time()
            try:
                resp = httpx.post(
                    f"{api_url}{endpoint}",
                    json={
                        "query": query,
                        "collection": collection,
                        "top_k": top_k,
                        "enable_reflection": enable_reflection,
                        "enable_reranker": enable_reranker,
                        "enable_hallucination_check": enable_hallucination,
                        "enable_grading": enable_grading,
                        "enable_trace": True,
                    },
                    timeout=120.0,
                )
                elapsed = time.time() - start

                if resp.status_code == 200:
                    result = resp.json()
                    _render_result(result, elapsed)
                else:
                    st.error(f"Error: {resp.text}")
            except httpx.ConnectError:
                st.error(f"Could not connect to AARS API at {api_url}")
            except Exception as e:
                st.error(f"Error: {e}")


def _render_result(result: dict, elapsed: float) -> None:
    """Render the full query result with agent trace."""
    # Answer
    st.subheader("Answer")
    st.markdown(result["answer"])

    # Metrics row
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{result['confidence']:.0%}")
    cols[1].metric("Time", f"{elapsed:.2f}s")
    cols[2].metric("Reranker", "Yes" if result.get("reranker_applied") else "No")

    halluc = result.get("hallucination_result")
    if halluc:
        cols[3].metric("Grounded", "Yes" if halluc["grounded"] else "No")

    # Self-RAG Evaluation
    eval_result = result.get("self_rag_evaluation")
    if eval_result:
        with st.expander("Self-RAG Evaluation Scores", expanded=True):
            eval_cols = st.columns(5)
            eval_cols[0].metric("Faithfulness", f"{eval_result['faithfulness']:.2f}")
            eval_cols[1].metric("Relevancy", f"{eval_result['answer_relevancy']:.2f}")
            eval_cols[2].metric("Ctx Precision", f"{eval_result['context_precision']:.2f}")
            eval_cols[3].metric("Ctx Recall", f"{eval_result['context_recall']:.2f}")
            eval_cols[4].metric("Overall", f"{eval_result['overall']:.2f}")

    # Citations
    if result.get("citations"):
        with st.expander("Citations", expanded=True):
            for cite in result["citations"]:
                st.markdown(f"**[{cite['doc_id']}]**: {cite['text']}")

    # Retrieval Plan
    if result.get("retrieval_plan"):
        with st.expander("Retrieval Plan"):
            plan = result["retrieval_plan"]
            plan_cols = st.columns(3)
            plan_cols[0].metric("Strategy", plan["strategy"])
            plan_cols[1].metric("Query Type", plan["query_type"])
            plan_cols[2].metric("Complexity", plan["complexity"])
            if plan.get("reasoning"):
                st.info(plan["reasoning"])
            if plan.get("decomposed_queries"):
                st.write("**Decomposed queries:**")
                for i, sq in enumerate(plan["decomposed_queries"], 1):
                    st.write(f"  {i}. {sq}")

    # Hallucination Details
    if halluc and halluc.get("ungrounded_claims"):
        with st.expander("Hallucination Check Details"):
            st.metric("Grounding Score", f"{halluc['score']:.2f}")
            st.write("**Ungrounded claims:**")
            for claim in halluc["ungrounded_claims"]:
                st.warning(claim)

    # Graded Documents
    if result.get("graded_documents"):
        with st.expander("Document Relevance Grades"):
            for grade in result["graded_documents"]:
                icon = "+" if grade["relevant"] else "-"
                st.markdown(f"**[{icon}] {grade['doc_id']}**: {grade.get('reasoning', '')}")

    # Reflection Loop
    if result.get("reflection_results"):
        with st.expander("Reflection Loop"):
            for i, ref in enumerate(result["reflection_results"]):
                st.markdown(f"**Iteration {i + 1}**")
                ref_cols = st.columns(2)
                ref_cols[0].metric("Sufficient", "Yes" if ref["sufficient"] else "No")
                ref_cols[1].metric("Confidence", f"{ref['confidence']:.0%}")
                if ref.get("missing_information"):
                    st.warning(f"Missing: {ref['missing_information']}")

    # Retrieved Documents
    if result.get("documents"):
        with st.expander(f"Retrieved Documents ({len(result['documents'])})"):
            for doc in result["documents"]:
                st.markdown(f"**{doc['id']}** (score: {doc['score']:.4f})")
                st.text(doc["content"][:500])
                st.divider()

    # Pipeline Trace
    if result.get("trace"):
        with st.expander("Pipeline Trace"):
            trace = result["trace"]
            st.caption(
                f"Trace ID: {trace['trace_id']} | "
                f"Total: {trace['total_duration_ms']:.0f}ms | "
                f"Tokens: {trace['total_tokens']} | "
                f"API Calls: {trace['total_api_calls']}"
            )

            # Timing bar chart
            step_names = [s["step"] for s in trace["steps"]]
            step_durations = [s["duration_ms"] for s in trace["steps"]]
            if step_durations:
                import pandas as pd
                chart_data = pd.DataFrame({"Step": step_names, "Duration (ms)": step_durations})
                st.bar_chart(chart_data.set_index("Step"))

            # Step details
            for step in trace["steps"]:
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**{step['step']}**")
                col2.markdown(f"`{step['duration_ms']:.0f}ms`")
                if step.get("details"):
                    st.json(step["details"])
