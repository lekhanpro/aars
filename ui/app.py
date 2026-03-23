"""AARS Streamlit Demo UI."""

from __future__ import annotations

import json
import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("AARS_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AARS — Agentic Adaptive Retrieval",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AARS — Agentic Adaptive Retrieval System")
st.markdown("Dynamic strategy selection for Retrieval-Augmented Generation")

# Sidebar — Configuration & Document Upload
with st.sidebar:
    st.header("⚙️ Settings")
    collection = st.text_input("Collection", value="default")
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)
    enable_reflection = st.checkbox("Enable Reflection", value=True)
    enable_trace = st.checkbox("Show Pipeline Trace", value=True)

    st.divider()
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt"],
        help="Upload PDF or text files to index",
    )
    if uploaded_file and st.button("📥 Ingest Document"):
        with st.spinner("Ingesting document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"collection": collection}
                resp = httpx.post(
                    f"{API_URL}/api/v1/ingest",
                    files=files,
                    data=data,
                    timeout=120.0,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"✅ Ingested {result['documents_ingested']} documents, "
                        f"{result['chunks_created']} chunks"
                    )
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()
    st.header("📊 System Status")
    if st.button("🔄 Check Health"):
        try:
            resp = httpx.get(f"{API_URL}/api/v1/health", timeout=5.0)
            health = resp.json()
            st.json(health)
        except Exception as e:
            st.error(f"API unreachable: {e}")

# Main content — Query Interface
st.header("💬 Ask a Question")

query = st.text_area(
    "Enter your query",
    placeholder="e.g., What is the relationship between transformers and attention mechanisms?",
    height=100,
)

if st.button("🚀 Submit Query", type="primary", disabled=not query):
    with st.spinner("Processing query through AARS pipeline..."):
        start = time.time()
        try:
            resp = httpx.post(
                f"{API_URL}/api/v1/query",
                json={
                    "query": query,
                    "collection": collection,
                    "top_k": top_k,
                    "enable_reflection": enable_reflection,
                    "enable_trace": enable_trace,
                },
                timeout=120.0,
            )
            elapsed = time.time() - start

            if resp.status_code == 200:
                result = resp.json()

                # Answer
                st.subheader("📝 Answer")
                st.markdown(result["answer"])
                st.caption(f"Confidence: {result['confidence']:.2%} | Time: {elapsed:.2f}s")

                # Citations
                if result.get("citations"):
                    with st.expander("📚 Citations", expanded=True):
                        for cite in result["citations"]:
                            st.markdown(f"**[{cite['doc_id']}]**: {cite['text']}")

                # Retrieval Plan
                if result.get("retrieval_plan"):
                    with st.expander("🗺️ Retrieval Plan"):
                        plan = result["retrieval_plan"]
                        cols = st.columns(3)
                        cols[0].metric("Strategy", plan["strategy"])
                        cols[1].metric("Query Type", plan["query_type"])
                        cols[2].metric("Complexity", plan["complexity"])
                        if plan.get("reasoning"):
                            st.info(plan["reasoning"])
                        if plan.get("rewritten_query"):
                            st.text(f"Rewritten query: {plan['rewritten_query']}")

                # Reflection Results
                if result.get("reflection_results"):
                    with st.expander("🔄 Reflection Loop"):
                        for i, ref in enumerate(result["reflection_results"]):
                            st.markdown(f"**Iteration {i + 1}**")
                            cols = st.columns(2)
                            cols[0].metric("Sufficient", "✅" if ref["sufficient"] else "❌")
                            cols[1].metric("Confidence", f"{ref['confidence']:.2%}")
                            if ref.get("missing_information"):
                                st.warning(f"Missing: {ref['missing_information']}")

                # Retrieved Documents
                if result.get("documents"):
                    with st.expander(f"📄 Retrieved Documents ({len(result['documents'])})"):
                        for doc in result["documents"]:
                            st.markdown(f"**{doc['id']}** (score: {doc['score']:.4f})")
                            st.text(doc["content"][:500])
                            st.divider()

                # Pipeline Trace
                if result.get("trace"):
                    with st.expander("🔬 Pipeline Trace"):
                        trace = result["trace"]
                        st.caption(
                            f"Trace ID: {trace['trace_id']} | "
                            f"Total: {trace['total_duration_ms']:.0f}ms | "
                            f"Tokens: {trace['total_tokens']} | "
                            f"API Calls: {trace['total_api_calls']}"
                        )
                        for step in trace["steps"]:
                            col1, col2 = st.columns([3, 1])
                            col1.markdown(f"**{step['step']}**")
                            col2.markdown(f"`{step['duration_ms']:.0f}ms`")
                            if step.get("details"):
                                st.json(step["details"])
            else:
                st.error(f"Error: {resp.text}")
        except httpx.ConnectError:
            st.error(
                "Could not connect to AARS API. "
                "Make sure the server is running at " + API_URL
            )
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.divider()
st.caption("AARS — Agentic Adaptive Retrieval System | Research Prototype")
