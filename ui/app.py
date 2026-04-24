"""AARS Streamlit Dashboard — 3-tab interface for query, benchmarks, and document management."""

from __future__ import annotations

import os

import streamlit as st

from ui.components import query_tab, benchmark_tab, document_tab

API_URL = os.getenv("AARS_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AARS — Agentic Adaptive Retrieval",
    page_icon="🔍",
    layout="wide",
)

st.title("AARS — Agentic Adaptive Retrieval System")
st.markdown("Dynamic strategy selection for Retrieval-Augmented Generation")

# Sidebar — Global settings
with st.sidebar:
    st.header("Settings")
    collection = st.text_input("Collection", value="default")
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)

    st.divider()
    st.header("Agent Toggles")
    settings = {
        "reflection": st.checkbox("Enable Reflection", value=True),
        "reranker": st.checkbox("Enable Cross-Encoder", value=True),
        "hallucination": st.checkbox("Enable Hallucination Check", value=True),
        "grading": st.checkbox("Enable Relevance Grading", value=True),
    }

    st.divider()
    st.caption(f"API: {API_URL}")

# Main content — 3 tabs
tab1, tab2, tab3 = st.tabs(["Query", "Benchmarks", "Documents"])

with tab1:
    query_tab.render(api_url=API_URL, collection=collection, top_k=top_k, settings=settings)

with tab2:
    benchmark_tab.render()

with tab3:
    document_tab.render(api_url=API_URL)
