"""Document management tab — upload, list, delete collections."""

from __future__ import annotations

import httpx
import streamlit as st


def render(api_url: str) -> None:
    """Render the document management tab."""
    st.header("Document Management")

    col1, col2 = st.columns(2)

    # Upload section
    with col1:
        st.subheader("Upload Documents")
        collection = st.text_input("Target collection", value="default", key="doc_upload_collection")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md"],
            help="Upload PDF, text, or markdown files to index",
        )
        if uploaded_file and st.button("Ingest Document"):
            with st.spinner("Ingesting document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {"collection": collection}
                    resp = httpx.post(
                        f"{api_url}/api/v1/ingest",
                        files=files,
                        data=data,
                        timeout=120.0,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(
                            f"Ingested {result['documents_ingested']} document(s), "
                            f"{result['chunks_created']} chunks into '{collection}'"
                        )
                    else:
                        st.error(f"Ingestion failed: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # Collections section
    with col2:
        st.subheader("Collections")
        if st.button("Refresh Collections"):
            st.rerun()

        try:
            resp = httpx.get(f"{api_url}/api/v1/collections", timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                collections = data.get("collections", [])
                if collections:
                    for col_info in collections:
                        with st.container():
                            c1, c2, c3 = st.columns([3, 1, 1])
                            c1.write(f"**{col_info['name']}**")
                            c2.write(f"{col_info['document_count']} docs")
                            if c3.button("Delete", key=f"del_{col_info['name']}"):
                                try:
                                    del_resp = httpx.delete(
                                        f"{api_url}/api/v1/collections/{col_info['name']}",
                                        timeout=10.0,
                                    )
                                    if del_resp.status_code == 200:
                                        st.success(f"Deleted '{col_info['name']}'")
                                        st.rerun()
                                    else:
                                        st.error(f"Delete failed: {del_resp.text}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                else:
                    st.info("No collections found.")
            else:
                st.error(f"Failed to fetch collections: {resp.text}")
        except httpx.ConnectError:
            st.error(f"Could not connect to AARS API at {api_url}")
        except Exception as e:
            st.error(f"Error: {e}")

    # System health
    st.divider()
    st.subheader("System Health")
    if st.button("Check Health"):
        try:
            resp = httpx.get(f"{api_url}/api/v1/health", timeout=5.0)
            health = resp.json()
            cols = st.columns(3)
            cols[0].metric("Status", health.get("status", "unknown"))
            cols[1].metric("Version", health.get("version", "unknown"))
            cols[2].metric("ChromaDB", "Connected" if health.get("chromadb_connected") else "Disconnected")
        except Exception as e:
            st.error(f"API unreachable: {e}")
