"""
Upload & Process Documents page.
"""

import streamlit as st
from pathlib import Path
from config.settings import UPLOAD_DIR, MAX_FILE_SIZE_MB


def render_upload_page():
    """Document upload, validation, and processing UI."""

    st.markdown("## 📄 Upload & Process Documents")

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.error("Pipeline not initialized. Please wait for models to load.")
        return

    # ── Chunking settings ─────────────────────────────────────────────
    with st.expander("⚙️ Chunking Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider(
                "Chunk size (characters)",
                200, 2000, st.session_state.get("chunk_size", 800), 50,
                help="Larger chunks give more context but may include irrelevant text.",
            )
            st.session_state["chunk_size"] = chunk_size
        with col2:
            chunk_overlap = st.slider(
                "Chunk overlap (characters)",
                0, 500, st.session_state.get("chunk_overlap", 150), 25,
                help="Overlap prevents losing information at chunk boundaries.",
            )
            st.session_state["chunk_overlap"] = chunk_overlap

    st.divider()

    # ── File uploader ─────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max file size: {MAX_FILE_SIZE_MB} MB per file.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)}** file(s) selected")

        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            saved_paths = []

            # Save uploaded files
            for uf in uploaded_files:
                size_mb = len(uf.getvalue()) / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"❌ {uf.name}: {size_mb:.1f} MB exceeds the {MAX_FILE_SIZE_MB} MB limit.")
                    continue

                dest = UPLOAD_DIR / uf.name
                dest.write_bytes(uf.getvalue())
                saved_paths.append(str(dest))

            if not saved_paths:
                st.error("No valid files to process.")
                return

            # Process
            with st.spinner("Extracting text, chunking, and building indices..."):
                result = pipeline.process_documents(saved_paths)

            if result["success"]:
                st.success("✅ Documents processed successfully!")

                # Stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📚 Documents", result["documents"])
                col2.metric("📄 Pages", result["pages"])
                col3.metric("📝 Characters", f"{result['characters']:,}")
                col4.metric("🧩 Chunks", result["chunks"])

                if result["warnings"]:
                    for w in result["warnings"]:
                        st.warning(f"⚠️ {w}")
            else:
                st.error("❌ Document processing failed.")
                for w in result["warnings"]:
                    st.warning(f"⚠️ {w}")

    # ── Current document status ───────────────────────────────────────
    if pipeline.chunks:
        st.divider()
        st.markdown("### 📂 Loaded Documents")

        for doc_name in pipeline.doc_names:
            doc_chunks = [c for c in pipeline.chunks if c["document_name"] == doc_name]
            doc_pages = set(c["page_number"] for c in doc_chunks)
            st.markdown(
                f"- **{doc_name}** — {len(doc_pages)} pages, {len(doc_chunks)} chunks"
            )

        with st.expander("🔍 Preview Chunks", expanded=False):
            max_preview = min(10, len(pipeline.chunks))
            for i, chunk in enumerate(pipeline.chunks[:max_preview]):
                st.markdown(
                    f"**Chunk {i+1}** | {chunk['document_name']} | "
                    f"Page {chunk['page_number']} | Section: {chunk['section_title']}"
                )
                st.text(chunk["chunk_text"][:300] + ("..." if len(chunk["chunk_text"]) > 300 else ""))
                st.divider()
