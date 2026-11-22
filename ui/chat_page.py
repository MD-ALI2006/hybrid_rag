"""
Chat with Documents page – the main Q&A interface.
"""

import streamlit as st
from src.utils import truncate_text


def _display_sources(sources: list[dict], citations: list[dict]):
    """Show cited sources with expandable text."""
    if not sources:
        return

    st.markdown("---")
    st.markdown("#### 📚 Retrieved Sources")

    # Show verification status of cited sources
    if citations:
        verified = sum(1 for c in citations if c.get("verified", False))
        total = len(citations)
        if verified == total:
            st.success(f"✅ All {total} citation(s) verified against retrieved content.")
        else:
            st.warning(f"⚠️ {verified}/{total} citation(s) verified.")

    for i, src in enumerate(sources, 1):
        sec = f" | Section: {src['section_title']}" if src.get("section_title") else ""
        label = f"📄 {src['document_name']} — Page {src['page_number']}{sec} (score: {src['score']:.4f})"
        with st.expander(label, expanded=False):
            st.text(src["chunk_text"])


def render_chat_page():
    """Chat interface for querying documents."""

    st.markdown("## 💬 Chat with Documents")

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.error("Pipeline not initialized.")
        return

    if not pipeline.chunks:
        st.warning("⚠️ No documents loaded. Please upload PDFs on the Upload page first.")
        return

    # ── Sidebar controls ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Chat Settings")

        mode = st.radio(
            "Retrieval Mode",
            ["hybrid", "bm25", "dense"],
            format_func=lambda m: {"hybrid": "🔀 Hybrid", "bm25": "🔤 BM25", "dense": "🧠 Dense"}[m],
            index=0,
            help="Select which retrieval method to use for finding relevant chunks.",
        )

        top_k = st.slider("Candidates to retrieve", 3, 20, 10, help="Number of chunks each retriever returns.")
        rerank_n = st.slider("Chunks after reranking", 1, 10, 5, help="Final chunks sent to the LLM.")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display past messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _display_sources(msg["sources"], msg.get("citations", []))

    # ── User input ────────────────────────────────────────────────────
    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        # Show user message
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                result = pipeline.query(
                    question=prompt,
                    mode=mode,
                    top_k=top_k,
                    rerank_top_n=rerank_n,
                )

            # Display answer
            st.markdown(result["answer"])

            # Timing info
            m = result["metrics"]
            mode_label = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}[result["mode"]]
            timing_text = (
                f"⏱️ **{mode_label}** | "
                f"Retrieval: {m.bm25_time + m.dense_time + m.hybrid_time:.3f}s | "
                f"Rerank: {m.rerank_time:.3f}s | "
                f"LLM: {m.llm_time:.3f}s | "
                f"Total: {m.total_time:.3f}s"
            )
            st.caption(timing_text)

            # Sources
            _display_sources(result["sources"], result["citations"])

            # Save to history
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "citations": result["citations"],
                }
            )
