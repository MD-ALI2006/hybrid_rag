"""
Dashboard page – system overview and stats.
"""

import streamlit as st


def render_dashboard():
    """Main dashboard with system status and document stats."""

    st.markdown("## 📊 Dashboard")

    # ── System status row ─────────────────────────────────────────────
    pipeline = st.session_state.get("pipeline")
    models_loaded = pipeline is not None and pipeline.models_loaded

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Models Status",
            "✅ Ready" if models_loaded else "⏳ Not loaded",
        )
    with col2:
        docs = len(pipeline.doc_names) if pipeline else 0
        st.metric("Documents", docs)
    with col3:
        pages = len(pipeline.pages) if pipeline else 0
        st.metric("Pages Extracted", pages)
    with col4:
        chunks = len(pipeline.chunks) if pipeline else 0
        st.metric("Text Chunks", chunks)

    st.divider()

    # ── Quick start guide ─────────────────────────────────────────────
    st.markdown("### 🚀 Quick Start Guide")

    steps = [
        ("1️⃣", "**Upload Documents** — Go to the Upload page and add your PDF files."),
        ("2️⃣", "**Ask Questions** — Use the Chat page to query your documents."),
        ("3️⃣", "**Compare Methods** — Visit Retrieval Comparison to see how BM25, Dense, and Hybrid retrieval differ."),
        ("4️⃣", "**Review Metrics** — Check the Evaluation Dashboard for performance analysis."),
    ]

    for icon, text in steps:
        st.markdown(f"{icon} {text}")

    st.divider()

    # ── How it works ──────────────────────────────────────────────────
    st.markdown("### 🔬 How Hybrid RAG Works")

    st.markdown("""
    This system combines **three retrieval strategies** to find the most relevant
    information from your documents:

    | Method | How it works | Best for |
    |--------|-------------|----------|
    | **BM25 (Keyword)** | Matches exact words and terms using statistical scoring | Technical terms, exact phrases |
    | **Dense (Semantic)** | Uses AI embeddings to find meaning-similar text | Paraphrased questions, conceptual queries |
    | **Hybrid (Combined)** | Fuses BM25 + Dense results with Reciprocal Rank Fusion | Best overall coverage |

    After retrieval, a **Cross-Encoder Reranker** re-scores the candidates for
    higher precision, then **Google Gemini** generates an answer grounded in the
    retrieved context with page-level citations.
    """)

    # ── Session metrics ───────────────────────────────────────────────
    if pipeline and pipeline.session_metrics.total_queries > 0:
        st.divider()
        st.markdown("### 📈 Session Statistics")
        sm = pipeline.session_metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Queries", sm.total_queries)
        with c2:
            st.metric("Avg Response Time", f"{sm.avg('total_time'):.2f}s")
        with c3:
            st.metric("Avg LLM Time", f"{sm.avg('llm_time'):.2f}s")

    # ── System info ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### ℹ️ System Information")
    from config.settings import EMBEDDING_MODEL, RERANKER_MODEL, GEMINI_MODEL

    info_data = {
        "Embedding Model": EMBEDDING_MODEL,
        "Reranker Model": RERANKER_MODEL,
        "LLM Model": GEMINI_MODEL,
        "Vector Store": "ChromaDB (persistent)",
        "Lexical Index": "BM25 (rank_bm25)",
        "Fusion Method": "Reciprocal Rank Fusion (RRF)",
    }

    for label, value in info_data.items():
        st.markdown(f"- **{label}:** `{value}`")

    st.info(
        "💡 **Privacy note:** Only the retrieved text chunks (not the full documents) "
        "are sent to Google Gemini for answer generation. Document processing, "
        "embedding, and retrieval happen entirely on your machine."
    )
