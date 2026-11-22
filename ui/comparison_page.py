"""
Retrieval Comparison page – side-by-side analysis of BM25, Dense, Hybrid.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def _results_to_df(results: list[dict], method: str) -> pd.DataFrame:
    """Convert retriever output to a DataFrame for display."""
    rows = []
    for r in results:
        c = r["chunk"]
        rows.append(
            {
                "Rank": r["rank"],
                "Score": round(r.get("rerank_score", r["score"]), 4),
                "Document": c["document_name"],
                "Page": c["page_number"],
                "Section": c.get("section_title", ""),
                "Preview": c["chunk_text"][:120] + "...",
                "Chunk ID": c["chunk_id"],
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render_comparison_page():
    """Side-by-side retrieval comparison."""

    st.markdown("## 🔍 Retrieval Comparison")

    pipeline = st.session_state.get("pipeline")
    if pipeline is None or not pipeline.chunks:
        st.warning("⚠️ Please upload and process documents first.")
        return

    # ── Explanation cards ─────────────────────────────────────────────
    with st.expander("📖 Understanding the retrieval methods", expanded=False):
        st.markdown("""
        | Method | Description |
        |--------|-------------|
        | **BM25** | Keyword matching using term frequency and document length normalization. Great for exact terms. |
        | **Dense** | Semantic similarity via neural embeddings. Finds conceptually related text even with different wording. |
        | **Hybrid** | Combines both using Reciprocal Rank Fusion (RRF) for the best of both worlds. |
        | **Reranker** | A cross-encoder re-scores the hybrid candidates for more precise final ranking. |

        **Reciprocal Rank Fusion (RRF)** merges ranked lists by scoring each document as:

        $$\\text{RRF}(d) = \\sum_{r \\in \\text{retrievers}} \\frac{1}{k + \\text{rank}_r(d)}$$

        where *k* is a constant (default 60). This balances contributions from each retriever
        without needing comparable raw scores.
        """)

    st.divider()

    # ── Query input ───────────────────────────────────────────────────
    query = st.text_input(
        "Enter a question to compare retrieval methods:",
        placeholder="e.g. What are the main findings of the study?",
    )

    top_k = st.slider("Top-K candidates per retriever", 3, 20, 10, key="comp_topk")

    if query and st.button("🔎 Compare", type="primary", use_container_width=True):
        with st.spinner("Running all retrieval methods..."):
            comp = pipeline.compare_retrieval(query, top_k=top_k)

        # ── Timing metrics ────────────────────────────────────────────
        st.markdown("### ⏱️ Retrieval Timing")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BM25", f"{comp['bm25_time']:.4f}s")
        c2.metric("Dense", f"{comp['dense_time']:.4f}s")
        c3.metric("Hybrid Fusion", f"{comp['hybrid_fusion_time']:.4f}s")
        c4.metric("Reranking", f"{comp['rerank_time']:.4f}s")

        st.divider()

        # ── Overlap analysis ──────────────────────────────────────────
        st.markdown("### 🔗 Overlap Analysis")
        ov = comp["overlap"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BM25 Only", ov["bm25_only"])
        c2.metric("Dense Only", ov["dense_only"])
        c3.metric("Both Methods", ov["both"])
        c4.metric("Hybrid Total", ov["hybrid_total"])

        # Overlap chart
        fig = go.Figure(data=[
            go.Bar(
                name="BM25 Only",
                x=["Overlap"],
                y=[ov["bm25_only"]],
                marker_color="#FF6B6B",
            ),
            go.Bar(
                name="Shared",
                x=["Overlap"],
                y=[ov["both"]],
                marker_color="#4ECDC4",
            ),
            go.Bar(
                name="Dense Only",
                x=["Overlap"],
                y=[ov["dense_only"]],
                marker_color="#45B7D1",
            ),
        ])
        fig.update_layout(
            barmode="stack",
            height=250,
            margin=dict(t=10, b=10),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Results tables ────────────────────────────────────────────
        st.markdown("### 📊 Results by Method")

        tab_bm25, tab_dense, tab_hybrid, tab_reranked = st.tabs(
            ["🔤 BM25", "🧠 Dense", "🔀 Hybrid", "🏆 Reranked"]
        )

        with tab_bm25:
            df = _results_to_df(comp["bm25"], "bm25")
            if df.empty:
                st.info("No BM25 results.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_dense:
            df = _results_to_df(comp["dense"], "dense")
            if df.empty:
                st.info("No Dense results.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_hybrid:
            df = _results_to_df(comp["hybrid"], "hybrid")
            if df.empty:
                st.info("No Hybrid results.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_reranked:
            df = _results_to_df(comp["reranked"], "reranked")
            if df.empty:
                st.info("No reranked results.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Score comparison chart ────────────────────────────────────
        st.divider()
        st.markdown("### 📈 Score Distribution")

        fig2 = go.Figure()
        for label, results, color in [
            ("BM25", comp["bm25"], "#FF6B6B"),
            ("Dense", comp["dense"], "#45B7D1"),
            ("Hybrid (RRF)", comp["hybrid"], "#4ECDC4"),
        ]:
            if results:
                scores = [r["score"] for r in results]
                ranks = [r["rank"] for r in results]
                fig2.add_trace(go.Scatter(
                    x=ranks, y=scores,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color),
                ))

        fig2.update_layout(
            xaxis_title="Rank",
            yaxis_title="Score",
            height=350,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
