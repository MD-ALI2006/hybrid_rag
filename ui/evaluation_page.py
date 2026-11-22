"""
Evaluation Dashboard page – timing, retrieval metrics, and analysis charts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_evaluation_page():
    """Display evaluation metrics and charts."""

    st.markdown("## 📈 Evaluation Dashboard")

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.warning("Pipeline not initialized.")
        return

    sm = pipeline.session_metrics
    if sm.total_queries == 0:
        st.info(
            "No queries yet. Use the **Chat** or **Comparison** page to run some "
            "queries first, then come back here to see the analysis."
        )
        return

    st.success(f"📊 Analysing **{sm.total_queries}** query/queries from this session.")

    # ── Aggregate metrics ─────────────────────────────────────────────
    st.markdown("### ⏱️ Average Timing Breakdown")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BM25", f"{sm.avg('bm25_time'):.4f}s")
    c2.metric("Dense", f"{sm.avg('dense_time'):.4f}s")
    c3.metric("Fusion", f"{sm.avg('hybrid_time'):.4f}s")
    c4.metric("Reranking", f"{sm.avg('rerank_time'):.4f}s")
    c5.metric("LLM", f"{sm.avg('llm_time'):.4f}s")

    st.divider()

    # ── Per-query table ───────────────────────────────────────────────
    st.markdown("### 📋 Per-Query Metrics")
    rows = []
    for i, q in enumerate(sm.queries, 1):
        rows.append(
            {
                "Query #": i,
                "Question": q.query[:60] + ("..." if len(q.query) > 60 else ""),
                "BM25 (s)": round(q.bm25_time, 4),
                "Dense (s)": round(q.dense_time, 4),
                "Fusion (s)": round(q.hybrid_time, 4),
                "Rerank (s)": round(q.rerank_time, 4),
                "LLM (s)": round(q.llm_time, 4),
                "Total (s)": round(q.total_time, 4),
                "Chunks Retrieved": q.bm25_count + q.dense_count + q.hybrid_count,
                "Chunks Reranked": q.reranked_count,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Timing chart ──────────────────────────────────────────────────
    st.markdown("### 📊 Timing Breakdown per Query")
    fig = go.Figure()
    query_labels = [f"Q{i+1}" for i in range(len(sm.queries))]

    for field, name, color in [
        ("bm25_time", "BM25", "#FF6B6B"),
        ("dense_time", "Dense", "#45B7D1"),
        ("hybrid_time", "Fusion", "#4ECDC4"),
        ("rerank_time", "Reranking", "#96CEB4"),
        ("llm_time", "LLM", "#FFEAA7"),
    ]:
        fig.add_trace(go.Bar(
            name=name,
            x=query_labels,
            y=[getattr(q, field) for q in sm.queries],
            marker_color=color,
        ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="Query",
        yaxis_title="Time (seconds)",
        height=400,
        margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Chunk count chart ─────────────────────────────────────────────
    st.markdown("### 🧩 Chunks Retrieved vs Reranked")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Retrieved",
        x=query_labels,
        y=[q.bm25_count + q.dense_count + q.hybrid_count for q in sm.queries],
        marker_color="#45B7D1",
    ))
    fig2.add_trace(go.Bar(
        name="After Reranking",
        x=query_labels,
        y=[q.reranked_count for q in sm.queries],
        marker_color="#4ECDC4",
    ))
    fig2.update_layout(
        barmode="group",
        xaxis_title="Query",
        yaxis_title="Chunk Count",
        height=350,
        margin=dict(t=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Note on evaluation ────────────────────────────────────────────
    st.divider()
    st.markdown("### 📝 Evaluation Methodology Note")
    st.info(
        "**Exploratory Analysis:** This dashboard shows real timing and retrieval "
        "statistics from your session. Since no ground-truth relevance labels are "
        "provided, the metrics above are **exploratory retrieval analysis** — not "
        "formal IR benchmarks. Precision@K, Recall@K, MRR, and Hit Rate require "
        "annotated query-relevance data.\n\n"
        "To run formal evaluation, provide a CSV file with columns "
        "`query, relevant_chunk_ids` and the system can compute standard IR metrics."
    )
