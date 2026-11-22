"""
Hybrid RAG – Main Streamlit Application

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

# ── Page configuration (must be the FIRST Streamlit call) ─────────────
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a professional look ────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    /* Hide the default streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Initialize pipeline in session state (runs once) ─────────────────
def _init_pipeline():
    """Load models once and cache the pipeline in session state."""
    if "pipeline" not in st.session_state:
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        st.session_state["pipeline"] = pipeline
        st.session_state["init_errors"] = []
        st.session_state["models_loading"] = True

    if st.session_state.get("models_loading", False):
        with st.spinner("🔄 Loading AI models (embedding, reranker, LLM)... This may take a minute on first run."):
            pipeline = st.session_state["pipeline"]
            errors = pipeline.load_models()
            st.session_state["init_errors"] = errors
            st.session_state["models_loading"] = False

        if errors:
            for e in errors:
                st.warning(f"⚠️ {e}")


_init_pipeline()


# ── Sidebar navigation ───────────────────────────────────────────────
st.sidebar.markdown('<p class="main-header">🔬 Hybrid RAG</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">AI Document Assistant</p>', unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📄 Upload & Process",
        "💬 Chat",
        "🔍 Retrieval Comparison",
        "📈 Evaluation",
    ],
    label_visibility="collapsed",
)

# ── Status indicators in sidebar ─────────────────────────────────────
st.sidebar.divider()
pipeline = st.session_state.get("pipeline")
if pipeline:
    # If LLM wasn't ready previously (e.g. user added API key after starting the app), attempt re-check
    if not pipeline.llm.is_ready:
        pipeline.llm.initialize()

    st.sidebar.markdown("**System Status**")
    st.sidebar.markdown(
        f"- Embedding: {'✅' if pipeline.dense._model else '❌'}\n"
        f"- Reranker: {'✅' if pipeline.reranker.is_ready else '❌'}\n"
        f"- LLM: {'✅' if pipeline.llm.is_ready else '❌'}\n"
        f"- Documents: {len(pipeline.doc_names)}\n"
        f"- Chunks: {len(pipeline.chunks)}"
    )

# ── Route to the selected page ───────────────────────────────────────
if page == "📊 Dashboard":
    from ui.dashboard import render_dashboard
    render_dashboard()
elif page == "📄 Upload & Process":
    from ui.upload_page import render_upload_page
    render_upload_page()
elif page == "💬 Chat":
    from ui.chat_page import render_chat_page
    render_chat_page()
elif page == "🔍 Retrieval Comparison":
    from ui.comparison_page import render_comparison_page
    render_comparison_page()
elif page == "📈 Evaluation":
    from ui.evaluation_page import render_evaluation_page
    render_evaluation_page()
