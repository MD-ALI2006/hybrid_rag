"""
Centralized configuration for the Hybrid RAG system.
All tuneable parameters live here so they are easy to find and change.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = _project_root / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = _project_root / "vector_db"

for _d in (UPLOAD_DIR, PROCESSED_DIR, VECTOR_DB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Gemini ───────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-3.1-flash-lite"          # active, highly responsive Gemini model

# ── Embedding model ─────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── Reranker model ──────────────────────────────────────────────────────
RERANKER_MODEL: str = "BAAI/bge-reranker-base"

# ── Chunking defaults ───────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: int = 800       # characters (~200 tokens)
DEFAULT_CHUNK_OVERLAP: int = 150    # characters

# ── Retrieval defaults ──────────────────────────────────────────────────
DEFAULT_TOP_K: int = 10             # candidates from each retriever
RERANK_TOP_N: int = 5              # final chunks after reranking
RRF_K: int = 60                    # RRF constant

# ── Upload limits ───────────────────────────────────────────────────────
MAX_FILE_SIZE_MB: int = 50
ALLOWED_EXTENSIONS: list[str] = [".pdf"]

# ── ChromaDB ────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "hybrid_rag_docs"
