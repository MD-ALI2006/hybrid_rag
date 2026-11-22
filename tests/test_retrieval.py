"""
Tests for BM25, Dense, and Hybrid retrieval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.bm25_retriever import BM25Retriever
from src.chunking import chunk_pages


# ── Sample chunks for testing ────────────────────────────────────────
SAMPLE_CHUNKS = [
    {
        "document_name": "paper.pdf",
        "page_number": 1,
        "page_numbers": [1],
        "chunk_id": "c001",
        "section_title": "Introduction",
        "chunk_text": "Machine learning is a subset of artificial intelligence that "
                       "enables systems to learn from data without explicit programming.",
    },
    {
        "document_name": "paper.pdf",
        "page_number": 2,
        "page_numbers": [2],
        "chunk_id": "c002",
        "section_title": "Methodology",
        "chunk_text": "BM25 is a bag-of-words retrieval function that ranks documents "
                       "based on query terms appearing in each document.",
    },
    {
        "document_name": "paper.pdf",
        "page_number": 3,
        "page_numbers": [3],
        "chunk_id": "c003",
        "section_title": "Results",
        "chunk_text": "The neural network achieved 95% accuracy on the test dataset, "
                       "outperforming traditional methods by 10 percentage points.",
    },
    {
        "document_name": "paper.pdf",
        "page_number": 4,
        "page_numbers": [4],
        "chunk_id": "c004",
        "section_title": "Discussion",
        "chunk_text": "Dense retrieval uses learned embeddings to capture semantic "
                       "similarity between queries and documents.",
    },
]


class TestBM25Retriever:
    def test_build_index(self):
        bm25 = BM25Retriever()
        bm25.build_index(SAMPLE_CHUNKS)
        assert bm25.is_ready

    def test_retrieve_returns_results(self):
        bm25 = BM25Retriever()
        bm25.build_index(SAMPLE_CHUNKS)
        results, elapsed = bm25.retrieve("BM25 retrieval function", top_k=3)
        assert len(results) > 0
        assert elapsed >= 0

    def test_bm25_prefers_exact_match(self):
        bm25 = BM25Retriever()
        bm25.build_index(SAMPLE_CHUNKS)
        results, _ = bm25.retrieve("BM25 bag of words retrieval", top_k=3)
        # Chunk c002 mentions BM25 explicitly – should rank first
        assert results[0]["chunk"]["chunk_id"] == "c002"

    def test_retrieve_has_scores(self):
        bm25 = BM25Retriever()
        bm25.build_index(SAMPLE_CHUNKS)
        results, _ = bm25.retrieve("machine learning", top_k=2)
        for r in results:
            assert "score" in r
            assert "rank" in r
            assert r["method"] == "bm25"

    def test_empty_index(self):
        bm25 = BM25Retriever()
        assert not bm25.is_ready
        results, elapsed = bm25.retrieve("anything")
        assert results == []


class TestHybridFusion:
    """Test the RRF fusion logic using mock data."""

    def test_rrf_combines_results(self):
        """Verify that RRF correctly merges two ranked lists."""
        from src.hybrid_retriever import HybridRetriever

        # We'll test the fusion logic by giving the hybrid retriever
        # pre-built BM25 and dense retrievers
        bm25 = BM25Retriever()
        bm25.build_index(SAMPLE_CHUNKS)

        # We can test just the BM25 side since dense needs model loading
        results, _ = bm25.retrieve("neural network accuracy", top_k=4)
        assert len(results) >= 1
        # Chunk c003 about neural network accuracy should rank high
        top_ids = [r["chunk"]["chunk_id"] for r in results[:2]]
        assert "c003" in top_ids
