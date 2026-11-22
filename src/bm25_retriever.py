"""
BM25 keyword-based retriever using rank_bm25.
"""

from __future__ import annotations
import re

from rank_bm25 import BM25Okapi

from src.utils import Timer


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowering tokeniser."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Build and query a BM25 index over document chunks."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: list[dict] = []

    # ── Building ──────────────────────────────────────────────────────
    def build_index(self, chunks: list[dict]) -> None:
        """Tokenize chunk texts and build a BM25 index."""
        self._chunks = chunks
        corpus = [_tokenize(c["chunk_text"]) for c in chunks]
        self._index = BM25Okapi(corpus)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    # ── Querying ──────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[dict], float]:
        """
        Return (results, elapsed_seconds).

        Each result dict:
            chunk  – original chunk record
            score  – BM25 score
            rank   – 1-based
            method – "bm25"
        """
        if not self.is_ready:
            return [], 0.0

        timer = Timer()
        with timer:
            tokens = _tokenize(query)
            scores = self._index.get_scores(tokens)

            ranked = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )[:top_k]

            results = []
            for rank, (idx, score) in enumerate(ranked, 1):
                results.append(
                    {
                        "chunk": self._chunks[idx],
                        "score": float(score),
                        "rank": rank,
                        "method": "bm25",
                    }
                )

        return results, timer.elapsed
