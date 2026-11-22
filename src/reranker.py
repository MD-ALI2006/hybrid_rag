"""
Cross-encoder reranker using BAAI/bge-reranker-base.

Re-scores (query, chunk) pairs to improve ranking quality.
Falls back gracefully if the model cannot be loaded.
"""

from __future__ import annotations

from src.utils import Timer
from config.settings import RERANKER_MODEL


class Reranker:
    """Score query–chunk pairs with a cross-encoder and re-order."""

    def __init__(self):
        self._model = None
        self._load_error: str | None = None

    # ── Loading ───────────────────────────────────────────────────────
    def load_model(self) -> str | None:
        """Load cross-encoder model. Returns error string or None."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(RERANKER_MODEL)
            return None
        except Exception as exc:
            self._load_error = str(exc)
            return self._load_error

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ── Reranking ─────────────────────────────────────────────────────
    def rerank(
        self, query: str, results: list[dict], top_n: int = 5
    ) -> tuple[list[dict], float]:
        """
        Re-score and re-order *results* (retriever output dicts).

        Returns (reranked_results, elapsed_seconds).
        Each result dict gets an added ``rerank_score`` key.
        """
        if not self.is_ready or not results:
            return results[:top_n], 0.0

        timer = Timer()
        with timer:
            pairs = [(query, r["chunk"]["chunk_text"]) for r in results]
            scores = self._model.predict(pairs)

            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)

            reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
            # update ranks
            for i, r in enumerate(reranked, 1):
                r["rerank_rank"] = i

        return reranked[:top_n], timer.elapsed
