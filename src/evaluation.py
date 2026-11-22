"""
Evaluation and metrics for retrieval analysis.

Tracks timing, overlap between methods, and optional IR metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryMetrics:
    """Timing and stats for a single query."""
    query: str = ""
    bm25_time: float = 0.0
    dense_time: float = 0.0
    hybrid_time: float = 0.0       # fusion time only
    rerank_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0
    bm25_count: int = 0
    dense_count: int = 0
    hybrid_count: int = 0
    reranked_count: int = 0


@dataclass
class SessionMetrics:
    """Accumulates metrics across the whole session."""
    queries: list[QueryMetrics] = field(default_factory=list)

    def add(self, m: QueryMetrics) -> None:
        self.queries.append(m)

    @property
    def total_queries(self) -> int:
        return len(self.queries)

    def avg(self, attr: str) -> float:
        if not self.queries:
            return 0.0
        return sum(getattr(q, attr) for q in self.queries) / len(self.queries)


def compute_overlap(
    bm25_results: list[dict],
    dense_results: list[dict],
    hybrid_results: list[dict],
) -> dict[str, Any]:
    """
    Compute chunk-id overlap between the three retrieval methods.

    Returns a dict with counts and sets (for display).
    """
    bm25_ids = {r["chunk"]["chunk_id"] for r in bm25_results}
    dense_ids = {r["chunk"]["chunk_id"] for r in dense_results}
    hybrid_ids = {r["chunk"]["chunk_id"] for r in hybrid_results}

    return {
        "bm25_only": len(bm25_ids - dense_ids),
        "dense_only": len(dense_ids - bm25_ids),
        "both": len(bm25_ids & dense_ids),
        "hybrid_total": len(hybrid_ids),
        "bm25_total": len(bm25_ids),
        "dense_total": len(dense_ids),
        "bm25_in_hybrid": len(bm25_ids & hybrid_ids),
        "dense_in_hybrid": len(dense_ids & hybrid_ids),
    }


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K – fraction of top-K that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K – fraction of relevant docs found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank for a single query."""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 if any relevant doc is in top-K, else 0."""
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0
