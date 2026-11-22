"""
Hybrid retriever: combines BM25 and Dense retrieval using
Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.utils import Timer
from config.settings import RRF_K


class HybridRetriever:
    """
    Fuse BM25 and dense retrieval results with Reciprocal Rank Fusion.

    RRF score for a document d:
        RRF(d) = Σ  1 / (k + rank_i(d))
    where the sum is over each retriever that returned d.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25 = bm25
        self.dense = dense

    @property
    def is_ready(self) -> bool:
        return self.bm25.is_ready and self.dense.is_ready

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int | None = None,
    ) -> tuple[list[dict], float, float, float]:
        """
        Returns (hybrid_results, bm25_time, dense_time, fusion_time).

        Each result dict: chunk, score (RRF), rank, method="hybrid",
        bm25_rank, dense_rank.
        """
        k = rrf_k or RRF_K

        bm25_results, bm25_time = self.bm25.retrieve(query, top_k=top_k)
        dense_results, dense_time = self.dense.retrieve(query, top_k=top_k)

        timer = Timer()
        with timer:
            # Build RRF scores
            scores: dict[str, float] = {}       # chunk_id -> score
            chunk_map: dict[str, dict] = {}     # chunk_id -> chunk
            bm25_ranks: dict[str, int] = {}
            dense_ranks: dict[str, int] = {}

            for r in bm25_results:
                cid = r["chunk"]["chunk_id"]
                bm25_ranks[cid] = r["rank"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
                chunk_map[cid] = r["chunk"]

            for r in dense_results:
                cid = r["chunk"]["chunk_id"]
                dense_ranks[cid] = r["rank"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
                chunk_map[cid] = r["chunk"]

            # Sort by RRF score descending
            sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)

            results = []
            for rank, cid in enumerate(sorted_ids[:top_k], 1):
                results.append(
                    {
                        "chunk": chunk_map[cid],
                        "score": scores[cid],
                        "rank": rank,
                        "method": "hybrid",
                        "bm25_rank": bm25_ranks.get(cid),
                        "dense_rank": dense_ranks.get(cid),
                    }
                )

        return results, bm25_time, dense_time, timer.elapsed
