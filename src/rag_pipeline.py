"""
RAG Pipeline – orchestrates the full retrieval → rerank → generate flow.

This is the central coordinator that the UI calls.
"""

from __future__ import annotations
import streamlit as st

from src.document_processor import extract_pages, validate_pdf, has_extractable_text
from src.chunking import chunk_pages
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.llm_service import LLMService
from src.citations import (
    extract_citations_from_answer,
    verify_citations,
    format_sources_section,
)
from src.evaluation import QueryMetrics, SessionMetrics, compute_overlap
from src.utils import Timer
from config.settings import DEFAULT_TOP_K, RERANK_TOP_N


class RAGPipeline:
    """End-to-end Hybrid RAG pipeline with comparison support."""

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = Reranker()
        self.llm = LLMService()
        self.chunks: list[dict] = []
        self.pages: list[dict] = []
        self.doc_names: list[str] = []
        self.session_metrics = SessionMetrics()
        self._models_loaded = False

    # ── Model loading ─────────────────────────────────────────────────
    def load_models(self) -> list[str]:
        """Load embedding, reranker, and LLM. Returns list of errors."""
        errors: list[str] = []

        err = self.dense.load_model()
        if err:
            errors.append(f"Embedding model: {err}")

        err = self.reranker.load_model()
        if err:
            errors.append(f"Reranker model: {err}")

        err = self.llm.initialize()
        if err:
            errors.append(f"LLM: {err}")

        self._models_loaded = True
        return errors

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    # ── Document processing ───────────────────────────────────────────
    def process_documents(self, file_paths: list) -> dict:
        """
        Validate, extract, chunk, and index a list of PDF paths.

        Returns a summary dict.
        """
        from pathlib import Path

        all_pages = []
        total_chars = 0
        warnings = []
        processed_docs = []

        for fp in file_paths:
            path = Path(fp)
            ok, msg = validate_pdf(path)
            if not ok:
                warnings.append(f"{path.name}: {msg}")
                continue

            if not has_extractable_text(path):
                warnings.append(
                    f"{path.name}: No extractable text found — may be a scanned PDF."
                )
                continue

            pages = extract_pages(path)
            if not pages:
                warnings.append(f"{path.name}: No text extracted.")
                continue

            all_pages.extend(pages)
            total_chars += sum(len(p["text"]) for p in pages)
            processed_docs.append(path.name)

        if not all_pages:
            return {
                "success": False,
                "warnings": warnings,
                "documents": 0,
                "pages": 0,
                "characters": 0,
                "chunks": 0,
            }

        self.pages = all_pages
        self.doc_names = list(set(processed_docs))

        # Chunk
        chunk_size = st.session_state.get("chunk_size", 800)
        chunk_overlap = st.session_state.get("chunk_overlap", 150)
        self.chunks = chunk_pages(all_pages, chunk_size, chunk_overlap)

        # Build indices
        self.bm25.build_index(self.chunks)
        self.dense.build_index(self.chunks)

        return {
            "success": True,
            "warnings": warnings,
            "documents": len(processed_docs),
            "pages": len(all_pages),
            "characters": total_chars,
            "chunks": len(self.chunks),
        }

    # ── Query ─────────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> dict:
        """
        Run the full RAG pipeline for *question*.

        Parameters
        ----------
        mode : "bm25" | "dense" | "hybrid"
        """
        tk = top_k or DEFAULT_TOP_K
        rn = rerank_top_n or RERANK_TOP_N
        metrics = QueryMetrics(query=question)

        total_timer = Timer()
        with total_timer:
            # 1. Retrieve
            if mode == "bm25":
                results, t = self.bm25.retrieve(question, tk)
                metrics.bm25_time = t
                metrics.bm25_count = len(results)
            elif mode == "dense":
                results, t = self.dense.retrieve(question, tk)
                metrics.dense_time = t
                metrics.dense_count = len(results)
            else:  # hybrid
                results, bt, dt, ft = self.hybrid.retrieve(question, tk)
                metrics.bm25_time = bt
                metrics.dense_time = dt
                metrics.hybrid_time = ft
                metrics.hybrid_count = len(results)

            # 2. Rerank
            if self.reranker.is_ready and results:
                reranked, rt = self.reranker.rerank(question, results, rn)
                metrics.rerank_time = rt
                metrics.reranked_count = len(reranked)
            else:
                reranked = results[:rn]
                metrics.reranked_count = len(reranked)

            # 3. Generate answer
            answer, lt = self.llm.generate_answer(question, reranked)
            metrics.llm_time = lt

        metrics.total_time = total_timer.elapsed
        self.session_metrics.add(metrics)

        # 4. Citations
        raw_citations = extract_citations_from_answer(answer)
        citations = verify_citations(raw_citations, reranked)
        sources = format_sources_section(reranked)

        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "retrieved": results,
            "reranked": reranked,
            "metrics": metrics,
            "mode": mode,
        }

    # ── Comparison (all three methods) ────────────────────────────────
    def compare_retrieval(self, question: str, top_k: int | None = None) -> dict:
        """Run all three retrieval modes and return comparison data."""
        tk = top_k or DEFAULT_TOP_K

        bm25_results, bm25_time = self.bm25.retrieve(question, tk)
        dense_results, dense_time = self.dense.retrieve(question, tk)
        hybrid_results, bt, dt, ft = self.hybrid.retrieve(question, tk)

        # Rerank hybrid results
        if self.reranker.is_ready and hybrid_results:
            reranked, rerank_time = self.reranker.rerank(question, hybrid_results)
        else:
            reranked = hybrid_results[:RERANK_TOP_N]
            rerank_time = 0.0

        overlap = compute_overlap(bm25_results, dense_results, hybrid_results)

        return {
            "bm25": bm25_results,
            "dense": dense_results,
            "hybrid": hybrid_results,
            "reranked": reranked,
            "bm25_time": bm25_time,
            "dense_time": dense_time,
            "hybrid_fusion_time": ft,
            "rerank_time": rerank_time,
            "overlap": overlap,
        }
