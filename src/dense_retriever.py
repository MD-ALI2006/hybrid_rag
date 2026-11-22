"""
Dense (semantic) vector retriever using Sentence Transformers + ChromaDB.
"""

from __future__ import annotations
from typing import Any

import chromadb

from src.utils import Timer
from config.settings import EMBEDDING_MODEL, VECTOR_DB_DIR, CHROMA_COLLECTION_NAME


class DenseRetriever:
    """Embed chunks, store in ChromaDB, and do semantic retrieval."""

    def __init__(self):
        self._model = None
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._chunks: list[dict] = []
        self._model_load_error: str | None = None

    # ── Model loading ─────────────────────────────────────────────────
    def load_model(self) -> str | None:
        """Load the sentence-transformer. Returns error string or None."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            return None
        except Exception as exc:
            self._model_load_error = str(exc)
            return self._model_load_error

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._collection is not None

    @property
    def model_load_error(self) -> str | None:
        return self._model_load_error

    # ── Indexing ──────────────────────────────────────────────────────
    def build_index(self, chunks: list[dict]) -> None:
        """Embed all chunks and store them in ChromaDB."""
        self._chunks = chunks

        # Setup ChromaDB persistent client
        self._client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
        )

        # Delete existing collection if it exists (re-index)
        try:
            self._client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Embed texts in batches
        texts = [c["chunk_text"] for c in chunks]
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "section_title": c["section_title"],
            }
            for c in chunks
        ]

        # ChromaDB add (batch ≤ 5000 at a time)
        batch = 5000
        for i in range(0, len(ids), batch):
            self._collection.add(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                embeddings=embeddings[i : i + batch].tolist(),
                metadatas=metadatas[i : i + batch],
            )

    # ── Querying ──────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[dict], float]:
        """
        Return (results, elapsed_seconds).
        Each result dict has: chunk, score, rank, method.
        """
        if not self.is_ready:
            return [], 0.0

        timer = Timer()
        with timer:
            query_emb = self._model.encode([query], convert_to_numpy=True).tolist()
            res = self._collection.query(
                query_embeddings=query_emb,
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            results = []
            if res and res["ids"] and res["ids"][0]:
                for rank, (cid, doc, meta, dist) in enumerate(
                    zip(
                        res["ids"][0],
                        res["documents"][0],
                        res["metadatas"][0],
                        res["distances"][0],
                    ),
                    1,
                ):
                    # ChromaDB cosine distance -> similarity
                    sim = 1.0 - dist
                    # find original chunk record
                    original = self._find_chunk(cid, doc, meta)
                    results.append(
                        {
                            "chunk": original,
                            "score": float(sim),
                            "rank": rank,
                            "method": "dense",
                        }
                    )

        return results, timer.elapsed

    def _find_chunk(self, cid: str, text: str, meta: dict) -> dict:
        """Locate the original chunk dict by id or rebuild from metadata."""
        for c in self._chunks:
            if c["chunk_id"] == cid:
                return c
        # fallback: build from retrieved data
        return {
            "document_name": meta.get("document_name", "unknown"),
            "page_number": meta.get("page_number", 0),
            "page_numbers": [meta.get("page_number", 0)],
            "chunk_id": cid,
            "section_title": meta.get("section_title", ""),
            "chunk_text": text,
        }
