"""
Shared utility functions used across the Hybrid RAG system.
"""

import time
import hashlib
from pathlib import Path


def generate_chunk_id(doc_name: str, page: int, index: int) -> str:
    """Create a deterministic unique ID for a text chunk."""
    raw = f"{doc_name}::page{page}::chunk{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class Timer:
    """Simple context-manager timer for measuring pipeline stages."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def truncate_text(text: str, max_len: int = 300) -> str:
    """Truncate text for UI display, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + " …"
