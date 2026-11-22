"""
Tests for document processing and chunking.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import fitz  # PyMuPDF

from src.document_processor import validate_pdf, extract_pages, has_extractable_text
from src.chunking import chunk_pages


# ── Helper: create a small test PDF ──────────────────────────────────
def _create_test_pdf(path: Path, pages: list[str]) -> Path:
    """Write a multi-page PDF with the given page texts."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return _create_test_pdf(
        tmp_path / "test.pdf",
        [
            "1. Introduction\nThis is the introduction section of the paper. "
            "We present a novel approach to hybrid retrieval.",
            "2. Methodology\nOur method combines BM25 and dense retrieval. "
            "We use reciprocal rank fusion for score combination.",
            "3. Results\nThe hybrid approach outperforms both baselines. "
            "Table 1 shows the precision and recall values.",
        ],
    )


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    doc.new_page()  # blank page
    doc.save(str(tmp_path / "empty.pdf"))
    doc.close()
    return tmp_path / "empty.pdf"


# ── Tests ─────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_pdf(self, sample_pdf: Path):
        ok, msg = validate_pdf(sample_pdf)
        assert ok is True
        assert msg == "OK"

    def test_missing_file(self, tmp_path: Path):
        ok, msg = validate_pdf(tmp_path / "nope.pdf")
        assert ok is False
        assert "not exist" in msg.lower()

    def test_wrong_extension(self, tmp_path: Path):
        txt = tmp_path / "file.txt"
        txt.write_text("hello")
        ok, msg = validate_pdf(txt)
        assert ok is False

    def test_empty_pdf_has_no_text(self, empty_pdf: Path):
        assert has_extractable_text(empty_pdf) is False


class TestExtraction:
    def test_page_count(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        assert len(pages) == 3

    def test_page_numbers_are_1_based(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        assert pages[0]["page_number"] == 1
        assert pages[2]["page_number"] == 3

    def test_document_name_preserved(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        assert pages[0]["document_name"] == "test.pdf"

    def test_text_not_empty(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        for p in pages:
            assert len(p["text"]) > 10


class TestChunking:
    def test_chunks_created(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 0

    def test_chunk_has_metadata(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=30)
        for c in chunks:
            assert "document_name" in c
            assert "page_number" in c
            assert "chunk_id" in c
            assert "chunk_text" in c
            assert c["page_number"] >= 1

    def test_chunk_size_respected(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        chunks = chunk_pages(pages, chunk_size=150, chunk_overlap=20)
        for c in chunks:
            # Allow some slack for word-boundary rounding
            assert len(c["chunk_text"]) <= 160

    def test_no_tiny_chunks(self, sample_pdf: Path):
        pages = extract_pages(sample_pdf)
        chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=30)
        for c in chunks:
            assert len(c["chunk_text"]) >= 30
