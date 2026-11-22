"""
Tests for citation extraction and verification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.citations import (
    extract_citations_from_answer,
    verify_citations,
    format_sources_section,
)


class TestExtractCitations:
    def test_basic_citation(self):
        answer = "The result is 42 [Source: paper.pdf, Page 5]."
        cits = extract_citations_from_answer(answer)
        assert len(cits) == 1
        assert cits[0]["document_name"] == "paper.pdf"
        assert cits[0]["page_number"] == 5

    def test_citation_with_section(self):
        answer = "Method uses X [Source: paper.pdf, Section: Methodology, Page 12]."
        cits = extract_citations_from_answer(answer)
        assert len(cits) == 1
        assert cits[0]["section"] == "Methodology"
        assert cits[0]["page_number"] == 12

    def test_multiple_citations(self):
        answer = (
            "Fact A [Source: a.pdf, Page 1]. "
            "Fact B [Source: b.pdf, Page 3]. "
            "Also [Source: a.pdf, Page 1]."  # duplicate
        )
        cits = extract_citations_from_answer(answer)
        assert len(cits) == 2  # deduplicated

    def test_no_citations(self):
        answer = "There are no citations here."
        cits = extract_citations_from_answer(answer)
        assert cits == []


class TestVerifyCitations:
    def test_verified_citation(self):
        citations = [{"document_name": "paper.pdf", "page_number": 3}]
        chunks = [
            {
                "chunk": {
                    "document_name": "paper.pdf",
                    "page_number": 3,
                    "page_numbers": [3],
                    "chunk_id": "x",
                    "section_title": "",
                    "chunk_text": "text",
                },
                "score": 0.9,
            }
        ]
        result = verify_citations(citations, chunks)
        assert result[0]["verified"] is True

    def test_unverified_citation(self):
        citations = [{"document_name": "paper.pdf", "page_number": 99}]
        chunks = [
            {
                "chunk": {
                    "document_name": "paper.pdf",
                    "page_number": 3,
                    "page_numbers": [3],
                    "chunk_id": "x",
                    "section_title": "",
                    "chunk_text": "text",
                },
                "score": 0.9,
            }
        ]
        result = verify_citations(citations, chunks)
        assert result[0]["verified"] is False


class TestFormatSources:
    def test_deduplication(self):
        chunks = [
            {
                "chunk": {
                    "document_name": "a.pdf",
                    "page_number": 1,
                    "section_title": "Intro",
                    "chunk_text": "Hello world",
                    "chunk_id": "c1",
                },
                "score": 0.9,
            },
            {
                "chunk": {
                    "document_name": "a.pdf",
                    "page_number": 1,
                    "section_title": "Intro",
                    "chunk_text": "Duplicate page",
                    "chunk_id": "c2",
                },
                "score": 0.8,
            },
        ]
        sources = format_sources_section(chunks)
        assert len(sources) == 1  # same doc + page → deduplicated
