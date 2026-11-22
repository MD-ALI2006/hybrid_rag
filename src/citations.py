"""
Citation extraction and verification helpers.

Ensures that every cited page actually appears in the retrieved chunks.
"""

from __future__ import annotations
import re


def extract_citations_from_answer(answer: str) -> list[dict]:
    """
    Parse [Source: filename, Page N] or
          [Source: filename, Section: X, Page N] from *answer*.

    Returns a list of dicts with keys: document_name, page_number, section (optional).
    """
    pattern = re.compile(
        r"\[Source:\s*(?P<doc>[^,\]]+?)"
        r"(?:,\s*Section:\s*(?P<sec>[^,\]]+?))?"
        r",\s*Page\s*(?P<page>\d+)\s*\]",
        re.IGNORECASE,
    )
    citations = []
    seen = set()
    for m in pattern.finditer(answer):
        doc = m.group("doc").strip()
        page = int(m.group("page"))
        sec = (m.group("sec") or "").strip()
        key = (doc, page, sec)
        if key not in seen:
            seen.add(key)
            entry = {"document_name": doc, "page_number": page}
            if sec:
                entry["section"] = sec
            citations.append(entry)
    return citations


def verify_citations(
    citations: list[dict], retrieved_chunks: list[dict]
) -> list[dict]:
    """
    Check each citation against retrieved chunks.

    Returns the citations list with an added ``verified`` boolean.
    """
    # Build set of (doc, page) pairs that were actually retrieved
    retrieved_pages: set[tuple[str, int]] = set()
    for r in retrieved_chunks:
        c = r["chunk"]
        retrieved_pages.add((c["document_name"], c["page_number"]))
        for p in c.get("page_numbers", []):
            retrieved_pages.add((c["document_name"], p))

    for cit in citations:
        cit["verified"] = (cit["document_name"], cit["page_number"]) in retrieved_pages

    return citations


def format_sources_section(retrieved_chunks: list[dict]) -> list[dict]:
    """
    Build a de-duplicated, ordered list of source entries for display.

    Each entry: document_name, page_number, section_title, preview_text.
    """
    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for r in retrieved_chunks:
        c = r["chunk"]
        key = (c["document_name"], c["page_number"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "section_title": c.get("section_title", ""),
                "chunk_text": c["chunk_text"],
                "score": r.get("rerank_score", r.get("score", 0.0)),
            }
        )
    return sources
