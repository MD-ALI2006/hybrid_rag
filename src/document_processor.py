"""
PDF document loading and text extraction using PyMuPDF.

Responsibilities
- Validate uploaded PDFs (size, extension, readability).
- Extract text page-by-page while preserving page numbers.
- Detect section headings heuristically.
- Return structured page records.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF


class PageRecord(TypedDict):
    document_name: str
    page_number: int          # 1-based for user display
    text: str
    section_title: str


# ── Heading detection heuristic ────────────────────────────────────────
_HEADING_RE = re.compile(
    r"^(?:\d+[\.\)]\s*)?[A-Z][A-Za-z0-9 :&,\-]{2,80}$"
)


def _detect_heading(line: str) -> str | None:
    """Return *line* if it looks like a section heading, else None."""
    stripped = line.strip()
    if not stripped:
        return None
    if _HEADING_RE.match(stripped) and len(stripped.split()) <= 12:
        return stripped
    return None


# ── Public API ─────────────────────────────────────────────────────────

def validate_pdf(path: Path, max_mb: int = 50) -> tuple[bool, str]:
    """Return (ok, message) after basic checks on *path*."""
    if not path.exists():
        return False, "File does not exist."
    if path.suffix.lower() != ".pdf":
        return False, "Only PDF files are accepted."
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"File is {size_mb:.1f} MB – exceeds {max_mb} MB limit."
    try:
        doc = fitz.open(str(path))
        if doc.page_count == 0:
            doc.close()
            return False, "PDF has zero pages."
        doc.close()
    except Exception as exc:
        return False, f"Cannot open PDF: {exc}"
    return True, "OK"


def extract_pages(path: Path) -> list[PageRecord]:
    """Extract text from every page, returning structured records."""
    doc = fitz.open(str(path))
    doc_name = path.name
    records: list[PageRecord] = []
    current_heading = "General"

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        raw_text = page.get_text("text") or ""
        # clean whitespace but keep paragraph separation
        text = re.sub(r"[ \t]+", " ", raw_text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text:
            continue  # skip blank pages (scanned / image-only)

        # try to pick up headings from the first few lines
        for line in text.split("\n")[:6]:
            h = _detect_heading(line)
            if h:
                current_heading = h
                break

        records.append(
            PageRecord(
                document_name=doc_name,
                page_number=page_idx + 1,  # 1-based
                text=text,
                section_title=current_heading,
            )
        )

    doc.close()
    return records


def has_extractable_text(path: Path) -> bool:
    """Quick check: does the PDF have any selectable text?"""
    doc = fitz.open(str(path))
    for i in range(min(doc.page_count, 5)):
        if (doc.load_page(i).get_text("text") or "").strip():
            doc.close()
            return True
    doc.close()
    return False
