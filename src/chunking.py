"""
Text chunking with configurable size, overlap, and metadata preservation.

Each chunk keeps:
  - document_name
  - page_number(s)
  - section_title
  - chunk_id
  - chunk_text
"""

from __future__ import annotations
from typing import TypedDict
from src.utils import generate_chunk_id


class ChunkRecord(TypedDict):
    document_name: str
    page_number: int
    page_numbers: list[int]
    chunk_id: str
    section_title: str
    chunk_text: str


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[ChunkRecord]:
    """
    Split page records into overlapping text chunks.

    Parameters
    ----------
    pages : list[dict]
        Output of ``extract_pages`` – each dict has *text*, *page_number*,
        *document_name*, *section_title*.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    """
    chunks: list[ChunkRecord] = []
    idx = 0

    for page in pages:
        text = page["text"]
        doc_name = page["document_name"]
        page_num = page["page_number"]
        section = page["section_title"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Avoid breaking mid-word if possible
            if end < len(text):
                last_space = chunk_text.rfind(" ")
                if last_space > chunk_size // 2:
                    end = start + last_space
                    chunk_text = text[start:end]

            chunk_text = chunk_text.strip()
            if len(chunk_text) < 30:
                # skip very tiny leftover fragments
                break

            cid = generate_chunk_id(doc_name, page_num, idx)
            chunks.append(
                ChunkRecord(
                    document_name=doc_name,
                    page_number=page_num,
                    page_numbers=[page_num],
                    chunk_id=cid,
                    section_title=section,
                    chunk_text=chunk_text,
                )
            )
            idx += 1
            start = end - chunk_overlap
            if start < 0:
                start = 0
            # prevent infinite loop
            if end >= len(text):
                break

    return chunks
