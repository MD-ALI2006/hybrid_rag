"""
Gemini LLM service for answer generation.

Wraps the Google Gen AI SDK, handles errors, and enforces that answers
are grounded in retrieved context only.
"""

from __future__ import annotations
from typing import Any

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from src.utils import Timer


# ── System prompt sent with every request ────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise document assistant. You answer user questions based ONLY on
the provided CONTEXT extracted from uploaded documents.

RULES:
1. Use ONLY the information in the CONTEXT below.
2. If the context does not contain enough information, say so clearly.
3. Do NOT invent facts, citations, or page numbers.
4. For every claim you make, cite the source using exactly this format:
   [Source: <document_name>, Page <page_number>]
   If a section title is available, include it:
   [Source: <document_name>, Section: <section>, Page <page_number>]
5. If multiple sources support a claim, cite all of them.
6. Structure your answer with clear paragraphs or bullet points when appropriate.
7. Distinguish facts from your interpretation.
8. If the question is unrelated to the documents, politely decline.
"""


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: list[str] = []
    for i, r in enumerate(chunks, 1):
        c = r["chunk"]
        section = c.get("section_title", "")
        sec_str = f" | Section: {section}" if section else ""
        header = (
            f"[Context {i}] Document: {c['document_name']}"
            f" | Page: {c['page_number']}{sec_str}"
        )
        parts.append(f"{header}\n{c['chunk_text']}")
    return "\n\n---\n\n".join(parts)


class LLMService:
    """Generate answers with Google Gemini."""

    def __init__(self):
        self._client = None
        self._init_error: str | None = None

    # ── Initialization ────────────────────────────────────────────────
    def initialize(self, api_key: str | None = None) -> str | None:
        """Configure the Gemini client. Returns error string or None."""
        # Always reload from environment or .env file to pick up runtime updates
        import os
        from pathlib import Path
        from dotenv import load_dotenv

        _project_root = Path(__file__).resolve().parent.parent
        load_dotenv(_project_root / ".env", override=True)
        env_key = os.getenv("GEMINI_API_KEY", "")

        key = api_key or env_key or GEMINI_API_KEY
        if not key or key == "your_api_key_here":
            self._client = None
            self._init_error = (
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to your .env file."
            )
            return self._init_error

        try:
            from google import genai
            self._client = genai.Client(api_key=key)
            self._init_error = None
            return None
        except Exception as exc:
            self._client = None
            self._init_error = f"Failed to initialize Gemini client: {exc}"
            return self._init_error

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    # ── Generation ────────────────────────────────────────────────────
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict],
        model: str | None = None,
    ) -> tuple[str, float]:
        """
        Generate an answer grounded in *retrieved_chunks*.

        Returns (answer_text, elapsed_seconds).
        """
        if not self.is_ready:
            return f"⚠️ LLM not initialized: {self._init_error}", 0.0

        if not retrieved_chunks:
            return (
                "I could not find sufficient information in the uploaded "
                "documents to answer this question reliably."
            ), 0.0

        context_block = _build_context_block(retrieved_chunks)
        user_message = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION:\n{query}\n\n"
            "Answer the question using ONLY the context above. "
            "Cite sources with [Source: filename, Page X] format."
        )

        timer = Timer()
        with timer:
            try:
                from google.genai import types
                response = self._client.models.generate_content(
                    model=model or GEMINI_MODEL,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        temperature=0.2,
                        max_output_tokens=2048,
                    ),
                )
                answer = response.text or "No response generated."
            except Exception as exc:
                error_str = str(exc).lower()
                if "quota" in error_str or "429" in error_str:
                    answer = "⚠️ API quota exceeded. Please wait and try again."
                elif "api key" in error_str or "401" in error_str or "403" in error_str:
                    answer = "⚠️ Invalid or unauthorized API key. Check your .env file."
                elif "not found" in error_str or "404" in error_str:
                    answer = f"⚠️ Model '{model or GEMINI_MODEL}' not found. Update GEMINI_MODEL in config/settings.py."
                else:
                    answer = f"⚠️ Gemini API error: {exc}"

        return answer, timer.elapsed
