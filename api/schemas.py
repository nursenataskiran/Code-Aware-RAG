from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask the RAG system.",
        examples=["How does the BM25Index class work?"],
    )


# ── Source representation ─────────────────────────────────────────────

class SourceItem(BaseModel):
    """Compact, serialisable representation of a single retrieved chunk."""

    id: str
    project: Optional[str] = None
    file: Optional[str] = None
    chunk_type: Optional[str] = None
    symbol: Optional[str] = None
    score: Optional[float] = None


# ── Response ──────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    # Populated when retrieval returns nothing useful
    message: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"


# ── Error ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
