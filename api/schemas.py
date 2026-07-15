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


# ── GitHub Ingestion ──────────────────────────────────────────────────

class SkippedFile(BaseModel):
    """A single file that was skipped during ingestion, with a reason."""

    path: str
    reason: str


class GitHubIngestRequest(BaseModel):
    repo_url: str = Field(
        ...,
        description="Public GitHub repository URL.",
        examples=["https://github.com/owner/repo"],
    )


class GitHubIngestResponse(BaseModel):
    status: str = Field(
        ...,
        description="'success' or 'already_ingested'.",
    )
    project_name: str = Field(
        ...,
        description="Normalised local name (owner__repo).",
    )
    repo_url: str
    downloaded_files: List[str] = Field(
        default_factory=list,
        description="Relative paths of files written to data/raw/.",
    )
    skipped_files: List[SkippedFile] = Field(
        default_factory=list,
        description="Files that were not downloaded and the reason why.",
    )
    indexed_chunks: int = Field(
        default=0,
        description="Number of chunks added to the Chroma collection.",
    )
    message: str
