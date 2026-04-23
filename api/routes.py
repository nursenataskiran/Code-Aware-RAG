from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from api.rate_limiter import rate_limit_dependency
from api.schemas import ChatRequest, ChatResponse, HealthResponse, SourceItem
from src.generation.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ── Shared pipeline instance ──────────────────────────────────────────
# Created once at import time so models are only loaded once.
# If startup is too slow, move this into a lifespan event in app.py.
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Return the shared pipeline, initialising it on first call."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising RAG pipeline (first request)...")
        _pipeline = RAGPipeline(
            use_hybrid=True,
            use_reranker=False,        # Keep startup fast; enable if needed
            use_query_expansion=False, # Same — cheap to turn on later
        )
        logger.info("RAG pipeline ready.")
    return _pipeline


# ── Routers ───────────────────────────────────────────────────────────

health_router = APIRouter(tags=["Health"])
chat_router = APIRouter(
    prefix="/api/v1",
    tags=["Chat"],
    dependencies=[Depends(rate_limit_dependency)],  # Rate-limit all chat routes
)


# ── Health ────────────────────────────────────────────────────────────

@health_router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    """Returns 200 OK when the service is up."""
    return HealthResponse(status="ok")


# ── Chat ──────────────────────────────────────────────────────────────

@chat_router.post("/chat", response_model=ChatResponse, summary="Ask the RAG system")
async def chat(body: ChatRequest) -> ChatResponse:
    pipeline = _get_pipeline()

    try:
        result = pipeline.query(question=body.query)
    except Exception as exc:
        logger.exception("Pipeline error for query: %r", body.query)
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline failed to process your request.",
        ) from exc

    raw_results = result.get("results", [])
    answer: str = result.get("answer", "")

    # The pipeline emits this exact phrase when it cannot ground its answer.
    FALLBACK_PHRASE = "I don't know based on the provided repository context."
    is_fallback = not raw_results or answer.strip().startswith(FALLBACK_PHRASE)

    if is_fallback:
        # Suppress sources — returning them alongside a "no context" answer
        # is misleading; the retrieved chunks were not actually useful.
        sources: List[SourceItem] = []
        message = "No relevant repository context was found for this question."
    else:
        sources = _build_sources(raw_results)
        message = None

    return ChatResponse(answer=answer, sources=sources, message=message)


# ── Helpers ───────────────────────────────────────────────────────────

def _build_sources(results: list) -> List[SourceItem]:
    sources = []
    for r in results:
        meta = r.metadata if hasattr(r, "metadata") else {}
        sources.append(
            SourceItem(
                id=r.id,
                project=meta.get("project_name"),
                file=meta.get("file_name"),
                chunk_type=meta.get("chunk_type"),
                symbol=meta.get("symbol_name"),
                score=round(r.score, 4) if r.score is not None else None,
            )
        )
    return sources
