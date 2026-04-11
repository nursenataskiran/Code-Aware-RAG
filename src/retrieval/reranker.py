"""
Cross-encoder re-ranker for second-pass relevance scoring.

After hybrid retrieval returns candidates, the cross-encoder
jointly encodes (query, document) pairs to produce much more
accurate relevance scores than bi-encoder or BM25 alone.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from src.retrieval.retriever import RetrievalResult


# Lightweight cross-encoder: ~80ms for 20 pairs on CPU
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singleton to avoid reloading the model on every call
_reranker_instance: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_instance


def rerank(
    query: str,
    results: List["RetrievalResult"],
    top_k: int = 5,
) -> List["RetrievalResult"]:
    """
    Re-rank retrieval results using a cross-encoder.

    Args:
        query: The user query.
        results: Candidate retrieval results from hybrid search.
        top_k: Number of results to return after re-ranking.

    Returns:
        Top-k results sorted by cross-encoder relevance score (descending).
    """
    if not results:
        return []

    if len(results) <= 1:
        return results[:top_k]

    model = _get_reranker()

    # Build (query, document) pairs for the cross-encoder
    pairs = [(query, r.text) for r in results]

    # Score all pairs
    scores = model.predict(pairs)

    # Attach scores and sort
    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Update the distance field with the cross-encoder score (higher = better)
    reranked = []
    for result, score in scored_results[:top_k]:
        result.distance = float(-score)  # Negate so lower = better (consistent with Chroma)
        reranked.append(result)

    return reranked
