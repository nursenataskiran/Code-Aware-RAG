from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HF_TOKEN,
)

# Ensure HF_TOKEN is available for model downloads at query time
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


@dataclass
class RetrievalResult:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None        # vector distance (lower = better)
    bm25_score: Optional[float] = None      # BM25 score (higher = better)
    reranker_score: Optional[float] = None   # cross-encoder score (higher = better)

    @property
    def score(self) -> Optional[float]:
        """
        Best available relevance score, normalized to higher = better.

        Priority: reranker_score > distance-derived > bm25_score.
        """
        if self.reranker_score is not None:
            return self.reranker_score
        if self.distance is not None:
            return 1 / (1 + self.distance)
        return self.bm25_score


class ChromaRetriever:
    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.client = chromadb.PersistentClient(path=str(chroma_path))

        # Use the same embedding function that was used to build the index
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
            )
        except Exception as exc:
            raise ValueError(
                f"Chroma collection '{collection_name}' was not found at '{chroma_path}'. "
                "Rebuild the vector store with the current chunks schema before querying."
            ) from exc

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        chunk_types: Optional[List[str]] = None,
        max_per_file: Optional[int] = None,
    ) -> List[RetrievalResult]:
        requested_k = k
        query_k = k
        if max_per_file is not None and max_per_file > 0:
            query_k = max(k * 3, k + 5)

        query_kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": query_k,
        }

        filters = {}

        if where is not None:
            filters.update(where)

        if chunk_types:
            if len(chunk_types) == 1:
                filters["chunk_type"] = chunk_types[0]
            else:
                filters["chunk_type"] = {"$in": chunk_types}

        if filters:
            query_kwargs["where"] = filters

        results = self.collection.query(**query_kwargs)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else []

        structured_results: List[RetrievalResult] = []

        for idx, chunk_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else None
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            text = documents[idx] if idx < len(documents) else ""

            # Use raw_text from metadata if available (the original chunk text)
            raw_text = metadata.pop("raw_text", None) or text

            structured_results.append(
                RetrievalResult(
                    id=chunk_id,
                    text=raw_text,
                    metadata=metadata,
                    distance=distance,
                )
            )

        if max_per_file is None or max_per_file <= 0:
            return structured_results[:requested_k]

        diversified_results: List[RetrievalResult] = []
        file_counts: Dict[str, int] = {}

        for result in structured_results:
            file_key = result.metadata.get("file_path") or result.metadata.get("file_name") or result.id
            current_count = file_counts.get(file_key, 0)
            if current_count >= max_per_file:
                continue

            diversified_results.append(result)
            file_counts[file_key] = current_count + 1

            if len(diversified_results) >= requested_k:
                break

        return diversified_results

    def retrieve_by_project(
        self,
        query: str,
        project_name: str,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Convenience wrapper for project-specific retrieval.
        """
        return self.retrieve(
            query=query,
            k=k,
            where={"project_name": project_name},
        )


class HybridRetriever:
    """
    Combines vector search (ChromaDB) + lexical search (BM25)
    using Reciprocal Rank Fusion, then optionally re-ranks with
    a cross-encoder.

    This is the recommended retriever for production use.
    """

    def __init__(
        self,
        use_reranker: bool = True,
        use_query_expansion: bool = True,
    ) -> None:
        self.chroma_retriever = ChromaRetriever()
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion

        # Lazy-load BM25 index
        from src.retrieval.bm25_index import BM25Index
        self.bm25_index = BM25Index()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        chunk_types: Optional[List[str]] = None,
        max_per_file: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval pipeline:
        1. Query expansion (optional)
        2. Vector search (ChromaDB)
        3. Lexical search (BM25)
        4. Reciprocal Rank Fusion
        5. Cross-encoder re-ranking (optional)
        6. Diversification (max_per_file)
        """
        # ── Step 1: Query expansion ──────────────────────────────────
        queries = [query]
        detected_project = None

        if self.use_query_expansion:
            from src.retrieval.query_expander import expand_query, detect_project
            queries = expand_query(query)
            detected_project = detect_project(query)

            # Auto-filter by project if detected and no explicit filter given
            if detected_project and where is None:
                where = {"project_name": detected_project}

        # ── Step 2: Vector search (run all query variants) ───────────
        candidate_k = max(k * 4, 20)  # Fetch more candidates for fusion
        all_vector_results: Dict[str, RetrievalResult] = {}
        vector_rankings: List[List[str]] = []

        for q in queries:
            results = self.chroma_retriever.retrieve(
                query=q,
                k=candidate_k,
                where=where,
                chunk_types=chunk_types,
            )
            ranking = []
            for r in results:
                if r.id not in all_vector_results:
                    all_vector_results[r.id] = r
                ranking.append(r.id)
            vector_rankings.append(ranking)

        # ── Step 3: BM25 search ──────────────────────────────────────
        bm25_rankings: List[List[str]] = []

        for q in queries:
            bm25_results = self.bm25_index.search(q, k=candidate_k)
            ranking = []
            for r in bm25_results:
                if r.id not in all_vector_results:
                    all_vector_results[r.id] = r
                ranking.append(r.id)
            bm25_rankings.append(ranking)

        # ── Step 4: Reciprocal Rank Fusion ───────────────────────────
        all_rankings = vector_rankings + bm25_rankings
        fused_scores = self._reciprocal_rank_fusion(all_rankings)

        # Sort by fused score (higher = better)
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        # Apply chunk_type filter on BM25-only results (they bypass Chroma's filter)
        if chunk_types:
            chunk_type_set = set(chunk_types)
            sorted_ids = [
                cid for cid in sorted_ids
                if all_vector_results[cid].metadata.get("chunk_type") in chunk_type_set
            ]

        # Apply project filter on BM25-only results
        if where and "project_name" in where:
            target_project = where["project_name"]
            sorted_ids = [
                cid for cid in sorted_ids
                if all_vector_results[cid].metadata.get("project_name") == target_project
            ]

        fused_results = [all_vector_results[cid] for cid in sorted_ids[:candidate_k]]

        # ── Step 5: Cross-encoder re-ranking ─────────────────────────
        if self.use_reranker and fused_results:
            from src.retrieval.reranker import rerank
            rerank_k = max(k * 2, 10)  # Re-rank more than we need
            fused_results = rerank(query, fused_results[:rerank_k], top_k=candidate_k)

        # ── Step 6: Diversification ──────────────────────────────────
        if max_per_file is not None and max_per_file > 0:
            diversified: List[RetrievalResult] = []
            file_counts: Dict[str, int] = {}

            for result in fused_results:
                file_key = result.metadata.get("file_path") or result.metadata.get("file_name") or result.id
                current_count = file_counts.get(file_key, 0)
                if current_count >= max_per_file:
                    continue
                diversified.append(result)
                file_counts[file_key] = current_count + 1
                if len(diversified) >= k:
                    break

            return diversified

        return fused_results[:k]

    @staticmethod
    def _reciprocal_rank_fusion(
        rankings: List[List[str]],
        k: int = 60,
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion (RRF).

        For each document d appearing in any ranking:
          score(d) = Σ 1 / (k + rank_i(d))

        k=60 is the standard constant from the original RRF paper.
        """
        scores: Dict[str, float] = {}

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank)

        return scores
