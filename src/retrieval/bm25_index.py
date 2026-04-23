"""
BM25 lexical index for hybrid search.

Complements vector search by catching exact keyword matches
(class names, function names, library calls) that embedding
models often compress into generic representations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH
from src.retrieval.retriever import RetrievalResult


def _code_aware_tokenize(text: str) -> List[str]:
    """
    Tokenizer designed for code + natural language.
    
    - Splits on whitespace and punctuation
    - Preserves snake_case and CamelCase identifiers
    - Expands CamelCase into sub-tokens (e.g. LSTMDataPreparer → lstm, data, preparer)
    - Lowercases everything
    """
    # Split on whitespace and common delimiters, but keep underscored identifiers
    raw_tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[0-9]+', text)

    tokens = []
    for token in raw_tokens:
        lower = token.lower()
        tokens.append(lower)

        # Expand CamelCase: LSTMDataPreparer → lstm, data, preparer
        parts = re.findall(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+', token)
        if len(parts) > 1:
            for part in parts:
                p = part.lower()
                if p != lower and len(p) > 1:
                    tokens.append(p)

        # Expand snake_case: clean_data → clean, data
        if "_" in token:
            for part in token.split("_"):
                p = part.lower()
                if p and p != lower and len(p) > 1:
                    tokens.append(p)

    return tokens


class BM25Index:
    """
    Lightweight BM25 lexical search index built from chunks.json.
    """

    def __init__(self, chunks_path: str | Path = CHUNKS_PATH):
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.bm25: Optional[BM25Okapi] = None

        self._build_index(chunks_path)

    def _build_index(self, chunks_path: str | Path) -> None:
        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        corpus = []
        for i, chunk in enumerate(self.chunks):
            # Use embedding_text if available (includes description), else raw text
            text = chunk.get("embedding_text", chunk.get("text", ""))
            tokens = _code_aware_tokenize(text)
            corpus.append(tokens)
            self.chunk_id_to_idx[chunk["id"]] = i

        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Search the BM25 index.

        Returns:
            List of RetrievalResult objects with bm25_score set,
            sorted by score descending.
        """
        if self.bm25 is None:
            return []

        query_tokens = _code_aware_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                id=chunk["id"],
                text=chunk.get("text", ""),
                metadata=chunk.get("metadata", {}),
                bm25_score=float(scores[idx]),
            ))

        return results
