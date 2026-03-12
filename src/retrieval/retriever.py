from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb


CHROMA_PATH = "data/vector_db/chroma"
COLLECTION_NAME = "github_projects_v1"


@dataclass
class RetrievalResult:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

    @property
    def score(self) -> Optional[float]:
        """
        Lower distance is better in Chroma.
        Optional helper to view a similarity-like score.
        """
        if self.distance is None:
            return None
        return 1 / (1 + self.distance)


class ChromaRetriever:
    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name=collection_name)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        chunk_types: Optional[List[str]] = None,
        ) -> List[RetrievalResult]:
        query_kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": k,
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

            structured_results.append(
                RetrievalResult(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                    distance=distance,
                )
            )

        return structured_results

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