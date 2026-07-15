"""
repo_indexer.py
---------------
Chunks a single downloaded project directory and adds the resulting chunks
to the persistent Chroma collection.

This module is the only place that touches Chroma for write operations
inside the ingestion flow. It is kept separate from the download logic
(github_ingestor.py) so each service has one clear responsibility.

Public API
~~~~~~~~~~
    is_project_indexed(project_name) -> bool
    index_project(project_dir, project_name) -> int   # returns chunk count
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.chunking.chunk_models import Chunk
from src.chunking.smart_chunker import SmartChunker
from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

#: Must match the batch size used in build_vector_store_v3.py so both paths
#: behave identically.
_CHROMA_BATCH_SIZE: int = 500

#: Extensions the SmartChunker can handle.  Intentionally kept in sync with
#: SUPPORTED_EXTENSIONS in github_ingestor, but defined independently here so
#: this module has no import dependency on the download layer.
_INDEXABLE_EXTENSIONS: frozenset[str] = frozenset({".py", ".md", ".ipynb"})

#: Files whose bare name is exactly this are skipped at collection time,
#: consistent with the download-time filter in github_ingestor.py.
_SKIP_FILENAMES: frozenset[str] = frozenset({"__init__.py"})


# ── Chroma client factory ─────────────────────────────────────────────────────


def _get_collection() -> chromadb.Collection:
    """
    Open the persistent Chroma collection, creating it if necessary.

    Uses ``get_or_create_collection`` so this is safe to call when the
    collection already exists (populated by the manual build script or a
    previous ingestion run).  The same embedding function and metadata that
    ``build_vector_store_v3.py`` uses are applied here.
    """
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "schema_version": "v4",
        },
    )
    return collection


# ── Public helpers ────────────────────────────────────────────────────────────


def is_project_indexed(project_name: str) -> bool:
    """
    Return ``True`` when at least one chunk from *project_name* already exists
    in the Chroma collection.

    Uses a ``limit=1`` query with a ``where`` filter so it is cheap regardless
    of collection size.  Returns ``False`` when the collection does not exist
    yet (catches the exception from ``get_or_create_collection``, which will
    simply create an empty collection in that case — harmless).
    """
    try:
        collection = _get_collection()
        results = collection.get(
            where={"project_name": project_name},
            limit=1,
            include=[],  # we only need to know *if* any document exists
        )
        return len(results.get("ids", [])) > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not check Chroma for project '%s': %s. "
            "Treating as not indexed.",
            project_name,
            exc,
        )
        return False


def index_project(project_dir: Path, project_name: str) -> int:
    """
    Chunk every supported file under *project_dir* and add the chunks to the
    persistent Chroma collection.

    Parameters
    ----------
    project_dir:
        Absolute path to the downloaded project directory
        (``data/raw/<project_name>/``).
    project_name:
        The normalised project identifier (``owner__repo``).  Stored as
        ``metadata.project_name`` on every chunk so the retriever can filter
        by project.

    Returns
    -------
    int
        Number of chunks successfully added to Chroma.

    Raises
    ------
    ChunkingError
        When chunking produces zero usable chunks from a non-empty file list.
    IndexingError
        When the Chroma ``collection.add()`` call fails.
    """
    # Import here to avoid a circular dependency if this module is ever
    # imported before the ingestor (both live in src.ingestion).
    from src.ingestion.github_ingestor import ChunkingError, IndexingError

    # ── 1. Collect files ──────────────────────────────────────────────────
    files = _collect_files(project_dir)
    logger.info(
        "index_project: %d file(s) to chunk in '%s'", len(files), project_name
    )

    if not files:
        raise ChunkingError(
            f"No indexable files found under '{project_dir}' for "
            f"project '{project_name}'."
        )

    # ── 2. Chunk each file ────────────────────────────────────────────────
    chunker = SmartChunker()
    all_chunks: list[Chunk] = []
    failed_files: list[str] = []

    for file_path in files:
        try:
            chunks = chunker.chunk_file(str(file_path), project_name)
            usable = [c for c in chunks if not c.is_too_short()]
            all_chunks.extend(usable)
            logger.debug(
                "Chunked '%s': %d chunk(s) (%d too short, dropped)",
                file_path.name,
                len(usable),
                len(chunks) - len(usable),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chunking failed for '%s': %s", file_path, exc)
            failed_files.append(str(file_path))

    logger.info(
        "Chunking complete: %d chunk(s) from %d file(s) (%d file(s) failed)",
        len(all_chunks),
        len(files),
        len(failed_files),
    )

    if not all_chunks:
        raise ChunkingError(
            f"Chunking produced zero usable chunks for project '{project_name}'. "
            f"{len(failed_files)} file(s) failed to chunk."
        )

    # ── 3. Build Chroma lists (same convention as build_vector_store_v3) ──
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    seen_ids: set[str] = set()
    for chunk in all_chunks:
        chunk_id = chunk.build_id()
        if chunk_id in seen_ids:
            logger.debug("Duplicate chunk ID skipped: %s", chunk_id)
            continue
        seen_ids.add(chunk_id)

        meta = chunk.metadata()
        meta["raw_text"] = chunk.text  # stored for retrieval display

        ids.append(chunk_id)
        documents.append(chunk.embedding_text())
        metadatas.append(meta)

    # ── 4. Add to Chroma in batches ───────────────────────────────────────
    try:
        collection = _get_collection()
        for start in range(0, len(ids), _CHROMA_BATCH_SIZE):
            end = min(start + _CHROMA_BATCH_SIZE, len(ids))
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            logger.debug("Chroma batch added: %d–%d", start, end)
    except Exception as exc:
        raise IndexingError(
            f"Failed to add chunks for project '{project_name}' to Chroma: {exc}"
        ) from exc

    logger.info(
        "Indexed %d chunk(s) for project '%s'.", len(ids), project_name
    )
    return len(ids)


# ── Private helpers ───────────────────────────────────────────────────────────


def _collect_files(project_dir: Path) -> List[Path]:
    """
    Return all indexable files under *project_dir*.

    Applies the same skip rules as the download-time filter:
    * Only ``_INDEXABLE_EXTENSIONS`` are included.
    * Files named exactly ``__init__.py`` are excluded.
    """
    files: List[Path] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _INDEXABLE_EXTENSIONS:
            continue
        if path.name in _SKIP_FILENAMES:
            continue
        files.append(path)
    return files
