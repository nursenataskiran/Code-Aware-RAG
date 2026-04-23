"""
Backward-compatibility shim.

All config has moved to src.config.
This file re-exports the old names so existing imports don't break.
Safe to delete once all imports are verified.
"""

from src.config import (  # noqa: F401
    CHUNKS_PATH,
    CHROMA_PATH,
    COLLECTION_FAMILY,
    CHUNK_SCHEMA_VERSION,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HF_TOKEN,
)
