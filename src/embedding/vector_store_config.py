import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Stable constants (paths, collection naming) ─────────────────────
CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = Path("data/vector_db/chroma")
COLLECTION_FAMILY = "github_projects"
CHUNK_SCHEMA_VERSION = "v3"
COLLECTION_NAME = f"{COLLECTION_FAMILY}_{CHUNK_SCHEMA_VERSION}"

# ── Environment-driven config ────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)

HF_TOKEN = os.getenv("HF_TOKEN")
