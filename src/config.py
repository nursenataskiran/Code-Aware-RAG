"""
Centralized configuration for Code-Aware-RAG.

Single source of truth for paths, API keys, and model names.
All other modules import from here — no scattered load_dotenv() calls.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ── Project paths ────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = Path("data/vector_db/chroma")
EVAL_DIR = Path("data/evaluation")
EVAL_REPORTS_DIR = Path("eval_reports")

# ── Vector store ─────────────────────────────────────────────────────
COLLECTION_FAMILY = "github_projects"
CHUNK_SCHEMA_VERSION = "v3"
COLLECTION_NAME = f"{COLLECTION_FAMILY}_{CHUNK_SCHEMA_VERSION}"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)

HF_TOKEN = os.getenv("HF_TOKEN")

# ── OpenRouter / LLM ────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── RAGAS evaluation ─────────────────────────────────────────────────
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "openai/gpt-4o-mini")
