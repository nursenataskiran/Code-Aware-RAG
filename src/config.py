import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project paths ────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks_v1.4_audit.json")
CHROMA_PATH = Path("data/vector_db/chroma")
EVAL_DIR = Path("data/evaluation")
EVAL_REPORTS_DIR = Path("eval_reports")
EVAL_TESTSET_PATH = Path(
    os.getenv("EVAL_TESTSET_PATH", str(EVAL_DIR / "ragas_testset_v4.jsonl"))
)

# ── Vector store ─────────────────────────────────────────────────────
COLLECTION_FAMILY = "github_projects"
CHUNK_SCHEMA_VERSION = "v4"
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

# ── Retrieval feature flags ───────────────────────────────────────────
# Both default to False to keep startup fast and match production behaviour.
# Set USE_RERANKER=true or USE_QUERY_EXPANSION=true in .env to enable.
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "false").lower() == "true"
