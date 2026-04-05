from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution


load_dotenv()

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
CHUNKS_PATH = Path("data/processed/chunks.json")
OUTPUT_DIR = Path("data/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_TESTSET_JSONL = OUTPUT_DIR / "ragas_testset_raw.jsonl"
RAW_TESTSET_CSV = OUTPUT_DIR / "ragas_testset_raw.csv"
REVIEW_CSV = OUTPUT_DIR / "ragas_testset_for_review.csv"


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
TESTSET_SIZE = int(os.getenv("RAGAS_TESTSET_SIZE", "5"))
GROUP_BY_FILE = os.getenv("RAGAS_GROUP_BY_FILE", "false").lower() == "true"
MAX_DOC_CHARS = int(os.getenv("RAGAS_MAX_DOC_CHARS", "12000"))

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "openai/gpt-4o-mini")

EMBED_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "openai/text-embedding-3-small")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "Code-Aware-RAG",
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("chunks.json must contain a list")

    return data


def trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def chunk_to_document(chunk: Dict[str, Any]) -> Document:
    metadata = chunk.get("metadata", {}).copy()
    metadata["chunk_id"] = chunk.get("id")

    return Document(
        page_content=(chunk.get("text") or "").strip(),
        metadata=metadata,
    )


def group_chunks_by_file(chunks: List[Dict[str, Any]]) -> List[Document]:
    grouped = defaultdict(list)

    for chunk in chunks:
        file_key = chunk.get("metadata", {}).get("file_path", "unknown")
        grouped[file_key].append(chunk)

    documents = []

    for _, file_chunks in grouped.items():
        file_chunks = sorted(
            file_chunks,
            key=lambda x: x.get("metadata", {}).get("chunk_index", 0),
        )

        combined_text = "\n\n".join(
            (c.get("text") or "").strip() for c in file_chunks
        )

        combined_text = trim_text(combined_text, MAX_DOC_CHARS)

        metadata = file_chunks[0].get("metadata", {}).copy()
        metadata["grouped_from"] = "file"

        documents.append(
            Document(
                page_content=combined_text,
                metadata=metadata,
            )
        )

    return documents


def build_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    if GROUP_BY_FILE:
        return group_chunks_by_file(chunks)
    return [chunk_to_document(chunk) for chunk in chunks]


def build_llm():
    if not LLM_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
        default_headers=OPENROUTER_HEADERS,
    )

    return LangchainLLMWrapper(llm)


def build_embeddings():
    if not EMBED_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=EMBED_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=OPENROUTER_HEADERS,
    )

    return LangchainEmbeddingsWrapper(embeddings)


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_testset_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "keep" not in result.columns:
        result["keep"] = ""

    if "notes" not in result.columns:
        result["notes"] = ""

    return result


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    if not chunks:
        raise ValueError("chunks.json is empty")

    print("Building documents...")
    documents = build_documents(chunks)

    if not documents:
        raise ValueError("No documents were built from chunks")

    print("Initializing models...")
    llm = build_llm()
    embeddings = build_embeddings()

    print("Creating generator...")
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
    )

    print("Generating testset...")
    testset = generator.generate_with_chunks(
        documents,
        testset_size=TESTSET_SIZE,
        query_distribution=default_query_distribution(llm),
    )

    print("Converting to dataframe...")
    df = testset.to_pandas()
    review_df = normalize_testset_dataframe(df)

    print("Saving...")
    df.to_csv(RAW_TESTSET_CSV, index=False, encoding="utf-8-sig")
    review_df.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    save_jsonl(df.to_dict(orient="records"), RAW_TESTSET_JSONL)

    print("Done!")


if __name__ == "__main__":
    main()