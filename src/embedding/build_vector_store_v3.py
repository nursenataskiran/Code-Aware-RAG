import json
import os
import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import (
    CHUNKS_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HF_TOKEN,
)


def load_chunks(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(recreate_collection: bool = True):

    print("Loading chunks...")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    chunks = load_chunks(CHUNKS_PATH)

    ids = [chunk["id"] for chunk in chunks]

    # Use embedding_text (includes semantic description) for the embedding,
    # but store the raw text as the document for retrieval display.
    documents = [chunk.get("embedding_text", chunk["text"]) for chunk in chunks]
    raw_texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Store raw text in metadata so we can retrieve it later
    for i, meta in enumerate(metadatas):
        meta["raw_text"] = raw_texts[i]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found in chunks.json. Rebuild chunk IDs before indexing.")

    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    # ── Authenticate with HuggingFace if token is available ──────────
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        print("HF_TOKEN detected — authenticated HuggingFace downloads enabled.")

    # ── Use explicit embedding function ──────────────────────────────
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    if recreate_collection:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    print(f"Creating collection: {COLLECTION_NAME}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "schema_version": "v3",
        },
    )

    print("Adding documents to collection...")

    # ChromaDB has a batch size limit, process in batches of 500
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Added batch {start}-{end}")

    print(f"Vector store created successfully! ({collection.count()} documents)")


if __name__ == "__main__":
    build_vector_store()
