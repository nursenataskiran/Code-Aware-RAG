import json
import chromadb
from pathlib import Path


CHUNK_FILE = "data/processed/chunks.json"
CHROMA_PATH = "data/vector_db/chroma"

COLLECTION_NAME = "github_projects_v1"


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store():

    print("Loading chunks...")
    chunks = load_chunks(CHUNK_FILE)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print(f"Total chunks: {len(chunks)}")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    print(f"Creating collection: {COLLECTION_NAME}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print("Adding documents to collection...")

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print("Vector store created successfully!")


if __name__ == "__main__":
    build_vector_store()