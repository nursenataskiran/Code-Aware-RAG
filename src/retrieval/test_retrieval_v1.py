import chromadb


CHROMA_PATH = "data/vector_db/chroma"
COLLECTION_NAME = "github_projects_v1"


def search(query, k=5):

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(docs, metas)):

        print("\n====================")
        print(f"Result {i+1}")
        print("====================")

        print("Project:", meta["project_name"])
        print("File:", meta["file_name"])
        print("Symbol:", meta.get("symbol_name"))

        print("\nSnippet:")
        print(doc[:300])


if __name__ == "__main__":

    query = "Where is the LSTM model defined?"

    search(query)