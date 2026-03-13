from src.retrieval.retriever import ChromaRetriever
from src.generation.context_builder import build_context


def main() -> None:
    retriever = ChromaRetriever()

    results = retriever.retrieve(
        query="Where is the LSTM model defined?",
        k=3
    )

    context = build_context(results)

    print("\nGenerated Context:\n")
    print(context)


if __name__ == "__main__":
    main()