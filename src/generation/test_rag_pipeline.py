from src.generation.rag_pipeline import run_rag_pipeline


def main() -> None:
    query = "Show me all files"

    output = run_rag_pipeline(
        query=query,
        k=3,

    )

    print("QUESTION:", query)
    print("ANSWER:", output["answer"])
    print("TOP FILES:", [r.metadata["file_name"] for r in output["results"]])


if __name__ == "__main__":
    main()