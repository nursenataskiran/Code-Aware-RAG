import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.retriever import ChromaRetriever


def print_result(result, rank: int) -> None:
    meta = result.metadata

    print("\n" + "=" * 50)
    print(f"Rank {rank}")
    print("=" * 50)

    print(f"ID: {result.id}")
    print(f"Project: {meta.get('project_name')}")
    print(f"File: {meta.get('file_name')}")
    print(f"Path: {meta.get('file_path')}")
    print(f"Chunk Type: {meta.get('chunk_type')}")
    print(f"Symbol: {meta.get('symbol_name')}")
    print(f"Parent Symbol: {meta.get('parent_symbol')}")
    print(f"Section: {meta.get('section_header')}")
    print(f"Cell: {meta.get('cell_index')} ({meta.get('cell_type')})")
    print(f"Lines: {meta.get('start_line')} - {meta.get('end_line')}")

    if result.distance is not None:
        print(f"Distance: {result.distance:.4f}")

    if result.score is not None:
        print(f"Score: {result.score:.4f}")

    print("\nText Preview:")
    print(result.text[:500])


def main() -> None:
    try:
        retriever = ChromaRetriever()
    except ValueError as exc:
        print(exc)
        return

    results = retriever.retrieve(
        query="How does the project process raw F1 data before feeding it to the model?",
        k=5,
        chunk_types=["notebook_code", "python_function"]
    )

    for i, result in enumerate(results, start=1):
        print_result(result, i)


if __name__ == "__main__":
    main()
