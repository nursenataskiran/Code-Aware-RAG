import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.generation.context_builder import build_context
from src.retrieval.retriever import RetrievalResult


def main() -> None:
    results = [
        RetrievalResult(
            id="demo_chunk",
            text="def train_model():\n    return 'ok'",
            metadata={
                "project_name": "demo-project",
                "file_name": "trainer.py",
                "file_path": "demo-project/src/trainer.py",
                "chunk_type": "python_function",
                "symbol_name": "train_model",
                "start_line": 10,
                "end_line": 20,
            },
        )
    ]

    context = build_context(results)

    print("\nGenerated Context:\n")
    print(context)


if __name__ == "__main__":
    main()
