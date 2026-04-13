import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.generation.context_builder import build_context
from src.generation.rag_pipeline import build_prompt, run_rag_pipeline
from src.retrieval.retriever import RetrievalResult


def main() -> None:
    query = "Show me all files"
    sample_results = [
        RetrievalResult(
            id="demo_chunk",
            text="README explains how the repository is structured.",
            metadata={
                "project_name": "demo-project",
                "file_name": "README.md",
                "file_path": "demo-project/README.md",
                "chunk_type": "markdown_section",
                "section_header": "Repository layout",
            },
        )
    ]
    prompt_preview = build_prompt(query, build_context(sample_results))

    print("PROMPT PREVIEW:\n")
    print(prompt_preview)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("\nSkipping live RAG run because OPENROUTER_API_KEY is not set.")
        return

    output = run_rag_pipeline(query=query, k=3)

    print("QUESTION:", query)
    print("ANSWER:", output["answer"])
    print("TOP FILES:", [r.metadata["file_name"] for r in output["results"]])


if __name__ == "__main__":
    main()
