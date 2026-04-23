from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.retrieval.retriever import ChromaRetriever, HybridRetriever
from src.generation.context_builder import build_context
from src.llm.openrouter_client import OpenRouterClient


PROMPT_TEMPLATE = """
You are an assistant that answers questions about a set of GitHub repositories.

Use the provided repository context to answer the question.
If the answer appears in multiple sources, combine them into a clear explanation.
Prefer concise explanations rather than copying long text blocks.
If the answer is not contained in the context, say:
"I don't know based on the provided repository context."

When possible, mention the file name and symbol name in your answer.

Context:
{context}

Question:
{question}

Answer:
"""


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )


class RAGPipeline:
    """
    Owns retriever + LLM client. Instantiate once, query many times.

    This avoids re-loading the embedding model, BM25 index, and
    cross-encoder on every query — saving ~6 seconds per call.
    """

    def __init__(
        self,
        use_hybrid: bool = True,
        use_reranker: bool = False,
        use_query_expansion: bool = False,
    ) -> None:
        if use_hybrid:
            self.retriever = HybridRetriever(
                use_reranker=use_reranker,
                use_query_expansion=use_query_expansion,
            )
        else:
            self.retriever = ChromaRetriever()

        self.llm = OpenRouterClient()

    def query(
        self,
        question: str,
        k: int = 5,
        chunk_types: Optional[List[str]] = None,
        max_per_file: Optional[int] = 2,
    ) -> Dict[str, Any]:
        """Run retrieval + generation for a single question."""
        results = self.retriever.retrieve(
            query=question,
            k=k,
            chunk_types=chunk_types,
            max_per_file=max_per_file,
        )

        context = build_context(results)
        prompt = build_prompt(question, context)
        answer = self.llm.generate(prompt)

        return {
            "query": question,
            "answer": answer,
            "context": context,
            "results": results,
        }


# ── Backward-compatibility wrapper ──────────────────────────────────
# Existing callers (evaluator, test scripts) can keep using this.
# It creates a fresh pipeline per call — same behavior as before.

def run_rag_pipeline(
    query: str,
    k: int = 5,
    chunk_types: list[str] | None = None,
    max_per_file: int | None = 2,
    use_hybrid: bool = True,
    use_reranker: bool = False,
    use_query_expansion: bool = False,
) -> dict:
    """
    Backward-compatible wrapper around RAGPipeline.

    For better performance, use RAGPipeline directly — it avoids
    re-loading models on every call.
    """
    pipeline = RAGPipeline(
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        use_query_expansion=use_query_expansion,
    )
    return pipeline.query(
        question=query,
        k=k,
        chunk_types=chunk_types,
        max_per_file=max_per_file,
    )
