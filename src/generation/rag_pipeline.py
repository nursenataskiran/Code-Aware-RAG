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
    RAG pipeline with toggleable retrieval features.

    Set use_hybrid=False (default) to use vector-only search (ChromaRetriever).
    Set use_hybrid=True to enable BM25 fusion, and optionally reranker/expansion.

    This lets us evaluate each improvement in isolation.
    """
    if use_hybrid:
        retriever = HybridRetriever(
            use_reranker=use_reranker,
            use_query_expansion=use_query_expansion,
        )
        results = retriever.retrieve(
            query=query,
            k=k,
            chunk_types=chunk_types,
            max_per_file=max_per_file,
        )
    else:
        retriever = ChromaRetriever()
        results = retriever.retrieve(
            query=query,
            k=k,
            chunk_types=chunk_types,
            max_per_file=max_per_file,
        )

    llm_client = OpenRouterClient()
    context = build_context(results)
    prompt = build_prompt(query, context)
    answer = llm_client.generate(prompt)

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "results": results,
    }
