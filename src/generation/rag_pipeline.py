from src.retrieval.retriever import ChromaRetriever
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
) -> dict:
    retriever = ChromaRetriever()
    llm_client = OpenRouterClient()

    results = retriever.retrieve(
        query=query,
        k=k,
        chunk_types=chunk_types,
    )

    context = build_context(results)
    prompt = build_prompt(query, context)
    answer = llm_client.generate(prompt)

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "results": results,
    }