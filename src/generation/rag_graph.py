"""
Usage
-----
    from src.generation.rag_graph import build_rag_graph

    graph = build_rag_graph()
    final_state = graph.invoke({"query": "What does the Retriever class do?"})

    print(final_state["answer"])
    print(final_state["judge_decision"])
    print(final_state["judge_reason"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.generation.context_builder import build_context
from src.generation.rag_pipeline import build_prompt
from src.generation.judge_node import JudgeDecision, judge_node
from src.llm.openrouter_client import OpenRouterClient
from src.retrieval.retriever import HybridRetriever, ChromaRetriever
from src.config import USE_RERANKER, USE_QUERY_EXPANSION

logger = logging.getLogger(__name__)

#: Maximum number of retrieval + generation attempts before giving up.
MAX_RETRIES: int = 2

# ---------------------------------------------------------------------------
# Graph state schema
# ---------------------------------------------------------------------------

class RAGState(TypedDict, total=False):
    """Shared state that flows through every node of the graph."""

    # Inputs 
    query: str                   # user question (required at invocation)

    # Retrieval
    context: str                 # formatted context string
    raw_results: list            # RetrievalResult objects

    # Generation
    answer: str                  # LLM-generated answer

    # Judge 
    judge_decision: JudgeDecision
    judge_reason: str

    # Control
    attempt: int                 # 0-based retry counter


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def _make_retrieve_node(retriever):
    """Return a LangGraph node function that uses the given retriever."""

    def retrieve(state: RAGState) -> Dict[str, Any]:
        attempt = state.get("attempt", 0)
        logger.info(
            "[rag_graph] retrieve — attempt=%d query=%r",
            attempt,
            state["query"][:120],
        )

        results = retriever.retrieve(
            query=state["query"],
            k=5,
            chunk_types=None,
            max_per_file=2,
        )
        context = build_context(results)

        return {
            "context": context,
            "raw_results": results,
            "attempt": attempt,
        }

    return retrieve


def _make_generate_node(llm: OpenRouterClient):
    """Return a LangGraph node function that uses the given LLM client."""

    def generate(state: RAGState) -> Dict[str, Any]:
        logger.info("[rag_graph] generate — attempt=%d", state.get("attempt", 0))

        prompt = build_prompt(
            question=state["query"],
            context=state["context"],
        )
        answer = llm.generate(prompt)

        logger.debug("[rag_graph] generated answer: %r", answer[:200])
        return {"answer": answer}

    return generate


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_judge(state: RAGState) -> str:

    decision = state.get("judge_decision")
    attempt = state.get("attempt", 0)

    if decision == JudgeDecision.PASS:
        logger.info("[rag_graph] Judge → PASS. Finishing.")
        return END

    if attempt >= MAX_RETRIES:
        logger.warning(
            "[rag_graph] Judge → RETRY but attempt=%d >= MAX_RETRIES=%d. "
            "Forcing end to avoid infinite loop.",
            attempt,
            MAX_RETRIES,
        )
        return END

    logger.info(
        "[rag_graph] Judge → RETRY (attempt %d / %d). Re-retrieving.",
        attempt + 1,
        MAX_RETRIES,
    )
    return "retrieve"


def _increment_attempt(state: RAGState) -> Dict[str, Any]:
    """
    Thin node that bumps the attempt counter before each re-retrieval.

    Placed between the judge's RETRY edge and the retrieve node so the
    counter is always accurate inside retrieve's log output.
    """
    return {"attempt": state.get("attempt", 0) + 1}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_rag_graph(
    use_hybrid: bool = True,
    use_reranker: bool = USE_RERANKER,
    use_query_expansion: bool = USE_QUERY_EXPANSION,
    llm: Optional[OpenRouterClient] = None,
    max_retries: int = MAX_RETRIES,
) -> Any:

    global MAX_RETRIES
    MAX_RETRIES = max_retries  # honour caller's preference in routing closure

    # ── Build shared components ──────────────────────────────────────────
    if use_hybrid:
        retriever = HybridRetriever(
            use_reranker=use_reranker,
            use_query_expansion=use_query_expansion,
        )
    else:
        retriever = ChromaRetriever()

    client = llm or OpenRouterClient()

    # ── Build node functions ─────────────────────────────────────────────
    retrieve_fn = _make_retrieve_node(retriever)
    generate_fn = _make_generate_node(client)

    # judge_node already matches the LangGraph node signature; wrap it so
    # we can inject the shared LLM client without repeating construction.
    def judge_fn(state: RAGState) -> Dict[str, Any]:
        return judge_node(state, llm=client)

    # ── Assemble the graph ───────────────────────────────────────────────
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_fn)
    builder.add_node("generate", generate_fn)
    builder.add_node("judge", judge_fn)
    builder.add_node("increment_attempt", _increment_attempt)

    # Normal flow
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "judge")

    # Conditional routing after the judge
    builder.add_conditional_edges(
        "judge",
        _route_after_judge,
        {
            "retrieve": "increment_attempt",   # bump counter then re-retrieve
            END: END,
        },
    )

    # After bumping the counter, loop back to retrieval
    builder.add_edge("increment_attempt", "retrieve")

    return builder.compile()
