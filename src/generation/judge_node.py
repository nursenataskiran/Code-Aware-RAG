"""
src/generation/judge_node.py
============================
LangGraph judge node for the Code-Aware RAG workflow.

Responsibility
--------------
Given the user query, the retrieved context, and the generated answer,
decide whether the answer is acceptable (PASS) or should trigger a fresh
retrieval + generation attempt (RETRY).

No ground truth, no scoring, no reference contexts are required — this is
a lightweight online quality gate that runs inside the LangGraph graph.

Output contract
---------------
The LLM is instructed to respond with exactly two lines::

    DECISION: PASS   (or RETRY)
    REASON: <one short sentence>

The node returns a :class:`JudgeResult` dataclass and also writes
``judge_decision`` and ``judge_reason`` back into the LangGraph state dict
so downstream nodes / edges can route on the decision.

Usage
-----
Wire it into a LangGraph StateGraph like this::

    from src.generation.judge_node import judge_node, JudgeDecision

    graph.add_node("judge", judge_node)
    graph.add_conditional_edges(
        "judge",
        lambda state: state["judge_decision"],
        {
            JudgeDecision.PASS: "respond",
            JudgeDecision.RETRY: "retrieve",
        },
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from src.llm.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum tokens the judge LLM should ever emit (DECISION + REASON is ~40).
_JUDGE_MAX_TOKENS: int = 80

#: Low temperature keeps the judge deterministic.
_JUDGE_TEMPERATURE: float = 0.0

#: Fallback decision when the LLM output cannot be parsed.
_FALLBACK_DECISION = "RETRY"
_FALLBACK_REASON = "Judge output could not be parsed; defaulting to retry."

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are a strict answer quality judge for a retrieval-augmented generation system.

You will be given:
- A user query
- Retrieved context (the chunks the system found)
- A generated answer

Your job is to decide whether the answer is acceptable or should be retried.

Rules:
- If the answer is clearly supported by the retrieved context → PASS
- If the answer says "I don't know" but relevant context exists → RETRY
- If the answer is generic and not tied to the context → RETRY
- If the answer ignores useful information in the context → RETRY

Respond with EXACTLY two lines and nothing else:

DECISION: PASS or RETRY
REASON: one short sentence

---

User Query:
{query}

Retrieved Context:
{context}

Generated Answer:
{answer}
"""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class JudgeDecision(str, Enum):
    """Possible outcomes of the judge node."""

    PASS = "PASS"
    RETRY = "RETRY"


@dataclass(frozen=True)
class JudgeResult:
    """Structured output of a single judge evaluation."""

    decision: JudgeDecision
    reason: str

    def __str__(self) -> str:  # noqa: D105
        return f"DECISION: {self.decision.value}\nREASON: {self.reason}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_judge_prompt(query: str, context: str, answer: str) -> str:
    """Render the judge prompt with the three required fields."""
    return _JUDGE_PROMPT_TEMPLATE.format(
        query=query.strip(),
        context=context.strip(),
        answer=answer.strip(),
    )


def _parse_judge_response(raw: str) -> JudgeResult:
    """
    Parse the LLM response into a :class:`JudgeResult`.

    Expected format (case-insensitive, lenient whitespace)::

        DECISION: PASS
        REASON: The answer directly uses the retrieved context.

    Falls back to RETRY if the response is malformed so the pipeline
    never silently degrades.
    """
    decision_match = re.search(
        r"DECISION\s*:\s*(PASS|RETRY)", raw, re.IGNORECASE
    )
    reason_match = re.search(
        r"REASON\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL
    )

    if not decision_match:
        logger.warning(
            "[judge_node] Could not parse DECISION from LLM output: %r", raw
        )
        return JudgeResult(
            decision=JudgeDecision(_FALLBACK_DECISION),
            reason=_FALLBACK_REASON,
        )

    decision_str = decision_match.group(1).upper()
    reason_str = (
        reason_match.group(1).strip().splitlines()[0].strip()
        if reason_match
        else "No reason provided."
    )

    try:
        decision = JudgeDecision(decision_str)
    except ValueError:
        logger.warning(
            "[judge_node] Unknown DECISION value %r; defaulting to RETRY.",
            decision_str,
        )
        decision = JudgeDecision.RETRY

    return JudgeResult(decision=decision, reason=reason_str)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def judge_node(
    state: Dict[str, Any],
    *,
    llm: OpenRouterClient | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node — evaluate the generated answer and decide PASS or RETRY.

    Parameters
    ----------
    state:
        The shared LangGraph state dict. Must contain:

        - ``"query"``   – the original user question (str)
        - ``"context"`` – the retrieved context string (str)
        - ``"answer"``  – the generated answer (str)

    llm:
        Optional pre-built :class:`~src.llm.openrouter_client.OpenRouterClient`.
        If ``None``, a new client is created using the environment defaults
        (``OPENROUTER_API_KEY`` / ``OPENROUTER_MODEL``). Injecting an ``llm``
        instance is useful in tests and when you want to share a single client
        across multiple nodes.

    Returns
    -------
    dict
        A partial state update with two new keys:

        - ``"judge_decision"`` – :class:`JudgeDecision` enum value
        - ``"judge_reason"``   – short explanation string

    Raises
    ------
    KeyError
        If ``state`` is missing any of the required keys
        (``query``, ``context``, ``answer``).

    Examples
    --------
    Direct call (outside a graph)::

        result_state = judge_node({
            "query": "What does the Retriever class do?",
            "context": "[Source 1]\\nClass Retriever ...",
            "answer": "The Retriever class fetches documents from ChromaDB.",
        })
        print(result_state["judge_decision"])   # JudgeDecision.PASS
        print(result_state["judge_reason"])

    Inside a LangGraph graph::

        graph.add_node("judge", judge_node)
    """
    # ── 1. Extract required state fields ──────────────────────────────
    query: str = state["query"]
    context: str = state["context"]
    answer: str = state["answer"]

    # ── 2. Build the judge prompt ──────────────────────────────────────
    prompt = _build_judge_prompt(query=query, context=context, answer=answer)

    # ── 3. Call the LLM ───────────────────────────────────────────────
    client = llm or OpenRouterClient()

    logger.info(
        "[judge_node] Evaluating answer for query: %r", query[:120]
    )

    raw_response = client.generate(
        prompt=prompt,
        temperature=_JUDGE_TEMPERATURE,
        max_tokens=_JUDGE_MAX_TOKENS,
    )

    logger.debug("[judge_node] Raw LLM output: %r", raw_response)

    # ── 4. Parse the response ─────────────────────────────────────────
    result = _parse_judge_response(raw_response)

    logger.info(
        "[judge_node] Decision=%s | Reason=%s",
        result.decision.value,
        result.reason,
    )

    # ── 5. Return partial state update ───────────────────────────────
    return {
        "judge_decision": result.decision,
        "judge_reason": result.reason,
    }
