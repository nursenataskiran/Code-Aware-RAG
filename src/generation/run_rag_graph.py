"""
src/generation/run_rag_graph.py
===============================
Live integration runner for the LangGraph RAG graph.

Builds the real graph (real retriever + real LLM), runs it against one or
more queries, and prints a structured report for each one.

Usage
-----
# Single query from the command line:
    python src/generation/run_rag_graph.py "What does the HybridRetriever class do?"

# Interactive mode (type queries one by one, empty line to quit):
    python src/generation/run_rag_graph.py

# Multiple queries from a file (one per line):
    python src/generation/run_rag_graph.py --file scripts/sample_queries.txt

Options
-------
--no-reranker         Disable cross-encoder re-ranking
--no-query-expansion  Disable query expansion
--max-retries N       Override the RETRY limit (default: 2)
--pure-vector         Use ChromaRetriever instead of HybridRetriever
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.generation.rag_graph import build_rag_graph  # noqa: E402
from src.generation.judge_node import JudgeDecision   # noqa: E402


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

WIDTH = 70


def _hr(char: str = "─") -> None:
    print(char * WIDTH)


def _header(text: str) -> None:
    _hr("═")
    print(f"  {text}")
    _hr("═")


def _section(text: str) -> None:
    print(f"\n  ▸ {text}")
    _hr()


def _wrap(label: str, value: str, indent: int = 4) -> None:
    prefix = " " * indent
    wrapped = textwrap.fill(
        value,
        width=WIDTH - indent,
        initial_indent=prefix,
        subsequent_indent=prefix,
    )
    print(f"  {label}:")
    print(wrapped)


def _decision_badge(decision: JudgeDecision) -> str:
    if decision == JudgeDecision.PASS:
        return "✅  PASS"
    return "🔄  RETRY (hit limit — answer kept)"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_query(graph, query: str, idx: int | None = None) -> dict:
    label = f"Query {idx}: {query!r}" if idx is not None else f"Query: {query!r}"
    _header(label)

    print(f"\n  Building state …")
    t0 = time.perf_counter()

    final = graph.invoke({"query": query})

    elapsed = time.perf_counter() - t0

    _section("Answer")
    _wrap("", final.get("answer", "(no answer)"))

    _section("Judge verdict")
    decision = final.get("judge_decision", JudgeDecision.RETRY)
    reason = final.get("judge_reason", "—")
    attempt = final.get("attempt", 0)
    print(f"    {_decision_badge(decision)}")
    print(f"    Reason  : {reason}")
    print(f"    Attempts: {attempt + 1}")

    raw = final.get("raw_results", [])
    if raw:
        _section(f"Retrieved chunks ({len(raw)})")
        for i, r in enumerate(raw, 1):
            meta = r.metadata if hasattr(r, "metadata") else {}
            score = f"{r.score:.4f}" if getattr(r, "score", None) is not None else "—"
            print(
                f"    [{i}] {meta.get('file_name', '?')} "
                f"· {meta.get('chunk_type', '?')} "
                f"· {meta.get('symbol_name', '?')} "
                f"· score={score}"
            )

    print(f"\n  ⏱  Total wall time: {elapsed:.2f}s")
    _hr()
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live integration runner for the LangGraph RAG graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query string. Omit to enter interactive mode.",
    )
    p.add_argument(
        "--file",
        metavar="PATH",
        help="Path to a file with one query per line.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        metavar="N",
        help="Max RETRY loops before the graph forces END (default: 2).",
    )
    p.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable cross-encoder re-ranking.",
    )
    p.add_argument(
        "--no-query-expansion",
        action="store_true",
        help="Disable query expansion.",
    )
    p.add_argument(
        "--pure-vector",
        action="store_true",
        help="Use ChromaRetriever instead of HybridRetriever.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # ── Build graph once (heavy models loaded here) ─────────────────────
    print("\nInitialising RAG graph … (this may take a few seconds)")
    t0 = time.perf_counter()
    graph = build_rag_graph(
        use_hybrid=not args.pure_vector,
        use_reranker=not args.no_reranker,
        use_query_expansion=not args.no_query_expansion,
        max_retries=args.max_retries,
    )
    print(f"Graph ready in {time.perf_counter() - t0:.1f}s.\n")

    # ── Collect queries ──────────────────────────────────────────────────
    queries: list[str] = []

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        queries = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"Loaded {len(queries)} queries from {path}.\n")

    elif args.query:
        queries = [args.query]

    else:
        # Interactive mode
        print("Interactive mode — enter a query and press Enter.")
        print("Leave the line empty and press Enter to quit.\n")
        while True:
            try:
                q = input("  Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            queries.append(q)
            # Run immediately so the user sees the result before the next prompt
            run_query(graph, q)
            queries.clear()          # already ran it
        print("\nBye.")
        return

    # ── Run all collected queries ────────────────────────────────────────
    for i, q in enumerate(queries, 1):
        run_query(graph, q, idx=i)

    print(f"\nDone — {len(queries)} query/queries processed.")


if __name__ == "__main__":
    main()
