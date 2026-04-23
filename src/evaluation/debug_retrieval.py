"""
Retrieval Debugging Tool
========================

Standalone diagnostic script to understand WHY retrieval fails
for specific questions. Run with a question from the testset to see:
- Vector search results vs BM25 results
- Expected reference chunks and their actual rank
- Score/distance details
- Whether hybrid search + reranking would have found it

Usage:
    python src/evaluation/debug_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.retriever import ChromaRetriever, HybridRetriever, RetrievalResult
from src.retrieval.bm25_index import BM25Index
from src.evaluation.evaluator import _is_match


def load_testset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def debug_single_question(
    question: str,
    reference_contexts: list[str],
    k: int = 10,
) -> None:
    """Run all retrieval methods and compare results."""
    
    print(f"\n{'='*70}")
    print(f"QUERY: {question}")
    print(f"{'='*70}")
    
    # ── Reference contexts ────────────────────────────────────────────
    print(f"\n📋 REFERENCE CONTEXTS ({len(reference_contexts)}):")
    for i, ref in enumerate(reference_contexts):
        print(f"  [{i+1}] {ref[:150]}...")
    
    # ── Vector search only ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("🔵 VECTOR SEARCH (ChromaDB only):")
    vector_results = []
    try:
        chroma = ChromaRetriever()
        vector_results = chroma.retrieve(query=question, k=k)
        _print_results(vector_results, reference_contexts)
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # ── BM25 search only ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("🟡 BM25 SEARCH (lexical only):")
    bm25_results = []
    try:
        bm25 = BM25Index()
        bm25_results = bm25.search(question, k=k)
        _print_results(bm25_results, reference_contexts)
        
        # Show BM25 scores
        print("\n  BM25 scores:")
        for r in bm25_results[:5]:
            symbol = r.metadata.get("symbol_name", r.metadata.get("section_header", ""))
            score_val = r.bm25_score or 0.0
            print(f"    score={score_val:.3f}  {r.metadata.get('file_name','')}  {symbol}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # ── Hybrid search (full pipeline) ─────────────────────────────────
    print(f"\n{'─'*70}")
    print("🟢 HYBRID SEARCH (vector + BM25 + reranker):")
    hybrid_results = []
    try:
        hybrid = HybridRetriever(use_reranker=True, use_query_expansion=True)
        hybrid_results = hybrid.retrieve(query=question, k=k)
        _print_results(hybrid_results, reference_contexts)
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("📊 MATCH SUMMARY:")
    
    methods = {
        "Vector": vector_results,
        "BM25": bm25_results,
        "Hybrid": hybrid_results,
    }
    
    for name, results in methods.items():
        hits = sum(
            1 for r in results
            if any(_is_match(r.text, ref) for ref in reference_contexts)
        )
        first_hit = None
        for rank, r in enumerate(results, 1):
            if any(_is_match(r.text, ref) for ref in reference_contexts):
                first_hit = rank
                break
        
        hit_str = "✅" if hits > 0 else "❌"
        rank_str = f"rank={first_hit}" if first_hit else "not found"
        print(f"  {hit_str} {name}: {hits}/{len(results)} relevant, first hit at {rank_str}")


def _print_results(
    results: list[RetrievalResult],
    reference_contexts: list[str],
) -> None:
    if not results:
        print("  (no results)")
        return
    
    for rank, r in enumerate(results, 1):
        is_relevant = any(_is_match(r.text, ref) for ref in reference_contexts)
        marker = "✅" if is_relevant else "  "
        
        meta = r.metadata
        symbol = meta.get("symbol_name", meta.get("section_header", "—"))
        file_name = meta.get("file_name", "—")
        chunk_type = meta.get("chunk_type", "—")
        project = meta.get("project_name", "—")
        
        dist_str = f"dist={r.distance:.4f}" if r.distance is not None else ""
        
        print(f"  {marker} [{rank}] {project}/{file_name} | {chunk_type} | {symbol} {dist_str}")
        print(f"        {r.text[:100]}...")


def main():
    testset_path = "data/evaluation/ragas_testset.jsonl"
    
    if not Path(testset_path).exists():
        print(f"❌ Testset not found: {testset_path}")
        print("\nRunning with a sample question instead...\n")
        
        debug_single_question(
            question="How does the project process raw F1 data before feeding it to the model?",
            reference_contexts=[
                "class SessionData:",
                "def clean_data(self):",
                "F1DataCleaner",
            ],
            k=10,
        )
        return
    
    testset = load_testset(testset_path)
    print(f"Loaded {len(testset)} questions from {testset_path}")
    
    # Debug first 3 questions (or all if fewer)
    for i, item in enumerate(testset[:3]):
        question = item["user_input"]
        reference_contexts = item.get("reference_contexts", [])
        debug_single_question(question, reference_contexts, k=10)


if __name__ == "__main__":
    main()
