"""
RAG Evaluation Pipeline
=======================
Nursena Taşkıran - v1.2 Hybrid Search

Evaluates a code-aware RAG system that answers questions about
GitHub ML projects using code and documentation.

Approach:
  - Retrieval metrics → Custom (deterministic, no LLM required)
  - Generation metrics → RAGAS (OpenRouter / gpt-4o-mini)
"""
import json
import time
import datetime
import pandas as pd
from pathlib import Path

from src.generation.rag_pipeline import RAGPipeline, run_rag_pipeline
from src.retrieval.retriever import RetrievalResult
import src.config  # noqa: F401 — ensures load_dotenv() runs


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — OPENROUTER YAPILANDIRMASI
# ══════════════════════════════════════════════════════════════════════════════

def get_ragas_llm():
    """
    RAGAS için OpenRouter üzerinden gpt-4o-mini bağlantısı.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    import os
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=1024,       # RAGAS metrics need short answers; caps cost per request
        max_retries=3,         # retry on transient failures
        request_timeout=60,    # seconds
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    """
    RAGAS Answer Relevancy metriği için embedding modeli.
    OpenRouter embedding desteklemediği için HuggingFace kullanıyoruz (ücretsiz).
    OpenAI embedding key'in varsa aşağıdaki alternatifi kullanabilirsin.
    """
    # Alternatif — OpenAI embedding varsa:
    # from langchain_openai import OpenAIEmbeddings
    # from ragas.embeddings import LangchainEmbeddingsWrapper
    # return LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(emb)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — TEST SETİ YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════

def load_testset(path: str) -> list[dict]:
    """
    testset_v2_merged.jsonl dosyasını yükler.
    Beklenen alanlar: user_input, reference, reference_contexts
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"✅ {len(rows)} soru yüklendi: {path}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — CUSTOM RETRİEVAL METRİKLERİ
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize_for_matching(text: str) -> set[str]:
    """Simple tokenizer for matching: lowercase, split on non-alphanumeric."""
    import re
    return set(re.findall(r'[a-z0-9_]+', text.lower()))


def _is_match(chunk_text: str, reference_text: str, threshold: float = 0.4) -> bool:
    """
    Chunk ile reference context eşleşiyor mu?

    Token-overlap approach: reference text'in token'larının
    yüzde kaçı chunk text'te bulunuyor?

    Bu yöntem fingerprint'e göre daha dayanıklıdır çünkü:
    - Sıra farklılıklarına toleranslıdır
    - RAGAS'ın reformatlama/kırpmasına dayanıklıdır
    - Kısmi eşleşmeleri de yakalar
    """
    chunk_tokens = _tokenize_for_matching(chunk_text)
    ref_tokens = _tokenize_for_matching(reference_text)

    if not ref_tokens:
        return False

    # What fraction of reference tokens appear in the chunk?
    overlap = len(chunk_tokens & ref_tokens)
    recall = overlap / len(ref_tokens)

    return recall >= threshold


def compute_hit_rate(
    results: list[RetrievalResult],
    reference_contexts: list[str],
) -> float:
    """
    Hit Rate @ k: retrieve edilen chunk'lar arasında en az bir
    reference context var mı? (0.0 veya 1.0)
    """
    for ref in reference_contexts:
        for r in results:
            if _is_match(r.text, ref):
                return 1.0
    return 0.0


def compute_context_precision(
    results: list[RetrievalResult],
    reference_contexts: list[str],
) -> float:
    """
    Context Precision @ k: retrieve edilenlerin kaçı relevant?
    relevant = reference_contexts'ten en az biriyle eşleşiyor
    """
    if not results:
        return 0.0
    relevant = sum(
        1 for r in results
        if any(_is_match(r.text, ref) for ref in reference_contexts)
    )
    return relevant / len(results)


def compute_context_recall(
    results: list[RetrievalResult],
    reference_contexts: list[str],
) -> float:
    """
    Context Recall @ k: reference context'lerin kaçı retrieve edildi?
    """
    if not reference_contexts:
        return 1.0
    found = sum(
        1 for ref in reference_contexts
        if any(_is_match(r.text, ref) for r in results)
    )
    return found / len(reference_contexts)


def compute_mrr(
    results: list[RetrievalResult],
    reference_contexts: list[str],
) -> float:
    """
    Mean Reciprocal Rank: ilk doğru chunk'ın sıralamadaki 1/rank değeri.
    results zaten ChromaRetriever'dan distance'a göre sıralı geliyor.
    """
    for rank, r in enumerate(results, start=1):
        if any(_is_match(r.text, ref) for ref in reference_contexts):
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    results: list[RetrievalResult],
    reference_contexts: list[str],
) -> dict:
    return {
        "hit_rate": compute_hit_rate(results, reference_contexts),
        "context_precision": compute_context_precision(results, reference_contexts),
        "context_recall": compute_context_recall(results, reference_contexts),
        "mrr": compute_mrr(results, reference_contexts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — RAGAS GENERATION METRİKLERİ
# ══════════════════════════════════════════════════════════════════════════════

def compute_generation_metrics_ragas(
    questions: list[str],
    answers: list[str],
    retrieved_contexts: list[list[str]],
    ground_truths: list[str],
) -> pd.DataFrame:
    """
    RAGAS üzerinden Faithfulness, Answer Relevancy, Answer Correctness hesaplar.
    retrieved_contexts: her soru için List[str] — chunk metinleri
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        answer_correctness,
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths,
    })

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    answer_correctness.llm = llm

    print("🤖 RAGAS generation metrikleri hesaplanıyor (OpenRouter / gpt-4o-mini)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=llm,
        embeddings=embeddings, 
    )
    return result.to_pandas()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — ANA EVALUATİON DÖNGÜSÜ
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    testset_path: str,
    version: str = "v1.0_baseline",
    output_dir: str = "eval_reports",
    k: int = 10,
    chunk_types: list[str] | None = None,
) -> dict:
    """
    Tam evaluation pipeline'ı çalıştırır ve versiyonlu rapor üretir.

    Returns:
        Özet metrik dict'i
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    testset = load_testset(testset_path)

    metric_cols = [
        "hit_rate", "context_precision", "context_recall", "mrr",
        "faithfulness", "answer_relevancy", "answer_correctness",
    ]

    # ── Her soru için RAG pipeline'ını çalıştır ────────────────────────────────
    print(f"\n🔍 RAG pipeline çalıştırılıyor ({len(testset)} soru, k={k})...")
    rows = []

    # Pipeline'ı bir kez oluştur, tüm sorularda tekrar kullan
    pipeline = RAGPipeline(use_hybrid=True)

    for i, item in enumerate(testset):
        question       = item["user_input"]
        reference_contexts = item["reference_contexts"]
        ground_truth   = item["reference"]

        print(f"  [{i+1:2}/{len(testset)}] {question[:72]}...")

        # Pipeline instance'ı tekrar kullanarak model yükleme süresinden tasarruf
        rag_out = pipeline.query(
            question=question,
            k=k,
            chunk_types=chunk_types,
        )

        results: list[RetrievalResult] = rag_out["results"]
        answer: str = rag_out["answer"]

        # Retrieval metrikleri — RetrievalResult listesi direkt geçiyor
        retrieval_scores = compute_retrieval_metrics(results, reference_contexts)

        rows.append({
            "question":           question,
            "ground_truth":       ground_truth,
            "answer":             answer,
            # RAGAS için chunk metinleri list[str] olarak
            "retrieved_texts":    [r.text for r in results],
            "reference_contexts": reference_contexts,
            "synthesizer":        item.get("synthesizer_name", "unknown"),
            **retrieval_scores,
        })

        time.sleep(0.3)  # rate limit önlemi

    df = pd.DataFrame(rows)

    # ── RAGAS generation metrikleri ────────────────────────────────────────────
    gen_df = compute_generation_metrics_ragas(
        questions=df["question"].tolist(),
        answers=df["answer"].tolist(),
        retrieved_contexts=df["retrieved_texts"].tolist(),
        ground_truths=df["ground_truth"].tolist(),
    )

    for col in ["faithfulness", "answer_relevancy", "answer_correctness"]:
        if col in gen_df.columns:
            df[col] = gen_df[col].values

    # ── Özet ──────────────────────────────────────────────────────────────────
    summary = {
        col: round(float(df[col].mean()), 4)
        for col in metric_cols
        if col in df.columns
    }

    # ── Versiyonlu rapor kaydet ────────────────────────────────────────────────
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    report_path = Path(output_dir) / f"eval_{version}_{timestamp}.json"
    csv_path    = Path(output_dir) / f"eval_{version}_{timestamp}.csv"

    report = {
        "version":       version,
        "timestamp":     timestamp,
        "testset_path":  testset_path,
        "k":             k,
        "chunk_types":   chunk_types,
        "num_questions": len(testset),
        "summary":       summary,
        "per_question":  df[[
            "question", "answer", "ground_truth", "synthesizer",
            *[c for c in metric_cols if c in df.columns],
        ]].to_dict(orient="records"),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    df[[
        "question",
        *[c for c in metric_cols if c in df.columns],
        "synthesizer",
    ]].to_csv(csv_path, index=False)

    # ── Konsola yazdır ─────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  RAG Evaluation — {version}")
    print(f"{'═'*55}")
    print(f"  Soru sayısı       : {len(testset)}  |  k = {k}")
    print(f"{'─'*55}")
    print(f"  RETRIEVAL METRİKLERİ")
    print(f"  Hit Rate          : {summary.get('hit_rate',          'N/A'):.4f}")
    print(f"  Context Precision : {summary.get('context_precision', 'N/A'):.4f}")
    print(f"  Context Recall    : {summary.get('context_recall',    'N/A'):.4f}")
    print(f"  MRR               : {summary.get('mrr',               'N/A'):.4f}")
    print(f"{'─'*55}")
    print(f"  GENERATION METRİKLERİ  (RAGAS / gpt-4o-mini)")
    print(f"  Faithfulness      : {summary.get('faithfulness',      'N/A'):.4f}")
    print(f"  Answer Relevancy  : {summary.get('answer_relevancy',  'N/A'):.4f}")
    print(f"  Answer Correctness: {summary.get('answer_correctness','N/A'):.4f}")
    print(f"{'═'*55}")
    print(f"  📄 JSON : {report_path}")
    print(f"  📊 CSV  : {csv_path}\n")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — VERSİYONLAR ARASI KARŞILAŞTIRMA
# ══════════════════════════════════════════════════════════════════════════════

def compare_versions(report_dir: str = "eval_reports") -> pd.DataFrame | None:
    """
    eval_reports/ klasöründeki tüm JSON raporları yükler,
    versiyon bazında karşılaştırma tablosu üretir.

    Thesis için: v1.0 → v1.1 → v1.2 iterasyonlarını yan yana gösterir.
    """
    report_files = sorted(Path(report_dir).glob("eval_*.json"))
    if not report_files:
        print("Karşılaştırılacak rapor bulunamadı.")
        return None

    rows = []
    for path in report_files:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({
            "version":   r["version"],
            "timestamp": r["timestamp"],
            "k":         r.get("k", "-"),
            **r["summary"],
        })

    df = pd.DataFrame(rows)
    print("\n📊 Versiyon Karşılaştırması:")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ÇALIŞTIRICI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Baseline çalıştırma:
    #   python src/evaluation/evaluator.py

    run_evaluation(
        testset_path="data/evaluation/ragas_testset_v3.jsonl",
        version="v1.2_hybrid_search",
        output_dir="eval_reports",
        k=5,
        chunk_types=None,   # tüm chunk tipleri — sonradan filtreyebilirsin
    )

    # Birden fazla versiyon çalıştırıldıktan sonra karşılaştır:
    # compare_versions("eval_reports")