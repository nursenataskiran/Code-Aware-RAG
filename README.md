# 🚀 Code-Aware RAG

A retrieval-augmented generation system that answers natural-language questions about machine learning and data science GitHub repositories by understanding their source code, notebooks, and documentation at a structural level.

Unlike generic RAG pipelines that treat code as flat text, this system uses **AST-based chunking** for Python, **heading-based splitting** for Markdown, and **cell-aware parsing** for Jupyter notebooks — so the retriever works with the same logical units a developer thinks in: classes, methods, sections, and notebook cells.

---

## 🧠 Why This Project Matters

LLMs are good at general programming questions, but they struggle with:

- *“What does this specific function do in this project?”*  
- *“How does the full pipeline work across multiple files?”*

The main challenge is that:

- Code is **structured**, not flat text  
- Important meaning is hidden in **functions, classes, and relationships**  
- Users ask in natural language, but answers live in **code identifiers**

This project focuses on bridging that gap with a **structure-aware RAG pipeline**.

---

## ⚙️ Key Features

- **AST-based chunking for Python** (functions, classes, methods)
- **Markdown & notebook-aware parsing** (sections + cell-level context)
- **Hybrid retrieval** (vector search + BM25 with RRF)
- **Semantic chunk descriptions** to improve embedding quality
- **Code-aware query expansion** (NL → code terms)
- **Evaluation pipeline** with custom retrieval metrics + RAGAS

---

## 🏗️ Architecture

```
                User Question
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Query Expansion                     │
│  (code vocabulary mapping, project detection)    │
└─────────────────────┬───────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ┌─────────────┐       ┌──────────────┐
   │ Vector Search│       │ BM25 Search  │
   │  (ChromaDB)  │       │ (rank_bm25)  │
   └──────┬──────┘       └──────┬───────┘
          │                     │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ Reciprocal Rank     │
          │ Fusion (RRF)        │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ Cross-Encoder       │
          │ Re-ranking          │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ Context Builder     │
          │ (metadata + text)   │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ LLM Generation      │
          │ (OpenRouter)        │
          └──────────┬──────────┘
                     ▼
                  Answer
```
---
## 📂 Project Structure
The project is organized into modular components for chunking, retrieval, generation, and evaluation.
```bash
src/
 ├── chunking/        # AST, Markdown, Notebook-based chunking
 ├── embedding/       # Vector store construction (ChromaDB)
 ├── retrieval/       # Hybrid search, BM25, reranking
 ├── generation/      # RAG pipeline & context building
 ├── evaluation/      # Metrics & evaluation pipeline
 ├── llm/             # OpenRouter client

data/
 ├── raw/             # Source repositories
 ├── processed/       # Chunked data (JSON)
 ├── vector_db/       # ChromaDB storage

eval_reports/         # Evaluation results (JSON / CSV)
```
---
## 🔄 How the System Works

### Chunking

The system uses a **structure-aware chunking strategy** instead of naive text splitting:

| File type | Strategy |
|:----------|:---------|
| `.py` | AST-based parsing → functions, classes, methods |
| `.md` | Split by headings (section-based) |
| `.ipynb` | Cell-based parsing with section context |

All chunks:
- are filtered if too short  
- split if too large (with overlap)  
- include metadata (file, symbol, line numbers)

---

### Embedding & Vector Store

- **Embedding model**: `sentence-transformers/all-mpnet-base-v2`  
- **Vector DB**: ChromaDB (persistent)

Chunks are embedded using a combination of:
- raw text  
- semantic description  

---

### Retrieval

The system uses a **hybrid retrieval approach**:

1. Query expansion (NL → code terms)
2. Vector search (semantic)
3. BM25 search (keyword-based)
4. Reciprocal Rank Fusion (RRF)
5. Optional re-ranking (cross-encoder)

A simpler **vector-only mode** is also available.

---

### Context Building

Retrieved chunks are formatted with metadata (file, symbol, lines) before being sent to the LLM, helping it produce grounded and traceable answers.

---

### Generation

- **LLM**: OpenRouter (configurable)
- Uses retrieved context only
- Mentions file & function names
- Low temperature (0.2) for consistency

---

### Evaluation

The system evaluates both retrieval and generation quality.

**Retrieval (deterministic, no LLM required)**

| Metric | Description |
|:-------|:------------|
| Hit Rate | At least one relevant chunk retrieved |
| Context Precision | Fraction of retrieved chunks that are relevant |
| Context Recall | Fraction of reference contexts retrieved |
| MRR | Rank of the first relevant chunk |

Matching is based on token overlap (≥40% of reference tokens present in a chunk).

---

**Generation (RAGAS, using `gpt-4o-mini`)**

| Metric | Description |
|:-------|:------------|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Answer Correctness | Does the answer match the ground truth? |

---

Each evaluation run produces versioned reports (JSON + CSV) for tracking improvements over time.

## 📈 Iterative Improvements

The system was developed through multiple iterations, focusing on improving both retrieval and generation quality.

---

### v1.0 → Initial Baseline

- Basic chunking  
- Vector search only  
- No query expansion or re-ranking  
- **LLM:** `deepseek/deepseek-chat-v3.1`  

**Results:**
- Hit Rate: ~0.22  
- Context Recall: ~0.16  
- MRR: ~0.15  

---

### v1.1 → Chunking & Embedding Improvements

- Removed near-empty chunks  
- Improved code chunk quality (class/method-level)  
- Added filtering for short chunks  
- Improved embedding inputs (more descriptive chunks)  
- **LLM:** `deepseek/deepseek-chat-v3.1`  

**Results:**
- Hit Rate: **0.56**
- Context Precision: **0.18**  
- Context Recall: **0.50**  
- MRR: **0.49**  


- Faithfulness: **0.71**  
- Answer Relevancy: **0.50**  
- Answer Correctness: **0.59**  

---

### v1.2 → Hybrid Retrieval (Current)

- Introduced **hybrid search** (vector + BM25)  
- Added **query expansion** (NL → code terms)  
- Implemented **Reciprocal Rank Fusion (RRF)**  
- Added optional **cross-encoder re-ranking**  
- Tested multiple LLMs and selected best-performing setup  
  - **Evaluation LLM:** `openai/gpt-4o-mini`  

**Results:**
- Hit Rate: **0.83**
- Context Precision: **0.19**  
- Context Recall: **0.72**  
- MRR: **0.46**  


- Faithfulness: **0.82**  
- Answer Relevancy: **0.73**  
- Answer Correctness: **0.68** 

---

### Key Takeaways

- Retrieval quality had the biggest impact on overall performance  
- Pure vector search was not sufficient for code understanding  
- Hybrid retrieval significantly improved recall  
- LLM choice mattered, but only after fixing retrieval

---
## 🚧 Limitations

- Evaluated on a relatively small corpus (limited number of projects and chunks)
- Context precision is still low (~0.19), meaning some irrelevant chunks are retrieved
- Query expansion is rule-based and not learned from data
- No production interface (API or UI) yet

---

## 🔮 Future Work

- Add a lightweight API layer for programmatic access
- Build a simple UI for interactive querying
- Improve retrieval precision with better filtering / ranking strategies
