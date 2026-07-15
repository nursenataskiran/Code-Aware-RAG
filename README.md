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

* **AST-based chunking for Python** (functions, classes, methods)
* **Markdown & notebook-aware parsing** (sections + cell-level context)
* **Hybrid retrieval** (vector search + BM25 with RRF)
* **Semantic chunk descriptions** to improve embedding quality
* **Code-aware query expansion** (NL → code terms)
* **Optional cross-encoder re-ranking**
* **Persistent ChromaDB storage** across application restarts
* **Public GitHub repository ingestion** through the GitHub API
* **Evaluation pipeline** with custom retrieval metrics + RAGAS
* **FastAPI-based API layer** (chat endpoint, ingestion endpoint, health checks, rate limiting, structured responses)
* **Docker & Docker Compose support** for reproducible local deployment


---

## 🏗️ Architecture

```text
                    User Question
                         │
                         ▼
       ┌─────────────────────────────────┐
       │     Optional Query Expansion    │
       │   Natural language → code terms │
       └────────────────┬────────────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      ┌──────────────┐      ┌──────────────┐
      │ Vector Search│      │ BM25 Search  │
      │  ChromaDB    │      │  rank_bm25   │
      └──────┬───────┘      └──────┬───────┘
             │                     │
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │ Reciprocal Rank     │
             │ Fusion — RRF        │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Optional            │
             │ Cross-Encoder       │
             │ Re-ranking          │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Context Builder     │
             │ Metadata + text     │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ LLM Generation      │
             │ OpenRouter          │
             └──────────┬──────────┘
                        │
                        ▼
                     Answer
```

Query expansion and cross-encoder re-ranking are implemented as **optional components**. They are disabled by default in the final configuration because the simpler **ChromaDB + BM25 + RRF** pipeline produced the most balanced performance on the final evaluation set.

---
## 📂 Project Structure

The project is organized into modular components for ingestion, chunking, retrieval, generation, evaluation, and API access.

```text
api/
 ├── app.py                 # FastAPI application
 ├── routes.py              # Chat and health endpoints
 ├── ingestion_routes.py    # GitHub repository ingestion endpoint
 ├── schemas.py             # Pydantic request/response models
 ├── rate_limiter.py        # In-memory API rate limiting
 └── errors.py              # Centralized error handlers

src/
 ├── chunking/              # AST, Markdown, and notebook chunking
 ├── embedding/             # ChromaDB vector-store construction
 ├── retrieval/             # Semantic search, BM25, RRF, and reranking
 ├── generation/            # RAG pipeline, context building, and experimental LangGraph workflow
 ├── ingestion/             # GitHub downloading and repository indexing
 ├── evaluation/            # Retrieval and RAGAS evaluation pipeline
 ├── llm/                   # OpenRouter client
 └── config.py              # Paths, models, and feature flags

data/
 ├── raw/                   # Downloaded repository files
 ├── processed/             # Generated chunks
 ├── evaluation/            # Evaluation test set
 └── vector_db/             # Persistent ChromaDB storage

Dockerfile
docker-compose.yml
requirements.txt
requirements-dev.txt
eval_reports/               # Versioned evaluation outputs
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
## 🌐 API Layer

The RAG system is exposed through a lightweight **FastAPI** backend for programmatic access.

### Endpoints

* **`GET /health`** → Returns the current service status
* **`POST /api/v1/chat`** → Answers questions using the indexed repository context
* **`POST /api/v1/ingest/github`** → Downloads, chunks, and indexes a public GitHub repository

### Repository Ingestion

The ingestion endpoint accepts a public GitHub repository URL and:

* **Downloads supported source files** (`.py`, `.md`, `.ipynb`)
* **Stores files under a project-specific directory**
* **Processes files with the structure-aware chunking pipeline**
* **Adds generated chunks to the persistent ChromaDB collection**
* **Returns downloaded and skipped files in a structured response**
* **Detects already-indexed repositories** and avoids unnecessary re-downloading or re-indexing

### API Features

* **Pydantic request and response schemas**
* **Structured source metadata** in chat responses
* **Built-in rate limiting** for chat requests
* **Centralized validation and exception handling**
* **Lazy initialization of the RAG pipeline**
* **Persistent ChromaDB usage** across application restarts
* **Swagger UI documentation** available at `/docs`
* **ReDoc documentation** available at `/redoc`


## 🐳 Docker Deployment

The application can be run using **Docker** and **Docker Compose**, without installing the Python dependencies directly on the host machine.

The Docker setup includes:

* **FastAPI served with Uvicorn**
* **Environment variables loaded from a local `.env` file**
* **Port mapping** for accessing the API from the host machine
* **Persistent ChromaDB storage** through a mounted volume
* **A reproducible Python environment** based on the project dependencies


## Evaluation

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

### v1.2 → Hybrid Retrieval

- Introduced hybrid search (ChromaDB vector search + BM25)
- Combined semantic and lexical rankings using Reciprocal Rank Fusion (RRF)
- Implemented optional query expansion and cross-encoder re-ranking
- Disabled query expansion and re-ranking by default after evaluation showed better recall and generation quality with the simpler hybrid setup
- Evaluation LLM: `openai/gpt-4o-mini`

**Results:**

- Hit Rate: **0.83**
- Context Precision: **0.21**
- Context Recall: **0.72**
- MRR: **0.45**

  
- Faithfulness: **0.75**
- Answer Relevancy: **0.68**
- Answer Correctness: **0.67**
> Note: Query expansion and cross-encoder re-ranking are implemented as optional features. However, evaluation showed that the simpler hybrid setup (ChromaDB + BM25 + RRF) performs better on the current test set, so both are disabled by default.
---

---

### v1.3 → Multi-Repository Ingestion, Improved Chunking & Final Evaluation (Current)

* **Added public GitHub repository ingestion**
* **Expanded the corpus to three machine learning repositories**
* **Added persistent ChromaDB storage** to preserve indexed repositories across restarts
* **Improved Python, Markdown, and notebook chunk generation**
* **Added semantic descriptions and richer metadata to embedding inputs**
* **Added an experimental LangGraph workflow** with an LLM-based answer judge and bounded retry mechanism
* **Reviewed and updated the evaluation dataset** to match the final chunk structure
* **Added project-aware filtering for repository-specific evaluation questions**
* **Added a FastAPI backend** with chat, ingestion, health-check, rate-limiting, and structured error handling
* **Added Docker and Docker Compose support**
* **Used top-5 retrieval (`k=5`)**
* **Evaluation LLM: `openai/gpt-4o-mini`**

Results:

* **Hit Rate: 0.8696**
* **Context Precision: 0.2174**
* **Context Recall: 0.8261**
* **MRR: 0.6232**
* **Faithfulness: 0.8727**
* **Answer Relevancy: 0.8849**
* **Answer Correctness: 0.6814**

The final configuration uses **ChromaDB semantic retrieval + BM25 lexical retrieval + Reciprocal Rank Fusion**.

Query expansion and cross-encoder re-ranking remain available as optional features. However, they are disabled by default because the simpler hybrid configuration produced the most balanced retrieval and generation results on the final evaluation set.

---

### Key Takeaways

* **Retrieval quality had the biggest impact on overall performance**
* **Pure vector search was not sufficient for code understanding**
* **BM25 improved retrieval of exact identifiers, function names, and configuration values**
* **Semantic search performed better for conceptual and natural-language questions**
* **Hybrid retrieval significantly improved context coverage**
* **Structure-aware chunking produced more meaningful retrieval units than flat text splitting**
* **Accurate and up-to-date reference contexts were essential for reliable evaluation**
* **The simpler hybrid pipeline performed better than enabling every optional retrieval component**
* **Persistent storage and Docker support made the system easier to run and reuse**

---

## ⚠️ Limitations

* **The system was evaluated on a relatively small corpus** of three machine learning repositories and 23 reviewed questions
* **Context precision remains lower than context recall**, because top-k retrieval may include supporting or partially related chunks
* **Only Python, Markdown, and Jupyter Notebook files are currently supported**
* **GitHub ingestion currently supports public repositories**
* **Query expansion is rule-based** rather than learned from repository-specific data
* **The FastAPI backend is stateless** and does not maintain conversational history between requests

---

## 🔭 Future Work

* **Add authentication and private repository ingestion**
* **Evaluate the system on a larger and more diverse repository corpus**
* **Add conversational memory for multi-turn repository questions**

