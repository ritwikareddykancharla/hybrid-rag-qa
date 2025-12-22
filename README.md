
# hybrid-rag-qa

Hybrid retrieval-augmented question answering pipeline combining sparse and dense retrieval with grounded LLM generation.

---

## Overview

This repository implements an end-to-end **Hybrid Retrieval-Augmented Question Answering (RAG) pipeline** designed to answer queries over heterogeneous data sources, including structured tables and unstructured documents.

The system explicitly separates **retrieval**, **reranking**, and **generation**, and surfaces retrieved evidence to improve grounding and reduce hallucinations. The architecture mirrors a production-style thin-client setup with a stateless frontend and a stable backend API.

---

## System Architecture

### Components

**Retrieval**
- Sparse retrieval using **BM25** (OpenSearch)
- Dense semantic retrieval using **Sentence-Transformers / E5**
- Vector similarity search via **FAISS**

**Reasoning & Generation**
- Retrieval and control-flow orchestration using **LangChain** and **LangGraph**
- Cross-encoder reranking to improve top-k relevance
- Grounded answer generation with explicit source attribution

**Serving**
- Backend API implemented in **FastAPI**
- Thin frontend demo (Lovable) communicating over a clean HTTP contract

### Data Flow

```text
User Query
    |
    v
+----------------------+
|    FastAPI Server    |
|     POST /query     |
+----------------------+
            |
            v
+----------------------------------+
|  Retrieval Orchestration Layer   |
|   (LangChain / LangGraph)        |
+----------------------------------+
        |                    |
        |                    |
        v                    v
+----------------+    +----------------------+
|  Sparse Search |    |  Dense Retrieval     |
|  BM25          |    |  (E5 / ST Embeddings)|
|  OpenSearch    |    +----------------------+
+----------------+              |
        |                        |
        +-----------+------------+
                    |
                    v
            +------------------+
            |   FAISS Index    |
            +------------------+
                    |
                    v
            +------------------+
            | Cross-Encoder    |
            |    Reranker      |
            +------------------+
                    |
                    v
            +------------------+
            |   LLM Generator  |
            | (Grounded QA)    |
            +------------------+
                    |
                    v
        Final Answer + Source Citations
````

---

## API Contract

### `POST /query`

**Request**

```json
{
  "question": "string"
}
```

**Response**

```json
{
  "answer": "string",
  "sources": [
    {
      "source_id": "string",
      "title": "string",
      "snippet": "string",
      "score": 0.0
    }
  ]
}
```

The API is intentionally minimal to allow easy integration with multiple clients.

---

## Evaluation

Offline evaluation is performed to measure both retrieval and generation quality:

* **Retrieval Recall@K** for sparse and dense components
* **Answer Faithfulness** by checking alignment between generated answers and retrieved evidence
* Structured error analysis to identify hallucinations, retrieval misses, and reranking failures

Evaluation scripts are designed to be run independently of the serving stack.

---

## Demo

A lightweight interactive demo is provided using **Lovable**, which acts purely as a UI layer.

* Frontend: Lovable (UI only)
* Backend: FastAPI
* Retrieval: OpenSearch (BM25) + FAISS
* Orchestration: LangChain + LangGraph

The demo explicitly displays retrieved sources alongside generated answers to make grounding behavior observable.

---

## Repository Structure

```text
hybrid-rag-qa/
├── retrieval/          # Sparse and dense retrieval modules
├── reranking/          # Cross-encoder rerankers
├── generation/         # LLM-based answer generation
├── orchestration/      # LangChain / LangGraph flows
├── api/                # FastAPI service
├── eval/               # Offline evaluation scripts
├── scripts/            # Data ingestion and indexing
├── README.md
├── LICENSE
└── .gitignore
```

---

## Design Principles

* Explicit separation of retrieval, reranking, and generation
* Evidence-first answer presentation
* Thin client, stateless frontend
* Evaluation-driven iteration
* Production-oriented system boundaries

---

## Notes

This project is intended as an applied ML systems demonstration and is structured to support extension to larger datasets, alternative retrievers, and different LLM backends.

---

## License

MIT License
