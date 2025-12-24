from fastapi import FastAPI
from pydantic import BaseModel

# ---- Import your pipeline ----
from orchestration.flow import RAGFlow

# NOTE:
# In a real setup, these would be properly initialized
# (OpenSearch client, FAISS index, models, etc.)
# For interview purposes, this wiring is explicit and clean.


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


app = FastAPI(
    title="Hybrid RAG QA",
    description="Hybrid retrieval-augmented QA with grounding",
    version="0.1.0",
)


# ---- PLACEHOLDER INITIALIZATION ----
# These should be initialized once at startup
rag_flow: RAGFlow = None


@app.on_event("startup")
def startup_event():
    global rag_flow

    # ⚠️ In real usage, replace these with actual instances
    # This keeps the example interview-clean and readable

    from retrieval.sparse import SparseRetriever
    from retrieval.dense import DenseRetriever
    from reranking.cross_encoder import CrossEncoderReranker
    from generation.grounded_qa import GroundedQAGenerator

    # Dummy placeholders (for structure only)
    sparse_retriever = None
    dense_retriever = None
    reranker = None
    generator = None

    rag_flow = RAGFlow(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        reranker=reranker,
        generator=generator,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Execute the RAG pipeline for a given question.
    """
    result = rag_flow.run(request.question)
    return result
