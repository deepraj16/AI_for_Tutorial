"""
FastAPI backend for the MSETCL RAG Chatbot
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ──────────────────────────────────────────
# APP INIT
# ──────────────────────────────────────────
app = FastAPI(
    title="MSETCL RAG API",
    description="RAG-powered Q&A over MSETCL technical manuals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────
# LOAD RAG PIPELINE ONCE AT STARTUP
# ──────────────────────────────────────────
from rag_graph import ask, chunks, metadata, faiss_index

TOTAL_CHUNKS  = len(chunks)
TOTAL_VECTORS = faiss_index.ntotal

# ──────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5          # reserved for future dynamic TOP_K support


class SourceDoc(BaseModel):
    text: str
    source: str
    filename: str
    page: str | int
    score: float
    is_neighbor: bool
    chunk_idx: int


class AnswerResponse(BaseModel):
    answer: str
    retrieved_docs: List[SourceDoc]


class StatsResponse(BaseModel):
    total_chunks: int
    total_vectors: int
    documents: List[dict]


# ──────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "MSETCL RAG API is running."}


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
def get_stats():
    """Return knowledge-base statistics shown in the UI sidebar."""
    documents = [
        {"name": "Transmission Lines Manual",      "pages": 288, "kind": "Digital"},
        {"name": "EHV Sub-station Manual Vol I",   "pages": 325, "kind": "Digital"},
        {"name": "Maintenance Procedure Manual",   "pages": 227, "kind": "Scanned"},
        {"name": "EHS Manual",                     "pages": 98,  "kind": "Scanned"},
        {"name": "Equipment Testing Manual Vol I", "pages": 390, "kind": "Scanned"},
    ]
    return StatsResponse(
        total_chunks=TOTAL_CHUNKS,
        total_vectors=TOTAL_VECTORS,
        documents=documents,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
def ask_question(body: QuestionRequest):
    """
    Run the full RAG pipeline for the given question.

    Returns the generated answer and all retrieved source chunks
    (both directly matched and neighboring context chunks).
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = ask(body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}")

    return AnswerResponse(
        answer=result["answer"],
        retrieved_docs=[SourceDoc(**doc) for doc in result["retrieved_docs"]],
    )


@app.get("/suggestions", tags=["Info"])
def get_suggestions():
    """Return example questions shown on the landing screen."""
    return {
        "suggestions": [
            "What is the daily log sheet procedure for EHV lines?",
            "Explain the principle of operation of a power transformer.",
            "What are the PPE requirements for MSETCL employees?",
            "What is the process for issuing a Permit to Work?",
            "How is the earth mat designed for a substation?",
            "What tests are performed during transformer pre-commissioning?",
        ]
    }


# ──────────────────────────────────────────
# DEV ENTRYPOINT
# ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)