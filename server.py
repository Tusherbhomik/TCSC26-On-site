"""
server.py
---------
FastAPI server exposing the RAG pipeline.

Endpoints
  GET  /health            — liveness check
  POST /query             — answer a question using RAG + OpenRouter LLM
  POST /retrieve          — raw semantic retrieval (no LLM)
  GET  /stats             — DB summary stats

Start:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import retrieve, build_vector_store, DB_PATH

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")   # repo-root .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL          = "google/gemini-3.1-flash-lite-preview"   # fast, free-tier on OpenRouter

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="arXiv RAG API",
    description="Semantic retrieval and question answering over arXiv papers.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: ensure vector store is ready ─────────────────────────────────────
@app.on_event("startup")
def startup_event():
    print("Checking vector store ...")
    build_vector_store(force=False)
    print("Server ready.")


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response models
# ══════════════════════════════════════════════════════════════════════════════
class QueryRequest(BaseModel):
    question:        str
    category_filter: Optional[str] = None
    year_filter:     Optional[int] = None
    n_results:       int = 5


class RetrieveRequest(BaseModel):
    query:           str
    category_filter: Optional[str] = None
    year_filter:     Optional[int] = None
    n_results:       int = 5


class SourceDoc(BaseModel):
    chunk_text:   str
    arxiv_id:     str
    title:        str
    category:     str
    year:         int
    pub_status:   str
    first_author: str
    distance:     float


class QueryResponse(BaseModel):
    question_id:     Optional[str] = None
    question:        str
    answer:          str
    sources:         list[SourceDoc]
    model_used:      str
    category_filter: Optional[str]
    year_filter:     Optional[int]


# ══════════════════════════════════════════════════════════════════════════════
# LLM call via OpenRouter
# ══════════════════════════════════════════════════════════════════════════════
def _call_llm(question: str, context_chunks: list[dict]) -> str:
    """Send question + retrieved context to OpenRouter and return the answer."""
    if not OPENROUTER_API_KEY:
        # Graceful fallback: return best-matching chunk summary
        if context_chunks:
            return (
                f"[LLM unavailable — top match] "
                f"{context_chunks[0]['title']}: {context_chunks[0]['chunk_text'][:300]}"
            )
        return "No relevant papers found."

    context = "\n\n".join(
        f"[{i+1}] {c['title']} ({c['category']}, {c['year']}, {c['pub_status']})\n"
        f"Author: {c['first_author']}\n{c['chunk_text']}"
        for i, c in enumerate(context_chunks)
    )
    system_prompt = (
        "You are a research assistant with expertise in AI and machine learning. "
        "Answer the user's question using ONLY the provided arXiv paper excerpts. "
        "Be concise (2-4 sentences). If the context does not contain enough information "
        "to answer confidently, say so."
    )
    user_prompt = (
        f"Context papers:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        resp = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "http://localhost:8000",
                "X-Title":       "arXiv RAG API",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.2,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"LLM call failed ({exc}). Top result: {context_chunks[0]['title'] if context_chunks else 'none'}"


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL}


@app.get("/stats")
def stats():
    """Return summary counts from the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total   = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            cats    = conn.execute(
                "SELECT primary_category, COUNT(*) FROM papers GROUP BY primary_category"
            ).fetchall()
            yr_min  = conn.execute("SELECT MIN(submitted_year) FROM papers").fetchone()[0]
            yr_max  = conn.execute("SELECT MAX(submitted_year) FROM papers").fetchone()[0]
        return {
            "total_papers":      total,
            "year_range":        [yr_min, yr_max],
            "papers_by_category": {c: n for c, n in cats},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrieve", response_model=list[SourceDoc])
def retrieve_endpoint(req: RetrieveRequest):
    """Raw semantic retrieval — returns ranked chunks without LLM."""
    try:
        results = retrieve(
            query=req.query,
            n_results=req.n_results,
            category_filter=req.category_filter,
            year_filter=req.year_filter,
        )
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Full RAG: retrieve relevant chunks then generate an answer via LLM."""
    try:
        sources = retrieve(
            query=req.question,
            n_results=req.n_results,
            category_filter=req.category_filter,
            year_filter=req.year_filter,
        )
        answer = _call_llm(req.question, sources)
        return QueryResponse(
            question=req.question,
            answer=answer,
            sources=[SourceDoc(**s) for s in sources],
            model_used=LLM_MODEL,
            category_filter=req.category_filter,
            year_filter=req.year_filter,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
