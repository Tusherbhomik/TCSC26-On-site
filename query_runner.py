"""
query_runner.py
---------------
Full RAG flow:
  1. Embed question via OpenRouter  (rag_pipeline._embed)
  2. Search ChromaDB vector store   (rag_pipeline.retrieve)
  3. Apply category / year filters
  4. Send retrieved chunks + question to Qwen LLM
  5. Write answers.json

Usage:
  python query_runner.py
  python query_runner.py --questions Test/questions.json --out answers.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

# ── RAG pipeline (semantic retrieval) ─────────────────────────────────────────
from rag_pipeline import retrieve

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL          = "google/gemini-3.1-flash-lite-preview"

# ── Defaults ───────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
DEFAULT_QUESTIONS = BASE_DIR / "Test" / "questions.json"
DEFAULT_OUTPUT    = BASE_DIR / "answers.json"

N_RESULTS   = 5    # chunks retrieved per question
RETRY_DELAY = 3.0


# ══════════════════════════════════════════════════════════════════════════════
# LLM answer generation  (Qwen3.5 via OpenRouter)
# ══════════════════════════════════════════════════════════════════════════════
def _call_llm(question: str, chunks: list[dict]) -> str:
    """
    Build a context from retrieved chunks and generate a grounded answer.
    Handles Qwen3.5's thinking-mode where content may be None and the
    actual answer is in the 'reasoning' field.
    """
    if not OPENROUTER_API_KEY:
        return "LLM unavailable — OPENROUTER key not set."

    context = "\n\n".join(
        f"[{i+1}] {c['title']} | {c['category']} | {c['year']} | {c['pub_status']}\n"
        f"Author: {c['first_author']}\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are a research analyst specialising in AI and machine learning literature. "
        "Answer the question using ONLY the provided arXiv paper excerpts. "
        "Write 3-5 focused sentences. "
        "Use precise technical terminology — method names, dataset names, metric names, "
        "algorithm names — exactly as they appear in the context. "
        "Do not invent information not in the context."
    )
    user_prompt = (
        f"Paper excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (use specific technical terms from the papers):"
    )

    for attempt in range(1, 5):
        try:
            resp = httpx.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "http://localhost:8000",
                    "X-Title":       "arXiv RAG Query Runner",
                },
                json={
                    "model":       LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "max_tokens":  700,
                    "temperature": 0.2,
                },
                timeout=60.0,
            )

            if resp.status_code == 429:
                print(f"  Rate limited — waiting {RETRY_DELAY * attempt}s ...")
                time.sleep(RETRY_DELAY * attempt)
                continue

            resp.raise_for_status()
            data   = resp.json()
            msg    = (data.get("choices") or [{}])[0].get("message") or {}
            content   = msg.get("content") or ""
            reasoning = msg.get("reasoning") or ""
            answer = (content or reasoning).strip()
            return answer if answer else "No answer returned by model."

        except Exception as exc:
            if attempt == 4:
                return f"LLM error: {exc}"
            time.sleep(RETRY_DELAY)

    return "LLM failed after retries."


# ══════════════════════════════════════════════════════════════════════════════
# Per-question processing  —  full RAG flow
# ══════════════════════════════════════════════════════════════════════════════
def process_question(q: dict) -> dict:
    qid      = str(q.get("id", q.get("question_id", "?")))
    question = q.get("question", "")
    cat_f    = q.get("category_filter") or None
    yr_raw   = q.get("year_filter")
    yr_f     = int(yr_raw) if yr_raw is not None else None

    # ── Step 1: Semantic retrieval from ChromaDB ───────────────────────────────
    chunks = retrieve(
        query=question,
        n_results=N_RESULTS,
        category_filter=cat_f,
        year_filter=yr_f,
    )

    # ── Step 2: LLM generates answer from retrieved chunks ────────────────────
    answer = _call_llm(question, chunks)

    # Sources (exclude chunk_text to keep answers.json readable)
    sources = [
        {k: v for k, v in c.items() if k != "chunk_text"}
        for c in chunks
    ]

    return {
        "question_id":     qid,
        "question":        question,
        "answer":          answer,
        "sources":         sources,
        "model_used":      LLM_MODEL,
        "category_filter": cat_f,
        "year_filter":     yr_f,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate answers.json using RAG pipeline (vector search + LLM)"
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out",       type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.questions.exists():
        print(f"ERROR: questions file not found: {args.questions}")
        raise SystemExit(1)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    print(f"Loaded   : {len(questions)} questions from {args.questions}")
    print(f"Retrieval: ChromaDB semantic search (rag_pipeline.retrieve)")
    print(f"LLM      : {LLM_MODEL}\n")

    answers = []
    for i, q in enumerate(questions, 1):
        qid = str(q.get("id", q.get("question_id", i)))
        print(f"[{i:02d}/{len(questions)}] Q{qid}: {q['question'][:72]}...")
        result = process_question(q)
        answers.append(result)
        print(f"         → {result['answer'][:120]}...\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(answers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Done — {len(answers)} answers written to {args.out}")


if __name__ == "__main__":
    main()
