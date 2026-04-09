"""
rag_pipeline.py
---------------
Responsibilities
  1. Load papers from data/arxiv.db
  2. Chunk abstracts (200 words, 40-word overlap)
  3. Embed via OpenRouter API  (sentence-transformers/all-minilm-l6-v2)
  4. Persist vectors into vector_store/ via ChromaDB
  5. Expose retrieve(query, ...) for the API server

Requires env var:  OPENROUTER = sk-or-v1-...
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import httpx
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER", "")
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL         = "sentence-transformers/all-minilm-l6-v2"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DB_PATH          = BASE_DIR / "data" / "arxiv.db"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "arxiv_papers"
CHUNK_SIZE      = 200    # words
CHUNK_OVERLAP   = 40     # words
EMBED_BATCH     = 64     # texts per API call (stay within rate limits)
RETRY_DELAY     = 3.0    # seconds to wait after a rate-limit (429)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding via OpenRouter
# ══════════════════════════════════════════════════════════════════════════════
def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using the OpenRouter embeddings API.
    Returns a list of float vectors in the same order as input.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER env var not set. Check your .env file.")

    for attempt in range(1, 5):
        try:
            resp = httpx.post(
                OPENROUTER_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "http://localhost:8000",
                    "X-Title":       "arXiv RAG Pipeline",
                },
                json={
                    "model":           EMBED_MODEL,
                    "input":           texts,
                    "encoding_format": "float",
                },
                timeout=60.0,
            )

            if resp.status_code == 429:
                wait = RETRY_DELAY * attempt
                print(f"  Rate limited — waiting {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()["data"]
            # sort by index to guarantee ordering
            data.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in data]

        except (httpx.HTTPStatusError, httpx.ReadError, httpx.ConnectError) as exc:
            if attempt == 4:
                raise
            print(f"  Network error ({type(exc).__name__}), retrying in {RETRY_DELAY * attempt}s ...")
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Embedding API failed after 4 attempts.")


# ══════════════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-count windows."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += size - overlap
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB client
# ══════════════════════════════════════════════════════════════════════════════
def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Build / refresh vector store
# ══════════════════════════════════════════════════════════════════════════════
def build_vector_store(force: bool = False) -> None:
    """
    Chunk all paper abstracts, embed via OpenRouter, and upsert into ChromaDB.
    - force=False : resume mode — skips chunks already stored, adds new ones.
    - force=True  : wipe collection and rebuild from scratch.
    """
    client     = _get_client()

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Existing collection deleted (force=True).")
        except Exception:
            pass

    collection = _get_collection(client)

    print("Reading papers from arxiv.db ...")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT arxiv_id, title, abstract, primary_category,
                   submitted_year, pub_status, first_author
            FROM   papers
            WHERE  abstract IS NOT NULL AND TRIM(abstract) != ''
        """).fetchall()
    print(f"  {len(rows):,} papers loaded.")

    # ── Resume: skip chunks already stored ────────────────────────────────────
    existing_ids = set(collection.get(include=[])["ids"])
    already = len(existing_ids)
    print(f"  {already:,} chunks already stored — resuming from where we left off.")

    all_ids, all_docs, all_meta = [], [], []
    for arxiv_id, title, abstract, category, year, pub_status, first_author in rows:
        for idx, chunk in enumerate(chunk_text(abstract)):
            chunk_id = f"{arxiv_id}__chunk{idx}"
            if chunk_id in existing_ids:
                continue
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_meta.append({
                "arxiv_id":    arxiv_id     or "",
                "title":       title        or "",
                "category":    category     or "",
                "year":        int(year)    if year else 0,
                "pub_status":  pub_status   or "",
                "first_author": first_author or "",
            })

    total = len(all_ids)
    if total == 0:
        print(f"  Vector store is complete — {already:,} chunks, nothing to add.")
        return
    print(f"  {total:,} new chunks to embed.")

    # ── Embed + upsert in batches ──────────────────────────────────────────────
    upserted = 0
    for start in range(0, total, EMBED_BATCH):
        end       = min(start + EMBED_BATCH, total)
        batch_ids  = all_ids[start:end]
        batch_docs = all_docs[start:end]
        batch_meta = all_meta[start:end]

        vecs = _embed(batch_docs)
        collection.upsert(
            ids=batch_ids,
            embeddings=vecs,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        upserted += len(batch_ids)
        print(f"  Progress: {upserted:,}/{total:,} chunks upserted ...", end="\r")

    print(f"\nVector store built: {collection.count():,} chunks in '{COLLECTION_NAME}'.")


# ══════════════════════════════════════════════════════════════════════════════
# Retrieve
# ══════════════════════════════════════════════════════════════════════════════
def retrieve(
    query: str,
    n_results: int = 5,
    category_filter: Optional[str] = None,
    year_filter: Optional[int] = None,
) -> list[dict]:
    """
    Semantic search over the ChromaDB vector store.

    Parameters
    ----------
    query           : natural-language question or keyword string
    n_results       : number of chunks to return
    category_filter : restrict to a specific primary_category (e.g. 'cs.AI')
    year_filter     : restrict to a specific submitted_year (e.g. 2023)

    Returns
    -------
    List of dicts:
        chunk_text, arxiv_id, title, category, year,
        pub_status, first_author, distance
    """
    client     = _get_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run build_vector_store() first.")

    # ── ChromaDB where-filter ──────────────────────────────────────────────────
    where: Optional[dict] = None
    if category_filter and year_filter:
        where = {"$and": [
            {"category": {"$eq": category_filter}},
            {"year":     {"$eq": int(year_filter)}},
        ]}
    elif category_filter:
        where = {"category": {"$eq": category_filter}}
    elif year_filter:
        where = {"year": {"$eq": int(year_filter)}}

    query_vec = _embed([query])

    results = collection.query(
        query_embeddings=query_vec,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for chunk, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "chunk_text":   chunk,
            "arxiv_id":     meta.get("arxiv_id",     ""),
            "title":        meta.get("title",         ""),
            "category":     meta.get("category",      ""),
            "year":         meta.get("year",          0),
            "pub_status":   meta.get("pub_status",    ""),
            "first_author": meta.get("first_author",  ""),
            "distance":     round(float(dist), 6),
        })

    return output


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_vector_store(force=False)
    print("\nSample retrieve('deep learning image classification'):")
    for r in retrieve("deep learning image classification", n_results=3):
        print(f"  [{r['distance']:.4f}] {r['arxiv_id']} — {r['title'][:60]}")
