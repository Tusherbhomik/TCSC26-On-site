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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
OPENROUTER_API_KEY   = os.getenv("OPENROUTER", "")
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL          = "sentence-transformers/all-minilm-l6-v2"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DB_PATH          = BASE_DIR / "data" / "arxiv.db"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
COLLECTION_NAME    = "arxiv_papers"
CHUNK_SIZE         = 200    # words
CHUNK_OVERLAP      = 40     # words
EMBED_BATCH        = 256    # texts per API call
MAX_EMBED_WORKERS  = 1      # sequential — parallel calls trigger API auth blocks
RETRY_DELAY        = 3.0    # seconds base wait after a rate-limit (429)

ALL_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]


# ══════════════════════════════════════════════════════════════════════════════
# Embedding via OpenRouter
# ══════════════════════════════════════════════════════════════════════════════
def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using the OpenRouter embeddings API.
    Returns a list of float vectors in the same order as input.
    Thread-safe — each call uses its own httpx client.
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
                timeout=120.0,
            )

            if resp.status_code == 429:
                wait = min(RETRY_DELAY * attempt, 15.0)
                print(f"  Rate limited — waiting {wait:.0f}s (attempt {attempt})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in data]

        except (httpx.HTTPStatusError, httpx.ReadError, httpx.ConnectError) as exc:
            if attempt == 4:
                raise
            wait = min(RETRY_DELAY * attempt, 15.0)
            print(f"  Network error ({type(exc).__name__}), retrying in {wait:.0f}s ...")
            time.sleep(wait)

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
def build_vector_store(
    force: bool = False,
    only_categories: Optional[list] = None,
) -> None:
    """
    Chunk paper abstracts, embed via OpenRouter, and upsert into ChromaDB.

    Parameters
    ----------
    force            : wipe collection and rebuild from scratch.
    only_categories  : if set, only process papers from these categories
                       (e.g. ['cs.CL', 'cs.CV', 'stat.ML']).
                       Useful to fill in missing categories without
                       re-processing what's already stored.
    """
    client = _get_client()

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Existing collection deleted (force=True).")
        except Exception:
            pass

    collection = _get_collection(client)

    # ── Load papers (optionally filtered by category) ─────────────────────────
    if only_categories:
        placeholders = ",".join("?" * len(only_categories))
        query = f"""
            SELECT arxiv_id, title, abstract, primary_category,
                   submitted_year, pub_status, first_author
            FROM   papers
            WHERE  abstract IS NOT NULL AND TRIM(abstract) != ''
              AND  primary_category IN ({placeholders})
        """
        params = only_categories
        print(f"Reading papers for categories {only_categories} ...")
    else:
        query = """
            SELECT arxiv_id, title, abstract, primary_category,
                   submitted_year, pub_status, first_author
            FROM   papers
            WHERE  abstract IS NOT NULL AND TRIM(abstract) != ''
        """
        params = []
        print("Reading all papers from arxiv.db ...")

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()
    print(f"  {len(rows):,} papers loaded.")

    # ── Resume: load existing IDs in pages (avoid segfault on large sets) ────
    existing_ids: set = set()
    _offset = 0
    while True:
        page = collection.get(include=[], limit=1000, offset=_offset)
        if not page["ids"]:
            break
        existing_ids.update(page["ids"])
        _offset += len(page["ids"])
    already = len(existing_ids)
    print(f"  {already:,} chunks already stored — resuming.")

    # ── Build list of new chunks ───────────────────────────────────────────────
    all_ids, all_docs, all_meta = [], [], []
    for arxiv_id, title, abstract, category, year, pub_status, first_author in rows:
        for idx, chunk in enumerate(chunk_text(abstract)):
            chunk_id = f"{arxiv_id}__chunk{idx}"
            if chunk_id in existing_ids:
                continue
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_meta.append({
                "arxiv_id":     arxiv_id      or "",
                "title":        title         or "",
                "category":     category      or "",
                "year":         int(year)     if year else 0,
                "pub_status":   pub_status    or "",
                "first_author": first_author  or "",
                "chunk_text":   chunk,
            })

    total = len(all_ids)
    if total == 0:
        print(f"  Nothing new to add — {already:,} chunks already complete.")
        return
    print(f"  {total:,} new chunks to embed (batch={EMBED_BATCH}, workers={MAX_EMBED_WORKERS}).")

    # ── Split into batches ────────────────────────────────────────────────────
    batches = []
    for start in range(0, total, EMBED_BATCH):
        end = min(start + EMBED_BATCH, total)
        batches.append((
            all_docs[start:end],
            all_ids[start:end],
            all_meta[start:end],
        ))

    n_batches = len(batches)
    print(f"  {n_batches} batches to process ...")

    # ── Parallel embed + sequential upsert ───────────────────────────────────
    # Embedding is I/O-bound (API call) → ThreadPoolExecutor speeds it up.
    # ChromaDB upserts happen on the main thread to avoid write contention.
    upserted   = 0
    completed  = 0

    with ThreadPoolExecutor(max_workers=MAX_EMBED_WORKERS) as pool:
        future_to_batch = {
            pool.submit(_embed, docs): (docs, ids, metas)
            for docs, ids, metas in batches
        }

        for fut in as_completed(future_to_batch):
            _, ids, metas = future_to_batch[fut]
            try:
                vecs = fut.result()
            except Exception as exc:
                print(f"\n  Batch failed: {exc} — skipping {len(ids)} chunks.")
                completed += 1
                continue

            collection.upsert(
                ids=ids,
                embeddings=vecs,
                documents=[m["chunk_text"] for m in metas],
                metadatas=[{k: v for k, v in m.items() if k != "chunk_text"} for m in metas],
            )
            upserted  += len(ids)
            completed += 1
            print(
                f"  [{completed}/{n_batches}] {upserted:,}/{total:,} chunks upserted ...",
                end="\r",
            )

    print(f"\nDone. Vector store now has {collection.count():,} chunks in '{COLLECTION_NAME}'.")


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
    # Always do a full resume build — skips already-stored chunk IDs so it is
    # safe to re-run at any time and only embeds what is genuinely missing.
    print("Running full resume build (skips already-stored chunks) ...")
    build_vector_store(force=False)

    print("\nSample retrieve('deep learning image classification'):")
    for r in retrieve("deep learning image classification", n_results=3):
        print(f"  [{r['distance']:.4f}] {r['arxiv_id']} — {r['title'][:60]}")
