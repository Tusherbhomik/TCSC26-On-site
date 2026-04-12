# arXiv Paper Analysis

A full data product over a sampled arXiv metadata dataset: ingestion pipeline,
SQL quality checks, visualisations, a RAG API server, and a benchmark answer
runner using semantic retrieval + LLM generation.

---

## Repository Structure

```
.
├── ingest.py           # Step 1 — load, filter, build all DB tables
├── clean.sql           # Step 2 — SQL quality checks & cleaning
├── visualize.py        # Step 3 — generate all four plots
├── rag_pipeline.py     # Step 4 — embed abstracts, build ChromaDB vector store, retrieve()
├── server.py           # Step 5 — FastAPI RAG server (4 endpoints)
├── query_runner.py     # Step 6 — benchmark answer generation → answers.json
├── ingest.ipynb        # Cell-by-cell verification notebook (mirrors ingest.py)
├── answers.json        # Generated answers for all 22 benchmark questions
├── data/
│   ├── kaggle_arxiv.csv    # Source dataset (100,000 rows) — gitignored
│   ├── papers_raw.json     # Exported raw records (JSON) — gitignored
│   └── arxiv.db            # SQLite database (all tables) — gitignored
├── vector_store/           # ChromaDB persistent store — gitignored (rebuild with rag_pipeline.py)
├── plots/
│   ├── 01_papers_per_category.png
│   ├── 02_submission_trend_over_time.png
│   ├── 03_publication_status_breakdown.png
│   └── 04_abstract_length_distribution.png
└── Test/
    └── questions.json      # 22 benchmark questions with grading criteria
```

---

## Requirements

```bash
pip install pandas matplotlib numpy httpx chromadb python-dotenv fastapi uvicorn pydantic
```

SQLite3 is part of the Python standard library — no extra install needed.

Create a `.env` file in the project root:

```
OPENROUTER=sk-or-v1-...
```

The OpenRouter key is used for both embeddings (`sentence-transformers/all-minilm-l6-v2`)
and LLM generation (`google/gemini-3.1-flash-lite-preview`).

---

## How to Run (Step-by-Step)

Run all commands from the project root.

---

### Step 1 — Ingest & build the database

```bash
python ingest.py
```

**What it does:**

1. Reads `data/kaggle_arxiv.csv` (100,000 rows, 13 columns).
2. Filters rows to keep only papers with at least one of the five
   supported categories: `cs.AI`, `cs.LG`, `cs.CL`, `stat.ML`, `cs.CV`.
3. Builds six tables and writes them to `data/arxiv.db`:

| Table | Rows | Description |
|---|---|---|
| `raw_papers` | ~100k | Direct mapping from the CSV |
| `papers` | ~100k | Enriched table with all derived fields |
| `category_stats` | 5 | Aggregated totals & publication rate per category |
| `yearly_trends` | varies | Paper counts per (year, category) pair |
| `publication_status` | 10 | Published vs Preprint count per category |
| `author_stats` | varies | One row per first author with activity summary |

4. Also writes `data/papers_raw.json` (JSON array, one object per paper).

**Expected output (tail):**

```
Saved 100,000 rows to table 'raw_papers'
Saved 100,000 rows to table 'papers'
Saved 5 rows to table 'category_stats'
...
Ingestion complete.
```

---

### Step 2 — SQL quality checks & cleaning

```bash
sqlite3 data/arxiv.db < clean.sql
```

**What it does:**

1. Verifies all five analytics tables exist.
2. Removes rows from `papers` where:
   - `title` is NULL or empty
   - `arxiv_id` is duplicated (keeps lowest rowid)
   - Any mandatory derived field is NULL or empty
3. Rebuilds `category_stats`, `yearly_trends`, `publication_status`, and
   `author_stats` from the cleaned `papers` table.
4. Prints a final assertion report — every row must show `✓ PASS` with
   `violations = 0`.

**Expected assertion report:**

```
check_name                          violations  status
----------------------------------  ----------  ------
papers: null/empty title            0           ✓ PASS
papers: duplicate arxiv_id          0           ✓ PASS
papers: null abstract_word_count    0           ✓ PASS
papers: null author_count           0           ✓ PASS
papers: null/empty first_author     0           ✓ PASS
papers: null submitted_year         0           ✓ PASS
papers: null/empty subject_area     0           ✓ PASS
papers: null/empty pub_status       0           ✓ PASS
```

---

### Step 3 — Generate visualisations

```bash
python visualize.py
```

Reads directly from `data/arxiv.db` and saves four PNG files to `plots/`.

---

### Step 4 — Build the vector store

```bash
python rag_pipeline.py
```

**What it does:**

1. Loads all papers with non-empty abstracts from `data/arxiv.db`.
2. Chunks each abstract into 200-word windows with 40-word overlap.
3. Embeds chunks in batches of 256 via OpenRouter
   (`sentence-transformers/all-minilm-l6-v2`, 384-dim vectors).
4. Upserts into a ChromaDB persistent collection (`arxiv_papers`) stored in
   `vector_store/` using cosine similarity.
5. **Resume-capable** — auto-detects missing categories and skips chunks
   already stored, so interrupted builds can be safely restarted.

---

### Step 5 — Start the API server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The server auto-builds the vector store on startup if needed (resume mode).

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns model name |
| `GET` | `/stats` | DB summary: total papers, year range, counts by category |
| `POST` | `/retrieve` | Semantic retrieval — returns ranked chunks, no LLM |
| `POST` | `/query` | Full RAG: retrieve + LLM answer generation |

**Example `/query` request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What regularization methods reduce overfitting?", "n_results": 5}'
```

Optional filters:

```json
{
  "question": "...",
  "category_filter": "cs.LG",
  "year_filter": 2022,
  "n_results": 5
}
```

---

### Step 6 — Generate benchmark answers

```bash
python query_runner.py
```

Reads `Test/questions.json` (22 benchmark questions), runs the full RAG
pipeline for each (semantic retrieval from ChromaDB + LLM via OpenRouter),
and writes results to `answers.json`.

**LLM:** `google/gemini-3.1-flash-lite-preview` via OpenRouter  
**Retrieval:** ChromaDB cosine similarity (`rag_pipeline.retrieve`)

Each entry in `answers.json`:

```json
{
  "question_id": "1",
  "question": "...",
  "answer": "...",
  "sources": [
    {
      "arxiv_id": "...",
      "title": "...",
      "category": "cs.LG",
      "year": 2022,
      "pub_status": "Published",
      "first_author": "...",
      "distance": 0.123
    }
  ],
  "model_used": "google/gemini-3.1-flash-lite-preview",
  "category_filter": "cs.LG",
  "year_filter": null
}
```

Override defaults with flags:

```bash
python query_runner.py --questions Test/questions.json --out answers.json
```

---

## Database Schema

### `raw_papers`
| Column | Type | Notes |
|---|---|---|
| arxiv_id | TEXT | arXiv paper ID |
| title | TEXT | Paper title |
| abstract | TEXT | Full abstract |
| authors | TEXT | Raw author string |
| categories | TEXT | Full category string (space-separated) |
| primary_category | TEXT | First token of categories |
| submitted_date | TEXT | ISO date (YYYY-MM-DD) |
| update_date | TEXT | ISO date (YYYY-MM-DD) |
| journal_ref | TEXT | Journal reference (nullable) |
| doi | TEXT | DOI (nullable) |
| comment | TEXT | Author comment (nullable) |

### `papers`  *(primary analytics table)*
| Column | Type | Notes |
|---|---|---|
| arxiv_id | TEXT | arXiv paper ID |
| title | TEXT | Paper title |
| abstract | TEXT | Full abstract |
| authors | TEXT | Raw author string |
| primary_category | TEXT | First listed category |
| submitted | TEXT | ISO submission date |
| abstract_word_count | INTEGER | Word count of abstract |
| author_count | INTEGER | Number of authors (comma-split) |
| first_author | TEXT | First name in author string |
| submitted_year | INTEGER | Year extracted from submitted date |
| subject_area | TEXT | Broad field mapped from category prefix |
| pub_status | TEXT | `Published` (has DOI or journal ref) / `Preprint` |

### `category_stats`
| Column | Type | Notes |
|---|---|---|
| category | TEXT | primary_category value |
| total_papers | INTEGER | Total papers in category |
| published_count | INTEGER | Papers with pub_status = Published |
| published_rate_pct | REAL | (published_count / total_papers) × 100 |

### `yearly_trends`
| Column | Type | Notes |
|---|---|---|
| year | INTEGER | submitted_year |
| category | TEXT | primary_category |
| paper_count | INTEGER | Papers for that (year, category) pair |

### `publication_status`
| Column | Type | Notes |
|---|---|---|
| pub_status | TEXT | `Published` or `Preprint` |
| category | TEXT | primary_category |
| paper_count | INTEGER | Papers for that (pub_status, category) pair |

### `author_stats`
| Column | Type | Notes |
|---|---|---|
| author | TEXT | first_author value |
| paper_count | INTEGER | Papers as first author |
| first_year | INTEGER | Earliest submitted_year |
| last_year | INTEGER | Latest submitted_year |
| top_category | TEXT | Category where author appears most as first author |

---

## Visualisations

### 01 — Papers per Category
**File:** `plots/01_papers_per_category.png`
- Bar chart of total papers per category
- Dual axis: red line overlays the publication rate (%) per category
- Bars annotated with exact counts

### 02 — Submission Trend Over Time
**File:** `plots/02_submission_trend_over_time.png`
- Top panel: multi-line time series (one line per category, year on x-axis)
- Peak year annotated on each line
- Bottom panel: year-over-year % growth bar chart (green = growth, red = decline)

### 03 — Publication Status Breakdown
**File:** `plots/03_publication_status_breakdown.png`
- Left panel: absolute stacked bar (Preprint + Published counts per category)
- Right panel: 100% normalised stacked bar (publication rate comparison)
- 50% reference line on the normalised panel
- Categories sorted by publication rate (highest first)

### 04 — Abstract Length Distribution
**File:** `plots/04_abstract_length_distribution.png`
- Top panel: horizontal box plots for all categories (IQR, whiskers, outlier dots)
- Bottom row: individual histogram per category with median and mean lines

---

## RAG Pipeline Architecture

```
kaggle_arxiv.csv
      │
      ▼
  ingest.py ──────────────► data/arxiv.db (SQLite, 6 tables)
                                  │
                                  ▼
                          rag_pipeline.py
                          ├── chunk abstracts (200 words / 40 overlap)
                          ├── embed via OpenRouter (all-minilm-l6-v2)
                          └── upsert → vector_store/ (ChromaDB)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              server.py                  query_runner.py
           (FastAPI REST API)          (batch answer generation)
           POST /query                  reads Test/questions.json
           POST /retrieve               writes answers.json
           GET  /stats
           GET  /health
```

---

## Supported Categories

| Category | Field |
|---|---|
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.CL` | Computation & Language (NLP) |
| `stat.ML` | Statistical Machine Learning |
| `cs.CV` | Computer Vision |

---

## Derived Field Logic

| Field | Logic |
|---|---|
| `primary_category` | First whitespace-token in the `categories` string |
| `abstract_word_count` | `len(abstract.split())` |
| `author_count` | Count of comma-separated tokens in `authors` |
| `first_author` | First comma-separated token in `authors` |
| `submitted_year` | `.year` from parsed `submitted` datetime |
| `subject_area` | Prefix map: `cs.*` → Computer Science, `stat.*` → Statistics |
| `pub_status` | `Published` if `doi` or `journal_ref` is non-null/non-empty, else `Preprint` |

---

## Verification Notebook

`ingest.ipynb` mirrors `ingest.py` cell-by-cell for interactive inspection:

```bash
jupyter notebook ingest.ipynb
```
