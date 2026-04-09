import pandas as pd
import sqlite3
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
CSV_PATH   = DATA_DIR / "kaggle_arxiv.csv"
JSON_PATH  = DATA_DIR / "papers_raw.json"
DB_PATH    = DATA_DIR / "arxiv.db"

# ── Supported categories ───────────────────────────────────────────────────────
SUPPORTED_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "stat.ML", "cs.CV"}

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Raw shape: {df.shape}")

# ── Filter rows that contain at least one supported category ───────────────────
def has_supported_category(cat_string):
    if pd.isna(cat_string):
        return False
    cats = set(cat_string.split())
    return bool(cats & SUPPORTED_CATEGORIES)

mask = df["categories"].apply(has_supported_category)
df = df[mask].reset_index(drop=True)
print(f"Filtered shape (supported categories only): {df.shape}")

# ── Build raw_papers table ─────────────────────────────────────────────────────
raw_papers = pd.DataFrame({
    "arxiv_id":         df["id"],
    "title":            df["title"],
    "abstract":         df["abstract"],
    "authors":          df["authors"],
    "categories":       df["categories"],
    "primary_category": df["categories"].str.split().str[0],
    "submitted_date":   pd.to_datetime(df["submitted"],   errors="coerce"),
    "update_date":      pd.to_datetime(df["update_date"], errors="coerce"),
    "journal_ref":      df["journal-ref"],
    "doi":              df["doi"],
    "comment":          df["comments"],
})

print(raw_papers.shape)
print(raw_papers.dtypes)
print(raw_papers.head())

# ── Export → papers_raw.json ───────────────────────────────────────────────────
print(f"\nWriting {JSON_PATH} ...")
records = raw_papers.copy()
records["submitted_date"] = records["submitted_date"].dt.strftime("%Y-%m-%d").where(
    records["submitted_date"].notna(), None
)
records["update_date"] = records["update_date"].dt.strftime("%Y-%m-%d").where(
    records["update_date"].notna(), None
)
records.to_json(JSON_PATH, orient="records", indent=2, force_ascii=False)
print(f"Saved {len(records):,} records to {JSON_PATH}")

# ── Export → arxiv.db (table: raw_papers) ─────────────────────────────────────
print(f"\nWriting {DB_PATH} ...")
db_df = raw_papers.copy()
db_df["submitted_date"] = db_df["submitted_date"].dt.strftime("%Y-%m-%d").where(
    db_df["submitted_date"].notna(), None
)
db_df["update_date"] = db_df["update_date"].dt.strftime("%Y-%m-%d").where(
    db_df["update_date"].notna(), None
)

with sqlite3.connect(DB_PATH) as conn:
    db_df.to_sql("raw_papers", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM raw_papers").fetchone()[0]

print(f"Saved {count:,} rows to table 'raw_papers' in {DB_PATH}")

# ── Helpers for papers table ───────────────────────────────────────────────────
def get_first_author(author_string):
    if pd.isna(author_string):
        return None
    return author_string.split(",")[0].strip()

def count_authors(author_string):
    if pd.isna(author_string):
        return 0
    return len([a for a in author_string.split(",") if a.strip()])

def get_subject_area(primary_category):
    if pd.isna(primary_category):
        return None
    prefix = primary_category.split(".")[0].lower()
    mapping = {
        "cs":      "Computer Science",
        "stat":    "Statistics",
        "math":    "Mathematics",
        "eess":    "Electrical Engineering",
        "econ":    "Economics",
        "q-bio":   "Quantitative Biology",
        "q-fin":   "Quantitative Finance",
        "physics": "Physics",
    }
    return mapping.get(prefix, prefix.upper())

def get_pub_status(row):
    has_doi     = pd.notna(row["doi"])        and str(row["doi"]).strip()        != ""
    has_journal = pd.notna(row["journal_ref"]) and str(row["journal_ref"]).strip() != ""
    return "Published" if (has_doi or has_journal) else "Preprint"

# ── Build papers table ─────────────────────────────────────────────────────────
print("\nBuilding papers table...")
papers = pd.DataFrame({
    "arxiv_id":            raw_papers["arxiv_id"],
    "title":               raw_papers["title"],
    "abstract":            raw_papers["abstract"],
    "authors":             raw_papers["authors"],
    "primary_category":    raw_papers["primary_category"],
    "submitted":           raw_papers["submitted_date"],
    "abstract_word_count": raw_papers["abstract"].fillna("").apply(lambda x: len(x.split())),
    "author_count":        raw_papers["authors"].apply(count_authors),
    "first_author":        raw_papers["authors"].apply(get_first_author),
    "submitted_year":      raw_papers["submitted_date"].dt.year.astype("Int64"),
    "subject_area":        raw_papers["primary_category"].apply(get_subject_area),
    "pub_status":          raw_papers[["doi", "journal_ref"]].apply(get_pub_status, axis=1),
})

print(papers.shape)
print(papers.dtypes)

# ── Export → arxiv.db (table: papers) ─────────────────────────────────────────
print(f"\nWriting papers table to {DB_PATH} ...")
db_papers = papers.copy()
db_papers["submitted"] = db_papers["submitted"].dt.strftime("%Y-%m-%d").where(
    db_papers["submitted"].notna(), None
)
db_papers["submitted_year"] = db_papers["submitted_year"].astype(object).where(
    db_papers["submitted_year"].notna(), None
)

with sqlite3.connect(DB_PATH) as conn:
    db_papers.to_sql("papers", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

print(f"Saved {count:,} rows to table 'papers' in {DB_PATH}")

# ── Build category_stats table ─────────────────────────────────────────────────
print("\nBuilding category_stats table...")
category_stats = (
    papers.groupby("primary_category", as_index=False)
    .agg(
        total_papers=("arxiv_id", "count"),
        published_count=("pub_status", lambda x: (x == "Published").sum()),
    )
    .rename(columns={"primary_category": "category"})
)
category_stats["published_rate_pct"] = (
    (category_stats["published_count"] / category_stats["total_papers"]) * 100
).round(2)

print(category_stats)

with sqlite3.connect(DB_PATH) as conn:
    category_stats.to_sql("category_stats", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM category_stats").fetchone()[0]

print(f"Saved {count:,} rows to table 'category_stats' in {DB_PATH}")

# ── Build yearly_trends table ──────────────────────────────────────────────────
print("\nBuilding yearly_trends table...")
yearly_trends = (
    papers.dropna(subset=["submitted_year"])
    .groupby(["submitted_year", "primary_category"], as_index=False)
    .agg(paper_count=("arxiv_id", "count"))
    .rename(columns={"submitted_year": "year", "primary_category": "category"})
    .sort_values(["year", "category"])
    .reset_index(drop=True)
)
yearly_trends["year"] = yearly_trends["year"].astype(int)

print(yearly_trends.head(10))

with sqlite3.connect(DB_PATH) as conn:
    yearly_trends.to_sql("yearly_trends", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM yearly_trends").fetchone()[0]

print(f"Saved {count:,} rows to table 'yearly_trends' in {DB_PATH}")

# ── Build publication_status table ────────────────────────────────────────────
print("\nBuilding publication_status table...")
publication_status = (
    papers.groupby(["pub_status", "primary_category"], as_index=False)
    .agg(paper_count=("arxiv_id", "count"))
    .rename(columns={"primary_category": "category"})
    .sort_values(["category", "pub_status"])
    .reset_index(drop=True)
)

print(publication_status)

with sqlite3.connect(DB_PATH) as conn:
    publication_status.to_sql("publication_status", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM publication_status").fetchone()[0]

print(f"Saved {count:,} rows to table 'publication_status' in {DB_PATH}")

# ── Build author_stats table ──────────────────────────────────────────────────
print("\nBuilding author_stats table...")
pa = papers.dropna(subset=["first_author"])

base = (
    pa.groupby("first_author", as_index=False)
    .agg(
        paper_count=("arxiv_id", "count"),
        first_year=("submitted_year", "min"),
        last_year=("submitted_year", "max"),
    )
)

top_cat = (
    pa.groupby(["first_author", "primary_category"])
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
    .drop_duplicates(subset="first_author")
    [["first_author", "primary_category"]]
    .rename(columns={"primary_category": "top_category"})
)

author_stats = (
    base.merge(top_cat, on="first_author", how="left")
    .rename(columns={"first_author": "author"})
    .sort_values("paper_count", ascending=False)
    .reset_index(drop=True)
)
author_stats["first_year"] = author_stats["first_year"].astype("Int64")
author_stats["last_year"]  = author_stats["last_year"].astype("Int64")

print(author_stats.head(10))

db_author = author_stats.copy()
db_author["first_year"] = db_author["first_year"].astype(object).where(db_author["first_year"].notna(), None)
db_author["last_year"]  = db_author["last_year"].astype(object).where(db_author["last_year"].notna(), None)

with sqlite3.connect(DB_PATH) as conn:
    db_author.to_sql("author_stats", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM author_stats").fetchone()[0]

print(f"Saved {count:,} rows to table 'author_stats' in {DB_PATH}")
print("\nIngestion complete.")
