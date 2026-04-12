-- ═══════════════════════════════════════════════════════════════════════════
-- clean.sql  —  Quality checks & cleaning for arxiv.db
-- Usage:  sqlite3 data/arxiv.db < clean.sql
-- ═══════════════════════════════════════════════════════════════════════════

-- ── 1. Verify all five analytics tables exist ────────────────────────────────
SELECT
    CASE WHEN COUNT(*) = 5
         THEN '✓ All 5 tables present'
         ELSE '✗ MISSING tables — found ' || COUNT(*) || '/5'
    END AS table_existence_check
FROM sqlite_master
WHERE type = 'table'
  AND name IN (
        'papers',
        'category_stats',
        'yearly_trends',
        'publication_status',
        'author_stats'
  );

-- ── 2. Remove rows with NULL or empty titles from papers ─────────────────────
DELETE FROM papers
WHERE title IS NULL OR TRIM(title) = '';

-- ── 3. Remove duplicate arxiv_id rows — keep the lowest rowid ───────────────
DELETE FROM papers
WHERE rowid NOT IN (
    SELECT MIN(rowid)
    FROM papers
    GROUP BY arxiv_id
);

-- ── 4. Remove rows missing any mandatory derived field ───────────────────────
DELETE FROM papers
WHERE abstract_word_count IS NULL
   OR author_count        IS NULL
   OR first_author        IS NULL  OR TRIM(first_author)   = ''
   OR submitted_year      IS NULL
   OR subject_area        IS NULL  OR TRIM(subject_area)   = ''
   OR pub_status          IS NULL  OR TRIM(pub_status)     = '';

-- ── 5. Rebuild category_stats from cleaned papers ───────────────────────────
DELETE FROM category_stats;
INSERT INTO category_stats (category, total_papers, published_count, published_rate_pct)
SELECT
    primary_category                                                        AS category,
    COUNT(*)                                                                AS total_papers,
    SUM(CASE WHEN pub_status = 'Published' THEN 1 ELSE 0 END)              AS published_count,
    ROUND(
        SUM(CASE WHEN pub_status = 'Published' THEN 1.0 ELSE 0 END)
        / COUNT(*) * 100, 2
    )                                                                       AS published_rate_pct
FROM papers
GROUP BY primary_category;

-- ── 6. Rebuild yearly_trends from cleaned papers ─────────────────────────────
DELETE FROM yearly_trends;
INSERT INTO yearly_trends (year, category, paper_count)
SELECT
    submitted_year   AS year,
    primary_category AS category,
    COUNT(*)         AS paper_count
FROM papers
WHERE submitted_year IS NOT NULL
GROUP BY submitted_year, primary_category
ORDER BY submitted_year, primary_category;

-- ── 7. Rebuild publication_status from cleaned papers ───────────────────────
DELETE FROM publication_status;
INSERT INTO publication_status (pub_status, category, paper_count)
SELECT
    pub_status,
    primary_category AS category,
    COUNT(*)         AS paper_count
FROM papers
GROUP BY pub_status, primary_category
ORDER BY primary_category, pub_status;

-- ── 8. Rebuild author_stats from cleaned papers ──────────────────────────────
DELETE FROM author_stats;
INSERT INTO author_stats (author, paper_count, first_year, last_year, top_category)
WITH author_cat_counts AS (
    SELECT
        first_author,
        primary_category,
        COUNT(*) AS cat_count
    FROM papers
    WHERE first_author IS NOT NULL
    GROUP BY first_author, primary_category
),
top_cats AS (
    SELECT first_author, primary_category AS top_category
    FROM (
        SELECT
            first_author,
            primary_category,
            ROW_NUMBER() OVER (
                PARTITION BY first_author
                ORDER BY cat_count DESC
            ) AS rn
        FROM author_cat_counts
    )
    WHERE rn = 1
)
SELECT
    p.first_author        AS author,
    COUNT(*)              AS paper_count,
    MIN(p.submitted_year) AS first_year,
    MAX(p.submitted_year) AS last_year,
    t.top_category
FROM papers p
LEFT JOIN top_cats t ON p.first_author = t.first_author
WHERE p.first_author IS NOT NULL
GROUP BY p.first_author, t.top_category;

-- ── 9. Final assertion report — all violations should be 0 ──────────────────
SELECT check_name, violations,
       CASE WHEN violations = 0 THEN '✓ PASS' ELSE '✗ FAIL' END AS status
FROM (
    SELECT 'papers: null/empty title'         AS check_name,
           COUNT(*)                            AS violations
    FROM papers WHERE title IS NULL OR TRIM(title) = ''

    UNION ALL
    SELECT 'papers: duplicate arxiv_id',
           COUNT(*) - COUNT(DISTINCT arxiv_id)
    FROM papers

    UNION ALL
    SELECT 'papers: null abstract_word_count',
           COUNT(*) FROM papers WHERE abstract_word_count IS NULL

    UNION ALL
    SELECT 'papers: null author_count',
           COUNT(*) FROM papers WHERE author_count IS NULL

    UNION ALL
    SELECT 'papers: null/empty first_author',
           COUNT(*) FROM papers WHERE first_author IS NULL OR TRIM(first_author) = ''

    UNION ALL
    SELECT 'papers: null submitted_year',
           COUNT(*) FROM papers WHERE submitted_year IS NULL

    UNION ALL
    SELECT 'papers: null/empty subject_area',
           COUNT(*) FROM papers WHERE subject_area IS NULL OR TRIM(subject_area) = ''

    UNION ALL
    SELECT 'papers: null/empty pub_status',
           COUNT(*) FROM papers WHERE pub_status IS NULL OR TRIM(pub_status) = ''
);
