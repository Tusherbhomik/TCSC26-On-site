import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH    = Path(__file__).parent / "data" / "arxiv.db"
PLOTS_DIR  = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────────────────
PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
GREY      = "#6c757d"
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

# ── Load tables ────────────────────────────────────────────────────────────────
with sqlite3.connect(DB_PATH) as conn:
    category_stats    = pd.read_sql("SELECT * FROM category_stats   ORDER BY total_papers DESC", conn)
    yearly_trends     = pd.read_sql("SELECT * FROM yearly_trends    ORDER BY year, category",    conn)
    pub_status        = pd.read_sql("SELECT * FROM publication_status ORDER BY category, pub_status", conn)
    papers            = pd.read_sql("SELECT abstract_word_count, primary_category FROM papers",  conn)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Papers per Category  (bar + publication-rate line)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 6))

cats   = category_stats["category"]
totals = category_stats["total_papers"]
rates  = category_stats["published_rate_pct"]
x      = np.arange(len(cats))
bar_w  = 0.6

bars = ax1.bar(x, totals, width=bar_w, color=PALETTE[:len(cats)], zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(cats, rotation=25, ha="right")
ax1.set_ylabel("Total Papers")
ax1.set_xlabel("Category")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=1)

# Annotate bars with counts
for bar, total in zip(bars, totals):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + totals.max() * 0.01,
        f"{total:,}", ha="center", va="bottom", fontsize=9, color="#333333",
    )

# Secondary axis: publication rate
ax2 = ax1.twinx()
ax2.plot(x, rates, color="#C44E52", marker="o", linewidth=2,
         markersize=7, label="Published rate (%)", zorder=3)
ax2.set_ylabel("Publication Rate (%)", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")
ax2.set_ylim(0, max(rates) * 1.35)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color("#C44E52")

# Legend
from matplotlib.lines import Line2D
legend_els = [
    plt.Rectangle((0, 0), 1, 1, color=PALETTE[0], label="Total Papers"),
    Line2D([0], [0], color="#C44E52", marker="o", linewidth=2, label="Published Rate (%)"),
]
ax1.legend(handles=legend_els, loc="upper right", framealpha=0.9)

ax1.set_title("Papers per Category with Publication Rate")
fig.tight_layout()
out = PLOTS_DIR / "01_papers_per_category.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Submission Trend Over Time  (multi-line, one per category)
# ══════════════════════════════════════════════════════════════════════════════

# Keep only years with at least 2 categories reporting (avoids noisy tails)
year_coverage = yearly_trends.groupby("year")["category"].count()
valid_years   = year_coverage[year_coverage >= 2].index
yt_filtered   = yearly_trends[yearly_trends["year"].isin(valid_years)]

pivot = (
    yt_filtered
    .pivot(index="year", columns="category", values="paper_count")
    .fillna(0)
    .astype(int)
)

# Order categories by total submissions (most prominent first)
cat_order = pivot.sum().sort_values(ascending=False).index.tolist()
pivot = pivot[cat_order]

fig, (ax_main, ax_growth) = plt.subplots(
    2, 1, figsize=(13, 10),
    gridspec_kw={"height_ratios": [3, 1.2]},
    sharex=True,
)

# ── Top panel: raw submission counts ──────────────────────────────────────────
for i, cat in enumerate(cat_order):
    color = PALETTE[i % len(PALETTE)]
    ax_main.plot(
        pivot.index, pivot[cat],
        marker="o", linewidth=2.2, markersize=5,
        color=color, label=cat, zorder=3,
    )
    # Annotate peak year
    peak_yr  = pivot[cat].idxmax()
    peak_val = pivot[cat].max()
    ax_main.annotate(
        f"peak {peak_yr}",
        xy=(peak_yr, peak_val),
        xytext=(4, 6), textcoords="offset points",
        fontsize=7.5, color=color, fontstyle="italic",
    )

ax_main.set_ylabel("Papers Submitted", fontsize=12)
ax_main.set_title(
    "arXiv Submission Trend Over Time by Category\n"
    "(based on sampled subset — relative trends are meaningful)",
    fontsize=14, fontweight="bold",
)
ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_main.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.45, zorder=1)
ax_main.legend(title="Category", framealpha=0.92, loc="upper left",
               fontsize=10, title_fontsize=10)

# ── Bottom panel: year-over-year % growth (total across all categories) ───────
total_per_year = pivot.sum(axis=1)
yoy_growth = total_per_year.pct_change() * 100

bars_pos = yoy_growth[yoy_growth >= 0]
bars_neg = yoy_growth[yoy_growth < 0]
ax_growth.bar(bars_pos.index, bars_pos.values, color="#55A868", alpha=0.8,
              label="Growth", zorder=2)
ax_growth.bar(bars_neg.index, bars_neg.values, color="#C44E52", alpha=0.8,
              label="Decline", zorder=2)
ax_growth.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax_growth.set_ylabel("YoY Growth (%)", fontsize=10)
ax_growth.set_xlabel("Year", fontsize=12)
ax_growth.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax_growth.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=1)
ax_growth.legend(fontsize=9, framealpha=0.9, loc="lower right")
ax_growth.set_title("Year-over-Year Growth (All Categories Combined)",
                    fontsize=11, fontweight="bold")

# Shared x-axis: integer years only
ax_growth.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.setp(ax_growth.get_xticklabels(), rotation=30, ha="right")

fig.tight_layout(h_pad=2.5)
out = PLOTS_DIR / "02_submission_trend_over_time.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Publication Status Breakdown  (stacked bar + normalised % panel)
# ══════════════════════════════════════════════════════════════════════════════
pivot_pub = (
    pub_status
    .pivot(index="category", columns="pub_status", values="paper_count")
    .fillna(0)
    .astype(int)
)
for col in ["Preprint", "Published"]:
    if col not in pivot_pub.columns:
        pivot_pub[col] = 0

pivot_pub["total"] = pivot_pub["Preprint"] + pivot_pub["Published"]
# Sort by published rate descending so the most-published category is on the left
pivot_pub["pub_rate"] = pivot_pub["Published"] / pivot_pub["total"] * 100
pivot_pub = pivot_pub.sort_values("pub_rate", ascending=False)

COLOR_PUBLISHED = "#2ecc71"   # green
COLOR_PREPRINT  = "#3498db"   # blue

fig, (ax_abs, ax_pct) = plt.subplots(
    1, 2, figsize=(14, 6),
    gridspec_kw={"width_ratios": [1.4, 1]},
)

x     = np.arange(len(pivot_pub))
bar_w = 0.55

# ── Left panel: absolute stacked counts ───────────────────────────────────────
ax_abs.bar(x, pivot_pub["Preprint"],  width=bar_w, label="Preprint",
           color=COLOR_PREPRINT,  alpha=0.88, zorder=2)
ax_abs.bar(x, pivot_pub["Published"], width=bar_w, label="Published",
           color=COLOR_PUBLISHED, alpha=0.88,
           bottom=pivot_pub["Preprint"], zorder=2)

# Annotate absolute counts inside each segment
for i, (_, row) in enumerate(pivot_pub.iterrows()):
    # Preprint count (bottom segment) — only if tall enough
    if row["Preprint"] > row["total"] * 0.06:
        ax_abs.text(i, row["Preprint"] / 2,
                    f"{int(row['Preprint']):,}",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
    # Published count (top segment)
    if row["Published"] > row["total"] * 0.06:
        ax_abs.text(i, row["Preprint"] + row["Published"] / 2,
                    f"{int(row['Published']):,}",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
    # Total label above bar
    ax_abs.text(i, row["total"] + pivot_pub["total"].max() * 0.012,
                f"{int(row['total']):,}",
                ha="center", va="bottom", fontsize=8, color="#333333")

ax_abs.set_xticks(x)
ax_abs.set_xticklabels(pivot_pub.index, rotation=30, ha="right", fontsize=10)
ax_abs.set_ylabel("Number of Papers", fontsize=11)
ax_abs.set_xlabel("Category", fontsize=11)
ax_abs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_abs.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=1)
ax_abs.set_title("Absolute Paper Counts\n(Published vs Preprint)", fontsize=12,
                 fontweight="bold")
ax_abs.legend(loc="upper right", framealpha=0.92, fontsize=10)

# ── Right panel: 100 % normalised stacked bar  ────────────────────────────────
pre_pct = pivot_pub["Preprint"]  / pivot_pub["total"] * 100
pub_pct = pivot_pub["Published"] / pivot_pub["total"] * 100

ax_pct.bar(x, pre_pct, width=bar_w, label="Preprint",
           color=COLOR_PREPRINT,  alpha=0.88, zorder=2)
ax_pct.bar(x, pub_pct, width=bar_w, label="Published",
           color=COLOR_PUBLISHED, alpha=0.88,
           bottom=pre_pct, zorder=2)

# Annotate % inside segments
for i, (pct_pre, pct_pub) in enumerate(zip(pre_pct, pub_pct)):
    if pct_pre > 5:
        ax_pct.text(i, pct_pre / 2, f"{pct_pre:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    if pct_pub > 5:
        ax_pct.text(i, pct_pre + pct_pub / 2, f"{pct_pub:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

# Reference line at 50 %
ax_pct.axhline(50, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
ax_pct.text(len(x) - 0.45, 51.5, "50 %", fontsize=8, color=GREY, style="italic")

ax_pct.set_xticks(x)
ax_pct.set_xticklabels(pivot_pub.index, rotation=30, ha="right", fontsize=10)
ax_pct.set_ylabel("Share of Papers (%)", fontsize=11)
ax_pct.set_xlabel("Category", fontsize=11)
ax_pct.set_ylim(0, 105)
ax_pct.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax_pct.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=1)
ax_pct.set_title("Publication Rate Comparison\n(100% normalised)", fontsize=12,
                 fontweight="bold")
ax_pct.legend(loc="lower right", framealpha=0.92, fontsize=10)

fig.suptitle("Publication Status Breakdown by Category", fontsize=15,
             fontweight="bold", y=1.02)
fig.tight_layout(w_pad=3)
out = PLOTS_DIR / "03_publication_status_breakdown.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Abstract Length Distribution  (box plot + overlapping histograms)
# ══════════════════════════════════════════════════════════════════════════════

# Sort categories by median word count descending (longest abstracts first)
cats_ordered = (
    papers.groupby("primary_category")["abstract_word_count"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)

# Build per-category data list (drop NaN, clip extreme outliers for readability)
P99        = papers["abstract_word_count"].quantile(0.99)
cat_data   = [
    papers.loc[
        papers["primary_category"] == cat, "abstract_word_count"
    ].dropna().clip(upper=P99).values
    for cat in cats_ordered
]
n_cats = len(cats_ordered)

fig = plt.figure(figsize=(14, 10))
# Two rows: top = box-plot comparison, bottom row = per-category histograms
gs  = fig.add_gridspec(2, n_cats, height_ratios=[1.4, 1.6], hspace=0.45, wspace=0.35)

# ── Top panel: horizontal box plots — all categories on one axis ──────────────
ax_box = fig.add_subplot(gs[0, :])

bp = ax_box.boxplot(
    cat_data,
    vert=False,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(linewidth=1.4),
    capprops=dict(linewidth=1.4),
    flierprops=dict(marker="o", markersize=2.5, alpha=0.25, linestyle="none"),
    notch=False,
)
for patch, color in zip(bp["boxes"], PALETTE[:n_cats]):
    patch.set_facecolor(color)
    patch.set_alpha(0.82)
for flier, color in zip(bp["fliers"], PALETTE[:n_cats]):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)

# Annotate median value beside each box
for i, data in enumerate(cat_data):
    med = float(np.median(data))
    ax_box.text(med + P99 * 0.012, i + 1, f"  med={med:.0f}",
                va="center", fontsize=8.5, color="#333333")

ax_box.set_yticks(range(1, n_cats + 1))
ax_box.set_yticklabels(cats_ordered, fontsize=11)
ax_box.set_xlabel("Abstract Word Count", fontsize=11)
ax_box.set_title(
    "Abstract Length by Category — Box Plot\n"
    "(whiskers = 1.5×IQR, dots = outliers, clipped at 99th percentile)",
    fontsize=12, fontweight="bold",
)
ax_box.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
ax_box.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# ── Bottom row: one histogram per category ────────────────────────────────────
for col_idx, (cat, data, color) in enumerate(zip(cats_ordered, cat_data, PALETTE)):
    ax_h = fig.add_subplot(gs[1, col_idx])

    ax_h.hist(data, bins=40, color=color, alpha=0.80,
              edgecolor="white", linewidth=0.3, zorder=2)

    med = float(np.median(data))
    mn  = float(np.mean(data))
    ax_h.axvline(med, color="black",   linestyle="--", linewidth=1.5,
                 label=f"Median\n{med:.0f}", zorder=3)
    ax_h.axvline(mn,  color="#C44E52", linestyle=":",  linewidth=1.5,
                 label=f"Mean\n{mn:.0f}", zorder=3)

    ax_h.set_title(f"{cat}\n(n={len(data):,})", fontsize=10, fontweight="bold")
    ax_h.set_xlabel("Word Count", fontsize=9)
    ax_h.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_h.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.45, zorder=1)
    ax_h.legend(fontsize=7.5, framealpha=0.88, loc="upper right",
                handlelength=1.2)

    # Label y-axis only on the leftmost subplot
    if col_idx == 0:
        ax_h.set_ylabel("Papers", fontsize=10)

    # Detect and flag unusually short/long abstracts
    short = int((data < 30).sum())
    long_ = int((data > P99 * 0.95).sum())
    note  = []
    if short > 0:
        note.append(f"{short} very short")
    if long_ > 0:
        note.append(f"{long_} very long")
    if note:
        ax_h.text(0.98, 0.97, " | ".join(note),
                  transform=ax_h.transAxes, fontsize=7, color=GREY,
                  ha="right", va="top", style="italic")

fig.suptitle(
    "Abstract Length Distribution by Category",
    fontsize=15, fontweight="bold", y=1.01,
)
out = PLOTS_DIR / "04_abstract_length_distribution.png"
fig.savefig(out, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved {out}")

print("\nAll 4 plots saved to", PLOTS_DIR)
