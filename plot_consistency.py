#!/usr/bin/env python3
"""
Visualize the table-difficulty consistency finding for the deck/report.

Reads per_table_accuracy.csv (written by consistency_analysis.py) and produces:
  - consistency_scatter.png : per-table accuracy, CSV vs SQL-schema, with the
    y=x line and Pearson r. Points hugging the diagonal = formats agree on which
    tables are hard -> difficulty is format-independent.
  - consistency_heatmap.png : the full cross-format correlation matrix.

Run on the login node:  python3 plot_consistency.py
"""
import csv, itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/eval_results"
CSV = f"{RES}/per_table_accuracy.csv"
FORMATS = ["csv", "json", "keyvalue", "markdown", "sql_schema"]

rows = list(csv.DictReader(open(CSV)))
acc = {f: [float(r[f]) for r in rows] for f in FORMATS}

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs)**0.5; vy = sum((y-my)**2 for y in ys)**0.5
    return cov/(vx*vy) if vx and vy else float("nan")

# --- scatter: CSV vs SQL-schema ---
x, y = acc["csv"], acc["sql_schema"]
r = pearson(x, y)
plt.figure(figsize=(5.5, 5.5))
plt.scatter(x, y, s=14, alpha=0.45, edgecolor="none")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
plt.xlabel("Per-table accuracy — CSV")
plt.ylabel("Per-table accuracy — SQL-schema")
plt.title(f"Same tables, same difficulty across formats (r = {r:.2f})")
plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{RES}/consistency_scatter.png", dpi=150)
print(f"wrote {RES}/consistency_scatter.png  (r={r:.3f})")

# --- heatmap: cross-format correlation matrix ---
mat = [[pearson(acc[a], acc[b]) for b in FORMATS] for a in FORMATS]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(FORMATS))); ax.set_xticklabels(FORMATS, rotation=45, ha="right")
ax.set_yticks(range(len(FORMATS))); ax.set_yticklabels(FORMATS)
for i, j in itertools.product(range(len(FORMATS)), repeat=2):
    ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center",
            color="white" if mat[i][j] < 0.6 else "black", fontsize=9)
ax.set_title("Cross-format per-table accuracy correlation")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{RES}/consistency_heatmap.png", dpi=150)
print(f"wrote {RES}/consistency_heatmap.png")
