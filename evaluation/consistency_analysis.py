#!/usr/bin/env python3
"""
Table-difficulty consistency analysis for the tabular-format ablation.

Question: are the SAME tables consistently easy/hard across seeds and across
formats? If table difficulty is shared, the near-chance accuracy reflects the
evaluation/data (truncation, hard distractors) rather than the model or the
serialization format.

Outputs:
  - printed cross-seed stability (per format)
  - printed cross-format correlation matrix
  - the consistently hardest / easiest tables
  - per_table_accuracy.csv  (table_id x format, averaged over seeds)

Run on the login node (no GPU):  python3 consistency_analysis.py

NOTE on coarseness: each table has ~5 pairs per seed, so a single-seed per-table
accuracy can only take values 0, .2, .4, .6, .8, 1.0. Correlations are therefore
attenuated; the difficulty estimate pools BOTH seeds (~10 pairs/table) to reduce
this. Read the cross-format correlation as the main signal.
"""
import json, collections, statistics as st, csv

RES = "/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/eval_results"
FORMATS = ["csv", "json", "keyvalue", "markdown", "sql_schema"]
SEEDS   = ["123", "2024"]

def load(fmt, seed):
    p = f"{RES}/results_{fmt}_seed{seed}_matched.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()]

def table_acc(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["table_id"]].append(bool(r["correct"]))
    return {t: sum(v) / len(v) for t, v in d.items()}

def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs)**0.5
    vy = sum((y-my)**2 for y in ys)**0.5
    return cov/(vx*vy) if vx > 0 and vy > 0 else float("nan")

# per (format, seed) table accuracy
acc_fs = {(f, s): table_acc(load(f, s)) for f in FORMATS for s in SEEDS}

# 1. cross-seed stability within each format
print("=" * 60)
print("CROSS-SEED STABILITY  (does a table's accuracy persist across seeds?)")
print("  r near 0  -> per-table result is mostly noise")
print("  r > ~0.3  -> table difficulty is a real, repeatable property")
print("=" * 60)
print(f"{'format':12} {'pearson r':>10}")
for f in FORMATS:
    a, b = acc_fs[(f, "123")], acc_fs[(f, "2024")]
    common = sorted(set(a) & set(b))
    print(f"{f:12} {pearson([a[t] for t in common], [b[t] for t in common]):>10.3f}")

# mean over seeds, per format
acc_f = {}
for f in FORMATS:
    a, b = acc_fs[(f, "123")], acc_fs[(f, "2024")]
    common = set(a) & set(b)
    acc_f[f] = {t: (a[t] + b[t]) / 2 for t in common}

common = sorted(set.intersection(*[set(acc_f[f]) for f in FORMATS]))
print(f"\n{len(common)} tables common to all formats.")

# 2. cross-format correlation matrix
print("\n" + "=" * 60)
print("CROSS-FORMAT CORRELATION  (do formats agree on which tables are hard?)")
print("  high values -> all formats struggle on the SAME tables")
print("              -> points to the eval/data, not the format")
print("=" * 60)
print(" " * 12 + "".join(f"{f[:8]:>9}" for f in FORMATS))
for f1 in FORMATS:
    row = [pearson([acc_f[f1][t] for t in common], [acc_f[f2][t] for t in common]) for f2 in FORMATS]
    print(f"{f1:12}" + "".join(f"{x:>9.3f}" for x in row))

# 3. average difficulty across formats; extremes
avg = {t: st.mean(acc_f[f][t] for f in FORMATS) for t in common}
ranked = sorted(common, key=lambda t: avg[t])
print("\n10 consistently HARDEST tables (mean acc across all formats):")
for t in ranked[:10]:
    print(f"  {avg[t]:.2f}  {t[:70]}")
print("\n10 consistently EASIEST tables:")
for t in ranked[-10:][::-1]:
    print(f"  {avg[t]:.2f}  {t[:70]}")

hard_all = sum(1 for t in common if all(acc_f[f][t] < 0.4 for f in FORMATS))
easy_all = sum(1 for t in common if all(acc_f[f][t] > 0.6 for f in FORMATS))
print(f"\nHard for ALL formats (<40%): {hard_all}/{len(common)} "
      f"({100*hard_all/len(common):.0f}%)")
print(f"Easy for ALL formats (>60%): {easy_all}/{len(common)} "
      f"({100*easy_all/len(common):.0f}%)")

# 4. CSV for sharing
out = f"{RES}/per_table_accuracy.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["table_id"] + FORMATS + ["mean_all_formats"])
    for t in ranked:
        w.writerow([t] + [f"{acc_f[fm][t]:.3f}" for fm in FORMATS] + [f"{avg[t]:.3f}"])
print(f"\nWrote {out}")
