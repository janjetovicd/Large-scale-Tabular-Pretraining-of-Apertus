#!/usr/bin/env python3
"""
Fair aggregate of the two-seed, matched-checkpoint ablation eval.

Reports, for every format x seed and the cross-seed mean:
  - per-pair accuracy        (fraction of statement/distractor pairs correct)
  - per-table all-correct     (fraction of tables where ALL pairs were correct)
  - per-table majority-correct(fraction of tables where >half of pairs correct)

FAIRNESS: sql_schema failed to serialize 12 tables (table_serialized = null),
so it is scored on fewer pairs than the other four formats. To compare like for
like, every format is restricted to the set of (table_id, statement) pairs that
are present in ALL five formats for that seed (the "common set"). This removes
the 12-table denominator mismatch entirely.

Run on the cluster login node (no GPU needed):
    python3 aggregate_eval_fair.py
"""
import json, collections, statistics as st, os

RES = "/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/eval_results"
FORMATS = ["csv", "json", "keyvalue", "markdown", "sql_schema"]
SEEDS   = ["123", "2024"]

def load(fmt, seed):
    p = f"{RES}/results_{fmt}_seed{seed}_matched.jsonl"
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return [json.loads(l) for l in open(p) if l.strip()]

def key(r):
    return (r["table_id"], r["statement"])

def metrics(rows):
    """per-pair, per-table all-correct, per-table majority-correct."""
    per_pair = sum(r["correct"] for r in rows) / len(rows)
    by_tbl = collections.defaultdict(list)
    for r in rows:
        by_tbl[r["table_id"]].append(bool(r["correct"]))
    n = len(by_tbl)
    t_all = sum(1 for v in by_tbl.values() if all(v)) / n
    t_maj = sum(1 for v in by_tbl.values() if sum(v) > len(v) / 2) / n
    return per_pair, t_all, t_maj, len(rows), n

agg = collections.defaultdict(lambda: collections.defaultdict(list))

print("=" * 74)
print("PER-SEED (common table set across all 5 formats)")
print("=" * 74)
print(f"{'format':12} {'seed':>5} {'pairs':>6} {'tables':>6} "
      f"{'per-pair':>9} {'tbl-all':>8} {'tbl-maj':>8}")

for seed in SEEDS:
    data = {fmt: {key(r): r for r in load(fmt, seed)} for fmt in FORMATS}
    common = set.intersection(*[set(d) for d in data.values()])
    for fmt in FORMATS:
        rows = [data[fmt][k] for k in common]
        pp, ta, tm, npairs, ntbl = metrics(rows)
        print(f"{fmt:12} {seed:>5} {npairs:>6} {ntbl:>6} "
              f"{pp:>9.3f} {ta:>8.3f} {tm:>8.3f}")
        agg[fmt]['pp'].append(pp)
        agg[fmt]['ta'].append(ta)
        agg[fmt]['tm'].append(tm)

print("\n" + "=" * 74)
print("CROSS-SEED MEAN (common table set) -- this is the fair comparison")
print("=" * 74)
print(f"{'format':12} {'per-pair':>9} {'tbl-all':>8} {'tbl-maj':>8}  (seeds)")
for fmt in FORMATS:
    pp = st.mean(agg[fmt]['pp']); ta = st.mean(agg[fmt]['ta']); tm = st.mean(agg[fmt]['tm'])
    seedstr = "/".join(f"{x:.3f}" for x in agg[fmt]['pp'])
    print(f"{fmt:12} {pp:>9.3f} {ta:>8.3f} {tm:>8.3f}  ({seedstr})")

# Noise reference: SE of a single proportion at p=0.5 for the common-set size.
# Differences smaller than ~2.6 SE between two formats are not significant.
any_seed = next(iter(SEEDS))
data = {fmt: {key(r): r for r in load(fmt, any_seed)} for fmt in FORMATS}
ncommon = len(set.intersection(*[set(d) for d in data.values()]))
se = (0.25 / ncommon) ** 0.5
print(f"\nCommon-set size ~{ncommon} pairs.  1 SE at chance = {se*100:.2f}%."
      f"  Two formats differ significantly only if their gap exceeds ~{se*2*1.41*100:.1f}%.")
