#!/usr/bin/env python3
"""
build_eval_plan.py — one-shot fixed-table-size matcher for checkpoints that are
SPLIT across multiple locations (a139 store dirs hold the early iters, the scratch
meg-runs dirs hold the later iters from the resumed runs).

It:
  1. Scans every --root for experiment dirs, and per (format, seed) builds the
     UNION of available checkpoints as {iter -> the dir that holds it}.
  2. Reads token_stats.json and computes the rows-equalized ("fixed table size")
     matched iteration for every run (see the algebra below).
  3. Snaps each matched iter to the nearest checkpoint that actually exists, in
     whichever location it lives, and builds a non-destructive symlink "view" dir
     so eval can load it.
  4. Writes eval_plan.json (consumed by 05_run_fixed_table_eval.sh) + prints a table.

ALGEBRA (why only per-format token counts matter)
  rows_seen(fmt, iter) ∝ iter / dataset_tokens(fmt)        (total_rows is shared)
  Equalize rows across all runs at the largest budget everyone can reach:
      rows_at_max(run) = max_iter(run) / dataset_tokens(fmt)
      budget           = min over runs of rows_at_max
      matched_iter(run)= round(budget * dataset_tokens(fmt))   -> snap to disk

Usage (login node):
  python3 build_eval_plan.py --token-stats token_stats.json
  python3 build_eval_plan.py --roots A B --views-dir /capstor/scratch/.../ckpt_views
"""

import os
import re
import json
import argparse
from collections import defaultdict

FORMATS = ["csv", "json", "keyvalue", "markdown", "sql_schema"]
EXP_RE = re.compile(r"^tabular-(?P<fmt>csv|json|keyvalue|markdown|sql_schema)"
                    r"-1p5b-seed(?P<seed>\d+)-(?P<jobid>\d+)$")
ITER_RE = re.compile(r"^iter_(\d+)$")

DEFAULT_ROOTS = [
    "/capstor/store/cscs/swissai/a139/djanjetovic/swiss-megatron-fork/logs/Meg-Runs/tabular-ablation",
    "/capstor/scratch/cscs/djanjetovic/meg-runs/tabular-ablation",
]


def find_ckpt_dir(exp_dir):
    for cand in (os.path.join(exp_dir, "checkpoints"), exp_dir):
        if os.path.isdir(cand) and any(ITER_RE.match(n) for n in os.listdir(cand)):
            return cand
    for dp, dn, _ in os.walk(exp_dir):
        if any(ITER_RE.match(d) for d in dn):
            return dp
    return None


def collect(roots):
    """Return {(fmt, seed): {iter: ckpt_dir}} merged across all roots/exp dirs."""
    runs = defaultdict(dict)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            m = EXP_RE.match(name)
            if not m:
                continue
            ck = find_ckpt_dir(os.path.join(root, name))
            if not ck:
                continue
            key = (m.group("fmt"), int(m.group("seed")))
            for n in os.listdir(ck):
                im = ITER_RE.match(n)
                if im and os.path.isdir(os.path.join(ck, n)):
                    it = int(im.group(1))
                    # later location wins ties (shouldn't matter — ranges disjoint)
                    runs[key][it] = ck
    return runs


def make_view(views_dir, fmt, seed, it, ckpt_dir, dry):
    view = os.path.join(views_dir, f"tabular-{fmt}-seed{seed}__iter{it}")
    src = os.path.join(ckpt_dir, f"iter_{it:07d}")
    link = os.path.join(view, f"iter_{it:07d}")
    if dry:
        return view
    os.makedirs(view, exist_ok=True)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(src, link)
    with open(os.path.join(view, "latest_checkpointed_iteration.txt"), "w") as fh:
        fh.write(str(it))
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-stats", default="token_stats.json")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--views-dir",
                    default="/capstor/scratch/cscs/djanjetovic/ckpt_views")
    ap.add_argument("--out", default="eval_plan.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token_stats = json.load(open(args.token_stats))
    dtok = {f: token_stats[f]["total_tokens"]
            for f in FORMATS if token_stats.get(f, {}).get("total_tokens")}

    runs = collect(args.roots)
    if not runs:
        raise SystemExit("No runs found. Check --roots.")

    # max iter per run, then the common rows budget
    max_iter = {k: max(v) for k, v in runs.items()}
    rows_at_max = {k: max_iter[k] / dtok[k[0]] for k in runs if k[0] in dtok}
    budget = min(rows_at_max.values())
    binding = min(rows_at_max, key=rows_at_max.get)

    print("=" * 92)
    print("FIXED TABLE SIZE — matched checkpoints (rows-equalized across all runs)")
    print(f"Binding run (stays at its max): {binding[0]} seed{binding[1]} "
          f"@ iter {max_iter[binding]}")
    print("=" * 92)
    print(f"{'format':<12}{'seed':>6}{'max_iter':>10}{'target':>9}"
          f"{'matched':>9}{'loc':>9}   view")
    print("-" * 92)

    plan = []
    for key in sorted(runs, key=lambda k: (k[0], k[1])):
        fmt, seed = key
        if fmt not in dtok:
            continue
        iters = sorted(runs[key])
        target = round(budget * dtok[fmt])
        snapped = min(iters, key=lambda it: abs(it - target))
        src_dir = runs[key][snapped]
        loc = "scratch" if "/scratch/" in src_dir else "store"
        view = make_view(args.views_dir, fmt, seed, snapped, src_dir, args.dry_run)
        plan.append({
            "format": fmt, "seed": seed,
            "max_iter": max_iter[key], "final_iter": max_iter[key],
            "target_iter": target, "matched_iter": snapped,
            "src_ckpt_dir": src_dir,
            "real_ckpt_dir": runs[key][max_iter[key]],  # dir holding the final (for --mode final)
            "view_ckpt_dir": view, "location": loc,
        })
        print(f"{fmt:<12}{seed:>6}{max_iter[key]:>10}{target:>9}"
              f"{snapped:>9}{loc:>9}   {os.path.basename(view)}")
    print("-" * 92)

    with open(args.out, "w") as fh:
        json.dump({"binding": f"{binding[0]}-seed{binding[1]}",
                   "dataset_tokens": dtok, "plan": plan}, fh, indent=2)
    print(f"\nWrote {args.out}  ({len(plan)} runs).")
    print("Views %s under %s"
          % ("WOULD be created" if args.dry_run else "created", args.views_dir))
    if not args.dry_run:
        print("\nNext: ./05_run_fixed_table_eval.sh")


if __name__ == "__main__":
    main()
