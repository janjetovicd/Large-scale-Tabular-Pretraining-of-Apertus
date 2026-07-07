#!/bin/bash
# resume_unfinished.sh — finish the ablation runs hands-off.
#
# For every (format, seed) it finds the FURTHEST checkpoint reached so far
# (across the original run dir and any resumed dirs), and if that's still short
# of the target it resubmits a resume job continuing from there. Runs that have
# essentially finished are skipped.
#
# RUN THIS ONLY WHEN NO tabular-ablation JOBS ARE ACTIVE (check `squeue --me`),
# otherwise you'll launch a duplicate of a run that's still going.
#
# Usage:
#   ./resume_unfinished.sh --dry-run     # preview what it would do
#   ./resume_unfinished.sh               # actually resubmit
set -uo pipefail

FORK=/capstor/store/cscs/swissai/a139/djanjetovic/swiss-megatron-fork/logs/Meg-Runs/tabular-ablation
SUBMIT=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/submit_tabular_ablation.sh
DONE_AT=28500          # last 500-step checkpoint before the 28,610 target = "done enough"
DRYRUN=${1:-}

declare -A best_iter best_dir
for d in $(ls -1 "$FORK" 2>/dev/null | grep -E '^tabular-.+-1p5b-seed[0-9]+-[0-9]+$'); do
    marker="$FORK/$d/checkpoints/latest_checkpointed_iteration.txt"
    [ -f "$marker" ] || continue
    it=$(cat "$marker" 2>/dev/null)
    [[ "$it" =~ ^[0-9]+$ ]] || continue
    fmt=$(echo "$d"  | sed -E 's/^tabular-(.+)-1p5b-seed[0-9]+-[0-9]+$/\1/')
    seed=$(echo "$d" | sed -E 's/^tabular-.+-1p5b-seed([0-9]+)-[0-9]+$/\1/')
    key="$fmt:$seed"
    if [ -z "${best_iter[$key]:-}" ] || [ "$it" -gt "${best_iter[$key]}" ]; then
        best_iter[$key]=$it
        best_dir[$key]="$FORK/$d/checkpoints"
    fi
done

if [ ${#best_iter[@]} -eq 0 ]; then
    echo "No run dirs found under $FORK"
    exit 1
fi

n_resume=0
for key in $(echo "${!best_iter[@]}" | tr ' ' '\n' | sort); do
    fmt=${key%:*}; seed=${key#*:}
    it=${best_iter[$key]}; ck=${best_dir[$key]}
    if [ "$it" -ge "$DONE_AT" ]; then
        echo "DONE    $key  (iter $it)"
    else
        echo "RESUME  $key  (iter $it < $DONE_AT)  <-  $ck"
        n_resume=$((n_resume + 1))
        if [ "$DRYRUN" != "--dry-run" ]; then
            RESUME_TRAINING=true sbatch "$SUBMIT" "$fmt" "$seed" "$ck"
        fi
    fi
done

echo
if [ "$DRYRUN" = "--dry-run" ]; then
    echo "DRY RUN — would resume $n_resume run(s). Re-run without --dry-run to submit."
else
    echo "Submitted $n_resume resume job(s). Watch with: squeue --me"
fi
