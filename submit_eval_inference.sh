#!/bin/bash
# Run BLiMP-style inference eval for one serialization format.
#
# Auto-submitted by submit_tabular_ablation.sh — you don't normally call this directly.
# If you need to run it manually:
#   sbatch submit_eval_inference.sh csv /path/to/checkpoints/dir
#
# Args:
#   $1 = format (csv | json | keyvalue | markdown | sql_schema)
#   $2 = checkpoint directory (the EXP_DIR/checkpoints path from training)

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=eval-inference
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron/logs/slurm/training/%x-%j.err

echo "START TIME: $(date)"
echo "Running on: $(hostname)"

# ── Guard: prevent accidental auto-eval ────────────────────────────────────────
# Eval is intentionally not auto-submitted from training. To run eval manually:
#   EVAL_ENABLED=1 sbatch submit_eval_inference.sh <format> <ckpt_dir> <seed>
# Without EVAL_ENABLED=1 the script exits immediately (safe no-op).
if [ "${EVAL_ENABLED:-0}" != "1" ]; then
    echo "EVAL_ENABLED is not set — eval skipped."
    echo "To run eval: EVAL_ENABLED=1 sbatch submit_eval_inference.sh <format> <ckpt_dir> <seed>"
    exit 0
fi

# ── Args ───────────────────────────────────────────────────────────────────────

FORMAT=${1:-csv}
CKPT_DIR=${2:-}
SEED=${3:-}        # optional; when provided, used to name output files

if [[ ! "$FORMAT" =~ ^(csv|json|keyvalue|markdown|sql_schema)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT'. Valid: csv json keyvalue markdown sql_schema"
    exit 1
fi

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint directory given. Pass it as the second argument."
    exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CKPT_DIR"
    exit 1
fi

# Build a unique run ID so each seed writes to its own output files and nothing
# overwrites anything else.
# - If a seed was passed explicitly (e.g. from an updated submit_tabular_ablation.sh),
#   use "<format>_seed<seed>" — clean and human-readable.
# - If no seed was given (the 10 jobs already submitted before this update use the
#   old call signature), fall back to the experiment name embedded in the checkpoint
#   path. That path looks like: .../tabular-csv-1p5b-<jobid>/checkpoints
#   so basename of its parent gives a unique ID like "tabular-csv-1p5b-2561889".
if [ -n "$SEED" ]; then
    RUN_ID="${FORMAT}_seed${SEED}"
else
    EXP_NAME=$(basename "$(dirname "$CKPT_DIR")")
    RUN_ID="$EXP_NAME"
fi

echo "Format:     $FORMAT"
echo "Checkpoint: $CKPT_DIR"
echo "Run ID:     $RUN_ID"

# ── Paths ──────────────────────────────────────────────────────────────────────

MEGATRON_LM_DIR=/capstor/store/cscs/swissai/a139/djanjetovic/swiss-megatron-fork
ABLATION_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation
CONTAINER_ENV=$ABLATION_DIR/tabular_container.toml

# STORE: permanent, backed-up project storage. Read eval pairs from store and
# write eval results to store so they survive scratch cleanup and stay available
# for analysis. Scratch (/iopsstor) purges anything untouched for 14 days.
STORE_DIR=/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation
PAIRS_DIR=$STORE_DIR/eval
OUTPUT_DIR=$STORE_DIR/eval_results

mkdir -p $OUTPUT_DIR
mkdir -p $MEGATRON_LM_DIR/logs/slurm/training

# ── Checks ─────────────────────────────────────────────────────────────────────

PAIRS_FILE=$PAIRS_DIR/pairs_${FORMAT}.jsonl
if [ ! -f "$PAIRS_FILE" ]; then
    echo "ERROR: Pairs file not found: $PAIRS_FILE"
    exit 1
fi

echo "Pairs file: $PAIRS_FILE ($(wc -l < $PAIRS_FILE) lines)"
echo "Output dir: $OUTPUT_DIR"

# ── Environment ────────────────────────────────────────────────────────────────

export MEGATRON_LM_DIR=$MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8889
export WORLD_SIZE=$SLURM_NPROCS

# ── Run eval ───────────────────────────────────────────────────────────────────

EVAL_CMD="python3 $ABLATION_DIR/run_eval_inference.py \
    --format     $FORMAT \
    --run-id     $RUN_ID \
    --ckpt-dir   $CKPT_DIR \
    --pairs-dir  $PAIRS_DIR \
    --output-dir $OUTPUT_DIR \
    --max-seq-len 4096"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID PYTHONPATH=$MEGATRON_LM_DIR MEGATRON_LM_DIR=$MEGATRON_LM_DIR $EVAL_CMD"

echo "END TIME: $(date)"
