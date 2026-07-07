# Tabular Ablation Study

Serialization format ablation study for large-scale tabular pretraining. Part of a semester project at EPFL / CSCS.

**Report:** [Tabular Datasets for the Posttraining Pipeline of Apertus](https://github.com/janjetovicd/tabular_ablation/blob/main/Semester_Project_Report_Dijana_Janjetovic_updated.pdf)

**Goal:** Train 5 small proxy models (1.54B parameters, ~30B tokens per run, 2 random seeds each) on the [T4 dataset](https://huggingface.co/datasets/mlfoundations/t4), one per serialization format, and evaluate which format leads to the best downstream tabular understanding — informing the format choice for the Apertus posttraining mixture (tabular data added on top of an already-trained Apertus).

**Comparison method:** formats are compared at matched **rows seen**, not matched tokens — token-efficient formats (CSV, SQL schema) pack ~2x more rows into the same token budget, so a fixed-token comparison would confound serialization quality with data quantity. See [`analysis/RUNBOOK.md`](analysis/RUNBOOK.md) for the full method and script order.

**Formats under test:** `csv`, `keyvalue`, `markdown`, `json`, `sql_schema`

---

## Pipeline overview

```
T4 parquet chunks (zip)
│
▼
[Phase 1] serialize_t4.py        → .jsonl files (one line per table segment)
│
▼
[Phase 2] submit_tokenize.sh     → .bin / .idx files (Megatron binary format)
│
▼
[Phase 3] submit_tabular_ablation.sh  → 5 proxy model checkpoints
│
▼
[Phase 4] Evaluation             → winning format
├── prepare_eval_data.py       → prompts + ground truth from held-out T4 tables
└── run_eval.sh                → log-prob scoring → accuracy per format
```

---
---

## Scripts

### `serialize_t4.py`

Reads T4 parquet files from a chunk zip archive, cleans and serializes every table using **Tabular Chunking** — splitting each table into multiple 3800-token segments so no row is discarded.

**Key design decisions:**
- **Tabular Chunking**: each table is split into multiple 3800-token segments using greedy row accumulation with per-row token pre-computation. Writes one `{"text": "..."}` JSON line per segment.
- **Header/schema repetition**: every segment includes the column header or SQL schema so the model always has semantic context.
- **Deterministic shuffle**: rows are shuffled with a seed derived from the filename (MD5 hash), ensuring all 5 format runs see identical row orderings for a fair comparison.
- **Cleaning**: drops artifact columns (`Unnamed:*`, `index`), all-null columns, and columns with any cell exceeding 500 characters (avoids free-text blobs and base64 data swamping the token budget).

**Usage:**
```bash
python serialize_t4.py \
    --chunk_zip /path/to/chunk-0000.zip \
    --format csv \
    --output /path/to/output/chunk-0000.jsonl \
    --tokenizer /path/to/cached/tokenizer \
    --token_budget 3800
```

**Formats:**

| ID | Name | Description |
|---|---|---|
| A | `keyvalue` | `The col is val.` per cell, one row per line. Most verbose — column names repeat every row. Baseline from T4/TabLLM/UniPredict. |
| B | `csv` | Standard CSV with header. Most token-efficient — column names appear once. ~2× more rows per segment than keyvalue. |
| C | `markdown` | Markdown table with separator row. Matches format seen in GitHub READMEs and Wikipedia. |
| D | `json` | JSON array of records. Common on the web; similar verbosity to keyvalue. |
| F | `sql_schema` | `CREATE TABLE` schema with type/range annotations prepended to CSV data. Novel application: mirrors how a data scientist reads a new dataset. |

---

### `submit_serialize.sh`

SLURM array job script. Submits one job per chunk for a given format. The `--array` range is passed at submission time (not hardcoded) to allow flexible test or full runs.

**For a full run (all 76 chunks, all formats):**
```bash
sbatch --array=0-75 submit_serialize.sh csv
sbatch --array=0-75 submit_serialize.sh sql_schema
sbatch --array=0-75 submit_serialize.sh keyvalue
sbatch --array=0-75 submit_serialize.sh markdown
sbatch --array=0-75 submit_serialize.sh json
```

**For this ablation study (24 chunks):**
```bash
sbatch --array=0-23 submit_serialize.sh csv
sbatch --array=0-23 submit_serialize.sh sql_schema
sbatch --array=0-23 submit_serialize.sh keyvalue
sbatch --array=0-23 submit_serialize.sh markdown
sbatch --array=0-23 submit_serialize.sh json
```

**SLURM config:** 1 node, 1 task, 16 CPUs, 12h wall time. No GPU needed for serialization.

**Important:** Compute nodes on Clariden have no internet access. The tokenizer must be pre-cached on the login node before submitting:
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('swiss-ai/Apertus-70B-2509')
tok.save_pretrained('/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus')
"
```

---

### `submit_tokenize.sh`

Converts `.jsonl` files to Megatron binary format (`.bin` + `.idx`) using datatrove's `MegatronDocumentTokenizer`.

**Why datatrove instead of Megatron's `preprocess_data.py`?**  
The swiss-ai Megatron fork explicitly recommends datatrove. `preprocess_data.py` was found to silently produce empty output on Clariden compute nodes because it tries to download the tokenizer at runtime (no internet on compute nodes). Datatrove with a pre-cached tokenizer path resolves this.

**Throughput:** ~70M tokens/second/node at 28 parallel workers (swiss-ai benchmark for Alps).

---

### `submit_tabular_ablation.sh`

Launches proxy model training on 8 GPU nodes (32 GH200s) using the Megatron-LM framework inside the provided container.

**Model:** 1.54B non-embedding parameters  
**Training:** ~30B tokens, 28,610 steps, global batch size 256 (≈1M tokens/step)

---

### `tabular_container.toml`

Container definition for GPU training jobs. Specifies `ngc-nemo:25.11.01-alps3`. Used by `submit_tabular_ablation.sh`.

---

## Evaluation

Training loss is tracked via WandB/TensorBoard but is **not** used to pick a format — verbose formats (JSON, key:value, Markdown) reach lower loss simply because their token streams are more predictable, not because the model understands tables better. Downstream evaluation uses a **BLiMP-style minimal-pair** protocol on held-out T4 data (chunks 24–25, not seen during training), run once per format per seed at the matched (rows-seen-equalized) checkpoint.

**Methodology:**
1. Sample 299 held-out tables (≥2 rows, header fits the 3,800-token budget in all 5 formats).
2. Truncate each table to the 3,800-token budget *before* generating any statements, so the cell needed to verify a statement can never be truncated away later.
3. Prompt LLaMA-3.1-8B-Instruct on the truncated context to generate one correct statement + one plausible distractor per **task type** — lookup, aggregation, comparison, inference, multihop — yielding up to 1,496 minimal pairs per format (~5/table). The same statement pairs are reused across all 5 formats, so generation noise can't bias the ranking.
4. Each proxy model scores both statements by mean NLL conditioned on the table context; the pair is correct if the true statement gets the higher likelihood (chance = 50%).
5. Report **per-pair accuracy** (fraction of pairs correct) and **per-table accuracy** (all-correct / majority-correct per table), on the common ~1,484-pair set every format could serialize.

A cross-format/cross-seed **consistency analysis** additionally checks whether the *same* tables are hard across formats and seeds (they are, r=0.41–0.61 cross-format) — evidence that near-chance accuracy reflects table/distractor difficulty rather than the model or the serialization format.

### `prepare_eval_data_v2.py`

Prepares the evaluation dataset: scans T4 chunks 24–25, filters valid tables, samples 299 with a fixed seed, and for each table builds truncated context prompts + generates the statement/distractor pairs per task type across all 5 formats simultaneously. `prepare_eval_data_v2_cpu.py` is the CPU-only variant for the login node.

### `run_eval_inference.py` / `submit_eval_inference.sh`

Runs inference-time NLL scoring of every statement/distractor pair against each proxy model checkpoint. Submitted per (format, seed) at the matched checkpoint from `analysis/build_eval_plan.py` / `analysis/03_fixed_table_checkpoints.py`.

### `aggregate_eval_fair.py`

Aggregates the two-seed, matched-checkpoint eval into per-pair and per-table (all-correct / majority) accuracy — this produces the final results table. Restricts every format to the common set of pairs present in all 5 formats (SQL schema fails to serialize 12/299 tables) so denominators match across formats.

### `consistency_analysis.py` / `plot_consistency.py`

Compute and plot the cross-format / cross-seed correlation of per-table accuracy (`consistency_heatmap.png`) — the check for whether accuracy is limited by table/distractor difficulty rather than by serialization format.

### `analysis/` (fixed-table-size matching pipeline)

Equalizes the 5 formats at the same number of **rows seen** rather than the same token budget before evaluation. Order: `01_assess_training.py` → `02_token_stats.py` → `03_fixed_table_checkpoints.py` (or `build_eval_plan.py` for checkpoints split across store/scratch) → `04_replot_loss_vs_rows.py` → `05_run_fixed_table_eval.sh`. `resume_unfinished.sh` resubmits any (format, seed) run that stalled short of the target step. Full walkthrough in [`analysis/RUNBOOK.md`](analysis/RUNBOOK.md).

---

## Data

**Source:** [mlf oundations/t4](https://huggingface.co/datasets/mlf oundations/t4) — a large-scale collection of tables scraped from the web, stored as parquet files organized in 76 chunks. Chunks 0–23 used for training; chunks 24–25 held out for evaluation.
