#!/bin/bash
#SBATCH --job-name=mcmc_cov
#SBATCH -o logs/mcmc_cov_%A_%a.out
#SBATCH --time=72:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compsci
#SBATCH --array=0-3

# Usage example:
#   sbatch run_mcmc_coverage_comparison.slurm
#
# You can override resources per submission (recommended to separate profiles):
#   - quick profile (5k/smoke): shorter walltime, lower memory
#   - long profile (5M): larger walltime/memory
#
# Example overrides:
#   sbatch --time=08:00:00 --mem=40G --array=0-3 run_mcmc_coverage_comparison.slurm
#   sbatch --time=120:00:00 --mem=160G --array=0-3 run_mcmc_coverage_comparison.slurm

DATASETS=(compas.csv fico.csv covertype.csv spambase.csv)
OUTPUT_DIR="results_mcmc_coverage_comparison"
DISTANCE_METRIC="euclidean"

# Core MCMC settings
N_STEPS_SMALL=5000
N_STEPS_LARGE=5000000
BURN_IN_SMALL=500
BURN_IN_LARGE=50000
TARGET_MODELS_SMALL=100
TARGET_MODELS_LARGE=300
N_TRIALS=1
BASE_SEED=42

# Colleague's requested settings
MH_VARIANT="standard"
PROPOSAL="mixture"
FINETUNE_FLAG="--finetune_coordinate"
FINETUNE_CORR_THRESHOLD=0.05
ELLIPSOID_AUGMENT_FLAG="--ellipsoid_augment"
ELLIPSOID_N_SAMPLES=50

IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$IDX]}
DS_STEM=${DATASET%.csv}

source /usr/xtmp/vb97/GAM_Rashomon_Sets/gams/bin/activate
cd /usr/xtmp/vb97/GAM_Rashomon_Sets/gam-rashoman-set/new_method_scripts

mkdir -p logs "$OUTPUT_DIR"

echo "Task $IDX: dataset=$DATASET distance=$DISTANCE_METRIC n_steps_small=$N_STEPS_SMALL n_steps_large=$N_STEPS_LARGE"

python3 run_mcmc_coverage_comparison.py \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --distance_metric "$DISTANCE_METRIC" \
  --n_trials "$N_TRIALS" \
  --base_seed "$BASE_SEED" \
  --n_steps_small "$N_STEPS_SMALL" \
  --n_steps_large "$N_STEPS_LARGE" \
  --burn_in_small "$BURN_IN_SMALL" \
  --burn_in_large "$BURN_IN_LARGE" \
  --target_models_small "$TARGET_MODELS_SMALL" \
  --target_models_large "$TARGET_MODELS_LARGE" \
  --proposal "$PROPOSAL" \
  --mh_variant "$MH_VARIANT" \
  $FINETUNE_FLAG \
  --finetune_corr_threshold "$FINETUNE_CORR_THRESHOLD" \
  $ELLIPSOID_AUGMENT_FLAG \
  --ellipsoid_n_samples "$ELLIPSOID_N_SAMPLES"

# Optional merge convenience after all tasks complete:
# python3 - <<'PY'
# from pathlib import Path
# import pandas as pd
# out = Path("results_mcmc_coverage_comparison")
# files = sorted(out.glob("*_coverage_summary.csv"))
# if files:
#     pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(
#         out / "coverage_summary_all_datasets.csv", index=False
#     )
#     print("Merged summaries written.")
# PY
