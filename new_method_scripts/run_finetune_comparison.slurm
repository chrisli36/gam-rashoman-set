#!/bin/bash
#SBATCH --job-name=finetune_cmp
#SBATCH -o logs/finetune_cmp_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compsci
#SBATCH --array=0-9

# Each array task runs ONE (dataset, variant): either no-finetune or finetune with one correlation threshold.
# Total jobs = 6 datasets * (1 no-finetune + 10 thresholds) = 66. Adjust array if you change datasets/thresholds.
# After each job, merge partial files for that dataset into <dataset>.csv.

DATASETS=(bike_binarized.csv)
CORR_THRESHOLDS=(0 0.02 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5)
OUTPUT_DIR="results_finetune_comparison"

N_D=${#DATASETS[@]}
N_T=${#CORR_THRESHOLDS[@]}
N_V=$((1 + N_T))
N_TOTAL=$((N_D * N_V))

IDX=${SLURM_ARRAY_TASK_ID}
D_IDX=$((IDX / N_V))
V_IDX=$((IDX % N_V))
DATASET=${DATASETS[$D_IDX]}
DS_STEM=${DATASET%.csv}

source /usr/xtmp/vb97/GAM_Rashomon_Sets/gams/bin/activate
cd /usr/xtmp/vb97/GAM_Rashomon_Sets/gam-rashoman-set/new_method_scripts

mkdir -p logs "$OUTPUT_DIR"

if [ $V_IDX -eq 0 ]; then
    echo "Task $IDX/$N_TOTAL: dataset=$DATASET (no-finetune)"
    python3 run_finetune_comparison.py \
        --dataset "$DATASET" \
        --no_finetune \
        --output "$OUTPUT_DIR/${DS_STEM}_no_finetune.csv"
else
    CORR_TH=${CORR_THRESHOLDS[$((V_IDX - 1))]}
    echo "Task $IDX/$N_TOTAL: dataset=$DATASET corr_threshold=$CORR_TH"
    python3 run_finetune_comparison.py \
        --dataset "$DATASET" \
        --corr_threshold $CORR_TH \
        --output "$OUTPUT_DIR/${DS_STEM}_${CORR_TH}.csv"
fi

python3 run_finetune_comparison.py --merge "$OUTPUT_DIR" --dataset "$DATASET"

