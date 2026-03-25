#!/bin/bash
#SBATCH --job-name=div_exp
#SBATCH -o logs_pfeat/div_exp_%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compsci
#SBATCH --array=0-110

# P_FEAT: prior probability of including each feature. Pass via: sbatch --export=P_FEAT=0.5 run_diversity_experiment.slurm
P_FEAT=0.03

# Each array task runs ONE (dataset, repulsion, diversity) config.
# Total tasks = N_DATASETS * N_REPULSION * N_DIVERSITY.
# After all jobs complete, run: python run_diversity_experiment.py --merge results_diversity_<P_FEAT>/<dataset>

DATASETS=(compas.csv fico.csv covertype.csv spambase.csv)
REPULSION=(0 0.00001 0.0001 0.001 0.01 0.02 0.03 0.04 0.05)
DIVERSITY=(monotonicity_entropy prediction_hamming support_jaccard)

# Epsilon (Rashomon set tolerance): default for all datasets, overrides per-dataset
EPS_DEFAULT=0.05
declare -A EPS_OVERRIDE
# EPS_OVERRIDE[heart.csv]=0.05
# EPS_OVERRIDE[fico.csv]=0.05
# EPS_OVERRIDE[compas.csv]=0.05
# EPS_OVERRIDE[spambase.csv]=0.05
# EPS_OVERRIDE[covertype.csv]=0.05
# EPS_OVERRIDE[mgh.csv]=0.05

N_D=${#DATASETS[@]}
N_R=${#REPULSION[@]}
N_V=${#DIVERSITY[@]}
N_TOTAL=$((N_D * N_R * N_V))

IDX=${SLURM_ARRAY_TASK_ID}
D_IDX=$((IDX / (N_R * N_V)))
R_IDX=$(((IDX % (N_R * N_V)) / N_V))
V_IDX=$((IDX % N_V))

DATASET=${DATASETS[$D_IDX]}
REP=${REPULSION[$R_IDX]}
DIV=${DIVERSITY[$V_IDX]}
EPS=${EPS_OVERRIDE[$DATASET]:-$EPS_DEFAULT}

OUTPUT_DIR="results_diversity_${P_FEAT}"

source /usr/xtmp/vb97/GAM_Rashomon_Sets/gams/bin/activate
cd /usr/xtmp/vb97/GAM_Rashomon_Sets/gam-rashoman-set/new_method_scripts

echo "Task $IDX/$N_TOTAL: dataset=$DATASET repulsion=$REP diversity=$DIV eps=$EPS p_feat=$P_FEAT"
python3 run_diversity_experiment.py \
    --dataset "$DATASET" \
    --repulsion "$REP" \
    --diversity "$DIV" \
    --eps "$EPS" \
    --p_feat "$P_FEAT" \
    --output_dir "$OUTPUT_DIR"

# Merge: run once per dataset after all its configs complete.
python3 run_diversity_experiment.py --merge "$OUTPUT_DIR/${DATASET%.csv}" --dataset $DATASET

# plot results for each dataset
python3 plot_diversity_results.py "$OUTPUT_DIR/${DATASET%.csv}/diversity_results.csv" -o "$OUTPUT_DIR/${DATASET%.csv}/diversity_plot.png"
