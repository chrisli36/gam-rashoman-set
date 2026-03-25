#!/bin/bash
#SBATCH --job-name=finetune_plot
#SBATCH -o logs/finetune_plot_%j.out
#SBATCH --time=0:05:00
#SBATCH --mem=4G
#SBATCH --partition=compsci
# Usage: sbatch --dependency=afterokarray:ARRAY_JOB_ID run_finetune_plot.slurm
# Submit after the finetune comparison array completes.

OUTPUT_DIR="results_finetune_comparison"

source /usr/xtmp/vb97/GAM_Rashomon_Sets/gams/bin/activate
cd /usr/xtmp/vb97/GAM_Rashomon_Sets/gam-rashoman-set/new_method_scripts

python3 run_finetune_comparison.py --plot "$OUTPUT_DIR"
echo "Plots saved under $OUTPUT_DIR/ (runtime, diversity metrics, avg_train_accuracy, n_models)"
