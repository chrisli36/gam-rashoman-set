# GAM Rashomon Set Diversity Experiments

Explores the diversity of Generalized Additive Models (GAMs) within the Rashomon set — the set of near-optimal models — by comparing two sampling methods:

- **Method 1 (MCMC + Ellipsoid):** Runs MCMC over support sets with a repulsive diversity kernel, then aggregates models across all sampled supports. Computes diversity over this full aggregated Rashomon set.
- **Method 2 (Ellipsoid Baseline):** For each support set discovered by MCMC, fits a local ellipsoid around the MAP estimate and samples models within it. Reports per-ellipsoid diversity (mean +/- std).

## Directory Structure

| File | Description |
|---|---|
| `mcmc_rashomon.py` | Core MCMC Rashomon sampler (`MCMCRashomonSampler`), data binarization, GAM shape plotting, diversity kernels |
| `ellipsoid.py` | `EllipsoidSampler` baseline — samples from a Hessian-defined ellipsoid around the MAP |
| `diversity_measures.py` | Pairwise diversity metrics: Hamming, prediction Hamming, weight L1, shape L1, structural cosine, monotonicity entropy, variable importance |
| `run_diversity_experiment.py` | Main experiment script — runs both methods for a (dataset, diversity, repulsion) config and saves JSON results |
| `run_diversity_experiment.slurm` | SLURM job array launcher — submits all configs in parallel |
| `plot_diversity_results.py` | Plots repulsion weight vs. diversity metric from merged CSV results |
| `run_finetune_comparison.py` | Ablation: compares coordinate finetuning on/off |
| `run_finetune_comparison.slurm` | SLURM launcher for the finetuning ablation |

## Running the Diversity Experiment via SLURM

### Quick Start

```bash
sbatch run_diversity_experiment.slurm
```

This submits a SLURM job array that runs all combinations of datasets, diversity functions, and repulsion weights in parallel. Each array task handles one `(dataset, diversity_fn, repulsion_weight)` configuration.

### How the Job Array Works

The slurm script defines three arrays that form a grid of configurations:

```bash
DATASETS=(heart.csv spambase.csv fico.csv compas.csv covertype.csv mgh.csv)
REPULSION=(1)
DIVERSITY=(monotonicity_entropy "support_jaccard" "l1" "monotonicity_entropy" "shape_l1")
```

The `--array=0-N` directive launches N+1 tasks. Each task ID is decomposed into a `(dataset, repulsion, diversity)` triple via integer arithmetic:

```
total_tasks = len(DATASETS) * len(REPULSION) * len(DIVERSITY)
dataset_idx  = task_id / (len(REPULSION) * len(DIVERSITY))
repulsion_idx = (task_id % (len(REPULSION) * len(DIVERSITY))) / len(DIVERSITY)
diversity_idx = task_id % len(DIVERSITY)
```

Make sure `--array=0-N` covers the full grid. For 6 datasets x 1 repulsion x 5 diversity functions = 30 tasks, use `--array=0-29`. The current script uses `--array=0-30` (31 tasks; the extra task will have an out-of-bounds index and exit harmlessly).

### Configurable Parameters

**P_FEAT** — prior probability of including each feature during MCMC. Set at the top of the slurm script (default `0.02`). To override at submission time:

```bash
sbatch --export=P_FEAT=0.5 run_diversity_experiment.slurm
```

**Epsilon (Rashomon tolerance)** — controls how far from the best model the Rashomon set extends. Configured per-dataset via the `EPS_OVERRIDE` associative array, with `EPS_DEFAULT=0.05` as the fallback:

```bash
EPS_DEFAULT=0.05
declare -A EPS_OVERRIDE
EPS_OVERRIDE[heart.csv]=0.05
EPS_OVERRIDE[fico.csv]=0.05
# ... etc.
```

Edit these values directly in the slurm script to tune per-dataset.

**Datasets** — edit the `DATASETS` array to run a subset:

```bash
DATASETS=(heart.csv fico.csv)
```

Remember to update `--array` accordingly (e.g. `--array=0-9` for 2 datasets x 1 repulsion x 5 diversity).

**Repulsion weights** — controls the strength of the repulsive diversity kernel during MCMC. Edit the `REPULSION` array:

```bash
REPULSION=(0 1 5 10)
```

**Diversity functions** — the repulsive kernel used during MCMC to encourage diverse model proposals. Edit the `DIVERSITY` array. Valid choices:

| Name | Description |
|---|---|
| `monotonicity_entropy` | Entropy of monotonicity classifications (increasing/decreasing/non-monotonic) across features |
| `support_jaccard` | Jaccard distance between feature support sets |
| `l1` | L1 distance between weight vectors |
| `shape_l1` | L1 distance between GAM shape functions |
| `prediction_hamming` | Hamming distance between model predictions |

### Output

Results are written to `results_diversity_<P_FEAT>/<dataset_stem>/`:

```
results_diversity_0.02/
  compas/
    l1_rep1.json                                    # Per-config JSON
    support_jaccard_rep1.json
    monotonicity_entropy_rep1.json
    diversity_results.json                          # Merged JSON (all configs)
    diversity_results.csv                           # Merged CSV (all configs)
    gam_shapes_l1_mcmc_rep1.png                     # GAM shape plots (Method 1)
    gam_shapes_l1_ellipsoid_rep1.png                # GAM shape plots (Method 2)
    loss_distributions_l1_rep1.png                  # Loss distribution histograms
    diversity_plot.png                              # Repulsion vs diversity plot
  heart/
    ...
```

Each per-config JSON contains:
- `n_models_mcmc` — number of accepted models (Method 1)
- `mcmc` — diversity metrics for MCMC (pred_hamming, support_hamming, weight_l1, shape_l1, monotonicity_entropy)
- `ellipsoid` — per-ellipsoid diversity metrics (mean +/- std)
- `acceptance_rate`, `best_error`, `error_bound`, `runtime`, `eps`

### Post-Processing (Merge & Plot)

Each SLURM task automatically runs the merge and plot steps after its experiment completes. If you need to re-run them manually:

**Merge** per-config JSONs into a single CSV/JSON:

```bash
python run_diversity_experiment.py --merge results_diversity_0.02/compas --dataset compas.csv
```

**Plot** repulsion weight vs. diversity:

```bash
python plot_diversity_results.py results_diversity_0.02/compas/diversity_results.csv \
    -o results_diversity_0.02/compas/diversity_plot.png
```

### Running a Single Config Locally (without SLURM)

```bash
python run_diversity_experiment.py \
    --dataset compas.csv \
    --repulsion 1 \
    --diversity monotonicity_entropy \
    --eps 0.05 \
    --p_feat 0.02 \
    --output_dir results_diversity_0.02
```

Full CLI options for `run_diversity_experiment.py`:

| Flag | Default | Description |
|---|---|---|
| `--dataset` | (required) | Dataset filename (resolved relative to `--data_dir`) |
| `--data_dir` | `<repo>/data/benchmark` | Directory containing CSV datasets |
| `--repulsion` | `-1,0,1,5,10,20` | Comma-separated repulsion weights |
| `--diversity` | `prediction_hamming,support_jaccard,l1,monotonicity_entropy` | Comma-separated diversity function names |
| `--eps` | `0.05` | Rashomon set tolerance |
| `--eps_config` | None | JSON object or path mapping dataset stems to per-dataset eps values |
| `--p_feat` | `0.02` | Prior probability of including each feature |
| `--output_dir` | `results_diversity` | Output directory (created under `new_method_scripts/`) |
| `--merge DIR` | None | Merge per-config JSONs in DIR instead of running an experiment |
| `--quick` | off | (Currently same as default; reserved for fast iteration) |

### Monitoring SLURM Jobs

```bash
# Check job status
squeue -u $USER

# View logs for a specific array task
cat logs_pfeat/div_exp_<JOB_ID>_<TASK_ID>.out

# Cancel all tasks
scancel <JOB_ID>
```

### Environment

The slurm script activates a Python virtualenv at:

```
/usr/xtmp/vb97/GAM_Rashomon_Sets/gams/bin/activate
```

Datasets are expected in:

```
/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark/
```
