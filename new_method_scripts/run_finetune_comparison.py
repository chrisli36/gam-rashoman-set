"""
Compare MCMC with vs without finetune_coordinate across correlation thresholds.

Each setting runs ``--n_trials`` times (default 5); CSV stores mean/std runtime
(``runtime_s``, ``runtime_s_std``), ``avg_train_accuracy`` / ``avg_train_accuracy_std``,
``n_models`` / ``n_models_std``, and mean/std per diversity metric
(``diversity_<metric>``, ``diversity_<metric>_std``). Plots use a shaded band (± std).
Figures: one PNG per diversity metric plus ``corr_threshold_vs_avg_train_accuracy.png``,
``corr_threshold_vs_n_models.png``, and ``corr_threshold_vs_acceptance_rate.png``; one subplot
per dataset (grid layout like runtime plots).

Batch mode (default): runs no-finetune once + finetune per threshold per dataset.
Single-job mode: use --no_finetune or --corr_threshold to run one variant only
(for SLURM array: one job per threshold).
Use --merge to combine partial CSVs after all jobs complete.

Usage:
    python run_finetune_comparison.py
    python run_finetune_comparison.py --n_trials 5
    python run_finetune_comparison.py --dataset compas.csv
    python run_finetune_comparison.py --dataset heart.csv --no_finetune --output results_finetune_comparison/heart_no_finetune.csv
    python run_finetune_comparison.py --dataset heart.csv --corr_threshold 0.05 --output results_finetune_comparison/heart_0.05.csv
    python run_finetune_comparison.py --merge results_finetune_comparison --dataset heart
"""

import argparse
import contextlib
import io
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple
from time import time

import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(42)
np.random.seed(42)
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_method_scripts.mcmc_rashomon import (
    MCMCRashomonSampler,
    RashomonResult,
    binarize_data,
)
from new_method_scripts.diversity_measures import (
    pairwise_hamming,
    pairwise_prediction_hamming,
    pairwise_support_jaccard,
    pairwise_weight_l1,
    pairwise_shape_distance,
    monotonicity_diversity,
)

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
BENCHMARK_DIR = Path(
    "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark"
)
EPS = 0.1
SIGMA2 = 2
P_FEAT = 0.1
MAX_FEATURES = 150
MAX_SAMPLES = 10000

METRIC_KEYS = [
    "pred_hamming",
    "support_hamming",
    "support_jaccard",
    "weight_l1",
    "shape_l1",
    "monotonicity_entropy",
]

# Plotted vs corr_threshold (one PNG each; one subplot per dataset, grid like runtime)
DIVERSITY_PLOT_METRICS = [
    ("pred_hamming", "Prediction Hamming", "corr_threshold_vs_pred_hamming.png"),
    ("support_jaccard", "Support Jaccard distance", "corr_threshold_vs_support_jaccard.png"),
    ("weight_l1", "Weight L1 distance", "corr_threshold_vs_weight_l1.png"),
    ("shape_l1", "Shape L1 distance", "corr_threshold_vs_shape_l1.png"),
    ("monotonicity_entropy", "Monotonicity entropy", "corr_threshold_vs_monotonicity_entropy.png"),
]


def merge_results(output_dir: Path, dataset_filter: str = None):
    """Merge partial CSV files (e.g. heart_no_finetune.csv, heart_0.05.csv) into per-dataset CSVs."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {output_dir}")

    # Group partial files by dataset: heart_no_finetune.csv, heart_0.05.csv -> heart
    dataset_files: dict = {}
    for f in output_dir.glob("*.csv"):
        stem = f.stem
        if stem.endswith("_no_finetune"):
            ds = stem.replace("_no_finetune", "")
        else:
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    float(parts[1])
                    ds = parts[0]
                except ValueError:
                    continue
            else:
                continue
        if dataset_filter and ds != Path(dataset_filter).stem:
            continue
        dataset_files.setdefault(ds, []).append(f)

    for ds, files in dataset_files.items():
        rows = []
        for f in sorted(files):
            df = pd.read_csv(f)
            rows.append(df)
        if not rows:
            continue
        combined = pd.concat(rows, ignore_index=True)
        # Sort: no_finetune first (corr_threshold NaN), then by corr_threshold
        combined = combined.sort_values(
            by="corr_threshold",
            ascending=True,
            na_position="first",
        )
        out_path = output_dir / f"{ds}.csv"
        combined.to_csv(out_path, index=False)
        print(f"Merged {len(files)} partial files -> {out_path}")


def load_finetune_results_by_dataset(results_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load all non–no-finetune CSVs under ``results_dir``, merge per dataset,
    dedupe by (corr_threshold, finetune_coordinate). Used for plotting.
    """
    results_dir = Path(results_dir)
    by_dataset: dict = defaultdict(list)
    for f in sorted(results_dir.glob("*.csv")):
        if f.name.startswith("merged") or "all" in f.name.lower():
            continue
        if f.stem.endswith("_no_finetune"):
            continue
        try:
            df = pd.read_csv(f)
            if "corr_threshold" not in df.columns or len(df) == 0:
                continue
            if "dataset" in df.columns:
                ds = str(df["dataset"].iloc[0])
            else:
                stem = f.stem
                parts = stem.rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        float(parts[1])
                        ds = parts[0]
                    except ValueError:
                        ds = stem
                else:
                    ds = stem
            by_dataset[ds].append(df)
        except Exception:
            continue

    all_data: List[Tuple[str, pd.DataFrame]] = []
    for ds in sorted(by_dataset.keys()):
        dfs = by_dataset[ds]
        df = pd.concat(dfs, ignore_index=True)
        if "finetune_coordinate" in df.columns:
            df = df.drop_duplicates(
                subset=["corr_threshold", "finetune_coordinate"],
                keep="last",
            )
        all_data.append((ds, df))
    return all_data


def plot_corr_threshold_vs_metric(
    results_dir: Path,
    value_col: str,
    std_col: Optional[str],
    ylabel: str,
    out_filename: str,
) -> Path:
    """
    One figure with a subplot per dataset (grid like ``plot_corr_threshold_vs_runtime``).
    Finetune rows only; shaded band = ± std across trials when ``std_col`` is present.
    """
    results_dir = Path(results_dir)
    all_data = load_finetune_results_by_dataset(results_dir)
    plot_path = results_dir / out_filename
    if not all_data:
        print(f"No valid CSV files found for plot ({out_filename}).")
        return plot_path

    plotable: List[Tuple[str, pd.DataFrame]] = []
    for dataset, df in all_data:
        if "finetune_coordinate" not in df.columns:
            continue
        if value_col not in df.columns:
            continue
        plotable.append((dataset, df))

    if not plotable:
        print(f"No CSV with column {value_col} for plot ({out_filename}).")
        return plot_path

    n_datasets = len(plotable)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    idx = -1
    for idx, (dataset, df) in enumerate(plotable):
        ax = axes.flat[idx]
        finetune_mask = df["finetune_coordinate"].astype(str).str.lower() == "true"
        finetune_df = df[finetune_mask].copy()
        finetune_df = finetune_df.dropna(subset=["corr_threshold", value_col])
        finetune_df = finetune_df.sort_values("corr_threshold")
        title_ds = str(dataset).replace("_", " ").title()

        if len(finetune_df) > 0:
            x = finetune_df["corr_threshold"].values
            y = finetune_df[value_col].values.astype(float)
            if (
                std_col
                and std_col in finetune_df.columns
                and finetune_df[std_col].notna().any()
            ):
                yerr = finetune_df[std_col].fillna(0.0).values.astype(float)
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    color="lightblue",
                    alpha=0.35,
                    linewidth=0,
                )
            ax.plot(
                x,
                y,
                marker="o",
                linestyle="-",
                markersize=6,
                color="C0",
                label="finetune (mean ± std)",
            )
            ax.legend(loc="best", fontsize=8)

        ax.set_xlabel("Correlation threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(title_ds)
        ax.grid(True, alpha=0.3)

    for j in range(idx + 1, len(axes.flat)):
        axes.flat[j].set_visible(False)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")
    return plot_path


def plot_corr_threshold_vs_diversity_metric(
    results_dir: Path,
    metric_key: str,
    ylabel: str,
    out_filename: str,
) -> Path:
    """Diversity metric vs threshold (wrapper around :func:`plot_corr_threshold_vs_metric`)."""
    return plot_corr_threshold_vs_metric(
        results_dir,
        f"diversity_{metric_key}",
        f"diversity_{metric_key}_std",
        ylabel,
        out_filename,
    )


def plot_corr_threshold_vs_avg_train_accuracy(results_dir: Path) -> Path:
    """Average train accuracy of models in the Rashomon set vs correlation threshold."""
    return plot_corr_threshold_vs_metric(
        results_dir,
        "avg_train_accuracy",
        "avg_train_accuracy_std",
        "Average train accuracy (Rashomon set)",
        "corr_threshold_vs_avg_train_accuracy.png",
    )


def plot_corr_threshold_vs_n_models(results_dir: Path) -> Path:
    """Number of accepted models vs correlation threshold."""
    return plot_corr_threshold_vs_metric(
        results_dir,
        "n_models",
        "n_models_std",
        "Number of models in Rashomon set",
        "corr_threshold_vs_n_models.png",
    )


def plot_corr_threshold_vs_acceptance_rate(results_dir: Path) -> Path:
    """MCMC acceptance rate vs correlation threshold."""
    return plot_corr_threshold_vs_metric(
        results_dir,
        "acceptance_rate",
        "acceptance_rate_std",
        "Acceptance rate",
        "corr_threshold_vs_acceptance_rate.png",
    )


def plot_all_diversity_figures(results_dir: Path) -> List[Path]:
    """Save diversity PNGs plus average train accuracy, model-count, and acceptance-rate figures."""
    paths: List[Path] = []
    for metric_key, ylabel, filename in DIVERSITY_PLOT_METRICS:
        paths.append(
            plot_corr_threshold_vs_diversity_metric(results_dir, metric_key, ylabel, filename)
        )
    paths.append(plot_corr_threshold_vs_avg_train_accuracy(results_dir))
    paths.append(plot_corr_threshold_vs_n_models(results_dir))
    paths.append(plot_corr_threshold_vs_acceptance_rate(results_dir))
    return paths


def plot_corr_threshold_vs_runtime(results_dir: Path) -> Path:
    """
    Plot correlation threshold vs runtime for each dataset.

    Combines all partial CSVs for the same dataset (e.g. heart_0.csv, heart_0.05.csv)
    into one curve per dataset. Only finetune runs are plotted. No-finetune runtime
    for the title comes only from ``{dataset}_no_finetune.csv`` (not from merged rows).
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_dir}")

    # No-finetune baseline runtimes live only in *_no_finetune.csv
    no_finetune_runtime: dict = {}
    no_finetune_std: dict = {}
    for f in sorted(results_dir.glob("*_no_finetune.csv")):
        try:
            df_nf = pd.read_csv(f)
            if len(df_nf) == 0 or "runtime_s" not in df_nf.columns:
                continue
            ds = f.stem.replace("_no_finetune", "")
            rt = df_nf["runtime_s"].iloc[0]
            if pd.notna(rt):
                no_finetune_runtime[ds] = float(rt)
            if "runtime_s_std" in df_nf.columns and pd.notna(df_nf["runtime_s_std"].iloc[0]):
                no_finetune_std[ds] = float(df_nf["runtime_s_std"].iloc[0])
        except Exception:
            continue

    all_data = load_finetune_results_by_dataset(results_dir)
    all_data = [(ds, df) for ds, df in all_data if "runtime_s" in df.columns]

    if not all_data:
        print("No valid CSV files found for plotting.")
        return results_dir / "corr_threshold_vs_runtime.png"

    n_datasets = len(all_data)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (dataset, df) in enumerate(all_data):
        ax = axes.flat[idx]
        finetune_mask = df["finetune_coordinate"].astype(str).str.lower() == "true"
        finetune_df = df[finetune_mask].copy()

        finetune_df = finetune_df.dropna(subset=["corr_threshold", "runtime_s"])
        finetune_df = finetune_df.sort_values("corr_threshold")
        if len(finetune_df) > 0:
            x = finetune_df["corr_threshold"].values
            y = finetune_df["runtime_s"].values
            if "runtime_s_std" in finetune_df.columns and finetune_df["runtime_s_std"].notna().any():
                yerr = finetune_df["runtime_s_std"].fillna(0.0).values
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    color="lightblue",
                    alpha=0.35,
                    linewidth=0,
                )
            ax.plot(
                x,
                y,
                marker="o",
                linestyle="-",
                markersize=6,
                color="C0",
                label="finetune (mean ± std)",
            )
        title_ds = str(dataset).replace("_", " ").title()
        rt = no_finetune_runtime.get(str(dataset))
        rt_std = no_finetune_std.get(str(dataset))
        if rt is not None:
            if rt_std is not None and rt_std > 0:
                title = f"{title_ds}: No-finetune - {rt:.2f} ± {rt_std:.2f}s"
            else:
                title = f"{title_ds}: No-finetune - {rt:.2f}s"
        else:
            title = title_ds
        ax.set_xlabel("Correlation threshold")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    for j in range(idx + 1, len(axes.flat)):
        axes.flat[j].set_visible(False)
    plt.tight_layout()
    plot_path = results_dir / "corr_threshold_vs_runtime.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved corr_threshold vs runtime plot to {plot_path}")
    return plot_path

DEFAULT_DATASETS = [
    "heart.csv",
    "spambase.csv",
    "fico.csv",
    "compas.csv",
    "covertype.csv",
    "mgh.csv",
]

EPS_OVERRIDE = {
    "heart": 0.3,
    "spambase": 0.1,
    "fico": 0.1,
    "compas": 0.1,
    "covertype": 0.1,
    "mgh": 0.5,
}

def select_features_l1(X, y, feature_names, feature_groups, max_features=MAX_FEATURES, C=100.0):
    n_cols = X.shape[1]
    if n_cols <= max_features:
        return X, feature_names, feature_groups
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            penalty="l1", solver="saga", fit_intercept=False,
            C=C, max_iter=1000,
        )
        clf.fit(X, y)
    coef = np.abs(clf.coef_.ravel())
    non_intercept = np.argsort(coef[1:])[::-1][: max_features - 1] + 1
    keep_idx = [0] + list(non_intercept)
    X_sel = X[:, keep_idx]
    new_index_map = {old: new for new, old in enumerate(keep_idx)}
    feature_names_sel = [feature_names[i] for i in keep_idx]
    feature_groups_sel = {}
    for feat_name, old_indices in feature_groups.items():
        new_indices = [new_index_map[old] for old in old_indices if old in new_index_map]
        if new_indices:
            feature_groups_sel[feat_name] = new_indices
    return X_sel, feature_names_sel, feature_groups_sel


def compute_diversity_metrics(result: RashomonResult, X, feature_groups):
    """Pairwise diversity metrics on a RashomonResult."""
    if result.n_accepted < 2:
        return {k: 0.0 for k in METRIC_KEYS}
    _, ph = pairwise_prediction_hamming(result, X)
    _, sh = pairwise_hamming(result)
    _, sj = pairwise_support_jaccard(result)
    _, wl = pairwise_weight_l1(result)
    sl = pairwise_shape_distance(result, feature_groups)[1] if feature_groups else 0.0
    mono = monotonicity_diversity(result, feature_groups) if feature_groups else {"mean_entropy": 0.0}
    return {
        "pred_hamming": ph,
        "support_hamming": sh,
        "support_jaccard": sj,
        "weight_l1": wl,
        "shape_l1": sl,
        "monotonicity_entropy": mono["mean_entropy"],
    }

def run_single(
    X, y, feature_groups,
    finetune_coordinate: bool,
    finetune_corr_threshold: float = 0.3,
    n_steps: int = 5000,
    burn_in: int = 500,
    target_models: int = 100,
    eps: float = EPS,
    trial_seed: int = 42,
):
    """Run MCMC once with given finetune_coordinate setting.
    finetune_corr_threshold is only used when finetune_coordinate=True."""
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    sampler = MCMCRashomonSampler(
        X, y, eps=eps, sigma2=SIGMA2, p_feat=P_FEAT,
        feature_groups=feature_groups,
    )
    t0 = time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sampler.sample(
            n_steps=n_steps,
            burn_in=burn_in,
            target_models=target_models,
            beta=0.5,
            proposal="mixture",
            mh_variant="standard",
            ellipsoid_augment=True,
            finetune_coordinate=finetune_coordinate,
            finetune_corr_threshold=finetune_corr_threshold,
        )
    runtime = time() - t0

    errs = np.asarray(result.errors, dtype=np.float64)
    if errs.size > 0:
        avg_loss = float(np.mean(errs))
        # Mean training accuracy across models in the Rashomon set (errors = train misclassification rate)
        avg_train_accuracy = float(np.mean(1.0 - errs))
    else:
        avg_loss = float("nan")
        avg_train_accuracy = float("nan")
    div = compute_diversity_metrics(result, X, feature_groups)

    return {
        "avg_loss": avg_loss,
        "best_error": result.best_error,
        "n_models": result.n_accepted,
        "avg_train_accuracy": avg_train_accuracy,
        "acceptance_rate": result.acceptance_rate,
        "runtime_s": runtime,
        "diversity": div,
    }


def run_single_trials(
    X, y, feature_groups,
    finetune_coordinate: bool,
    n_trials: int = 5,
    base_seed: int = 42,
    finetune_corr_threshold: float = 0.3,
    n_steps: int = 5000,
    burn_in: int = 500,
    target_models: int = 100,
    eps: float = EPS,
):
    """
    Run ``run_single`` ``n_trials`` times with different RNG seeds.
    Returns mean/std of runtime, n_models, avg_train_accuracy, and diversity metrics across trials.
    """
    runtimes = []
    trial_results = []
    for t in range(n_trials):
        seed = base_seed + t * 7919
        res = run_single(
            X, y, feature_groups,
            finetune_coordinate=finetune_coordinate,
            finetune_corr_threshold=finetune_corr_threshold,
            n_steps=n_steps,
            burn_in=burn_in,
            target_models=target_models,
            eps=eps,
            trial_seed=seed,
        )
        runtimes.append(res["runtime_s"])
        trial_results.append(res)
    runtimes = np.asarray(runtimes, dtype=np.float64)
    mean_rt = float(np.mean(runtimes))
    std_rt = float(np.std(runtimes, ddof=1)) if n_trials > 1 else 0.0

    def _mean(key, sub=None):
        if sub is None:
            vals = [r[key] for r in trial_results]
        else:
            vals = [r[key][sub] for r in trial_results]
        return float(np.nanmean(vals))

    div_mean = {
        k: _mean("diversity", k)
        for k in METRIC_KEYS
    }
    div_std = {}
    for k in METRIC_KEYS:
        vals = np.asarray([r["diversity"][k] for r in trial_results], dtype=np.float64)
        div_std[k] = float(np.std(vals, ddof=1)) if n_trials > 1 else 0.0

    n_models_vals = np.asarray([r["n_models"] for r in trial_results], dtype=np.float64)
    n_models_std = float(np.std(n_models_vals, ddof=1)) if n_trials > 1 else 0.0

    at_vals = np.asarray([r["avg_train_accuracy"] for r in trial_results], dtype=np.float64)
    avg_train_accuracy_std = float(np.std(at_vals, ddof=1)) if n_trials > 1 else 0.0

    ar_vals = np.asarray([r["acceptance_rate"] for r in trial_results], dtype=np.float64)
    acceptance_rate_std = float(np.std(ar_vals, ddof=1)) if n_trials > 1 else 0.0

    return {
        "avg_loss": _mean("avg_loss"),
        "best_error": _mean("best_error"),
        "n_models": _mean("n_models"),
        "n_models_std": n_models_std,
        "avg_train_accuracy": _mean("avg_train_accuracy"),
        "avg_train_accuracy_std": avg_train_accuracy_std,
        "acceptance_rate": _mean("acceptance_rate"),
        "acceptance_rate_std": acceptance_rate_std,
        "runtime_s": mean_rt,
        "runtime_s_std": std_rt,
        "diversity": div_mean,
        "diversity_std": div_std,
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare MCMC with vs without finetune_coordinate"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single dataset (e.g. compas.csv). If not set, runs all default datasets.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(BENCHMARK_DIR),
    )
    parser.add_argument(
        "--output",
        default="results_finetune_comparison.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Rashomon epsilon (default: 0.05, overridden per dataset)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--target_models",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--corr_thresholds",
        type=str,
        default="0.05,0.1,0.2,0.25,0.3,0.5",
        help="Comma-separated correlation thresholds for finetune. Used only in batch mode.",
    )
    parser.add_argument(
        "--no_finetune",
        action="store_true",
        help="Single-job mode: run only the non-finetuned variant.",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=None,
        help="Single-job mode: run only finetune with this correlation threshold.",
    )
    parser.add_argument(
        "--merge",
        metavar="DIR",
        help="Merge partial CSV files into per-dataset CSVs. Pass output directory.",
    )
    parser.add_argument(
        "--plot",
        metavar="DIR",
        nargs="?",
        const="results_finetune_comparison",
        help=(
            "Plot corr_threshold vs runtime, vs each diversity metric, vs avg train accuracy, "
            "vs n_models, and vs acceptance_rate from CSVs (one subplot per dataset per figure). "
            "Pass results directory (default: results_finetune_comparison)."
        ),
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of MCMC repeats per setting; report mean/std runtime (default: 5).",
    )
    args = parser.parse_args()

    if args.plot is not None:
        plot_corr_threshold_vs_runtime(Path(args.plot))
        plot_all_diversity_figures(Path(args.plot))
        return

    if args.merge:
        merge_results(Path(args.merge), Path(args.dataset).stem if args.dataset else None)
        plot_corr_threshold_vs_runtime(Path(args.merge))
        plot_all_diversity_figures(Path(args.merge))
        return

    single_job = args.no_finetune or (args.corr_threshold is not None)
    if args.no_finetune and args.corr_threshold is not None:
        parser.error("Cannot use both --no_finetune and --corr_threshold")

    corr_thresholds = [float(x.strip()) for x in args.corr_thresholds.split(",")]
    if not single_job:
        print(f"Correlation thresholds for finetune: {corr_thresholds}")

    data_dir = Path(args.data_dir)
    datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS

    rows = []
    for ds_name in datasets:
        data_path = data_dir / ds_name if not Path(ds_name).is_absolute() else Path(ds_name)
        if not data_path.exists():
            data_path = data_dir / Path(ds_name).name
        if not data_path.exists():
            print(f"Skip {ds_name}: not found")
            continue

        dataset_stem = data_path.stem
        eps = args.eps if args.eps is not None else EPS_OVERRIDE.get(dataset_stem, EPS)

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_stem} (eps={eps})")
        print("=" * 60)

        data = pd.read_csv(data_path)
        # fill nan values with mean of the column
        data = data.fillna(data.mean())
        if len(data) > MAX_SAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(data), size=MAX_SAMPLES, replace=False)
            data = data.iloc[idx].reset_index(drop=True)
            print(f"Subsampled to {MAX_SAMPLES} rows")

        X, y, feature_names, feature_groups = binarize_data(data)
        if X.shape[1] > MAX_FEATURES:
            X, feature_names, feature_groups = select_features_l1(
                X, y, feature_names, feature_groups
            )
            print(f"L1 feature selection -> {MAX_FEATURES} features")

        # Non-finetuned variant: run once (independent of correlation threshold)
        if not single_job or args.no_finetune:
            print(
                f"\n  Running finetune_coordinate=False (no-finetune baseline), "
                f"n_trials={args.n_trials} ..."
            )
            try:
                res = run_single_trials(
                    X, y, feature_groups,
                    finetune_coordinate=False,
                    n_trials=args.n_trials,
                    finetune_corr_threshold=0.3,
                    n_steps=args.n_steps,
                    burn_in=args.burn_in,
                    target_models=args.target_models,
                    eps=eps,
                )
                row = {
                    "dataset": dataset_stem,
                    "finetune_coordinate": False,
                    "corr_threshold": float("nan"),
                    "avg_loss": res["avg_loss"],
                    "best_error": res["best_error"],
                    "n_models": res["n_models"],
                    "n_models_std": res["n_models_std"],
                    "avg_train_accuracy": res["avg_train_accuracy"],
                    "avg_train_accuracy_std": res["avg_train_accuracy_std"],
                    "acceptance_rate": res["acceptance_rate"],
                    "acceptance_rate_std": res["acceptance_rate_std"],
                    "runtime_s": res["runtime_s"],
                    "runtime_s_std": res["runtime_s_std"],
                    "n_trials": res["n_trials"],
                }
                for k in METRIC_KEYS:
                    row[f"diversity_{k}"] = res["diversity"][k]
                    row[f"diversity_{k}_std"] = res["diversity_std"][k]
                rows.append(row)
                print(
                    f"    avg_loss={res['avg_loss']:.4f}, n_models={res['n_models']:.1f}±{res['n_models_std']:.1f}, "
                    f"avg_train_acc={res['avg_train_accuracy']:.4f}±{res['avg_train_accuracy_std']:.4f}, "
                    f"runtime={res['runtime_s']:.1f}±{res['runtime_s_std']:.1f}s, "
                    f"pred_hamming={res['diversity']['pred_hamming']:.4f}"
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                rows.append({
                    "dataset": dataset_stem,
                    "finetune_coordinate": False,
                    "corr_threshold": float("nan"),
                    "avg_loss": float("nan"),
                    "best_error": float("nan"),
                    "n_models": 0,
                    "n_models_std": float("nan"),
                    "avg_train_accuracy": float("nan"),
                    "avg_train_accuracy_std": float("nan"),
                    "acceptance_rate": float("nan"),
                    "acceptance_rate_std": float("nan"),
                    "runtime_s": float("nan"),
                    "runtime_s_std": float("nan"),
                    "n_trials": args.n_trials,
                    **{f"diversity_{k}": float("nan") for k in METRIC_KEYS},
                    **{f"diversity_{k}_std": float("nan") for k in METRIC_KEYS},
                })

        # Finetuned variants: one run per correlation threshold (or single threshold in single-job mode)
        if single_job and args.no_finetune:
            thresholds_to_run = []  # no finetune variants when running no-finetune job
        elif single_job and args.corr_threshold is not None:
            thresholds_to_run = [args.corr_threshold]
        else:
            thresholds_to_run = corr_thresholds
        for corr_th in thresholds_to_run:
            print(
                f"\n  Running finetune_coordinate=True (corr_threshold={corr_th}), "
                f"n_trials={args.n_trials} ..."
            )
            try:
                res = run_single_trials(
                    X, y, feature_groups,
                    finetune_coordinate=True,
                    n_trials=args.n_trials,
                    finetune_corr_threshold=corr_th,
                    n_steps=args.n_steps,
                    burn_in=args.burn_in,
                    target_models=args.target_models,
                    eps=eps,
                )
                row = {
                    "dataset": dataset_stem,
                    "finetune_coordinate": True,
                    "corr_threshold": corr_th,
                    "avg_loss": res["avg_loss"],
                    "best_error": res["best_error"],
                    "n_models": res["n_models"],
                    "n_models_std": res["n_models_std"],
                    "avg_train_accuracy": res["avg_train_accuracy"],
                    "avg_train_accuracy_std": res["avg_train_accuracy_std"],
                    "acceptance_rate": res["acceptance_rate"],
                    "acceptance_rate_std": res["acceptance_rate_std"],
                    "runtime_s": res["runtime_s"],
                    "runtime_s_std": res["runtime_s_std"],
                    "n_trials": res["n_trials"],
                }
                for k in METRIC_KEYS:
                    row[f"diversity_{k}"] = res["diversity"][k]
                    row[f"diversity_{k}_std"] = res["diversity_std"][k]
                rows.append(row)
                print(
                    f"    avg_loss={res['avg_loss']:.4f}, n_models={res['n_models']:.1f}±{res['n_models_std']:.1f}, "
                    f"avg_train_acc={res['avg_train_accuracy']:.4f}±{res['avg_train_accuracy_std']:.4f}, "
                    f"runtime={res['runtime_s']:.1f}±{res['runtime_s_std']:.1f}s, "
                    f"pred_hamming={res['diversity']['pred_hamming']:.4f}"
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                rows.append({
                    "dataset": dataset_stem,
                    "finetune_coordinate": True,
                    "corr_threshold": corr_th,
                    "avg_loss": float("nan"),
                    "best_error": float("nan"),
                    "n_models": 0,
                    "n_models_std": float("nan"),
                    "avg_train_accuracy": float("nan"),
                    "avg_train_accuracy_std": float("nan"),
                    "acceptance_rate": float("nan"),
                    "acceptance_rate_std": float("nan"),
                    "runtime_s": float("nan"),
                    "runtime_s_std": float("nan"),
                    "n_trials": args.n_trials,
                    **{f"diversity_{k}": float("nan") for k in METRIC_KEYS},
                    **{f"diversity_{k}_std": float("nan") for k in METRIC_KEYS},
                })

    if rows:
        df = pd.DataFrame(rows)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
