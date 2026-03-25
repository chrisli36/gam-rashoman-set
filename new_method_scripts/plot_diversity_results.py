"""
Plot diversity experiment results: repulsion weight vs diversity metrics.

Parses CSV results and plots MCMC+ellipsoid (Method 1) vs EllipsoidSampler baseline
(Method 2) for each repulsive kernel (diversity function) in separate subplots.

Usage:
    python plot_diversity_results.py results_diversity/compas/diversity_results.csv
    python plot_diversity_results.py results_diversity/compas/diversity_results.csv -o plot.png

Each subplot shows the matching metric: prediction_hamming -> pred_hamming,
support_jaccard -> support_hamming, l1 -> weight_l1.
"""

import argparse
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Map each diversity function (repulsive kernel) to its matching metric
DIVERSITY_TO_METRIC = {
    "prediction_hamming": "pred_hamming",
    "support_jaccard": "support_hamming",
    "l1": "weight_l1",
    "monotonicity_entropy": "monotonicity_entropy",
}
DIVERSITY_FN_LABELS = {
    "prediction_hamming": "Prediction Hamming",
    "support_jaccard": "Support Jaccard",
    "l1": "L1",
    "monotonicity_entropy": "Monotonicity Entropy",
}


def parse_results(csv_path: Union[str, Path]):
    """
    Parse diversity experiment results from CSV.

    Args:
        csv_path: Path to the CSV file (e.g. diversity_results.csv).

    Returns:
        df: DataFrame with columns diversity_fn, repulsion_weight, mcmc_*, ellipsoid_*, etc.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["repulsion_weight"] = df["repulsion_weight"].astype(float)
    return df


def _get_mcmc_col(df, metric: str) -> Optional[str]:
    """Return MCMC column name (new: mcmc_*, legacy: overall_*)."""
    for prefix in ("mcmc_", "overall_"):
        col = f"{prefix}{metric}"
        if col in df.columns:
            return col
    return None


def _get_ellipsoid_cols(df, metric: str) -> tuple:
    """Return (avg_col, std_col) for Ellipsoid (new: ellipsoid_*, legacy: persup_*)."""
    for prefix in ("ellipsoid_", "persup_"):
        avg_col = f"{prefix}{metric}_avg"
        std_col = f"{prefix}{metric}_std"
        if avg_col in df.columns:
            return (avg_col, std_col if std_col in df.columns else None)
    return (None, None)


def plot_diversity_results(
    csv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 5),
):
    """
    Plot repulsion weights vs the relevant diversity metric for each diversity function.

    One subplot per diversity function. In each subplot, X = repulsion weight,
    Y = diversity value. Plots:
      - Method 1: MCMC+ellipsoid (mcmc_<metric>)
      - Method 2: EllipsoidSampler per support (ellipsoid_<metric>_avg ± std)

    Args:
        csv_path: Path to the CSV results file.
        output_path: Path to save the figure. If None, displays interactively.
        figsize: Figure size (width, height).
    """
    df = parse_results(csv_path)

    diversity_fns = df["diversity_fn"].unique()
    n_subplots = len(diversity_fns)
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize, sharey=False)
    if n_subplots == 1:
        axes = [axes]

    for ax, div_fn in zip(axes, diversity_fns):
        metric = DIVERSITY_TO_METRIC.get(div_fn)
        if not metric:
            continue
        sub = df[df["diversity_fn"] == div_fn].sort_values("repulsion_weight")
        x = sub["repulsion_weight"].values
        # if there is a 0 in x (or x < 1e-10) replace it with 1e-10 
        x = np.where(x < 1e-10, 1e-10, x)

        mcmc_col = _get_mcmc_col(sub, metric)
        ell_avg_col, ell_std_col = _get_ellipsoid_cols(sub, metric)

        if mcmc_col is not None:
            ax.plot(x, sub[mcmc_col].values, "o-", color="C0",
                    label="MCMC + ellipsoid", linewidth=2, markersize=6, alpha=0.7)
        if ell_avg_col is not None:
            y = sub[ell_avg_col].values
            yerr = sub[ell_std_col].values if ell_std_col else None
            ax.errorbar(x, y, yerr=yerr, fmt="s-", color="C1",
                        label="EllipsoidSampler (per support)", linewidth=2, markersize=6,
                        capsize=8, capthick=1.5, alpha=0.7)

        ax.set_xlabel("Repulsion weight", fontsize=11)
        ax.set_ylabel("Diversity", fontsize=11)
        ax.set_title(DIVERSITY_FN_LABELS.get(div_fn, div_fn), fontsize=12)
        ax.set_xscale("log")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot diversity experiment: repulsion vs diversity metrics"
    )
    parser.add_argument(
        "csv_path",
        help="Path to results CSV (e.g. results_compas_diversity/compas_diversity_results.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output figure path (default: display only)",
    )
    args = parser.parse_args()

    plot_diversity_results(
        args.csv_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
