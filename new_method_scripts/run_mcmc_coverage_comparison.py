"""
Compare Rashomon-set coverage between short (5k) and long (5M) MCMC runs.

The script computes directed nearest-neighbor coverage statistics:
    small -> large  and  large -> small
using configurable model distance (default: euclidean).
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from new_method_scripts.ellipsoid import EllipsoidSampler
from new_method_scripts.mcmc_rashomon import MCMCRashomonSampler, RashomonResult, binarize_data
from new_method_scripts.run_finetune_comparison import (
    BENCHMARK_DIR,
    EPS,
    EPS_OVERRIDE,
    MAX_FEATURES,
    MAX_SAMPLES,
    P_FEAT,
    SIGMA2,
    DEFAULT_DATASETS,
    select_features_l1,
)
from new_method_scripts.set_coverage_metrics import bidirectional_coverage_summary


def _run_single_mcmc(
    X: np.ndarray,
    y: np.ndarray,
    feature_groups: Dict[str, List[int]],
    eps: float,
    n_steps: int,
    burn_in: int,
    target_models: int,
    proposal: str,
    mh_variant: str,
    ellipsoid_augment: bool,
    ellipsoid_n_samples: int,
    repulsion_weight: float,
    finetune_coordinate: bool,
    finetune_corr_threshold: float,
    use_hessian_update: bool,
    seed: int,
) -> RashomonResult:
    random.seed(seed)
    np.random.seed(seed)
    sampler = MCMCRashomonSampler(
        X=X,
        y=y,
        eps=eps,
        sigma2=SIGMA2,
        p_feat=P_FEAT,
        feature_groups=feature_groups,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sampler.sample(
            n_steps=n_steps,
            burn_in=burn_in,
            target_models=target_models,
            beta=0.5,
            proposal=proposal,
            mh_variant=mh_variant,
            ellipsoid_augment=ellipsoid_augment,
            ellipsoid_n_samples=ellipsoid_n_samples,
            repulsion_weight=repulsion_weight,
            finetune_coordinate=finetune_coordinate,
            finetune_corr_threshold=finetune_corr_threshold,
            use_hessian_update=use_hessian_update,
        )


def _run_single_ellipsoid(
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    n_samples: int,
    seed: int,
    w_center: Optional[np.ndarray] = None,
    best_nll: Optional[float] = None,
) -> RashomonResult:
    random.seed(seed)
    np.random.seed(seed)
    sampler = EllipsoidSampler(X, y, eps=eps, sigma2=SIGMA2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sampler.sample(
            n_samples=n_samples,
            sampling="uniform",
            eps=eps,
            w_center=w_center,
            best_nll=best_nll,
        )


def _summarize_trials(rows: List[Dict[str, float]], key: str) -> tuple[float, float]:
    arr = np.asarray([r[key] for r in rows], dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 5k vs 5M MCMC Rashomon sets with directed coverage metrics."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single dataset file (default: run all defaults).",
    )
    parser.add_argument("--data_dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--output_dir", default="results_mcmc_coverage_comparison")
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--n_steps_small", type=int, default=5_000)
    parser.add_argument("--n_steps_large", type=int, default=5_000_000)
    parser.add_argument("--burn_in_small", type=int, default=500)
    parser.add_argument("--burn_in_large", type=int, default=50_000)
    parser.add_argument("--target_models_small", type=int, default=100)
    parser.add_argument("--target_models_large", type=int, default=300)
    parser.add_argument("--proposal", default="mixture")
    parser.add_argument("--mh_variant", default="standard", choices=["standard", "repulsive"])
    parser.add_argument("--ellipsoid_augment", action="store_true")
    parser.add_argument("--ellipsoid_n_samples", type=int, default=50)
    parser.add_argument("--repulsion_weight", type=float, default=0.03)
    parser.add_argument("--finetune_coordinate", action="store_true")
    parser.add_argument("--finetune_corr_threshold", type=float, default=0.05)
    parser.add_argument("--disable_hessian_update", action="store_true")
    parser.add_argument(
        "--distance_metric",
        default="euclidean",
        choices=["euclidean", "l1", "support_jaccard", "prediction_hamming"],
    )
    parser.add_argument(
        "--include_ellipsoid",
        action="store_true",
        help="Also compare a standalone ellipsoid set to the large MCMC set.",
    )
    parser.add_argument(
        "--ellipsoid_baseline_n_samples",
        type=int,
        default=10000,
        help="Candidates for standalone ellipsoid baseline (--include_ellipsoid).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS
    summary_rows: List[Dict[str, object]] = []

    for ds_name in datasets:
        data_path = Path(ds_name)
        if not data_path.is_absolute():
            data_path = data_dir / ds_name
        if not data_path.exists():
            data_path = data_dir / Path(ds_name).name
        if not data_path.exists():
            print(f"Skip {ds_name}: dataset not found")
            continue

        dataset_stem = data_path.stem
        eps = args.eps if args.eps is not None else EPS_OVERRIDE.get(dataset_stem, EPS)
        print(f"\n{'=' * 72}")
        print(f"Dataset={dataset_stem} eps={eps} trials={args.n_trials}")
        print(f"{'=' * 72}")

        data = pd.read_csv(data_path)
        data = data.fillna(data.mean(numeric_only=True))
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

        trial_rows: List[Dict[str, object]] = []
        for trial_idx in range(args.n_trials):
            seed = args.base_seed + trial_idx * 7919
            print(f"Trial {trial_idx + 1}/{args.n_trials}, seed={seed}")

            t_small = time()
            small_result = _run_single_mcmc(
                X=X,
                y=y,
                feature_groups=feature_groups,
                eps=eps,
                n_steps=args.n_steps_small,
                burn_in=args.burn_in_small,
                target_models=args.target_models_small,
                proposal=args.proposal,
                mh_variant=args.mh_variant,
                ellipsoid_augment=args.ellipsoid_augment,
                ellipsoid_n_samples=args.ellipsoid_n_samples,
                repulsion_weight=args.repulsion_weight,
                finetune_coordinate=args.finetune_coordinate,
                finetune_corr_threshold=args.finetune_corr_threshold,
                use_hessian_update=not args.disable_hessian_update,
                seed=seed,
            )
            runtime_small = time() - t_small

            t_large = time()
            large_result = _run_single_mcmc(
                X=X,
                y=y,
                feature_groups=feature_groups,
                eps=eps,
                n_steps=args.n_steps_large,
                burn_in=args.burn_in_large,
                target_models=args.target_models_large,
                proposal=args.proposal,
                mh_variant=args.mh_variant,
                ellipsoid_augment=args.ellipsoid_augment,
                ellipsoid_n_samples=args.ellipsoid_n_samples,
                repulsion_weight=args.repulsion_weight,
                finetune_coordinate=args.finetune_coordinate,
                finetune_corr_threshold=args.finetune_corr_threshold,
                use_hessian_update=not args.disable_hessian_update,
                seed=seed + 1_000_003,
            )
            runtime_large = time() - t_large

            x_eval = X if args.distance_metric == "prediction_hamming" else None
            cov = bidirectional_coverage_summary(
                set_a_models=small_result.w_models,
                set_b_models=large_result.w_models,
                distance=args.distance_metric,
                x_eval=x_eval,
            )

            trial_row: Dict[str, object] = {
                "dataset": dataset_stem,
                "trial_idx": trial_idx,
                "seed_small": seed,
                "seed_large": seed + 1_000_003,
                "distance_metric": args.distance_metric,
                "n_steps_small": args.n_steps_small,
                "n_steps_large": args.n_steps_large,
                "n_models_small": int(small_result.n_accepted),
                "n_models_large": int(large_result.n_accepted),
                "acceptance_rate_small": float(small_result.acceptance_rate),
                "acceptance_rate_large": float(large_result.acceptance_rate),
                "runtime_small_s": float(runtime_small),
                "runtime_large_s": float(runtime_large),
                "small_to_large_mean_epsilon": cov["a_to_b"].mean_epsilon,
                "small_to_large_min_epsilon": cov["a_to_b"].min_epsilon,
                "small_to_large_max_epsilon": cov["a_to_b"].max_epsilon,
                "large_to_small_mean_epsilon": cov["b_to_a"].mean_epsilon,
                "large_to_small_min_epsilon": cov["b_to_a"].min_epsilon,
                "large_to_small_max_epsilon": cov["b_to_a"].max_epsilon,
            }

            if args.include_ellipsoid:
                ellipsoid_result = _run_single_ellipsoid(
                    X=X,
                    y=y,
                    eps=eps,
                    n_samples=args.ellipsoid_baseline_n_samples,
                    seed=seed + 2_000_003,
                    w_center=small_result.w_opt,
                    best_nll=small_result.best_nll,
                )
                ell_cov = bidirectional_coverage_summary(
                    set_a_models=ellipsoid_result.w_models,
                    set_b_models=large_result.w_models,
                    distance=args.distance_metric,
                    x_eval=x_eval,
                )
                trial_row.update(
                    {
                        "n_models_ellipsoid": int(ellipsoid_result.n_accepted),
                        "ellipsoid_to_large_mean_epsilon": ell_cov["a_to_b"].mean_epsilon,
                        "ellipsoid_to_large_min_epsilon": ell_cov["a_to_b"].min_epsilon,
                        "ellipsoid_to_large_max_epsilon": ell_cov["a_to_b"].max_epsilon,
                        "large_to_ellipsoid_mean_epsilon": ell_cov["b_to_a"].mean_epsilon,
                        "large_to_ellipsoid_min_epsilon": ell_cov["b_to_a"].min_epsilon,
                        "large_to_ellipsoid_max_epsilon": ell_cov["b_to_a"].max_epsilon,
                    }
                )

            trial_rows.append(trial_row)

        summary: Dict[str, object] = {
            "dataset": dataset_stem,
            "distance_metric": args.distance_metric,
            "n_trials": args.n_trials,
            "n_steps_small": args.n_steps_small,
            "n_steps_large": args.n_steps_large,
            "mh_variant": args.mh_variant,
            "proposal": args.proposal,
            "finetune_coordinate": args.finetune_coordinate,
            "finetune_corr_threshold": args.finetune_corr_threshold,
            "ellipsoid_augment": args.ellipsoid_augment,
        }
        stats_keys = [
            "small_to_large_mean_epsilon",
            "small_to_large_min_epsilon",
            "small_to_large_max_epsilon",
            "large_to_small_mean_epsilon",
            "large_to_small_min_epsilon",
            "large_to_small_max_epsilon",
            "n_models_small",
            "n_models_large",
            "acceptance_rate_small",
            "acceptance_rate_large",
            "runtime_small_s",
            "runtime_large_s",
        ]
        if args.include_ellipsoid:
            stats_keys.extend(
                [
                    "n_models_ellipsoid",
                    "ellipsoid_to_large_mean_epsilon",
                    "ellipsoid_to_large_min_epsilon",
                    "ellipsoid_to_large_max_epsilon",
                    "large_to_ellipsoid_mean_epsilon",
                    "large_to_ellipsoid_min_epsilon",
                    "large_to_ellipsoid_max_epsilon",
                ]
            )

        for key in stats_keys:
            mean_val, std_val = _summarize_trials(trial_rows, key)
            summary[key] = mean_val
            summary[f"{key}_std"] = std_val

        summary_rows.append(summary)

        dataset_json = output_dir / f"{dataset_stem}_coverage_trials.json"
        dataset_csv = output_dir / f"{dataset_stem}_coverage_trials.csv"
        dataset_summary_csv = output_dir / f"{dataset_stem}_coverage_summary.csv"
        dataset_json.write_text(json.dumps({"summary": summary, "trials": trial_rows}, indent=2))
        pd.DataFrame(trial_rows).to_csv(dataset_csv, index=False)
        pd.DataFrame([summary]).to_csv(dataset_summary_csv, index=False)
        print(f"Saved: {dataset_json}")
        print(f"Saved: {dataset_csv}")
        print(f"Saved: {dataset_summary_csv}")

    if summary_rows:
        merged_summary = output_dir / "coverage_summary_all_datasets.csv"
        pd.DataFrame(summary_rows).to_csv(merged_summary, index=False)
        print(f"\nMerged summary saved: {merged_summary}")
    else:
        print("No datasets completed; nothing to save.")


if __name__ == "__main__":
    main()
