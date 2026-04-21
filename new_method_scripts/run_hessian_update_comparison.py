"""
Benchmark runtime impact of Hessian updates with finetune coordinate updates.

Default behavior runs:
    (finetune_coordinate, use_hessian_update) in
    {(True, False), (True, True)}

When finetune_coordinate=True, this script enforces:
    finetune_corr_threshold = 0.05

Example:
    python run_hessian_update_comparison.py --dataset compas.csv --n_trials 5
"""

import argparse
import random
import sys
import warnings
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_method_scripts.mcmc_rashomon import MCMCRashomonSampler, binarize_data
from new_method_scripts.run_finetune_comparison import (
    BENCHMARK_DIR,
    EPS,
    EPS_OVERRIDE,
    MAX_FEATURES,
    MAX_SAMPLES,
    P_FEAT,
    SIGMA2,
    select_features_l1,
)


def run_single_trial(
    X: np.ndarray,
    y: np.ndarray,
    feature_groups: Dict[str, List[int]],
    eps: float,
    finetune_coordinate: bool,
    use_hessian_update: bool,
    n_steps: int,
    burn_in: int,
    target_models: int,
    seed: int,
) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    sampler = MCMCRashomonSampler(
        X, y, eps=eps, sigma2=SIGMA2, p_feat=P_FEAT, feature_groups=feature_groups
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
            mh_variant="repulsive",
            ellipsoid_augment=True,
            finetune_coordinate=finetune_coordinate,
            finetune_corr_threshold=0.05 if finetune_coordinate else 0.3,
            use_hessian_update=use_hessian_update,
        )
    runtime_s = time() - t0
    return {
        "runtime_s": float(runtime_s),
        "n_models": float(result.n_accepted),
        "acceptance_rate": float(result.acceptance_rate),
    }


def summarize_trials(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare use_hessian_update True vs False for selected finetune setting(s)."
    )
    parser.add_argument("--dataset", default="compas.csv", help="Dataset file name.")
    parser.add_argument("--data_dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--burn_in", type=int, default=500)
    parser.add_argument("--target_models", type=int, default=100)
    parser.add_argument("--output", default="results_hessian_update_comparison.csv")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Override Rashomon epsilon (otherwise uses dataset-specific default).",
    )
    parser.add_argument(
        "--finetune_coordinate",
        type=parse_bool,
        default=None,
        help="Set finetune_coordinate explicitly (True/False).",
    )
    parser.add_argument(
        "--use_hessian_update",
        type=parse_bool,
        default=None,
        help="Set use_hessian_update explicitly (True/False).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_path = Path(args.dataset)
    if not data_path.is_absolute():
        data_path = data_dir / args.dataset
    if not data_path.exists():
        data_path = data_dir / Path(args.dataset).name
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    dataset_stem = data_path.stem
    eps = args.eps if args.eps is not None else EPS_OVERRIDE.get(dataset_stem, EPS)
    print(f"Dataset: {dataset_stem}, eps={eps}, n_trials={args.n_trials}")

    data = pd.read_csv(data_path)
    data = data.fillna(data.mean())
    if len(data) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=MAX_SAMPLES, replace=False)
        data = data.iloc[idx].reset_index(drop=True)
        print(f"Subsampled to {MAX_SAMPLES} rows")

    X, y, feature_names, feature_groups = binarize_data(data)
    if X.shape[1] > MAX_FEATURES:
        X, feature_names, feature_groups = select_features_l1(X, y, feature_names, feature_groups)
        print(f"L1 feature selection -> {MAX_FEATURES} features")

    if (args.finetune_coordinate is None) != (args.use_hessian_update is None):
        raise ValueError(
            "Provide both --finetune_coordinate and --use_hessian_update, or neither."
        )

    if args.finetune_coordinate is None:
        settings = [
            (True, False),
            (True, True),
        ]
    else:
        settings = [
            (args.finetune_coordinate, args.use_hessian_update),
        ]
    rows = []

    for finetune_coordinate, use_hessian_update in settings:
        runtimes = []
        n_models = []
        accept_rates = []
        label = f"finetune={finetune_coordinate}, hessian_update={use_hessian_update}"
        print(f"Running setting: {label}")
        for t in range(args.n_trials):
            seed = args.base_seed + t * 7919
            out = run_single_trial(
                X=X,
                y=y,
                feature_groups=feature_groups,
                eps=eps,
                finetune_coordinate=finetune_coordinate,
                use_hessian_update=use_hessian_update,
                n_steps=args.n_steps,
                burn_in=args.burn_in,
                target_models=args.target_models,
                seed=seed,
            )
            runtimes.append(out["runtime_s"])
            n_models.append(out["n_models"])
            accept_rates.append(out["acceptance_rate"])
        rt_mean, rt_std = summarize_trials(runtimes)
        nm_mean, nm_std = summarize_trials(n_models)
        ar_mean, ar_std = summarize_trials(accept_rates)
        rows.append(
            {
                "dataset": dataset_stem,
                "finetune_coordinate": finetune_coordinate,
                "use_hessian_update": use_hessian_update,
                "finetune_corr_threshold": 0.05 if finetune_coordinate else np.nan,
                "runtime_s": rt_mean,
                "runtime_s_std": rt_std,
                "n_models": nm_mean,
                "n_models_std": nm_std,
                "acceptance_rate": ar_mean,
                "acceptance_rate_std": ar_std,
                "n_trials": args.n_trials,
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nResults:")
    print(df[["finetune_coordinate", "use_hessian_update", "runtime_s", "runtime_s_std", "n_models"]].to_string(index=False))

    print("\nSpeedup from Hessian updates (False_runtime / True_runtime):")
    printed_speedup = False
    for finetune_val in sorted(df["finetune_coordinate"].dropna().unique()):
        sub = df[df["finetune_coordinate"] == finetune_val]
        has_no = bool((sub["use_hessian_update"] == False).any())
        has_yes = bool((sub["use_hessian_update"] == True).any())
        if not (has_no and has_yes):
            continue
        rt_no = float(sub[sub["use_hessian_update"] == False]["runtime_s"].iloc[0])
        rt_yes = float(sub[sub["use_hessian_update"] == True]["runtime_s"].iloc[0])
        speedup = rt_no / max(rt_yes, 1e-12)
        print(f"  finetune={finetune_val}: {speedup:.3f}x")
        printed_speedup = True
    if not printed_speedup:
        print("  Not enough settings to compute speedup (need both hessian_update=False and True).")

    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
