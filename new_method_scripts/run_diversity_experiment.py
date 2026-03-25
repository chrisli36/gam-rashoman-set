"""
Diversity Experiment: MCMC + Ellipsoid vs EllipsoidSampler baseline.

Two methods:

  Method 1 (MCMC + ellipsoid): For each support set from MCMC, create an ellipsoid
  around the optimal weights, sample ~200 models, aggregate all across support sets,
  filter by Rashomon threshold. Compute diversity over this full aggregated set.
  Plot GAM shapes for the full aggregated Rashomon set.

  Method 2 (EllipsoidSampler per support): For each support set from MCMC and its
  MAP, run EllipsoidSampler. For each ellipsoid, compute average pairwise diversity.
  Report mean ± std across ellipsoids. Randomly choose one ellipsoid and plot its
  GAM shape functions.

Usage:
    python run_diversity_experiment.py --dataset compas.csv
    python run_diversity_experiment.py --dataset compas.csv --repulsion 1 --diversity prediction_hamming
    python run_diversity_experiment.py --merge results_diversity/compas
"""

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_method_scripts.mcmc_rashomon import (
    MCMCRashomonSampler,
    RashomonResult,
    binarize_data,
    diversity_min_l1,
    diversity_support_jaccard,
    make_diversity_monotonicity_entropy,
    make_diversity_prediction_hamming,
    plot_gam_shapes,
)
from new_method_scripts.ellipsoid import EllipsoidSampler
from new_method_scripts.diversity_measures import (
    monotonicity_diversity,
    pairwise_hamming,
    pairwise_prediction_hamming,
    pairwise_shape_distance,
    pairwise_weight_l1,
)
import numpy as np
import random

np.random.seed(42)
random.seed(42)
# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
BENCHMARK_DIR = Path(
    "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark"
)
DEFAULT_REPULSION = [-1, 0, 1, 5, 10, 20]
DEFAULT_DIVERSITY = ["prediction_hamming", "support_jaccard", "l1", "monotonicity_entropy"]

EPS = 0.05
SIGMA2 = 1
P_FEAT = 0.02
ELLIPSOID_N_SAMPLES = 20  # Per support set for both methods

METRIC_KEYS = ["pred_hamming", "support_hamming", "weight_l1", "shape_l1", "monotonicity_entropy"]

MAX_FEATURES = 150
MAX_SAMPLES = 10000


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


def parse_list_arg(s: str, parser, choices=None):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if choices and parts:
        bad = [p for p in parts if p not in choices]
        if bad:
            parser.error(f"Invalid values: {bad}. Choices: {choices}")
    return parts


def parse_repulsion(s: str, parser):
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            parser.error(f"Repulsion weight must be numeric: {p}")
    return out


def parse_eps_config(s: str, parser) -> dict:
    """
    Parse dataset->eps mapping from JSON string or path to JSON file.
    Keys are dataset stems (e.g. 'compas', 'fico') without .csv.
    """
    s = s.strip()
    if not s:
        return {}
    path = Path(s)
    if path.exists():
        with open(path) as fp:
            data = json.load(fp)
    else:
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            parser.error(f"Invalid --eps_config JSON: {e}")
    if not isinstance(data, dict):
        parser.error("--eps_config must be a JSON object: {dataset: eps, ...}")
    out = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            parser.error(f"eps_config value for '{k}' must be numeric: {v}")
    return out


def repulsion_to_suffix(rw: float) -> str:
    s = str(int(rw)) if rw == int(rw) else str(rw).replace(".", "_")
    return "rep" + s


# ------------------------------------------------------------------ #
#  Diversity
# ------------------------------------------------------------------ #

def compute_diversity_metrics(result: RashomonResult, X, feature_groups):
    """Pairwise diversity metrics on a RashomonResult."""
    if result.n_accepted < 2:
        return {k: 0.0 for k in METRIC_KEYS}
    _, ph = pairwise_prediction_hamming(result, X)
    _, sh = pairwise_hamming(result)
    _, wl = pairwise_weight_l1(result)
    _, sl = pairwise_shape_distance(result, feature_groups)
    mono = monotonicity_diversity(result, feature_groups)
    return {
        "pred_hamming": ph,
        "support_hamming": sh,
        "weight_l1": wl,
        "shape_l1": sl,
        "monotonicity_entropy": mono["mean_entropy"],
    }


def _support_to_map(result: RashomonResult):
    """Map each unique support set to its MAP model (first model per support)."""
    out = {}
    for i, S in enumerate(result.support_sets):
        key = tuple(sorted(S))
        if key not in out:
            out[key] = result.w_models[i].copy()
    return out


# ------------------------------------------------------------------ #
#  Single run
# ------------------------------------------------------------------ #

def run_single(
    X, y, feature_groups, diversity_fn, repulsion_weight,
    n_steps, burn_in, target_models,
    eps: float = EPS,
    p_feat: float = P_FEAT,
):
    """
    Run one config. Returns:
      - Method 1 (MCMC): full aggregated Rashomon set, diversity, GAM plot
      - Method 2 (EllipsoidSampler): per-ellipsoid diversity mean±std, GAM plot of one
    """
    sampler = MCMCRashomonSampler(
        X, y, eps=eps, sigma2=SIGMA2, p_feat=p_feat,
        feature_groups=feature_groups,
    )

    # ---- Method 1: MCMC + ellipsoid ----
    result = sampler.sample(
        n_steps=n_steps,
        burn_in=burn_in,
        target_models=target_models,
        beta=0.5,
        proposal="mixture",
        mh_variant="repulsive",
        ellipsoid_augment=False,
        ellipsoid_n_samples=ELLIPSOID_N_SAMPLES,
        repulsion_weight=repulsion_weight,
        diversity_fn=diversity_fn,
        finetune_coordinate=False,
    )

    mcmc_diversity = compute_diversity_metrics(result, X, feature_groups)
    total_models_sampled_mcmc = len(result.w_models)

    # ---- Method 2: EllipsoidSampler per support ----
    support_to_map = _support_to_map(result)
    ell_sampler = EllipsoidSampler(X, y, eps=eps, sigma2=SIGMA2)

    # Compute best_nll for log-likelihood-based ellipsoid bound (same as mcmc_rashomon)
    best_nll = float("inf")
    for supp_key, w_map in support_to_map.items():
        if len(supp_key) == 0:
            continue
        idx = list(supp_key)
        theta = w_map[idx]
        X_S = X[:, idx]
        proba = 1.0 / (1.0 + np.exp(-np.clip(X_S @ theta, -20, 20)))
        nll = EllipsoidSampler.total_nll(proba, y, theta, SIGMA2)
        best_nll = min(best_nll, nll)

    per_ellipsoid = {k: [] for k in METRIC_KEYS}
    ellipsoid_results = []

    for supp_key, w_map in support_to_map.items():
        if len(supp_key) == 0:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ell_result = ell_sampler.sample(
                    n_samples=total_models_sampled_mcmc*2,
                    w_center=w_map,
                    sampling="uniform",
                    eps=eps,
                    best_nll=best_nll if best_nll < float("inf") else None,
                )
        except Exception:
            continue
        if ell_result.n_accepted < 2:
            continue
        m = compute_diversity_metrics(ell_result, X, feature_groups)
        for k in METRIC_KEYS:
            per_ellipsoid[k].append(m[k])
        ellipsoid_results.append(ell_result)

    ellipsoid_diversity = {}
    for k in METRIC_KEYS:
        vals = per_ellipsoid[k]
        arr = np.array(vals) if vals else np.array([0.0])
        ellipsoid_diversity[k + "_avg"] = float(arr.mean())
        ellipsoid_diversity[k + "_std"] = float(arr.std())
    ellipsoid_diversity["n_ellipsoids"] = len(ellipsoid_results)
    
    return {
        "n_models_mcmc": result.n_accepted,
        "n_explored": result.n_explored,
        "best_error": result.best_error,
        "error_bound": result.error_bound,
        "acceptance_rate": result.acceptance_rate,
        "runtime": result.runtime,
        "mcmc": mcmc_diversity,
        "ellipsoid": ellipsoid_diversity,
        "result": result,
        "ellipsoid_results": ellipsoid_results,
    }


def _plot_gam_shapes_for_config(res, data, feature_groups, feature_names, output_dir, div_name, rw):
    """Plot GAM shapes for MCMC (full set) and EllipsoidSampler (one random ellipsoid)."""
    if not feature_groups:
        return
    X_raw = data.iloc[:, :-1]
    fg = {k: v for k, v in feature_groups.items() if k in X_raw.columns}
    if not fg:
        return

    def _plot(result, method, suffix):
        if result.n_accepted == 0:
            return
        out_path = output_dir / f"gam_shapes_{div_name}_{suffix}_{repulsion_to_suffix(rw)}.png"
        plot_gam_shapes(
            result, fg, data,
            max_models=min(200, result.n_accepted),
            save_path=str(out_path),
            title_suffix=f" ({method}, repulsion={rw})",
            alpha=0.7,
            normalize=False,
            feature_names=feature_names,
        )
        print(f"  GAM plot saved: {out_path}")

    # MCMC: full aggregated Rashomon set
    _plot(res["result"], "MCMC + ellipsoid", "mcmc")

    # EllipsoidSampler: one random ellipsoid
    ell_results = res.get("ellipsoid_results", [])
    if ell_results:
        idx = np.random.randint(len(ell_results))
        _plot(ell_results[idx], "EllipsoidSampler (single support)", "ellipsoid")


def _plot_loss_distributions(res, X, y, output_dir, div_name, rw):
    """Plot misclassification error and log posterior distributions for MCMC and Ellipsoid methods."""
    import matplotlib.pyplot as plt

    def _compute_nlls(result, X, y, sigma2):
        """Compute NLL for each model in result."""
        if result.n_accepted == 0:
            return np.array([])
        nlls = []
        for w in result.w_models:
            logits = X @ w
            proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            nll = EllipsoidSampler.total_nll(proba, y, w, sigma2)
            nlls.append(nll)
        return np.array(nlls)

    mcmc_result = res["result"]
    errors_mcmc = mcmc_result.errors
    nlls_mcmc = _compute_nlls(mcmc_result, X, y, SIGMA2)

    ell_results = res.get("ellipsoid_results", [])
    if ell_results:
        idx = np.random.randint(len(ell_results))
        ell_result = ell_results[idx]
        errors_ell = ell_result.errors
        nlls_ell = _compute_nlls(ell_result, X, y, SIGMA2)
    else:
        errors_ell = np.array([])
        nlls_ell = np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Row 0: MCMC
    ax0, ax1 = axes[0]
    if len(errors_mcmc) > 0:
        ax0.hist(errors_mcmc, bins=min(50, len(errors_mcmc)), edgecolor="black", alpha=0.7)
    ax0.set_title("MCMC: Misclassification Error")
    ax0.set_xlabel("Error")
    if len(nlls_mcmc) > 0:
        ax1.hist(nlls_mcmc, bins=min(50, len(nlls_mcmc)), edgecolor="black", alpha=0.7)
    ax1.set_title("MCMC: Log Posterior (NLL)")
    ax1.set_xlabel("NLL")

    # Row 1: Ellipsoid
    ax2, ax3 = axes[1]
    if len(errors_ell) > 0:
        ax2.hist(errors_ell, bins=min(50, len(errors_ell)), edgecolor="black", alpha=0.7)
    ax2.set_title("Ellipsoid: Misclassification Error")
    ax2.set_xlabel("Error")
    if len(nlls_ell) > 0:
        ax3.hist(nlls_ell, bins=min(50, len(nlls_ell)), edgecolor="black", alpha=0.7)
    ax3.set_title("Ellipsoid: Log Posterior (NLL)")
    ax3.set_xlabel("NLL")

    plt.suptitle(f"Loss Distributions ({div_name}, repulsion={rw})", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = output_dir / f"loss_distributions_{div_name}_{repulsion_to_suffix(rw)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss distribution plot saved: {out_path}")


# ------------------------------------------------------------------ #
#  Merge
# ------------------------------------------------------------------ #

def merge_results(dir_path: Path):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir_path}")
    pattern = "*_rep*.json"
    files = sorted(dir_path.glob(pattern))
    if not files:
        print(f"No {pattern} files found in {dir_path}")
        return

    all_results = {"dataset": dir_path.name}
    rows = []

    for f in files:
        stem = f.stem
        if "_rep" not in stem:
            continue
        div_part, rep_part = stem.rsplit("_rep", 1)
        try:
            rw = float(rep_part.replace("_", "."))
        except ValueError:
            continue
        div_name = div_part
        with open(f) as fp:
            data = json.load(fp)
        if "error" in data:
            continue
        all_results.setdefault(div_name, {})[str(rw)] = data

        row = {
            "dataset": dir_path.name,
            "diversity_fn": div_name,
            "repulsion_weight": rw,
            "n_models": data["n_models_mcmc"],
            "accept_rate": data["acceptance_rate"],
        }
        if "eps" in data:
            row["eps"] = data["eps"]
        for k in METRIC_KEYS:
            row[f"mcmc_{k}"] = data["mcmc"][k]
            row[f"ellipsoid_{k}_avg"] = data["ellipsoid"][k + "_avg"]
            row[f"ellipsoid_{k}_std"] = data["ellipsoid"][k + "_std"]
        row["n_ellipsoids"] = data["ellipsoid"]["n_ellipsoids"]
        row["runtime_s"] = data.get("total_time", 0)
        rows.append(row)

    json_path = dir_path / "diversity_results.json"
    all_save = {"dataset": all_results["dataset"]}
    for div_name, configs in all_results.items():
        if div_name == "dataset":
            continue
        all_save[div_name] = {
            rw: {k: v for k, v in cfg.items() if k not in ("result", "ellipsoid_results")}
            for rw, cfg in configs.items()
        }
    with open(json_path, "w") as fp:
        json.dump(all_save, fp, indent=2)
    print(f"Merged {len(files)} configs -> {json_path}")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = dir_path / "diversity_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Diversity: MCMC+ellipsoid vs EllipsoidSampler")
    parser.add_argument("--dataset", required=True, help="Dataset filename or path")
    parser.add_argument("--data_dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--repulsion", default=",".join(str(x) for x in DEFAULT_REPULSION))
    parser.add_argument("--diversity", default=",".join(DEFAULT_DIVERSITY))
    parser.add_argument("--eps", type=float, default=None,
                        help="Rashomon epsilon (default: 0.05). Overridden by --eps_config for specific datasets.")
    parser.add_argument("--eps_config", default=None,
                        help="JSON object or path: {dataset_stem: eps}. E.g. '{\"compas\":0.05,\"fico\":0.1}'")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_dir", default="results_diversity")
    parser.add_argument("--p_feat", type=float, default=None,
                        help="Prior probability of including each feature (default: 0.5)")
    parser.add_argument("--merge", metavar="DIR", help="Merge per-config JSONs")
    args = parser.parse_args()

    if args.merge:
        merge_results(Path(args.merge))
        return

    repulsion_weights = parse_repulsion(args.repulsion, parser)
    diversity_names = parse_list_arg(
        args.diversity, parser,
        choices=["prediction_hamming", "support_jaccard", "l1", "shape_l1", "monotonicity_entropy"],
    )
    if not diversity_names:
        parser.error("At least one diversity function required")

    eps_config = parse_eps_config(args.eps_config, parser) if args.eps_config else {}
    eps_default = args.eps if args.eps is not None else EPS
    p_feat = args.p_feat if args.p_feat is not None else P_FEAT

    n_steps = 10000 if args.quick else 10000
    burn_in = 500 if args.quick else 500
    target_models = 100 if args.quick else 200
    C = 1

    data_path = Path(args.dataset)
    if not data_path.is_absolute():
        data_path = Path(args.data_dir) / data_path.name
    if not data_path.exists():
        parser.error(f"Dataset not found: {data_path}")

    dataset_name = data_path.stem
    output_dir = Path(__file__).parent / args.output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {data_path}")
    data = pd.read_csv(data_path)
    if len(data) > MAX_SAMPLES:
        n_orig = len(data)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_orig, size=MAX_SAMPLES, replace=False)
        data = data.iloc[idx].reset_index(drop=True)
        print(f"Subsampled to {MAX_SAMPLES} rows (original had {n_orig})")

    X, y, feature_names, feature_groups = binarize_data(data, num_estimators=500, max_thresholds_per_feat=50)
    print(f"X: {X.shape}  ({len(feature_groups)} features -> {X.shape[1]-1} thresholds + intercept)")

    if X.shape[1] > MAX_FEATURES:
        print(f"L1 feature selection -> {MAX_FEATURES} features")
        X, feature_names, feature_groups = select_features_l1(X, y, feature_names, feature_groups, max_features=MAX_FEATURES, C=C)

    eps = eps_config.get(dataset_name, eps_default)
    print(f"Parameters: eps={eps}, sigma2={SIGMA2}, p_feat={p_feat}, ellipsoid_n={ELLIPSOID_N_SAMPLES}")
    print(f"Repulsion: {repulsion_weights}, Diversity: {diversity_names}\n")

    div_fn_map = {
        "prediction_hamming": make_diversity_prediction_hamming(X),
        "support_jaccard": diversity_support_jaccard,
        "l1": diversity_min_l1,
        "monotonicity_entropy": make_diversity_monotonicity_entropy(feature_groups),
    }

    all_results = {"dataset": dataset_name}
    t_total = time()

    for div_name in diversity_names:
        diversity_fn = div_fn_map[div_name]
        all_results[div_name] = {}

        for rw in repulsion_weights:
            print(f"[{div_name}]  repulsion_weight={rw}")
            t0 = time()
            try:
                res = run_single(X, y, feature_groups, diversity_fn, rw, n_steps, burn_in, target_models, eps=eps, p_feat=p_feat)
            except Exception as exc:
                res = {"error": str(exc)}
                import traceback
                traceback.print_exc()

            elapsed = time() - t0
            if "error" not in res:
                res["total_time"] = elapsed
                res["eps"] = eps
            all_results[div_name][str(rw)] = res

            if "error" in res:
                print(f"  ERROR: {res['error']}")
            else:
                print(f"  MCMC: {res['n_models_mcmc']} models, best_err={res['best_error']:.4f}")
                print(f"  Method 1 (MCMC+ellipsoid) diversity:")
                for k in METRIC_KEYS:
                    print(f"    {k:>20s}: {res['mcmc'][k]:.6f}")
                print(f"  Method 2 (EllipsoidSampler per support) diversity (mean ± std, {res['ellipsoid']['n_ellipsoids']} ellipsoids):")
                for k in METRIC_KEYS:
                    avg, std = res["ellipsoid"][k + "_avg"], res["ellipsoid"][k + "_std"]
                    print(f"    {k:>20s}: {avg:.6f} ± {std:.6f}")
                if rw == 0.05 and "result" in res:
                    _plot_gam_shapes_for_config(res, data, feature_groups, feature_names, output_dir, div_name, rw)
                    _plot_loss_distributions(res, X, y, output_dir, div_name, rw)

            print(f"  Time: {elapsed:.1f}s\n")

    total_time = time() - t_total

    single_config = len(repulsion_weights) == 1 and len(diversity_names) == 1

    if single_config:
        div_name = diversity_names[0]
        rw = repulsion_weights[0]
        res = all_results.get(div_name, {}).get(str(rw), {})
        if "error" not in res:
            res_save = {k: v for k, v in res.items() if k not in ("result", "ellipsoid_results")}
            json_path = output_dir / f"{div_name}_{repulsion_to_suffix(rw)}.json"
            with open(json_path, "w") as f:
                json.dump(res_save, f, indent=2)
            print(f"Results saved to {json_path}")
    else:
        all_save = {"dataset": dataset_name}
        for div_name in diversity_names:
            all_save[div_name] = {
                rw: {k: v for k, v in res.items() if k not in ("result", "ellipsoid_results")}
                for rw, res in all_results.get(div_name, {}).items()
            }
        json_path = output_dir / "diversity_results.json"
        with open(json_path, "w") as f:
            json.dump(all_save, f, indent=2)
        print(f"Results JSON saved to {json_path}")

        rows = []
        for div_name in diversity_names:
            for rw in repulsion_weights:
                res = all_results.get(div_name, {}).get(str(rw), {})
                if "error" in res:
                    continue
                row = {
                    "dataset": dataset_name,
                    "diversity_fn": div_name,
                    "repulsion_weight": rw,
                    "n_models": res["n_models_mcmc"],
                    "accept_rate": res["acceptance_rate"],
                    "n_ellipsoids": res["ellipsoid"]["n_ellipsoids"],
                    "runtime_s": res.get("total_time", 0),
                    "eps": eps,
                }
                for k in METRIC_KEYS:
                    row[f"mcmc_{k}"] = res["mcmc"][k]
                    row[f"ellipsoid_{k}_avg"] = res["ellipsoid"][k + "_avg"]
                    row[f"ellipsoid_{k}_std"] = res["ellipsoid"][k + "_std"]
                rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_dir / "diversity_results.csv", index=False)
            print(f"CSV saved to {output_dir / 'diversity_results.csv'}")

    if not single_config:
        print(f"\n{'='*100}")
        print(f"SUMMARY — {dataset_name}")
        print(f"{'='*100}")
        print(f"{'Div':<22} {'RW':>4} | {'#Mod':>5} | " + " | ".join(f"{'mcmc_'+k:>12}" for k in METRIC_KEYS) + " | " + " | ".join(f"{'ell_'+k:>16}" for k in METRIC_KEYS))
        print("-" * 100)
        for div_name in diversity_names:
            for rw in repulsion_weights:
                res = all_results.get(div_name, {}).get(str(rw), {})
                if "error" in res:
                    print(f"{div_name:<22} {rw:>4} | ERROR")
                    continue
                parts = [f"{div_name:<22} {rw:>4} | {res['n_models_mcmc']:>5} | "]
                parts.append(" | ".join(f"{res['mcmc'][k]:>12.6f}" for k in METRIC_KEYS))
                parts.append(" | ")
                parts.append(" | ".join(f"{res['ellipsoid'][k+'_avg']:>8.4f}±{res['ellipsoid'][k+'_std']:<10.4f}" for k in METRIC_KEYS))
                print("".join(parts))
            print("-" * 100)

    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
