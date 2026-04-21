
import numpy as np
import pandas as pd
import warnings
import argparse
import sys
from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression, LinearRegression
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
from functools import partial
from time import time
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import pearsonr
import time
from binarize_augmented_datasets import binarize_dataset as binarize_dataset_gbdt
import random
from sklearn.linear_model import LogisticRegression, LinearRegression
from mcmc_rashomon import (
    binarize_data,
    plot_gam_shapes,
)
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
random.seed(42)
np.random.seed(42)
MAX_EXTRA_THRESHOLDS_PER_FEAT = 1
DIVERSITY_PAIR_SAMPLE_LIMIT = 20_000
DEFAULT_DIVERSITY_FN = "prediction_hamming"
# options: prediction_hamming, support_jaccard, l1, monotonicity_entropy

# Ensure imports like `method_scripts.*` resolve in batch/Slurm contexts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Flush stdout/stderr promptly in batch jobs so logs appear in real time.
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

def fit_logistic(X_S, y, sigma2=10.0, penalty="l2", max_iter=1000):
        """Fit L2-regularized logistic regression. Returns (theta, probabilities)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty=penalty, C=sigma2, fit_intercept=False,
                solver="liblinear", max_iter=max_iter,
            )
            clf.fit(X_S, y)
        return clf.coef_.ravel(), clf.predict_proba(X_S)[:, 1]

def _total_nll(proba, y, theta, sigma2):
    """Total negative log-posterior (NLL + L2 prior)."""
    _e = 1e-12
    nll = -np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
    return nll + 0.5 * np.dot(theta, theta) / sigma2

def _hessian_matrix(X_S, proba, sigma2, jitter=0.0):
    p = np.clip(proba, 1e-8, 1 - 1e-8)
    W = p * (1 - p)
    d = X_S.shape[1]
    return (X_S.T * W) @ X_S + (1 / sigma2 + jitter) * np.eye(d)

def _sample_ellipsoid(
    theta_center,
    X_S,
    y,
    sigma2,
    n_samples,
    ub,
    eigvals=None,
    eigvecs=None,
    sampling_method="uniform",
    rejection_min_distance=None,
):
    """
    Sample n_samples weight vectors from the Hessian-defined ellipsoid
    around theta_center with bound ub, and return them with their
    misclassification errors.

    When eigvals, eigvecs are provided (from _hessian_eig), skips Hessian
    computation to avoid redundant work.

    Returns (theta_samples, errors, nlls) where theta_samples is (n_samples, d),
    errors is (n_samples,), nlls is (n_samples,) for Rashomon filtering.
    """
    if n_samples <= 0 or ub <= 0:
        return np.zeros((0, len(theta_center))), np.array([]), np.array([])

    d = len(theta_center)
    if eigvals is None or eigvecs is None:
        proba_center = 1.0 / (1.0 + np.exp(-np.clip(X_S @ theta_center, -20, 20)))
        H = _hessian_matrix(X_S, proba_center, sigma2, jitter=1e-6)
        eigvals, eigvecs = np.linalg.eigh(H)
    a = np.sqrt(2 * ub / np.maximum(eigvals, 1e-12))

    if sampling_method == "uniform":
        u = np.random.randn(n_samples, d)
        u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
        r = np.random.random(n_samples) ** (1.0 / d)
        z = u * r[:, None]  # uniform in unit ball
        # Rotate then scale using eigvecs as columns
        deltas = (eigvecs * a) @ z.T  # shape (d, n_samples)
        theta_samples = theta_center[:, None] + deltas  # (d, n_samples)
        theta_samples = theta_samples.T  # (n_samples, d)
    elif sampling_method == "rejection":
        # Rejection sampling with a minimum pairwise distance in theta-space.
        # Major axis length is 2 * max(semi-axis length) = 2 * max(a).
        major_axis_length = 2.0 * float(np.max(a))
        min_distance = (
            0.01 * major_axis_length
            if rejection_min_distance is None
            else float(rejection_min_distance)
        )
        max_attempts = 5000
        accepted = []
        attempts = 0
        for i in range(max_attempts):
            u = np.random.randn(1, d)
            u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
            r = np.random.random(1) ** (1.0 / d)
            z = u * r[:, None]
            delta = (eigvecs * a) @ z.T
            candidate = (theta_center[:, None] + delta).ravel()
            if not accepted:
                accepted.append(candidate)
                continue
            dists = np.linalg.norm(np.asarray(accepted) - candidate, axis=1)
            if np.min(dists) >= min_distance:
                accepted.append(candidate)
            if len(accepted) >= n_samples:
                break
        theta_samples = (
            np.asarray(accepted, dtype=np.float64)
            if accepted
            else np.zeros((0, d), dtype=np.float64)
        )
    else:
        raise ValueError(
            "ellipsoid sampling method must be one of {'uniform', 'rejection'}."
        )

    logits = X_S @ theta_samples.T
    proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    preds = (logits >= 0).astype(np.float64)
    errors = np.mean(preds != y[:, None], axis=0)
    _eps = 1e-12
    nll_data = -np.sum(
    y[:, None] * np.log(proba + _eps) + (1 - y[:, None]) * np.log(1 - proba + _eps),axis=0)                                                    # (n_samples,)
    nll_reg  = 0.5 * np.sum(theta_samples**2, axis=1) / sigma2  # (n_samples,)
    nlls     = nll_data + nll_reg  
    return theta_samples, errors, nlls

def random_support_set_ellipsoid_sampling(
    X,
    y,
    num_trials=30,
    n_ellipsoid_samples=20,
    ellipsoid_sampling_method="uniform",
    rejection_min_distance=None,
    rashomon_loss_bound=None,
    sigma2=10.0,
    support_set_size=None,
):
    """Randomly sample support sets and draw ellipsoid candidates around each MAP fit.

    If ``rashomon_loss_bound`` is provided, keep only candidates with
    L2-regularized loss <= that absolute bound.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array with the same number of rows as X.")

    n, p = X.shape
    num_trials = max(1, int(num_trials))
    if support_set_size is None:
        raise ValueError("support_set_size must be provided for deterministic sparsity.")
    support_set_size = int(np.clip(int(np.rint(support_set_size)), 1, p))

    all_models = []
    all_errors = []
    all_nlls = []
    support_sets = []
    support_sizes = []
    w_centers = []

    for i in tqdm(range(num_trials)):
        support_set = np.sort(np.random.choice(p, support_set_size, replace=False))
        support_sets.append(support_set.tolist())
        support_sizes.append(support_set_size)

        X_support_set = X[:, support_set]
        theta_center, proba_center = fit_logistic(X_support_set, y, sigma2=sigma2)

        # Use a per-support quadratic slack around the center NLL.
        nll_center = _total_nll(proba_center, y, theta_center, sigma2)
        if nll_center > rashomon_loss_bound:
            continue
        ub = max(1e-6, 0.05 * nll_center)
        H = _hessian_matrix(X_support_set, proba_center, sigma2, jitter=1e-6)
        eigvals, eigvecs = np.linalg.eigh(H)

        theta_samples, errors, nlls = _sample_ellipsoid(
            theta_center=theta_center,
            X_S=X_support_set,
            y=y,
            sigma2=sigma2,
            n_samples=n_ellipsoid_samples,
            ub=ub,
            eigvals=eigvals,
            eigvecs=eigvecs,
            sampling_method=ellipsoid_sampling_method,
            rejection_min_distance=rejection_min_distance,
        )

        w_center = np.zeros(p, dtype=np.float64)
        w_center[support_set] = theta_center
        w_centers.append(w_center)

        if rashomon_loss_bound is not None:
            keep_mask = nlls <= float(rashomon_loss_bound)
            theta_samples = theta_samples[keep_mask]
            errors = errors[keep_mask]
            nlls = nlls[keep_mask]

        w_samples_full = np.zeros((theta_samples.shape[0], p), dtype=np.float64)
        w_samples_full[:, support_set] = theta_samples
        all_models.append(w_samples_full)
        all_errors.append(errors)
        all_nlls.append(nlls)

    w_models = np.vstack(all_models) if all_models else np.zeros((0, p), dtype=np.float64)
    return {
        "support_sets": support_sets,
        "support_sizes": np.asarray(support_sizes, dtype=int),
        "w_centers": np.vstack(w_centers) if w_centers else np.zeros((0, p), dtype=np.float64),
        "w_models": w_models,
        "errors": np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float64),
        "nlls": np.concatenate(all_nlls) if all_nlls else np.array([], dtype=np.float64),
        "n_explored": int(w_models.shape[0]),
        "X_eval": X,
    }

def bootstrap_optimal_regularized_logistic_sampling(
    X,
    y,
    n_bootstrap_iters=30,
    c_values=None,
    penalties=("l2",),
    rashomon_rel_tol=1e-3,
    rashomon_loss_bound=None,
    sigma2=10.0,
    max_iter=1000,
    random_state=42,
):
    """
    Baseline: bootstrap + optimal regularized logistic models.

    For each bootstrap iteration, fit regularized logistic models.
    By default, this baseline uses the same L2 regularization strength as the
    other logistic baselines in this file (``C = sigma2``). If ``c_values`` is
    provided, that explicit grid is used instead.
    The bootstrap Rashomon set is the collection of models whose bootstrap NLL is
    within ``(1 + rashomon_rel_tol)`` of that bootstrap's best NLL.
    If ``rashomon_loss_bound`` is provided, additionally keep only models with
    full-data L2-regularized loss <= that absolute bound.

    Returns a dict consistent with other baselines:
      - support_sets: supports of accepted models
      - support_sizes: support cardinalities
      - w_centers: best model per bootstrap (full p-vectors)
      - w_models: all accepted Rashomon models across bootstraps
      - errors, nlls: evaluated on the original full dataset
      - n_explored: total candidate models evaluated
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array with the same number of rows as X.")
    if n_bootstrap_iters <= 0:
        return {
            "support_sets": [],
            "support_sizes": np.array([], dtype=int),
            "w_centers": np.zeros((0, X.shape[1]), dtype=np.float64),
            "w_models": np.zeros((0, X.shape[1]), dtype=np.float64),
            "errors": np.array([], dtype=np.float64),
            "nlls": np.array([], dtype=np.float64),
            "n_explored": 0,
            "X_eval": X,
        }

    if c_values is None:
        c_values = np.array([sigma2], dtype=np.float64)
    c_values = np.asarray(c_values, dtype=np.float64).ravel()
    if c_values.size == 0:
        raise ValueError("c_values must contain at least one value.")
    if isinstance(penalties, str):
        penalties = (penalties,)
    else:
        penalties = tuple(penalties)
    if not penalties:
        raise ValueError("penalties must contain at least one penalty string.")

    n, p = X.shape
    rng = np.random.default_rng(random_state)

    all_models = []
    all_errors = []
    all_nlls = []
    support_sets = []
    support_sizes = []
    w_centers = []
    n_explored = 0

    _eps = 1e-12
    for _ in tqdm(range(n_bootstrap_iters)):
        boot_idx = rng.integers(0, n, size=n)
        Xb = X[boot_idx]
        yb = y[boot_idx]

        candidates = []
        for penalty in penalties:
            for c in c_values:
                # liblinear supports both l1 and l2 for binary logistic.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(
                        penalty=penalty,
                        C=float(c),
                        fit_intercept=False,
                        solver="liblinear",
                        max_iter=max_iter,
                    )
                    try:
                        clf.fit(Xb, yb)
                    except Exception:
                        continue
                theta = clf.coef_.ravel().astype(np.float64, copy=False)
                proba_b = clf.predict_proba(Xb)[:, 1]
                nll_b = _total_nll(proba_b, yb, theta, sigma2=sigma2)
                candidates.append((theta, nll_b))
                n_explored += 1

        if not candidates:
            continue

        best_idx = int(np.argmin([nll_b for _, nll_b in candidates]))
        theta_best, nll_best = candidates[best_idx]
        nll_cutoff = nll_best * (1.0 + float(rashomon_rel_tol))
        accepted_thetas = [theta for theta, nll_b in candidates if nll_b <= nll_cutoff]

        # Track one center per bootstrap iteration.
        w_centers.append(theta_best.copy())

        for theta in accepted_thetas:
            logits = X @ theta
            proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            err = float(np.mean((logits >= 0).astype(np.float64) != y))
            nll = float(
                -np.sum(y * np.log(proba + _eps) + (1 - y) * np.log(1 - proba + _eps))
                + 0.5 * np.dot(theta, theta) / sigma2
            )
            if rashomon_loss_bound is not None and nll > float(rashomon_loss_bound):
                continue

            support = np.flatnonzero(np.abs(theta) > 1e-10).tolist()
            support_sets.append(support)
            support_sizes.append(len(support))
            all_models.append(theta.copy())
            all_errors.append(err)
            all_nlls.append(nll)

    w_models = np.vstack(all_models) if all_models else np.zeros((0, p), dtype=np.float64)
    return {
        "support_sets": support_sets,
        "support_sizes": np.asarray(support_sizes, dtype=int),
        "w_centers": np.vstack(w_centers) if w_centers else np.zeros((0, p), dtype=np.float64),
        "w_models": w_models,
        "errors": np.asarray(all_errors, dtype=np.float64),
        "nlls": np.asarray(all_nlls, dtype=np.float64),
        "n_explored": int(n_explored),
        "X_eval": X,
    }


def swapping_baseline(
    dataset_name,
    eps,
    n_samples=100,
    l0=5.0,
    l2=0.1,
    m=1.05,
    ne=None,
    k=3,
    sigma2=10.0,
    rashomon_loss_bound=None,
    dataset_path=None,
):
    """
    Baseline wrapper around the existing FasterRisk swapping method.

    Returns a dict consistent with other baselines:
      - support_sets, support_sizes
      - w_centers (single best model)
      - w_models, errors, nlls
      - n_explored
    """
    from method_scripts.swapping import SwappingMethod
    from method_scripts.results import Results
    from gam_rs_utils.utils import DatasetUtils

    if ne is None:
        raise ValueError("swapping_baseline requires `ne` (num_estimators) for binarized dataset creation.")

    # Pre-build binarized cache with an explicit dataset path when provided.
    # This avoids fallback to method_scripts default 'datasets/<name>.csv'
    # in Slurm runs.
    if dataset_path is not None:
        X_one_hot, y_cached, header, header_new = DatasetUtils.get_binned_dataset(str(dataset_path), ne)
        binarized_data = {
            "X": X_one_hot,
            "y": y_cached,
            "header": header,
            "header_new": header_new,
            "num_estimators": ne,
            "sample_proportion": X_one_hot.sum(0) / X_one_hot.shape[0],
        }
        Results.save_dataset(dataset_name, {"num_estimators": ne}, binarized_data)

    method = SwappingMethod()
    result_obj = method.run_dataset(
        dn=dataset_name,
        eps=eps,
        n_samples=n_samples,
        l0=l0,
        l2=l2,
        m=m,
        ne=ne,
        k=k,
    )

    data = Results.create_binarized_dataset(dataset_name, ne)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64).ravel()

    w_models = np.asarray(result_obj.w_rset, dtype=np.float64)
    if w_models.ndim == 1:
        w_models = w_models.reshape(1, -1)
    if w_models.size == 0:
        w_models = np.zeros((0, X.shape[1]), dtype=np.float64)

    if w_models.shape[0] > 0:
        logits = X @ w_models.T
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        preds = (logits >= 0).astype(np.float64)
        errors = np.mean(preds != y[:, None], axis=0)
        _eps = 1e-12
        nll_data = -np.sum(
            y[:, None] * np.log(proba + _eps) + (1 - y[:, None]) * np.log(1 - proba + _eps),
            axis=0,
        )
        nll_reg = 0.5 * np.sum(w_models ** 2, axis=1) / sigma2
        nlls = nll_data + nll_reg
    else:
        errors = np.array([], dtype=np.float64)
        nlls = np.array([], dtype=np.float64)

    if rashomon_loss_bound is not None and w_models.shape[0] > 0:
        keep_mask = nlls <= float(rashomon_loss_bound)
        w_models = w_models[keep_mask]
        errors = errors[keep_mask]
        nlls = nlls[keep_mask]

    support_sets = [np.flatnonzero(np.abs(w) > 1e-10).tolist() for w in w_models]
    support_sizes = np.asarray([len(s) for s in support_sets], dtype=int)
    w_center = np.asarray(result_obj.w_opt, dtype=np.float64).reshape(1, -1)

    return {
        "support_sets": support_sets,
        "support_sizes": support_sizes,
        "w_centers": w_center,
        "w_models": w_models,
        "errors": errors,
        "nlls": nlls,
        "n_explored": int(result_obj.n_samples),
        "X_eval": X,
    }


def mcmc_baseline(
    X,
    y,
    feature_groups=None,
    sigma2=1.0,
    p_feat=0.02,
    n_steps=1000,
    burn_in=500,
    target_models=200,
    beta=0.5,
    proposal="mixture",
    l1_C=0.1,
    mh_variant="repulsive",
    ellipsoid_augment=True,
    ellipsoid_n_samples=20,
    repulsion_weight=0.05,
    diversity_fn=None,
    finetune_coordinate=False,
    finetune_corr_threshold=0.1,
    use_hessian_update=False,
    rashomon_loss_bound=None,
):
    """
    Baseline wrapper around ``MCMCRashomonSampler.sample``.

    Returns a dict consistent with other baselines:
      - support_sets, support_sizes
      - w_centers (single best model)
      - w_models, errors, nlls
      - n_explored

    Uses an absolute Rashomon bound for model selection (``rashomon_loss_bound``)
    and disables relative-epsilon filtering inside the sampler.
    """
    from mcmc_rashomon import MCMCRashomonSampler
    if diversity_fn is None:
        diversity_fn = DEFAULT_DIVERSITY_FN

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array with the same number of rows as X.")

    div_fn_map = {
        "prediction_hamming": make_diversity_prediction_hamming(X),
        "support_jaccard": diversity_support_jaccard,
        "l1": diversity_min_l1,
    }
    if feature_groups:
        div_fn_map["monotonicity_entropy"] = make_diversity_monotonicity_entropy(feature_groups)
    diversity_callable = div_fn_map[diversity_fn]

    sampler = MCMCRashomonSampler(
        X=X,
        y=y,
        eps=0.05, 
        sigma2=sigma2,
        p_feat=p_feat,
    )
    print(f"Using diversity function: {diversity_fn}")
    result = sampler.sample(
        n_steps=n_steps,
        burn_in=burn_in,
        target_models=target_models,
        beta=beta,
        proposal=proposal,
        l1_C=l1_C,
        mh_variant=mh_variant,
        ellipsoid_augment=ellipsoid_augment,
        ellipsoid_n_samples=ellipsoid_n_samples,
        repulsion_weight=repulsion_weight,
        diversity_fn=diversity_callable,
        finetune_coordinate=finetune_coordinate,
        finetune_corr_threshold=finetune_corr_threshold,
        use_hessian_update=use_hessian_update,
    )

    w_models = np.asarray(result.w_models, dtype=np.float64)
    if w_models.ndim == 1:
        w_models = w_models.reshape(1, -1)
    if w_models.size == 0:
        w_models = np.zeros((0, X.shape[1]), dtype=np.float64)

    if w_models.shape[0] > 0:
        logits = X @ w_models.T
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        preds = (logits >= 0).astype(np.float64)
        errors = np.mean(preds != y[:, None], axis=0)
        _eps = 1e-12
        nll_data = -np.sum(
            y[:, None] * np.log(proba + _eps) + (1 - y[:, None]) * np.log(1 - proba + _eps),
            axis=0,
        )
        nll_reg = 0.5 * np.sum(w_models ** 2, axis=1) / sigma2
        nlls = nll_data + nll_reg
    else:
        errors = np.array([], dtype=np.float64)
        nlls = np.array([], dtype=np.float64)

    support_sets = list(result.support_sets)
    if rashomon_loss_bound is not None and w_models.shape[0] > 0:
        keep_mask = nlls <= float(rashomon_loss_bound)
        w_models = w_models[keep_mask]
        errors = errors[keep_mask]
        nlls = nlls[keep_mask]
        support_sets = [s for s, keep in zip(support_sets, keep_mask) if keep]

    support_sizes = np.asarray([len(s) for s in support_sets], dtype=int)
    w_center = np.asarray(result.w_opt, dtype=np.float64).reshape(1, -1)

    return {
        "support_sets": support_sets,
        "support_sizes": support_sizes,
        "w_centers": w_center,
        "w_models": w_models,
        "errors": errors,
        "nlls": nlls,
        "n_explored": int(result.n_explored),
        "best_nll": float(result.best_nll),
        "X_eval": X,
    }

# baselines
# 1. Random sampling of support sets + ellipsoid sampling (uniform)
# 2. Random sampling of support sets + ellipsoid sampling (rejection)
# 3. Swapping method: beam search
# 4. Bootstrapping + optimal regularized logistic models

# criteria:
# diversity (a bunch of metrics), throughput, runtime


def _sample_model_pairs(n_models: int, max_pairs: int = DIVERSITY_PAIR_SAMPLE_LIMIT) -> Tuple[np.ndarray, np.ndarray]:
    """Sample model index pairs (i, j) for pairwise diversity estimates."""
    if n_models < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    total_pairs = n_models * (n_models - 1) // 2
    if total_pairs <= max_pairs:
        return np.triu_indices(n_models, k=1)

    rng = np.random.default_rng(42)
    i_idx = rng.integers(0, n_models, size=max_pairs, endpoint=False)
    j_idx = rng.integers(0, n_models, size=max_pairs, endpoint=False)
    same = i_idx == j_idx
    while np.any(same):
        j_idx[same] = rng.integers(0, n_models, size=int(np.sum(same)), endpoint=False)
        same = i_idx == j_idx
    return i_idx, j_idx


def _mean_pairwise_prediction_hamming(preds: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray, chunk_size: int = 512) -> float:
    if i_idx.size == 0:
        return np.nan
    vals = []
    for start in range(0, i_idx.size, chunk_size):
        end = min(start + chunk_size, i_idx.size)
        vals.append(np.mean(preds[:, i_idx[start:end]] != preds[:, j_idx[start:end]], axis=0))
    return float(np.mean(np.concatenate(vals)))


def _compute_monotonicity_entropy(
    w_models: np.ndarray,
    feature_groups: Optional[Dict[str, List[int]]],
) -> float:
    if not feature_groups:
        return np.nan
    from new_method_scripts.diversity_measures import monotonicity_diversity

    out = monotonicity_diversity(SimpleNamespace(w_models=w_models), feature_groups)
    return float(out.get("mean_entropy", np.nan))


def _compute_diversity_metrics(
    w_models: np.ndarray,
    X_eval: Optional[np.ndarray] = None,
    feature_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """Compute diversity metrics for one method's model set."""
    w_models = np.asarray(w_models, dtype=np.float64)
    if w_models.ndim == 1:
        w_models = w_models.reshape(1, -1)
    n_models = int(w_models.shape[0])

    empty_metrics = {
        "diversity_support_jaccard": np.nan,
        "diversity_weight_l1": np.nan,
        "diversity_prediction_hamming": np.nan,
        "diversity_monotonicity_entropy": np.nan,
    }
    if n_models == 0:
        return empty_metrics

    active = np.abs(w_models) > 1e-10

    i_idx, j_idx = _sample_model_pairs(n_models, DIVERSITY_PAIR_SAMPLE_LIMIT)
    if i_idx.size == 0:
        out = dict(empty_metrics)
        out["diversity_monotonicity_entropy"] = _compute_monotonicity_entropy(w_models, feature_groups)
        return out

    inter = np.sum(active[i_idx] & active[j_idx], axis=1).astype(np.float64)
    union = np.sum(active[i_idx] | active[j_idx], axis=1).astype(np.float64)
    support_jaccard = np.mean(1.0 - np.where(union > 0, inter / union, 1.0))
    norms = np.linalg.norm(w_models, axis=1, keepdims=True)
    w_models_unit = np.divide(
        w_models,
        np.where(norms > 0.0, norms, 1.0),
    )
    weight_l1 = np.mean(np.mean(np.abs(w_models_unit[i_idx] - w_models_unit[j_idx]), axis=1))

    pred_hamming = np.nan
    if X_eval is not None:
        X_eval = np.asarray(X_eval, dtype=np.float64)
        if X_eval.ndim == 2 and X_eval.shape[1] == w_models.shape[1]:
            logits = X_eval @ w_models.T
            preds = logits >= 0
            pred_hamming = _mean_pairwise_prediction_hamming(preds, i_idx, j_idx)

    return {
        "diversity_support_jaccard": float(support_jaccard),
        "diversity_weight_l1": float(weight_l1),
        "diversity_prediction_hamming": float(pred_hamming) if not np.isnan(pred_hamming) else np.nan,
        "diversity_monotonicity_entropy": _compute_monotonicity_entropy(w_models, feature_groups),
    }


def _summarize_method(
    method_name: str,
    runtime_s: float,
    out: Dict[str, Any],
    feature_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    w_models = np.asarray(out.get("w_models", np.zeros((0, 0))), dtype=np.float64)
    if w_models.ndim == 1:
        w_models = w_models.reshape(1, -1)
    errors = np.asarray(out.get("errors", np.array([])), dtype=np.float64)
    nlls = np.asarray(out.get("nlls", np.array([])), dtype=np.float64)
    support_sizes = np.asarray(out.get("support_sizes", np.array([])), dtype=np.float64)
    diversity_metrics = _compute_diversity_metrics(
        w_models,
        out.get("X_eval"),
        feature_groups=feature_groups,
    )

    row = {
        "method": method_name,
        "runtime_s": float(runtime_s),
        "n_models": int(w_models.shape[0]),
        "n_explored": int(out.get("n_explored", w_models.shape[0])),
        "best_error": float(np.min(errors)) if errors.size else np.nan,
        "mean_error": float(np.mean(errors)) if errors.size else np.nan,
        "best_nll": float(np.min(nlls)) if nlls.size else np.nan,
        "mean_nll": float(np.mean(nlls)) if nlls.size else np.nan,
        "mean_support_size": float(np.mean(support_sizes)) if support_sizes.size else np.nan,
    }
    row.update(diversity_metrics)
    return row


def _num_models_in_out(out: Dict[str, Any]) -> int:
    w_models = np.asarray(out.get("w_models", np.zeros((0, 0))), dtype=np.float64)
    if w_models.ndim == 1:
        w_models = w_models.reshape(1, -1)
    return int(w_models.shape[0])


def _downsample_out_models(out: Dict[str, Any], target_n: int, seed: int) -> Dict[str, Any]:
    """Downsample model-level outputs to at most target_n models."""
    target_n = max(0, int(target_n))
    n_models = _num_models_in_out(out)
    if n_models <= target_n:
        return out

    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(n_models, size=target_n, replace=False))
    out_new: Dict[str, Any] = dict(out)

    for key in ("w_models", "errors", "nlls"):
        arr = np.asarray(out_new.get(key))
        if arr.ndim > 0 and arr.shape[0] == n_models:
            out_new[key] = arr[keep_idx]

    support_sets = out_new.get("support_sets")
    if isinstance(support_sets, list) and len(support_sets) == n_models:
        out_new["support_sets"] = [support_sets[i] for i in keep_idx]

    support_sizes = np.asarray(out_new.get("support_sizes", np.array([])))
    if support_sizes.ndim > 0 and support_sizes.shape[0] == n_models:
        out_new["support_sizes"] = support_sizes[keep_idx]

    return out_new


def _apply_absolute_rashomon_bound(out: Dict[str, Any], rashomon_loss_bound: float) -> Dict[str, Any]:
    """Filter model-level outputs to models with nll <= rashomon_loss_bound."""
    n_models = _num_models_in_out(out)
    if n_models == 0:
        return out

    nlls = np.asarray(out.get("nlls", np.array([])), dtype=np.float64)
    if nlls.ndim == 0 or nlls.shape[0] != n_models:
        return out

    keep_mask = nlls <= float(rashomon_loss_bound)
    out_new: Dict[str, Any] = dict(out)

    for key in ("w_models", "errors", "nlls"):
        arr = np.asarray(out_new.get(key))
        if arr.ndim > 0 and arr.shape[0] == n_models:
            out_new[key] = arr[keep_mask]

    support_sets = out_new.get("support_sets")
    if isinstance(support_sets, list) and len(support_sets) == n_models:
        out_new["support_sets"] = [s for s, keep in zip(support_sets, keep_mask) if keep]

    support_sizes = np.asarray(out_new.get("support_sizes", np.array([])))
    if support_sizes.ndim > 0 and support_sizes.shape[0] == n_models:
        out_new["support_sizes"] = support_sizes[keep_mask]

    return out_new


def _match_baseline_model_count(
    run_fn: Callable[[int], Dict[str, Any]],
    initial_budget: int,
    target_n_models: int,
    budget_name: str,
    method_name: str,
    tolerance_frac: float = 0.25,
    max_retries: int = 1,
) -> Tuple[Dict[str, Any], int]:
    """
    Run a baseline and try to match accepted model count to target_n_models.

    The callback ``run_fn`` takes one integer budget parameter and returns
    a baseline output dict.
    """
    budget = max(1, int(initial_budget))
    out = run_fn(budget)
    n_models = _num_models_in_out(out)

    if target_n_models <= 0:
        return out, budget

    tol = max(0.0, float(tolerance_frac))
    lower = max(1, int(np.floor(target_n_models * (1.0 - tol))))
    upper = int(np.ceil(target_n_models * (1.0 + tol)))

    retries = 0
    while n_models < lower and retries < max_retries:
        scale = target_n_models / max(1, n_models)
        budget = int(np.ceil(budget * max(1.5, min(scale, 4.0))))
        print(
            f"{method_name}: n_models={n_models} < {lower}; "
            f"increasing {budget_name} to {budget} and retrying."
        )
        out = run_fn(budget)
        n_models = _num_models_in_out(out)
        retries += 1

    if n_models > upper:
        print(
            f"{method_name}: n_models={n_models} > {upper}; "
            f"downsampling to target {target_n_models} for fair comparison."
        )
        out = _downsample_out_models(out, target_n_models, seed=42)

    return out, budget


def _plot_method_gams(
    method_name: str,
    out: Dict[str, Any],
    feature_groups: Optional[Dict[str, List[int]]],
    data: pd.DataFrame,
    feature_names: Optional[List[str]],
    save_path: Path,
    max_models: int = 200,
    alpha: float = 0.25,
    normalize: bool = True,
) -> bool:
    """Plot GAM shape functions for one method output."""
    if not feature_groups:
        print(f"Skipping {method_name} GAM plot: feature_groups unavailable.")
        return False

    w_models = np.asarray(out.get("w_models", np.zeros((0, 0))), dtype=np.float64)
    nlls = np.asarray(out.get("nlls", np.array([])), dtype=np.float64)
    if w_models.ndim == 1:
        w_models = w_models.reshape(1, -1)
    if w_models.shape[0] == 0:
        print(f"Skipping {method_name} GAM plot: no accepted models.")
        return False

    if nlls.size == w_models.shape[0]:
        w_opt = w_models[int(np.argmin(nlls))]
    else:
        w_opt = w_models[0]

    result_like = SimpleNamespace(
        w_models=w_models,
        w_opt=np.asarray(w_opt, dtype=np.float64),
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_gam_shapes(
        result=result_like,
        feature_groups=feature_groups,
        data=data,
        max_models=max_models,
        alpha=alpha,
        normalize=normalize,
        save_path=str(save_path),
        title_suffix=f" - {method_name}",
        feature_names=feature_names,
    )
    print(f"Saved {method_name} GAM plot: {save_path}")
    return True


def run_baselines_comparison_main() -> None:
    """
    Compare baseline methods on one dataset with a shared absolute Rashomon bound.

    The bound is defined from MCMC's best NLL as:
        B = (1 + eps) * best_nll.
    """
    parser = argparse.ArgumentParser(description="Compare baselines on one dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset filename, e.g. compas.csv")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--max_features", type=int, default=50)
    parser.add_argument("--num_estimators", type=int, default=200)
    parser.add_argument("--max_thresholds_per_feat", type=int, default=20)

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument(
        "--diversity_fn",
        type=str,
        default=DEFAULT_DIVERSITY_FN,
        choices=["prediction_hamming", "support_jaccard", "l1", "monotonicity_entropy"],
        help="Diversity function used by the MCMC baseline.",
    )

    # Random support + ellipsoid baseline params
    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--n_ellipsoid_samples", type=int, default=20)
    parser.add_argument("--rejection_min_distance", type=float, default=None)
    parser.add_argument("--gam_plot_max_models", type=int, default=200)
    # Bootstrap baseline params
    parser.add_argument("--n_bootstrap_iters", type=int, default=30)
    parser.add_argument("--bootstrap_rashomon_rel_tol", type=float, default=1e-3)
    parser.add_argument(
        "--bootstrap_penalties",
        type=str,
        default="l1",
        help="Comma-separated penalties for bootstrap logistic models (e.g. 'l1' or 'l1,l2').",
    )
    parser.add_argument(
        "--bootstrap_c_scale",
        type=float,
        default=1.0,
        help="Scale applied to sigma2 when setting bootstrap C (smaller => stronger regularization).",
    )

    # Swapping baseline params
    parser.add_argument("--swapping_n_samples", type=int, default=100)
    parser.add_argument("--swapping_l0", type=float, default=5.0)
    parser.add_argument("--swapping_l2", type=float, default=0.1)
    parser.add_argument("--swapping_m", type=float, default=1.05)
    parser.add_argument("--swapping_k", type=int, default=3)

    args = parser.parse_args()
    if args.output is None:
        args.output = (
            f"results_baselines_comparison_{args.diversity_fn}_eps_{args.eps}/"
            "baselines_comparison.csv"
        )

    # Local import to avoid circular import at module load.

    random.seed(args.seed)
    np.random.seed(args.seed)
    print("Reading dataset")
    data_path = Path(args.dataset)
    if not data_path.is_absolute():
        data_path = Path(args.data_dir) / args.dataset
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    dataset_stem = data_path.stem
    data = pd.read_csv(data_path)
    print("Data read")
    if len(data) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(data), size=args.max_samples, replace=False)
        data = data.iloc[keep].reset_index(drop=True)
    print("Binarizing dataset")
    X, y, feature_names, feature_groups = binarize_data(
        data,
        num_estimators=args.num_estimators,
        max_thresholds_per_feat=args.max_thresholds_per_feat,
    )

    # if X.shape[1] > args.max_features:
    #     X = X[:, :args.max_features]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta_ref, proba_ref = fit_logistic(X, y, sigma2=args.sigma2, penalty="l1")
    reference_loss = _total_nll(proba_ref, y, theta_ref, sigma2=args.sigma2)
    print("Binarized dataset and calculated reference loss")
    print(
        f"Dataset={dataset_stem}, X={X.shape}, sigma2={args.sigma2}, "
        f"reference_loss={reference_loss:.6f}"
    )

    rows: List[Dict[str, Any]] = []
    method_outputs: List[Tuple[str, Dict[str, Any]]] = []
    print("Running mcmc")
    t0 = time.perf_counter()
    mcmc_method_name = "mcmc_repulsive"

    out_mcmc_unbounded = mcmc_baseline(
        X=X,
        y=y,
        feature_groups=feature_groups,
        sigma2=args.sigma2,
        diversity_fn=args.diversity_fn,
        mh_variant="repulsive",
        rashomon_loss_bound=None,
    )
    mcmc_best_nll = float(out_mcmc_unbounded.get("best_nll", np.nan))
    if not np.isfinite(mcmc_best_nll):
        mcmc_nlls = np.asarray(out_mcmc_unbounded.get("nlls", np.array([])), dtype=np.float64)
        if mcmc_nlls.size > 0:
            mcmc_best_nll = float(np.min(mcmc_nlls))
        else:
            mcmc_best_nll = float(reference_loss)

    rashomon_loss_bound = float((1.0 + float(args.eps)) * mcmc_best_nll)
    print(
        "Absolute Rashomon bound from MCMC best_nll: "
        f"best_nll={mcmc_best_nll:.6f}, eps={args.eps:.4f}, B={rashomon_loss_bound:.6f}"
    )

    out_mcmc = _apply_absolute_rashomon_bound(out_mcmc_unbounded, rashomon_loss_bound)
    mcmc_runtime = time.perf_counter() - t0
    rows.append(
        _summarize_method(
            mcmc_method_name,
            mcmc_runtime,
            out_mcmc,
            feature_groups=feature_groups,
        )
    )
    method_outputs.append((mcmc_method_name, out_mcmc))
    target_n_models = _num_models_in_out(out_mcmc)
    mcmc_num_trials = int(out_mcmc.get("n_explored", args.num_trials))
    if mcmc_num_trials <= 0:
        mcmc_num_trials = max(1, int(args.num_trials))

    mcmc_support_sizes = np.asarray(out_mcmc.get("support_sizes", np.array([])), dtype=np.float64)
    if mcmc_support_sizes.size > 0:
        mcmc_avg_sparsity = float(np.mean(mcmc_support_sizes / X.shape[1]))
    else:
        mcmc_avg_sparsity = 1.0 / float(X.shape[1])
    ellipsoid_support_size = int(np.clip(np.rint(mcmc_avg_sparsity * X.shape[1]), 1, X.shape[1]))
    print(f"MCMC produced {target_n_models} accepted models.")
    print(
        "Using MCMC-matched ellipsoid settings: "
        f"num_trials={mcmc_num_trials}, avg_sparsity={mcmc_avg_sparsity:.6f}, "
        f"support_set_size={ellipsoid_support_size}"
    )

    print("Running random_ellipsoid_uniform")
    t0 = time.perf_counter()
    out_uniform, used_uniform_ellipsoid_samples = _match_baseline_model_count(
        run_fn=lambda n_ellipsoid_samples: random_support_set_ellipsoid_sampling(
            X=X,
            y=y,
            num_trials=mcmc_num_trials,
            n_ellipsoid_samples=n_ellipsoid_samples,
            ellipsoid_sampling_method="uniform",
            rashomon_loss_bound=rashomon_loss_bound,
            sigma2=args.sigma2,
            support_set_size=ellipsoid_support_size,
        ),
        initial_budget=args.n_ellipsoid_samples,
        target_n_models=target_n_models,
        budget_name="n_ellipsoid_samples",
        method_name="random_ellipsoid_uniform",
    )
    print(
        "random_ellipsoid_uniform final model count="
        f"{_num_models_in_out(out_uniform)} with n_ellipsoid_samples={used_uniform_ellipsoid_samples}"
    )
    rows.append(
        _summarize_method(
            "random_ellipsoid_uniform",
            time.perf_counter() - t0,
            out_uniform,
            feature_groups=feature_groups,
        )
    )
    method_outputs.append(("random_ellipsoid_uniform", out_uniform))

    print("Running random_ellipsoid_rejection")
    t0 = time.perf_counter()
    out_reject, used_reject_ellipsoid_samples = _match_baseline_model_count(
        run_fn=lambda n_ellipsoid_samples: random_support_set_ellipsoid_sampling(
            X=X,
            y=y,
            num_trials=mcmc_num_trials,
            n_ellipsoid_samples=n_ellipsoid_samples,
            ellipsoid_sampling_method="rejection",
            rejection_min_distance=args.rejection_min_distance,
            rashomon_loss_bound=rashomon_loss_bound,
            sigma2=args.sigma2,
            support_set_size=ellipsoid_support_size,
        ),
        initial_budget=args.n_ellipsoid_samples,
        target_n_models=target_n_models,
        budget_name="n_ellipsoid_samples",
        method_name="random_ellipsoid_rejection",
    )
    print(
        "random_ellipsoid_rejection final model count="
        f"{_num_models_in_out(out_reject)} with n_ellipsoid_samples={used_reject_ellipsoid_samples}"
    )
    rows.append(
        _summarize_method(
            "random_ellipsoid_rejection",
            time.perf_counter() - t0,
            out_reject,
            feature_groups=feature_groups,
        )
    )
    method_outputs.append(("random_ellipsoid_rejection", out_reject))

    print("Running bootstrap_regularized_logreg")
    t0 = time.perf_counter()
    bootstrap_penalties = tuple(
        p.strip() for p in str(args.bootstrap_penalties).split(",") if p.strip()
    )
    if not bootstrap_penalties:
        raise ValueError("bootstrap_penalties must contain at least one penalty.")
    bootstrap_c = max(1e-8, float(args.sigma2) * float(args.bootstrap_c_scale))
    out_boot, used_bootstrap_iters = _match_baseline_model_count(
        run_fn=lambda n_bootstrap_iters: bootstrap_optimal_regularized_logistic_sampling(
            X=X,
            y=y,
            n_bootstrap_iters=n_bootstrap_iters,
            c_values=np.array([bootstrap_c], dtype=np.float64),
            penalties=bootstrap_penalties,
            rashomon_rel_tol=args.bootstrap_rashomon_rel_tol,
            rashomon_loss_bound=rashomon_loss_bound,
            sigma2=args.sigma2,
            random_state=args.seed,
        ),
        initial_budget=args.n_bootstrap_iters,
        target_n_models=target_n_models,
        budget_name="n_bootstrap_iters",
        method_name="bootstrap_regularized_logreg",
    )
    print(
        "bootstrap_regularized_logreg final model count="
        f"{_num_models_in_out(out_boot)} with n_bootstrap_iters={used_bootstrap_iters}"
    )
    rows.append(
        _summarize_method(
            "bootstrap_regularized_logreg",
            time.perf_counter() - t0,
            out_boot,
            feature_groups=feature_groups,
        )
    )
    method_outputs.append(("bootstrap_regularized_logreg", out_boot))

    print("Running swapping")
    # t0 = time.perf_counter()
    # out_swap = swapping_baseline(
    #     dataset_name=dataset_stem,
    #     eps=args.eps,
    #     n_samples=args.swapping_n_samples,
    #     l0=args.swapping_l0,
    #     l2=args.swapping_l2,
    #     m=args.swapping_m,
    #     ne=args.num_estimators,
    #     k=args.swapping_k,
    #     sigma2=args.sigma2,
    #     rashomon_loss_bound=rashomon_loss_bound,
    #     dataset_path=data_path,
    # )
    # rows.append(_summarize_method("swapping", time.perf_counter() - t0, out_swap))

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    gam_plot_dir = out_path.parent / "gam_plots"
    for method_name, method_out in method_outputs:
        safe_method_name = method_name.replace("/", "_").replace(" ", "_")
        gam_plot_path = gam_plot_dir / f"{dataset_stem}_{safe_method_name}_gams.png"
        _plot_method_gams(
            method_name=method_name,
            out=method_out,
            feature_groups=feature_groups,
            data=data,
            feature_names=feature_names,
            save_path=gam_plot_path,
            max_models=max(1, int(args.gam_plot_max_models)),
            alpha=0.25,
            normalize=False,
        )

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", dataset_stem)
    df.insert(1, "rashomon_loss_bound", rashomon_loss_bound)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nMethod comparison:")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    run_baselines_comparison_main()