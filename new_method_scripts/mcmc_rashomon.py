"""
Pure MCMC Rashomon Set Sampler

Samples models from the Rashomon set defined by log posterior:

    R(eps) = { w : nll(w) <= reference_nll * (1 + eps) }

where nll = negative log posterior (NLL + L2 prior). Models are kept if their
log posterior is within the specified bound of the best.

Approach:
  1. Find initial sparse model via L1-regularized logistic regression
  2. Run Metropolis-Hastings over support sets using Laplace-approximated
     marginal likelihood as the MH target
  3. For each sampled support set, fit L2-regularized logistic regression
  4. Keep models whose negative log posterior falls within the bound

Usage:
    sampler = MCMCRashomonSampler(X, y, eps=0.05)
    result = sampler.sample(n_steps=5000, proposal="swap")
    print(result.w_models.shape, result.best_error, result.error_bound)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
from functools import partial
from time import time
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import pearsonr
import time
from binarize_augmented_datasets import binarize_dataset as binarize_dataset_gbdt
import random
random.seed(42)
np.random.seed(42)
MAX_EXTRA_THRESHOLDS_PER_FEAT = 1


def _select_additional_thresholds(
    values: np.ndarray,
    existing_thresholds: List[float],
    max_extra: int = MAX_EXTRA_THRESHOLDS_PER_FEAT,
) -> List[float]:
    """Select up to ``max_extra`` observed thresholds not already in use."""
    if max_extra <= 0:
        return []

    unique_vals = np.sort(pd.unique(pd.Series(values).dropna()))
    remaining = [
        float(v)
        for v in unique_vals
        if not any(np.isclose(v, t, rtol=1e-9, atol=1e-12) for t in existing_thresholds)
    ]

    if len(remaining) <= max_extra:
        return remaining

    # Spread the added thresholds across the remaining candidates so we expand
    # coverage of the feature rather than concentrating all extras in one region.
    bin_edges = np.linspace(0, len(remaining), num=max_extra + 1, dtype=int)
    return [
        remaining[start]
        for start, end in zip(bin_edges[:-1], bin_edges[1:])
        if start < end
    ]


def binarize_data(
    data: pd.DataFrame,
    num_estimators: int = 500,
    max_thresholds_per_feat: int = 20,
    extra_thresholds_per_feat: int = MAX_EXTRA_THRESHOLDS_PER_FEAT,
):
    """
    Binarize a dataset using GBDT-based thresholds from binarize_augmented_datasets.
    For each feature, creates binary columns ``feature <= v`` using learned
    thresholds, then appends extra observed thresholds that were not previously
    selected. An intercept column of ones is prepended.

    Args:
        data: DataFrame where the last column is the target (binary 0/1).
        num_estimators: Number of GBDT estimators for threshold learning (default: 50).
        max_thresholds_per_feat: Max GBDT-selected thresholds per feature (default: 10).
        extra_thresholds_per_feat: Max additional observed thresholds per feature
            beyond the GBDT-selected ones (default: 20).

    Returns:
        X: (n, 1 + total_thresholds) numpy array.  Column 0 is the intercept.
        y: (n,) numpy array with values in {0, 1}.
        feature_names: list of column name strings (length = X.shape[1]).
        feature_groups: dict mapping original feature name -> list of column
                        indices in X (useful for group-level proposals).
    """
    df_binary, _, header, _ = binarize_dataset_gbdt(
        data,
        num_estimators=num_estimators,
        max_thresholds_per_feat=max_thresholds_per_feat,
    )
    X_binary = df_binary.iloc[:, :-1].values.astype(np.float64)
    y = df_binary.iloc[:, -1].values.astype(np.float64)
    header = list(header)

    existing_thresholds: Dict[str, List[float]] = {}
    existing_signatures: Dict[str, set] = {}
    for col_idx, colname in enumerate(header):
        if "<=" not in colname:
            continue
        feat = colname.split("<=", 1)[0].strip()
        threshold = _parse_threshold_from_colname(colname, feat)
        if threshold is not None:
            existing_thresholds.setdefault(feat, []).append(threshold)
        existing_signatures.setdefault(feat, set()).add(
            X_binary[:, col_idx].astype(np.uint8, copy=False).tobytes()
        )

    extra_columns: List[np.ndarray] = []
    extra_headers: List[str] = []
    X_raw = data.iloc[:, :-1]
    for feat in X_raw.columns:
        series = X_raw[feat]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        feature_values = series.to_numpy(dtype=np.float64, copy=False)
        candidate_thresholds = _select_additional_thresholds(
            feature_values,
            existing_thresholds.get(feat, []),
            max_extra=extra_thresholds_per_feat,
        )
        feature_signatures = existing_signatures.setdefault(feat, set())
        for threshold in candidate_thresholds:
            binary_col = (feature_values <= threshold).astype(np.float64)
            signature = binary_col.astype(np.uint8, copy=False).tobytes()
            if signature in feature_signatures:
                continue
            extra_columns.append(binary_col)
            extra_headers.append(f"{feat}<={threshold}")
            feature_signatures.add(signature)

    if extra_columns:
        X_binary = np.column_stack([X_binary] + extra_columns)
        header.extend(extra_headers)

    n = X_binary.shape[0]
    X = np.hstack([np.ones((n, 1)), X_binary])
    feature_names = ["intercept"] + list(header)

    # Build feature_groups from header: parse "feat<=v" -> group by feat
    feature_groups: Dict[str, List[int]] = {}
    for col_idx, colname in enumerate(header):
        if "<=" in colname:
            feat = colname.split("<=", 1)[0].strip()
            if feat not in feature_groups:
                feature_groups[feat] = []
            feature_groups[feat].append(col_idx + 1)  # +1 for intercept at 0

    print(X.shape)
    return X, y, feature_names, feature_groups


def _normalize_shape(shape):
    """Center (subtract mean) and normalize (divide by L2 norm) a shape vector."""
    shape = shape - shape.mean()
    norm = np.linalg.norm(shape)
    if norm > 1e-12:
        shape = shape / norm
    return shape


def _parse_threshold_from_colname(colname: str, feat: str) -> Optional[float]:
    """Extract threshold value from column name like 'feat<=v'."""
    if f"{feat}<=" not in colname:
        return None
    try:
        return float(colname.split("<=", 1)[1].strip())
    except (ValueError, IndexError):
        return None


def plot_gam_shapes(
    result: "RashomonResult",
    feature_groups: Dict[str, List[int]],
    data: pd.DataFrame,
    max_models: int = 200,
    alpha: float = 0.7,
    normalize: bool = True,
    figsize_per_feat: Tuple[float, float] = (5.5, 4.0),
    save_path: Optional[str] = None,
    title_suffix: str = "",
    feature_names: Optional[List[str]] = None,
):
    """
    Plot GAM shape functions for every original feature.

    With threshold binarization (column j = ``feature <= v_j``), the shape
    function at input value x equals the sum of weights whose threshold is
    >= x, i.e. a reverse cumulative sum of the weight vector.

    When ``normalize=True`` (default), each shape function is centered
    (mean-subtracted) and divided by its L2 norm so that all features are
    shown on a comparable scale.

    All Rashomon-set models are drawn in red; the optimal model in dark gray.

    Args:
        feature_names: Optional list of column names (e.g. ``feat<=v``). When
            provided and L1 feature selection reduces columns, thresholds are
            extracted from these names for correct alignment with indices.
    """
    import matplotlib.pyplot as plt

    X_raw = data.iloc[:, :-1]
    n_feats = len(feature_groups)
    n_cols = min(n_feats, 4)
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_feat[0] * n_cols, figsize_per_feat[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    w_all = result.w_models
    if w_all.shape[0] > max_models:
        idx = np.random.choice(w_all.shape[0], max_models, replace=False)
        w_all = w_all[idx]

    for ax, (feat, indices) in zip(axes_flat, feature_groups.items()):
        thresholds_raw = np.sort(np.unique(X_raw[feat].values))
        if feature_names is not None and len(indices) != len(thresholds_raw):
            thresholds = []
            for i in indices:
                if i < len(feature_names):
                    v = _parse_threshold_from_colname(feature_names[i], feat)
                    if v is not None:
                        thresholds.append(v)
            thresholds = np.array(thresholds) if thresholds else np.array([])
        else:
            thresholds = thresholds_raw

        if len(thresholds) != len(indices):
            continue

        perm = np.argsort(thresholds)
        thresholds = thresholds[perm]

        def _apply_perm(shape_arr):
            return shape_arr[perm]

        for w in w_all:
            shape = np.cumsum(w[indices][::-1])[::-1]
            shape = _apply_perm(shape)
            if normalize:
                shape = _normalize_shape(shape)
            ax.step(thresholds, shape, where="post",
                    color="red", alpha=alpha, linewidth=0.8)

        shape_opt = np.cumsum(result.w_opt[indices][::-1])[::-1]
        shape_opt = _apply_perm(shape_opt)
        if normalize:
            shape_opt = _normalize_shape(shape_opt)
        ax.step(thresholds, shape_opt, where="post",
                color="dimgray", linewidth=2.5, label="optimal")

        ax.set_xlabel(feat, fontsize=14)
        ylabel = "normalized shape" if normalize else "predicted logit"
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10)

    for ax in axes_flat[n_feats:]:
        ax.set_visible(False)

    title = "GAM Shape Functions (Rashomon Set)" + title_suffix
    if normalize:
        title += " — centered & normalized"
    fig.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ================================================================== #
#  Diversity metrics for repulsive MH
# ================================================================== #
# Each has signature (accepted_samples: List[ndarray], candidate: ndarray) -> float.
# Higher return value = more diverse (further from accepted). Used in log ratio:
#   log_alpha += repulsion_weight * (log(diversity(..., w_prop) + eps) - log(diversity(..., w_cur) + eps))


def diversity_min_l1(
    accepted_samples: List[np.ndarray], candidate: np.ndarray
) -> float:
    """
    Minimum mean absolute (L1) distance to any accepted sample.
    Favours proposals that are far from the nearest collected model.
    """
    if not accepted_samples:
        return 0.0
    arr = np.array(accepted_samples)
    dists = np.mean(np.abs(arr - candidate), axis=1)
    return float(dists.min())


def diversity_min_l2(
    accepted_samples: List[np.ndarray], candidate: np.ndarray
) -> float:
    """
    Minimum L2 (Euclidean) distance to any accepted sample.
    More sensitive to large coordinate-wise differences than L1.
    """
    if not accepted_samples:
        return 0.0
    arr = np.array(accepted_samples)
    dists = np.sqrt(np.sum((arr - candidate) ** 2, axis=1))
    return float(dists.min())


def diversity_mean_l1(
    accepted_samples: List[np.ndarray], candidate: np.ndarray
) -> float:
    """
    Mean L1 distance to all accepted samples.
    Favours proposals that are far from the whole set on average (not just the nearest).
    """
    if not accepted_samples:
        return 0.0
    arr = np.array(accepted_samples)
    dists = np.mean(np.abs(arr - candidate), axis=1)
    return float(np.mean(dists))


def diversity_mean_l2(
    accepted_samples: List[np.ndarray], candidate: np.ndarray
) -> float:
    """
    Mean L2 distance to all accepted samples.
    """
    if not accepted_samples:
        return 0.0
    arr = np.array(accepted_samples)
    dists = np.sqrt(np.sum((arr - candidate) ** 2, axis=1))
    return float(np.mean(dists))


def diversity_min_cosine(
    accepted_samples: List[np.ndarray], candidate: np.ndarray
) -> float:
    """
    One minus the maximum cosine similarity with any accepted sample.
    Higher when the candidate points in a different direction (orthogonal = 1).
    Uses only the non-zero support of candidate to avoid trivial zeros.
    """
    if not accepted_samples:
        return 0.0
    c_norm = np.linalg.norm(candidate)
    if c_norm < 1e-12:
        return 0.0
    c_unit = candidate / c_norm
    max_cos = -np.inf
    for w in accepted_samples:
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        cos = float(np.dot(w, c_unit) / w_norm)
        max_cos = max(max_cos, cos)
    if max_cos == -np.inf:
        return 1.0
    # 1 - cos in [0, 2]; clip so diversity is in [0, 2], typically [0, 1]
    return float(1.0 - np.clip(max_cos, -1.0, 1.0))


def diversity_support_jaccard(
    accepted_samples: List[np.ndarray], candidate: np.ndarray, thresh: float = 1e-8
) -> float:
    """
    One minus the mean Jaccard similarity of support sets.
    Favours proposals whose active (non-zero) features differ from accepted models.
    Supports are derived by thresholding weights: |w[j]| > thresh.
    """
    if not accepted_samples:
        return 0.0
    sup_c = set(np.where(np.abs(candidate) > thresh)[0].tolist())
    if not sup_c:
        return 0.0
    jaccards = []
    for w in accepted_samples:
        sup_w = set(np.where(np.abs(w) > thresh)[0].tolist())
        if not sup_w:
            jaccards.append(1.0)
            continue
        inter = len(sup_c & sup_w)
        union = len(sup_c | sup_w)
        jaccards.append(inter / union if union else 1.0)
    return float(1.0 - np.mean(jaccards))

def make_diversity_prediction_hamming(
    X: np.ndarray,
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Factory that returns a prediction-level Hamming diversity function.

    Computes logits z = X @ w for each model, converts to binary predictions
    (z >= 0 → 1, else 0), and returns the mean Hamming distance between the
    candidate's predictions and those of all accepted models.

    Usage::

        diversity_fn = make_diversity_prediction_hamming(X)
        result = sampler.sample(..., diversity_fn=diversity_fn)
    """
    def _diversity(accepted_samples: List[np.ndarray], candidate: np.ndarray) -> float:
        if not accepted_samples:
            return 0.0
        pred_cand = (X @ candidate >= 0).astype(np.int8)
        n = len(pred_cand)
        hamming_sum = 0.0
        for w in accepted_samples:
            pred_w = (X @ w >= 0).astype(np.int8)
            hamming_sum += np.count_nonzero(pred_cand != pred_w) / n
        return float(hamming_sum / len(accepted_samples))
    return _diversity


def make_diversity_prediction_logit_l2(
    X: np.ndarray,
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Factory that returns a logit-space L2 diversity function.

    Computes logits z = X @ w and returns the minimum L2 distance
    between the candidate's logit vector and those of all accepted models.
    Captures continuous differences in model confidence, not just label flips.

    Usage::

        diversity_fn = make_diversity_prediction_logit_l2(X)
        result = sampler.sample(..., diversity_fn=diversity_fn)
    """
    def _diversity(accepted_samples: List[np.ndarray], candidate: np.ndarray) -> float:
        if not accepted_samples:
            return 0.0
        z_cand = X @ candidate
        min_dist = np.inf
        for w in accepted_samples:
            d = np.sqrt(np.sum((z_cand - X @ w) ** 2))
            if d < min_dist:
                min_dist = d
        return float(min_dist)
    return _diversity


def make_diversity_prediction_rank(
    X: np.ndarray,
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Factory that returns a rank-correlation diversity function.

    Computes logits z = X @ w, ranks them, and returns 1 minus the mean
    Spearman correlation between the candidate's ranking and those of
    accepted models. Captures when models order observations differently
    even if they agree on labels.

    Usage::

        diversity_fn = make_diversity_prediction_rank(X)
        result = sampler.sample(..., diversity_fn=diversity_fn)
    """
    from scipy.stats import spearmanr

    def _diversity(accepted_samples: List[np.ndarray], candidate: np.ndarray) -> float:
        if not accepted_samples:
            return 0.0
        z_cand = X @ candidate
        corr_sum = 0.0
        for w in accepted_samples:
            rho, _ = spearmanr(z_cand, X @ w)
            corr_sum += rho if np.isfinite(rho) else 1.0
        return float(1.0 - corr_sum / len(accepted_samples))
    return _diversity


def make_diversity_monotonicity_entropy(
    feature_groups: Dict[str, List[int]],
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Factory that returns a monotonicity-entropy diversity function.

    For the set {accepted_samples} ∪ {candidate}, computes the mean entropy
    across shape functions (increasing vs decreasing vs non-monotonic).
    Higher entropy = more diverse monotonicity patterns.

    Usage::

        diversity_fn = make_diversity_monotonicity_entropy(feature_groups)
        result = sampler.sample(..., diversity_fn=diversity_fn)
    """
    from new_method_scripts.diversity_measures import monotonicity_diversity

    def _diversity(accepted_samples: List[np.ndarray], candidate: np.ndarray) -> float:
        if not feature_groups:
            return 0.0
        w_all = np.array(accepted_samples + [candidate])
        if w_all.shape[0] < 2:
            return 0.0
        # Minimal RashomonResult for monotonicity_diversity
        fake_result = RashomonResult(
            w_models=w_all,
            w_opt=w_all[0],
            support_sets=[[]] * len(w_all),
            errors=np.zeros(len(w_all)),
            best_error=0.0,
            error_bound=0.0,
            eps=0.0,
            n_explored=len(w_all),
            n_accepted=len(w_all),
            runtime=0.0,
            acceptance_rate=0.0,
        )
        out = monotonicity_diversity(fake_result, feature_groups)
        return float(out["mean_entropy"])

    return _diversity


def make_diversity_shape_l1(
    feature_groups: Dict[str, List[int]],
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Factory that returns a shape-L1 diversity function.

    For the set {accepted_samples} ∪ {candidate}, computes the mean pairwise
    L1 distance between GAM shape functions. Higher = more diverse shapes.

    Usage::

        diversity_fn = make_diversity_shape_l1(feature_groups)
        result = sampler.sample(..., diversity_fn=diversity_fn)
    """
    from new_method_scripts.diversity_measures import pairwise_shape_distance

    def _diversity(accepted_samples: List[np.ndarray], candidate: np.ndarray) -> float:
        if not feature_groups:
            return 0.0
        w_all = np.array(accepted_samples + [candidate])
        if w_all.shape[0] < 2:
            return 0.0
        fake_result = RashomonResult(
            w_models=w_all,
            w_opt=w_all[0],
            support_sets=[[]] * len(w_all),
            errors=np.zeros(len(w_all)),
            best_error=0.0,
            error_bound=0.0,
            eps=0.0,
            n_explored=len(w_all),
            n_accepted=len(w_all),
            runtime=0.0,
            acceptance_rate=0.0,
        )
        _, mean_dist = pairwise_shape_distance(fake_result, feature_groups)
        return float(mean_dist)

    return _diversity


# Default diversity for repulsive MH (same as original inline behaviour)
DEFAULT_DIVERSITY_FN = diversity_support_jaccard


@dataclass
class RashomonResult:
    """Results from MCMC Rashomon set sampling."""
    w_models: np.ndarray
    w_opt: np.ndarray
    support_sets: List[List[int]]
    errors: np.ndarray
    best_error: float
    error_bound: float
    eps: float
    n_explored: int
    n_accepted: int
    runtime: float
    acceptance_rate: float
    w_mcmc: Optional[np.ndarray] = None  # MAP models only (when ellipsoid_augment)
    w_ellipsoid: Optional[np.ndarray] = None  # Ellipsoid samples only
    ellipsoid_by_support: Optional[Dict[tuple, List[np.ndarray]]] = None  # support_key -> ellipsoid samples
    log_alpha_history: Optional[List[Dict[str, float]]] = None  # Per-step terms when mh_variant="repulsive"
    best_nll: Optional[float] = None  # Best NLL found so far


class MCMCRashomonSampler:
    """
    MCMC sampler for the misclassification-error Rashomon set.

    Uses Metropolis-Hastings over support sets (feature subsets) with the
    Laplace-approximated marginal likelihood as the MH target distribution.
    For each sampled support set, fits L2-regularized logistic regression
    and keeps models whose 0-1 error is within (1+eps) of the best.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 0.05,
        sigma2: float = 10.0,
        p_feat: float = 0.01,
        feature_groups: Optional[Dict[str, List[int]]] = None,
    ):
        """
        Args:
            X: (n, p) design matrix. Include a column of ones for an intercept.
            y: (n,) binary labels in {0, 1}.
            eps: Rashomon set tolerance.
            sigma2: Prior variance on weights (= C in sklearn LogisticRegression).
            p_feat: Prior probability of including each feature (for Laplace score).
            feature_groups: Optional dict mapping group names to column index lists.
                            Enables group-level swap proposals.
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.n, self.p = self.X.shape
        self.eps = eps
        self.sigma2 = sigma2
        self.p_feat = p_feat
        self.feature_groups = feature_groups
        # Precompute |corr(feature_i, feature_j)| once for fast lookups in MH.
        self.abs_feature_corr = self.compute_feature_correlation_matrix(self.X)
        # Per-run caches populated during `sample`.
        self._mh_hessian_cache: Dict[tuple, np.ndarray] = {}
        self._rashomon_hessian_cache: Dict[tuple, np.ndarray] = {}
        self._reset_hessian_timing_stats()

    def _reset_hessian_timing_stats(self) -> None:
        """Reset scalar timing accumulators for update vs scratch comparisons."""
        self._timing_hessian_update_s = 0.0
        self._timing_hessian_update_n = 0
        self._timing_scratch_s = 0.0
        self._timing_scratch_n = 0

    def _print_hessian_timing_stats(self, label: str) -> None:
        """Print timing summary for Hessian-update and scratch paths."""
        if (
            self._timing_hessian_update_n == 0
            and self._timing_scratch_n == 0
        ):
            return

        print(f"{label} timing summary:")
        if self._timing_hessian_update_n > 0:
            print(
                "  Hessian update: "
                f"total={self._timing_hessian_update_s:.6f}s "
                f"(n={self._timing_hessian_update_n})"
            )
        if self._timing_scratch_n > 0:
            print(
                "  Hessian from scratch: "
                f"total={self._timing_scratch_s:.6f}s "
                f"(n={self._timing_scratch_n})"
            )

    @staticmethod
    def fit_logistic(X_S, y, sigma2=10.0, max_iter=1000):
        """Fit L2-regularized logistic regression. Returns (theta, probabilities)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l2", C=sigma2, fit_intercept=False,
                solver="liblinear", max_iter=max_iter,
            )
            clf.fit(X_S, y)
        return clf.coef_.ravel(), clf.predict_proba(X_S)[:, 1]

    @staticmethod
    def compute_feature_correlation_matrix(X: np.ndarray) -> np.ndarray:
        """Compute NaN-safe absolute pairwise feature correlation matrix."""
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.abs(corr).astype(np.float32, copy=False)

    @staticmethod
    def misclassification_error(proba, y):
        """0-1 misclassification rate from predicted probabilities."""
        return float(np.mean((proba >= 0.5).astype(float) != y))

    @staticmethod
    def predict_error(X, w, y):
        """Misclassification error from a full (p,) weight vector."""
        preds = (X @ w >= 0).astype(float)
        return float(np.mean(preds != y))

    @staticmethod
    def total_nll(proba, y, theta, sigma2):
        """Total negative log-posterior (NLL + L2 prior)."""
        _e = 1e-12
        nll = -np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
        return nll + 0.5 * np.dot(theta, theta) / sigma2

    @staticmethod
    def _hessian_matrix(X_S, proba, sigma2, jitter=0.0):
        p = np.clip(proba, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        d = X_S.shape[1]
        return (X_S.T * W) @ X_S + (1 / sigma2 + jitter) * np.eye(d)

    @staticmethod
    def _hessian_add_feature_update(
        existing_hessian: np.ndarray,
        X_prev: np.ndarray,
        proba_prev: np.ndarray,
        x_added: np.ndarray,
        proba_new: np.ndarray,
        sigma2: float,
        add_pos: int,
        tau: float = 0.005,
    ) -> np.ndarray:
        """Cheap Hessian update when proposal adds exactly one feature."""
        p_prev = np.clip(proba_prev, 1e-8, 1 - 1e-8)
        p_new = np.clip(proba_new, 1e-8, 1 - 1e-8)
        W_prev = p_prev * (1 - p_prev)
        W_new = p_new * (1 - p_new)
        delta_W = W_new - W_prev

        # percentile_tau = np.percentile(np.abs(delta_W), (1 - tau) * 100)

        # idx = np.where(np.abs(delta_W) > percentile_tau)[0]
        # hessian_update = (X_prev[idx, :] * delta_W[idx, None]).T @ X_prev[idx, :]
        # + hessian_update

        updated_prev_block = existing_hessian 

        cross = (X_prev.T * W_new) @ x_added
        diag = float((x_added * W_new) @ x_added + 1.0 / sigma2)

        d_prev = existing_hessian.shape[0]
        d_new = d_prev + 1
        H_new = np.empty((d_new, d_new), dtype=np.float64)

        # Fast insertion without keep_pos/np.ix_ indexing overhead.
        if add_pos == d_prev:
            H_new[:-1, :-1] = updated_prev_block
            H_new[:-1, -1] = cross
            H_new[-1, :-1] = cross
        elif add_pos == 0:
            H_new[1:, 1:] = updated_prev_block
            H_new[1:, 0] = cross
            H_new[0, 1:] = cross
        else:
            H_new[:add_pos, :add_pos] = updated_prev_block[:add_pos, :add_pos]
            H_new[:add_pos, add_pos + 1 :] = updated_prev_block[:add_pos, add_pos:]
            H_new[add_pos + 1 :, :add_pos] = updated_prev_block[add_pos:, :add_pos]
            H_new[add_pos + 1 :, add_pos + 1 :] = updated_prev_block[add_pos:, add_pos:]

            H_new[:add_pos, add_pos] = cross[:add_pos]
            H_new[add_pos + 1 :, add_pos] = cross[add_pos:]
            H_new[add_pos, :add_pos] = cross[:add_pos]
            H_new[add_pos, add_pos + 1 :] = cross[add_pos:]
        H_new[add_pos, add_pos] = diag
        return H_new

    @staticmethod
    def _hessian_remove_feature_update(
        existing_hessian: np.ndarray,
        X_new: np.ndarray,
        proba_prev: np.ndarray,
        proba_new: np.ndarray,
        remove_pos: int,
        tau: float = 0.01,
    ) -> np.ndarray:
        """Cheap Hessian update when proposal removes exactly one feature."""
        p_prev = np.clip(proba_prev, 1e-8, 1 - 1e-8)
        p_new = np.clip(proba_new, 1e-8, 1 - 1e-8)
        W_prev = p_prev * (1 - p_prev)
        W_new = p_new * (1 - p_new)
        delta_W = W_prev - W_new

        d_prev = existing_hessian.shape[0]
        d_new = d_prev - 1
        reduced = np.empty((d_new, d_new), dtype=existing_hessian.dtype)
        if remove_pos == 0:
            reduced[:, :] = existing_hessian[1:, 1:]
        elif remove_pos == d_prev - 1:
            reduced[:, :] = existing_hessian[:-1, :-1]
        else:
            reduced[:remove_pos, :remove_pos] = existing_hessian[:remove_pos, :remove_pos]
            reduced[:remove_pos, remove_pos:] = existing_hessian[:remove_pos, remove_pos + 1 :]
            reduced[remove_pos:, :remove_pos] = existing_hessian[remove_pos + 1 :, :remove_pos]
            reduced[remove_pos:, remove_pos:] = existing_hessian[remove_pos + 1 :, remove_pos + 1 :]
        percentile_tau = np.percentile(np.abs(delta_W), (1-tau)*100)
        idx = np.where(np.abs(delta_W) > percentile_tau)[0]
        update = (
            X_new[idx, :] * delta_W[idx, None]
        ).T @ X_new[idx, :]
        return reduced - update

    @staticmethod
    def _laplace_score_from_hessian(
        X_S: np.ndarray,
        theta: np.ndarray,
        proba: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        p_feat: float,
        K: int,
        hessian: np.ndarray,
    ) -> float:
        """Laplace score using a precomputed Hessian matrix."""
        _e = 1e-12
        ll = np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
        lp_theta = -0.5 * np.dot(theta, theta) / sigma2
        _, logdet = np.linalg.slogdet(hessian)
        k = X_S.shape[1]
        lp_S = k * np.log(p_feat) + (K - k) * np.log(1 - p_feat)
        return float(ll + lp_theta - 0.5 * logdet + lp_S)

    def _hessian_logdet(
        self,
        X_S,
        proba,
        sigma2,
        jitter=1e-6,
        existing_hessian: Optional[np.ndarray] = None,
        prev_X_S: Optional[np.ndarray] = None,
        prev_proba: Optional[np.ndarray] = None,
        idx_prev: Optional[List[int]] = None,
        idx_new: Optional[List[int]] = None,
    ):
        H = None
        if (
            existing_hessian is not None
            and prev_X_S is not None
            and prev_proba is not None
            and idx_prev is not None
            and idx_new is not None
            and len(idx_new) > 10
            and set(idx_prev).issubset(set(idx_new))
        ):
            
            if len(idx_new) - len(idx_prev) == 1:
                t_hessian_update = time.perf_counter()
                added = [j for j in idx_new if j not in set(idx_prev)]
                add_pos = idx_new.index(added[0])
                H = MCMCRashomonSampler._hessian_add_feature_update(
                    existing_hessian=existing_hessian,
                    X_prev=prev_X_S,
                    proba_prev=prev_proba,
                    x_added=X_S[:, add_pos],
                    proba_new=proba,
                    sigma2=sigma2,
                    add_pos=add_pos,
                )
                self._timing_hessian_update_s += time.perf_counter() - t_hessian_update
                self._timing_hessian_update_n += 1
            elif len(idx_new) - len(idx_prev) == -1:
                t_hessian_update = time.perf_counter()
                removed = [j for j in idx_prev if j not in set(idx_new)]
                remove_pos = idx_prev.index(removed[0])
                H = MCMCRashomonSampler._hessian_remove_feature_update(
                    existing_hessian=existing_hessian,
                    X_new=self.X[:, idx_new],
                    proba_prev=prev_proba,
                    proba_new=proba,
                    remove_pos=remove_pos,
                )
                self._timing_hessian_update_s += time.perf_counter() - t_hessian_update
                self._timing_hessian_update_n += 1
            
        if H is None:
            t_scratch = time.perf_counter()
            H = MCMCRashomonSampler._hessian_matrix(X_S, proba, sigma2, jitter=jitter)
            self._timing_scratch_s += time.perf_counter() - t_scratch
            self._timing_scratch_n += 1
        _, logdet = np.linalg.slogdet(H)
        return logdet, H

    def laplace_score(
        self,
        X_S,
        theta,
        proba,
        y,
        sigma2,
        p_feat,
        K,
        existing_hessian: Optional[np.ndarray] = None,
        prev_X_S: Optional[np.ndarray] = None,
        prev_proba: Optional[np.ndarray] = None,
        idx_prev: Optional[List[int]] = None,
        idx_new: Optional[List[int]] = None,
        return_hessian: bool = False,
    ):
        """
        Laplace-approximated log marginal likelihood + structure prior.
        J(S) = log p(D|S,theta) + log p(theta|S) - 0.5 log|H_S| + log p(S)
        """
        _e = 1e-12
        ll = np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
        lp_theta = -0.5 * np.dot(theta, theta) / sigma2
        logdet, hessian = self._hessian_logdet(
            X_S,
            proba,
            sigma2,
            existing_hessian=existing_hessian,
            prev_X_S=prev_X_S,
            prev_proba=prev_proba,
            idx_prev=idx_prev,
            idx_new=idx_new,
        )
        k = X_S.shape[1]
        lp_S = k * np.log(p_feat) + (K - k) * np.log(1 - p_feat)
        sc = ll + lp_theta - 0.5 * logdet + lp_S
        if return_hessian:
            return sc, hessian
        return sc

    def score_support(self, S):
        """Fit logistic regression on support S and return (score, theta, proba)."""
        idx = sorted(S)
        return self.score_support_idx(idx)

    def score_support_idx(self, idx: List[int]):
        """Fit logistic regression on a precomputed support index list."""
        sc, theta, proba, _ = self.score_support_idx_with_hessian(idx)
        return sc, theta, proba

    def score_support_idx_with_hessian(self, idx: List[int]):
        """Like score_support_idx but also returns the Hessian matrix."""
        X_S = self.X[:, idx]
        theta, proba = self.fit_logistic(X_S, self.y, self.sigma2)
        sc, hessian = self.laplace_score(
            X_S, theta, proba, self.y, self.sigma2, self.p_feat, self.p, return_hessian=True
        )
        return sc, theta, proba, hessian

    def _fit_logistic_warm_start(self, X_S: np.ndarray, y: np.ndarray, sigma2: float, x0: np.ndarray, max_iter: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit L2-regularized logistic regression with warm-started feature coefficients.

        Uses ``LogisticRegression`` with ``warm_start=True``, same hyperparameters as
        ``fit_logistic``. Initializes ``coef_`` from ``x0`` and ``intercept_`` to zero
        before ``fit`` so sag starts from the warm start (sklearn appends intercept
        to the coefficient vector internally when optimizing).
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        n_features = X_S.shape[1]
        if x0.shape[0] != n_features:
            raise ValueError(f"x0 length {x0.shape[0]} != n_features {n_features}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l2",
                C=sigma2,
                fit_intercept=False,
                solver="liblinear",
                max_iter=max_iter,
                warm_start=True,
            )
            clf.coef_ = x0.reshape(1, n_features)
            clf.fit(X_S, y)
        theta = clf.coef_.ravel()
        proba = clf.predict_proba(X_S)[:, 1]
        return theta, proba

    def _score_support_warm_start_add_remove(
        self,
        S: set,
        S_p: set,
        theta_cur: np.ndarray,
        idx_cur: List[int],
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Warm-start fit for add or remove (exactly one feature changed).
        - Delete: x0 = current weights (excluding removed feature).
        - Add: x0 = current weights + weight of most correlated existing feature.
        """
        added = S_p - S
        removed = S - S_p
        if len(added) == 0 and len(removed) == 1:
            # Delete one feature
            idx_p = sorted(S_p)
            feat_to_coef_dict = dict(zip(idx_cur, theta_cur))
            x0 = np.array([feat_to_coef_dict.get(j, 0.0) for j in idx_p], dtype=np.float64)
        elif len(added) == 1 and len(removed) == 0:
            # Add one feature: use weight of most correlated existing feature
            f_add = next(iter(added))
            feat_to_coef = dict(zip(idx_cur, theta_cur))
            best_w = 0.0
            max_corr = 0.0
            for f in S:
                corr = float(self.abs_feature_corr[f_add, f])
                if corr > max_corr:
                    max_corr = corr
                    best_w = feat_to_coef.get(f, 0.0)
            idx_p = sorted(S_p)
            x0 = np.zeros(len(idx_p), dtype=np.float64)
            for i, j in enumerate(idx_p):
                if j in S:
                    x0[i] = theta_cur[idx_cur.index(j)]
                else:
                    x0[i] = best_w
        else:
            return None

        X_Sp = self.X[:, idx_p]
        theta_p, proba_p = self._fit_logistic_warm_start(
            X_Sp, self.y, self.sigma2, x0
        )
        sc = self.laplace_score(
            X_Sp, theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p
        )
        return sc, theta_p, proba_p

    def _score_support_finetune_swap(
        self,
        S: set,
        S_p: set,
        theta_cur: np.ndarray,
        idx_cur: List[int],
        corr_threshold: float,
        existing_hessian: Optional[np.ndarray] = None,
        proba_cur: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Fast approximation for k-swap proposals (k=1,2,3) when swapped features
        are correlated. Fit only the incoming features' weights; keep others fixed.

        Finetune is applied only if for each incoming feature f_add, there exists
        at least one outgoing feature f_rem with |corr(f_add, f_rem)| >= tau.

        Returns (sc, theta_p, proba_p) or None if finetune cannot be applied.
        """
        added = S_p - S
        removed = S - S_p
        k = len(added)
        if k == 0 or k != len(removed) or k > 3:
            return None

        added_list = sorted(added)
        removed_list = sorted(removed)

        # Correlation check: each incoming feature must have corr >= tau with
        # at least one outgoing feature
        X_add = self.X[:, added_list].astype(np.float64)
        for f_add in added_list:
            max_corr = float(np.max(self.abs_feature_corr[f_add, removed_list]))
            if max_corr < corr_threshold:
                return None

        feat_to_coef = dict(zip(idx_cur, theta_cur))
        fixed_outcome = self.X[:, idx_cur] @ theta_cur
        for f_rem in removed_list:
            w_rem = feat_to_coef.get(f_rem, 0.0)
            fixed_outcome = fixed_outcome - self.X[:, f_rem] * w_rem
        prior_sum_sq = np.dot(theta_cur, theta_cur) - sum(
            feat_to_coef.get(f, 0.0) ** 2 for f in removed_list
        )
        _e = 1e-12

        def objective(theta_add_arr: np.ndarray) -> float:
            theta_add = theta_add_arr.astype(np.float64)
            logits = np.clip(fixed_outcome + X_add @ theta_add, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            nll = -np.sum(
                self.y * np.log(p + _e) + (1 - self.y) * np.log(1 - p + _e)
            )
            prior = 0.5 * (prior_sum_sq + np.dot(theta_add, theta_add)) / self.sigma2
            fval = nll + prior
            # grad_nll = (X_add.T @ (p - self.y))
            # grad_prior = theta_add / self.sigma2
            # grad = grad_nll + grad_prior
            return float(fval)

        x0 = np.array([feat_to_coef.get(f, 0.0) for f in removed_list], dtype=np.float64)
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(-30.0, 30.0)] * k,
            options={"maxiter": 500},
        )
        theta_add_new = result.x.astype(np.float64)

        idx_p = sorted(S_p)
        add_to_idx = {f: i for i, f in enumerate(added_list)}
        theta_p = np.zeros(len(idx_p))
        for i, j in enumerate(idx_p):
            if j in added_list:
                theta_p[i] = theta_add_new[add_to_idx[j]]
            elif j in S:
                theta_p[i] = feat_to_coef.get(j, 0.0)

        X_Sp = self.X[:, idx_p]
        logits = np.clip(X_Sp @ theta_p, -20.0, 20.0)
        proba_p = 1.0 / (1.0 + np.exp(-logits))

        hessian_p = None
        if (
            existing_hessian is not None
            and proba_cur is not None
            and len(idx_p) > 10
            and len(added_list) == 1
            and len(removed_list) == 1
        ):
            f_add = added_list[0]
            f_rem = removed_list[0]
            idx_add = sorted(set(idx_cur) | {f_add})
            add_pos = idx_add.index(f_add)
            rem_pos_in_add = idx_add.index(f_rem)

            feat_to_coef = dict(zip(idx_cur, theta_cur))
            theta_add = np.zeros(len(idx_add), dtype=np.float64)
            theta_add_from_final = theta_p[idx_p.index(f_add)]
            for i, f in enumerate(idx_add):
                if f == f_add:
                    theta_add[i] = theta_add_from_final
                else:
                    theta_add[i] = feat_to_coef.get(f, 0.0)
            proba_add = 1.0 / (1.0 + np.exp(-np.clip(self.X[:, idx_add] @ theta_add, -20.0, 20.0)))

            t_hessian_update = time.perf_counter()
            h_plus = self._hessian_add_feature_update(
                existing_hessian=existing_hessian,
                X_prev=self.X[:, idx_cur],
                proba_prev=proba_cur,
                x_added=self.X[:, f_add],
                proba_new=proba_add,
                sigma2=self.sigma2,
                add_pos=add_pos,
            )
            hessian_p = self._hessian_remove_feature_update(
                existing_hessian=h_plus,
                X_new=X_Sp,
                proba_prev=proba_add,
                proba_new=proba_p,
                remove_pos=rem_pos_in_add,
            )
            self._timing_hessian_update_s += time.perf_counter() - t_hessian_update
            self._timing_hessian_update_n += 1

        if hessian_p is None:
            t_scratch = time.perf_counter()
            hessian_p = self._hessian_matrix(
                X_Sp, proba_p, self.sigma2, jitter=1e-6
            )
            self._timing_scratch_s += time.perf_counter() - t_scratch
            self._timing_scratch_n += 1
        sc_p = self._laplace_score_from_hessian(
            X_Sp, theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p
        )
        return sc_p, theta_p, proba_p, hessian_p

    def _score_support_add_remove_support(
        self,
        S: set,
        S_p: set,
        idx_cur: List[int],
        theta_cur: np.ndarray,
        corr_threshold: float = 0.3,
        existing_hessian: Optional[np.ndarray] = None,
        proba_cur: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Score support with add or remove (exactly one feature changed).
        """
        added = S_p - S
        removed = S - S_p
        idx_p = sorted(S_p)
        if len(added) == 1 and len(removed) == 0:
            feature_to_add = list(added)[0]
            corr_vector = self.abs_feature_corr[feature_to_add, idx_cur].astype(np.float64, copy=False)
            max_corr = np.max(corr_vector)
            if max_corr > corr_threshold:
                return None
            fixed_outcome = self.X[:, idx_cur] @ theta_cur
            X_add = self.X[:, feature_to_add].astype(np.float64)
            theta_add_init = np.random.randn(1)
            _e = 1e-12
            prior_sum_sq = np.dot(theta_cur, theta_cur)
            def objective(theta_add: np.ndarray) -> float:
                logits = np.clip(fixed_outcome + X_add * theta_add, -20.0, 20.0)
                p = 1.0 / (1.0 + np.exp(-logits))
                nll = -np.sum(
                    self.y * np.log(p + _e) + (1 - self.y) * np.log(1 - p + _e)
                )
                prior = 0.5 * (prior_sum_sq + np.dot(theta_add, theta_add)) / self.sigma2
                fval = nll + prior
                return float(fval)
            result = minimize(
                objective,
                x0=theta_add_init,
                method="L-BFGS-B",
                bounds=[(-30.0, 30.0)],
            )
            theta_add_new = float(np.asarray(result.x, dtype=np.float64).reshape(-1)[0])
            theta_p = np.zeros(len(idx_p))
            feat_to_coef_p = {f:i for i, f in enumerate(idx_cur)}
            for i, f in enumerate(idx_p):
                if f in S:
                    theta_p[i] = theta_cur[feat_to_coef_p[f]]
                else:
                    theta_p[i] = theta_add_new

            logits = np.clip(self.X[:, idx_p] @ theta_p, -20.0, 20.0)
            proba_p = 1.0 / (1.0 + np.exp(-logits))
            if (
                existing_hessian is not None
                and proba_cur is not None
                and len(idx_p) > 50
            ):
                t_hessian_update = time.perf_counter()
                add_pos = idx_p.index(feature_to_add)
                hessian_p = self._hessian_add_feature_update(
                    existing_hessian=existing_hessian,
                    X_prev=self.X[:, idx_cur],
                    proba_prev=proba_cur,
                    x_added=self.X[:, feature_to_add],
                    proba_new=proba_p,
                    sigma2=self.sigma2,
                    add_pos=add_pos,
                )
                self._timing_hessian_update_s += time.perf_counter() - t_hessian_update
                self._timing_hessian_update_n += 1
                sc_p = self._laplace_score_from_hessian(
                    self.X[:, idx_p], theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p
                )
            else:
                t_scratch = time.perf_counter()
                hessian_p = self._hessian_matrix(
                    self.X[:, idx_p], proba_p, self.sigma2, jitter=1e-6
                )
                self._timing_scratch_s += time.perf_counter() - t_scratch
                self._timing_scratch_n += 1
                sc_p = self._laplace_score_from_hessian(
                    self.X[:, idx_p], theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p
                )
            return sc_p, theta_p, proba_p, hessian_p

        elif len(added) == 0 and len(removed) == 1:
            feature_to_remove = list(removed)[0]
            remaining_features = [f for f in idx_cur if f != feature_to_remove]
            corr_vector = self.abs_feature_corr[feature_to_remove, remaining_features].astype(np.float64, copy=False)
            max_corr = np.max(corr_vector)
            if max_corr > corr_threshold:
                return None
            k = 3
            k_eff = min(k, len(remaining_features))
            top_k_linear = np.argsort(corr_vector)[::-1][:k_eff]
            top_k_correlated_features = sorted(
                remaining_features[i] for i in top_k_linear
            )
            remaining_features_fixed = [
                i for i in remaining_features if i not in top_k_correlated_features
            ]
            feat_to_coef_cur = {f:i for i, f in enumerate(idx_cur)}
            theta_rf = theta_cur[
                np.array([feat_to_coef_cur[f] for f in remaining_features_fixed], dtype=np.intp)
            ]
            fixed_outcome = self.X[:, remaining_features_fixed] @ theta_rf
            theta_modified_init = np.random.randn(k_eff)
            _e = 1e-12
            prior_sum_sq = np.dot(theta_rf, theta_rf)
            def objective(theta_modified: np.ndarray) -> float:
                logits = np.clip(fixed_outcome + self.X[:, top_k_correlated_features] @ theta_modified, -20.0, 20.0)
                p = 1.0 / (1.0 + np.exp(-logits))
                nll = -np.sum(
                    self.y * np.log(p + _e) + (1 - self.y) * np.log(1 - p + _e)
                )
                prior = 0.5 * (prior_sum_sq + np.dot(theta_modified, theta_modified)) / self.sigma2
                fval = nll + prior
                return float(fval)
            result = minimize(
                objective,
                x0=theta_modified_init,
                method="L-BFGS-B",
            )
            pos_in_cur = {f: i for i, f in enumerate(idx_cur)}
            theta_modified_new = result.x.astype(np.float64)
            theta_from_topk = {
                f: theta_modified_new[i] for i, f in enumerate(top_k_correlated_features)
            }
            theta_p = np.empty(len(idx_p), dtype=np.float64)
            for i, f in enumerate(idx_p):
                theta_p[i] = theta_from_topk.get(f, theta_cur[pos_in_cur[f]])
            logits = np.clip(self.X[:, idx_p] @ theta_p, -20.0, 20.0)
            proba_p = 1.0 / (1.0 + np.exp(-logits))
            if existing_hessian is not None and proba_cur is not None:
                remove_pos = idx_cur.index(feature_to_remove)
                t_hessian_update = time.perf_counter()
                hessian_p = self._hessian_remove_feature_update(
                    existing_hessian=existing_hessian,
                    X_new=self.X[:, idx_p],
                    proba_prev=proba_cur,
                    proba_new=proba_p,
                    remove_pos=remove_pos,
                )
                self._timing_hessian_update_s += time.perf_counter() - t_hessian_update
                self._timing_hessian_update_n += 1
                sc_p = self._laplace_score_from_hessian(
                    self.X[:, idx_p], theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p
                )
            else:
                t_scratch = time.perf_counter()
                hessian_p = self._hessian_matrix(
                    self.X[:, idx_p], proba_p, self.sigma2, jitter=1e-6
                )
                self._timing_scratch_s += time.perf_counter() - t_scratch
                self._timing_scratch_n += 1
                sc_p = self._laplace_score_from_hessian(
                    self.X[:, idx_p], theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p
                )
            return sc_p, theta_p, proba_p, hessian_p
        else:
            return None

    def score_support_with_error(self, S):
        """Like score_support but also returns misclassification error."""
        idx = sorted(S)
        X_S = self.X[:, idx]
        theta, proba = self.fit_logistic(X_S, self.y, self.sigma2)
        sc = self.laplace_score(
            X_S, theta, proba, self.y, self.sigma2, self.p_feat, self.p
        )
        err = self.misclassification_error(proba, self.y)
        return sc, theta, proba, err

    @staticmethod
    def propose_swap(S, K, score):
        """Drop one feature, add one from complement. Symmetric; fixed support size."""
        S_list = list(S)
        compl = [j for j in range(K) if j not in S]
        if not S_list or not compl:
            return S, 0.0, 0.0
        drop = S_list[np.random.randint(len(S_list))]
        add = compl[np.random.randint(len(compl))]
        return (S - {drop}) | {add}, 0.0, 0.0

    @staticmethod
    def propose_multi_swap(S, K, score, n_swaps=2):
        """Swap n_swaps features simultaneously. Symmetric; fixed support size."""
        S_list = list(S)
        compl = [j for j in range(K) if j not in S]
        n = min(n_swaps, len(S_list), len(compl))
        if n == 0:
            return S, 0.0, 0.0
        drops = set(np.random.choice(S_list, n, replace=False))
        adds = set(np.random.choice(compl, n, replace=False))
        return (S - drops) | adds, 0.0, 0.0

    @staticmethod
    def propose_add_or_remove(S, K, score, add_prob=0.5):
        """Add or remove a single feature. Asymmetric; changes support size."""
        compl = [j for j in range(K) if j not in S]
        if np.random.random() < add_prob and compl:
            add = compl[np.random.randint(len(compl))]
            S_prop = S | {add}
            lq_f = -np.log(len(compl))
            lq_b = -np.log(len(S_prop))
        elif len(S) > 1:
            S_list = list(S)
            drop = S_list[np.random.randint(len(S_list))]
            S_prop = S - {drop}
            lq_f = -np.log(len(S_list))
            lq_b = -np.log(len(compl) + 1)
        else:
            return S, 0.0, 0.0
        return S_prop, lq_f, lq_b

    @staticmethod
    def propose_weighted_swap(S, K, score, corr_scores=None,
                              add_power=1.0, drop_power=1.0):
        """Swap weighted by feature-target correlations. Asymmetric."""
        _e = 1e-12
        S_list = list(S)
        compl = [j for j in range(K) if j not in S]
        if not S_list or not compl:
            return S, 0.0, 0.0
        if corr_scores is None:
            return MCMCRashomonSampler.propose_swap(S, K, score)

        s_in = corr_scores[S_list]
        w_drop = (s_in + _e) ** (-drop_power)
        w_drop /= w_drop.sum()
        di = np.random.choice(len(S_list), p=w_drop)
        drop = S_list[di]

        s_out = corr_scores[compl]
        w_add = (s_out + _e) ** add_power
        w_add /= w_add.sum()
        ai = np.random.choice(len(compl), p=w_add)
        add = compl[ai]

        lq_f = np.log(w_drop[di]) + np.log(w_add[ai])

        S_prop = (S - {drop}) | {add}
        Sp_list = list(S_prop)
        compl_b = [j for j in range(K) if j not in S_prop]
        s_in_b = corr_scores[Sp_list]
        w_drop_b = (s_in_b + _e) ** (-drop_power)
        w_drop_b /= w_drop_b.sum()
        s_out_b = corr_scores[compl_b]
        w_add_b = (s_out_b + _e) ** add_power
        w_add_b /= w_add_b.sum()
        lq_b = (np.log(w_drop_b[Sp_list.index(add)])
                + np.log(w_add_b[compl_b.index(drop)]))

        return S_prop, lq_f, lq_b

    @staticmethod
    def propose_swap_or_resize(S, K, score, swap_prob=0.6):
        """
        With probability swap_prob: swap one feature (fixed size).
        With probability (1-swap_prob)/2: add a random feature.
        With probability (1-swap_prob)/2: remove a random feature.

        Correctly computes the MH proposal ratio for the mixed move.
        """
        S_list = list(S)
        compl = [j for j in range(K) if j not in S]
        k = len(S_list)
        k_out = len(compl)

        u = np.random.random()

        grow_prob = (1 - swap_prob) / 2
        shrink_prob = (1 - swap_prob) / 2

        if u < swap_prob:
            # --- swap ---
            if not S_list or not compl:
                return S, 0.0, 0.0
            drop = S_list[np.random.randint(k)]
            add = compl[np.random.randint(k_out)]
            S_prop = (S - {drop}) | {add}
            # q_fwd = swap_prob * 1/k * 1/k_out
            # q_bwd = swap_prob * 1/k * 1/k_out  (same sizes after swap)
            return S_prop, 0.0, 0.0

        elif u < swap_prob + grow_prob:
            # --- add a feature ---
            if not compl:
                return S, 0.0, 0.0
            add = compl[np.random.randint(k_out)]
            S_prop = S | {add}
            # q_fwd = grow_prob * 1/k_out
            # q_bwd = shrink_prob * 1/(k+1)
            lq_f = np.log(grow_prob) - np.log(k_out)
            lq_b = np.log(shrink_prob) - np.log(k + 1)
            return S_prop, lq_f, lq_b
        else:
            # --- remove a feature ---
            if k <= 1:
                return S, 0.0, 0.0
            drop = S_list[np.random.randint(k)]
            S_prop = S - {drop}
            # q_fwd = shrink_prob * 1/k
            # q_bwd = grow_prob * 1/(k_out+1)
            lq_f = np.log(shrink_prob) - np.log(k)
            lq_b = np.log(grow_prob) - np.log(k_out + 1)
            return S_prop, lq_f, lq_b

    @staticmethod
    def _softmax_probs(values: np.ndarray, temperature: float = 0.25, invert: bool = False) -> np.ndarray:
        """Stable softmax probabilities; optionally invert to prefer smaller values."""
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size == 0:
            return v
        t = max(float(temperature), 1e-8)
        if invert:
            z = -v / t
        else:
            z = v / t
        z = z - np.max(z)
        p = np.exp(z)
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            return np.ones_like(p) / len(p)
        return p / p_sum

    @staticmethod
    def propose_promise_score_mixture(
        S,
        K,
        score,
        feature_scores=None,
        swap_prob: float = 0.55,
        add_prob: float = 0.30,
        remove_prob: float = 0.15,
        temperature: float = 0.25,
    ):
        """
        Mixture proposal over {swap, add, remove} with score-guided sampling.

        - Adds prefer high-score features outside support (softmax on scores).
        - Removes prefer low-score features inside support (softmax on -scores).
        - Swap combines one remove + one add with the same biases.

        Returns (S_prop, log q_forward, log q_backward).
        """
        if feature_scores is None:
            return MCMCRashomonSampler.propose_swap_or_resize(
                S, K, score, swap_prob=swap_prob
            )

        S_list = list(S)
        compl = [j for j in range(K) if j not in S]
        k = len(S_list)
        k_out = len(compl)
        if k == 0:
            return S, 0.0, 0.0

        raw_move_probs = {
            "swap": float(swap_prob),
            "add": float(add_prob),
            "remove": float(remove_prob),
        }
        feasible = {
            "swap": (k > 0 and k_out > 0),
            "add": (k_out > 0),
            "remove": (k > 1),  # keep at least one active feature
        }
        move_names = [m for m in ("swap", "add", "remove") if feasible[m] and raw_move_probs[m] > 0]
        if not move_names:
            return S, 0.0, 0.0
        move_weights = np.array([raw_move_probs[m] for m in move_names], dtype=np.float64)
        move_probs = move_weights / move_weights.sum()

        move = move_names[np.random.choice(len(move_names), p=move_probs)]
        p_move_f = move_probs[move_names.index(move)]

        if move == "swap":
            s_in = np.asarray(feature_scores[S_list], dtype=np.float64)
            w_drop = MCMCRashomonSampler._softmax_probs(
                s_in, temperature=temperature, invert=True
            )
            di = int(np.random.choice(len(S_list), p=w_drop))
            drop = S_list[di]

            s_out = np.asarray(feature_scores[compl], dtype=np.float64)
            w_add = MCMCRashomonSampler._softmax_probs(
                s_out, temperature=temperature, invert=False
            )
            ai = int(np.random.choice(len(compl), p=w_add))
            add = compl[ai]

            S_prop = (S - {drop}) | {add}
            lq_f = np.log(p_move_f) + np.log(w_drop[di]) + np.log(w_add[ai])

            Sp_list = list(S_prop)
            compl_b = [j for j in range(K) if j not in S_prop]
            s_in_b = np.asarray(feature_scores[Sp_list], dtype=np.float64)
            w_drop_b = MCMCRashomonSampler._softmax_probs(
                s_in_b, temperature=temperature, invert=True
            )
            s_out_b = np.asarray(feature_scores[compl_b], dtype=np.float64)
            w_add_b = MCMCRashomonSampler._softmax_probs(
                s_out_b, temperature=temperature, invert=False
            )
            lq_b = (
                np.log(p_move_f)
                + np.log(w_drop_b[Sp_list.index(add)])
                + np.log(w_add_b[compl_b.index(drop)])
            )
            return S_prop, float(lq_f), float(lq_b)

        if move == "add":
            s_out = np.asarray(feature_scores[compl], dtype=np.float64)
            w_add = MCMCRashomonSampler._softmax_probs(
                s_out, temperature=temperature, invert=False
            )
            ai = int(np.random.choice(len(compl), p=w_add))
            add = compl[ai]
            S_prop = S | {add}
            lq_f = np.log(p_move_f) + np.log(w_add[ai])

            Sp_list = list(S_prop)
            s_in_b = np.asarray(feature_scores[Sp_list], dtype=np.float64)
            w_drop_b = MCMCRashomonSampler._softmax_probs(
                s_in_b, temperature=temperature, invert=True
            )
            remove_idx = Sp_list.index(add)
            p_remove_b = move_probs[move_names.index("remove")]
            lq_b = np.log(p_remove_b) + np.log(w_drop_b[remove_idx])
            return S_prop, float(lq_f), float(lq_b)

        s_in = np.asarray(feature_scores[S_list], dtype=np.float64)
        w_drop = MCMCRashomonSampler._softmax_probs(
            s_in, temperature=temperature, invert=True
        )
        di = int(np.random.choice(len(S_list), p=w_drop))
        drop = S_list[di]
        S_prop = S - {drop}
        lq_f = np.log(p_move_f) + np.log(w_drop[di])

        compl_b = [j for j in range(K) if j not in S_prop]
        s_out_b = np.asarray(feature_scores[compl_b], dtype=np.float64)
        w_add_b = MCMCRashomonSampler._softmax_probs(
            s_out_b, temperature=temperature, invert=False
        )
        add_idx = compl_b.index(drop)
        p_add_b = move_probs[move_names.index("add")]
        lq_b = np.log(p_add_b) + np.log(w_add_b[add_idx])
        return S_prop, float(lq_f), float(lq_b)

    def propose_feature_group_swap(self, S, K, score):
        """Swap all columns of one feature group for another. Symmetric."""
        if self.feature_groups is None:
            return self.propose_swap(S, K, score)
        g_in = [(n, i) for n, i in self.feature_groups.items() if set(i) & S]
        g_out = [(n, i) for n, i in self.feature_groups.items()
                 if not (set(i) & S)]
        if not g_in or not g_out:
            return S, 0.0, 0.0
        _, d_idx = g_in[np.random.randint(len(g_in))]
        _, a_idx = g_out[np.random.randint(len(g_out))]
        return (S - set(d_idx)) | set(a_idx), 0.0, 0.0

    @staticmethod
    def make_mixture_proposal(proposal_fns, weights=None):
        """Create a proposal that randomly selects from multiple proposals."""
        if weights is None:
            weights = np.ones(len(proposal_fns))
        w = np.array(weights, dtype=float)
        w /= w.sum()

        def _propose(S, K, score):
            idx = np.random.choice(len(proposal_fns), p=w)
            return proposal_fns[idx](S, K, score)
        return _propose

    @staticmethod
    def compute_correlation_scores(X, y):
        """Absolute Pearson correlation of each feature with y."""
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            r, _ = pearsonr(X[:, j], y)
            out[j] = np.abs(r) if np.isfinite(r) else 0.0
        return out

    def mh_sample(
        self,
        init_support: List[int],
        n_steps: int = 5000,
        burn_in: int = 1000,
        skip: int = 80,
        beta: float = 1.0,
        proposal_fn: Optional[Callable] = None,
        finetune_coordinate: bool = False,
        finetune_corr_threshold: float = 0.3,
        use_hessian_update: bool = True,
    ) -> Tuple[List[List[int]], List[float], float, List[np.ndarray], int, int]:
        """
        Standard MH over support sets.

        Returns (support_sets, scores, acceptance_rate, collected_w, n_finetune, n_full).
        Collects unique support sets at evenly spaced intervals after burn-in.
        collected_w[i] is the MAP weight vector (full p-vector) for support_sets[i].
        """
        if proposal_fn is None:
            proposal_fn = self.propose_swap

        S = set(init_support)
        idx_cur = sorted(S)
        sc, theta_cur, proba_cur, hessian_cur = self.score_support_idx_with_hessian(idx_cur)
        self._reset_hessian_timing_stats()

        skip = max(1, int(skip))
        n_acc = 0
        n_finetune, n_full = 0, 0
        results, result_scores, result_thetas = [], [], []
        result_hessians: Dict[tuple, np.ndarray] = {}
        seen = set()

        for t in tqdm(range(n_steps), desc="MH sampling"):
            S_p, lq_f, lq_b = proposal_fn(S, self.p, sc)
            if not S_p:
                continue
            idx_p = sorted(S_p)
            # try:
            hessian_p = None
            if finetune_coordinate:
                if len(S_p) == len(S): # swapping
                    # t0 = time.perf_counter()
                    t0 = time.perf_counter()
                    existing_hessian = hessian_cur if use_hessian_update else None
                    proba_for_update = proba_cur if use_hessian_update else None
                    finetune_result_swap = self._score_support_finetune_swap(
                        S,
                        S_p,
                        theta_cur,
                        idx_cur,
                        finetune_corr_threshold,
                        existing_hessian=existing_hessian,
                        proba_cur=proba_for_update,
                    )
                    time_taken = time.perf_counter() - t0
                    
                    if finetune_result_swap is not None:
                        # print(f"Time taken for finetune_swap: {time_taken:.4f} seconds")
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_swap
                        n_finetune += 1
                    else:
                        t0 = time.perf_counter()
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        time_taken = time.perf_counter() - t0
                        # print(f"Time taken for full finetune: {time_taken:.4f} seconds")
                        n_full += 1
                else:
                    t0 = time.perf_counter()
                    existing_hessian = hessian_cur if use_hessian_update else None
                    proba_for_update = proba_cur if use_hessian_update else None
                    finetune_result_add_remove = self._score_support_add_remove_support(
                        S,
                        S_p,
                        idx_cur,
                        theta_cur,
                        finetune_corr_threshold,
                        existing_hessian=existing_hessian,
                        proba_cur=proba_for_update,
                    )
                    time_taken = time.perf_counter() - t0
                    # print(f"Time taken for finetune_add_remove: {time_taken:.4f} seconds")
                    if finetune_result_add_remove is not None:
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_add_remove
                        n_finetune += 1
                    else:
                        t0 = time.perf_counter()
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        time_taken = time.perf_counter() - t0
                        # print(f"Time taken for full finetune: {time_taken:.4f} seconds")
                        n_full += 1
            else:
                t0 = time.perf_counter()
                sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                time_taken = time.perf_counter() - t0
                # print(f"Time taken for full finetune: {time_taken:.4f} seconds")
                n_full += 1

            log_a = beta * (sc_p - sc) + lq_b - lq_f
            if np.log(np.random.random() + 1e-300) < log_a:
                S, sc = S_p, sc_p
                theta_cur, idx_cur = theta_p, idx_p
                proba_cur = proba_p
                if hessian_p is None:
                    hessian_cur = self._hessian_matrix(self.X[:, idx_cur], proba_cur, self.sigma2)
                else:
                    hessian_cur = hessian_p
                n_acc += 1
                if t >= burn_in and (t - burn_in) % skip == 0:
                    key = tuple(idx_cur)
                    if key not in seen:
                        seen.add(key)
                        results.append(idx_cur.copy())
                        result_scores.append(sc)
                        w = np.zeros(self.p)
                        w[idx_cur] = theta_cur
                        result_thetas.append(w)
                        result_hessians[key] = hessian_cur.copy()

        self._print_hessian_timing_stats("MH sampling")
        return results, result_scores, n_acc / max(n_steps, 1), result_thetas, n_finetune, n_full

    def mh_sample_repulsive(
        self,
        init_support: List[int],
        n_steps: int = 5000,
        burn_in: int = 1000,
        skip: int = 80,
        beta: float = 1.0,
        proposal_fn: Optional[Callable] = None,
        repulsion_weight: float = 1.0,
        diversity_fn: Optional[Callable[[List[np.ndarray], np.ndarray], float]] = None,
        ellipsoid_augment: bool = False,
        ellipsoid_n_samples: int = 50,
        finetune_coordinate: bool = False,
        finetune_corr_threshold: float = 0.25,
        use_hessian_update: bool = True,
    ) -> Tuple[List[List[int]], List[float], float, List[np.ndarray], List[Dict[str, float]], int, int]:
        """
        Diversity-seeking MH over support sets.

        Like ``mh_sample`` but the acceptance probability includes a repulsion
        term that favours proposals far from already collected models::

            log alpha = beta * (score_proposed - score_current)
                      + (log_q_bwd - log_q_fwd)
                      + repulsion_weight * (log d_prop - log d_cur)

        where d = diversity(accepted_samples, sample). Higher diversity value =
        more diverse. Before any samples are collected (during burn-in) the
        repulsion term is zero.

        Returns (support_sets, scores, acceptance_rate, collected_w, log_alpha_history, n_finetune, n_full).
        collected_w[i] is the MAP weight vector for support_sets[i].
        log_alpha_history is a list of dicts per step with keys: term_score,
        term_proposal, term_repulsion, log_alpha, accepted, d_prop, d_cur.

        Args:
            repulsion_weight: Strength of the diversity bonus (0 = standard MH).
            diversity_fn: Callable (accepted_samples, candidate) -> float;
                higher = more diverse. Default: diversity_min_l1. Other options
                in this module: diversity_min_l2, diversity_mean_l1,
                diversity_mean_l2, diversity_min_cosine, diversity_support_jaccard.
            ellipsoid_augment: If True, compute repulsion diversity as the mean
                diversity over ellipsoid samples around candidate centers.
            ellipsoid_n_samples: Number of ellipsoid samples used for repulsion
                diversity averaging when ``ellipsoid_augment=True``.
        """
        if proposal_fn is None:
            proposal_fn = self.propose_swap
        if diversity_fn is None:
            diversity_fn = DEFAULT_DIVERSITY_FN

        S = set(init_support)
        idx_cur = sorted(S)
        sc_cur, theta_cur, proba_cur, hessian_cur = self.score_support_idx_with_hessian(idx_cur)
        self._reset_hessian_timing_stats()
        w_cur = np.zeros(self.p)
        w_cur[idx_cur] = theta_cur

        skip = max(1, int(skip))
        n_acc = 0
        n_finetune, n_full = 0, 0
        results, result_scores = [], []
        result_hessians: Dict[tuple, np.ndarray] = {}
        seen = set()
        collected_w: List[np.ndarray] = []
        log_alpha_history: List[Dict[str, float]] = []
        repulsion_ellipsoid_n = max(1, int(ellipsoid_n_samples))

        def _mean_repulsion_diversity(
            center_w: np.ndarray,
            center_idx: List[int],
            center_theta: np.ndarray,
            center_proba: np.ndarray,
            center_hessian: Optional[np.ndarray],
        ) -> float:
            max_repulsion_refs = 100
            if len(collected_w) > max_repulsion_refs:
                sampled_idx = np.random.choice(
                    len(collected_w), size=max_repulsion_refs, replace=False
                )
                repulsion_refs = [collected_w[int(i)] for i in sampled_idx]
            else:
                repulsion_refs = collected_w

            if not ellipsoid_augment:
                return float(diversity_fn(repulsion_refs, center_w))

            hessian_use = center_hessian
            if hessian_use is None:
                hessian_use = self._hessian_matrix(
                    self.X[:, center_idx], center_proba, self.sigma2, jitter=1e-6
                )
            eigvals, eigvecs = self._hessian_eig_from_matrix(hessian_use)
            nll_center = self.total_nll(center_proba, self.y, center_theta, self.sigma2)
            ub = max(1e-6, 0.05 * float(nll_center))
            theta_samples, _, _ = self._sample_ellipsoid(
                theta_center=center_theta,
                X_S=self.X[:, center_idx],
                y=self.y,
                sigma2=self.sigma2,
                n_samples=repulsion_ellipsoid_n,
                ub=ub,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            if theta_samples.shape[0] == 0:
                return float(diversity_fn(repulsion_refs, center_w))
            w_samples_full = np.zeros((theta_samples.shape[0], self.p), dtype=np.float64)
            w_samples_full[:, center_idx] = theta_samples
            d_vals = [float(diversity_fn(repulsion_refs, w_s)) for w_s in w_samples_full]
            return float(np.mean(d_vals))

        for t in tqdm(range(n_steps), desc="MH repulsive"):
            S_p, lq_f, lq_b = proposal_fn(S, self.p, sc_cur)
            if not S_p:
                continue
            idx_p = sorted(S_p)
            hessian_p = None
            if finetune_coordinate:
                if len(S_p) == len(S): # swapping
                    existing_hessian = hessian_cur if use_hessian_update else None
                    proba_for_update = proba_cur if use_hessian_update else None
                    finetune_result_swap = self._score_support_finetune_swap(
                            S,
                            S_p,
                            theta_cur,
                            idx_cur,
                            finetune_corr_threshold,
                            existing_hessian=existing_hessian,
                            proba_cur=proba_for_update,
                    )
                    if finetune_result_swap is not None:
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_swap
                        n_finetune += 1
                    else:
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        n_full += 1
                else:
                    existing_hessian = hessian_cur if use_hessian_update else None
                    proba_for_update = proba_cur if use_hessian_update else None
                    finetune_result_add_remove = self._score_support_add_remove_support(
                        S,
                        S_p,
                        idx_cur,
                        theta_cur,
                        finetune_corr_threshold,
                        existing_hessian=existing_hessian,
                        proba_cur=proba_for_update,
                    )
                    if finetune_result_add_remove is not None:
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_add_remove
                        n_finetune += 1
                    else:
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        n_full += 1
            else:
                sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                n_full += 1

            w_prop = np.zeros(self.p)
            w_prop[idx_p] = theta_p

            # Standard MH ratio
            term_score = beta * (sc_p - sc_cur)
            term_proposal = lq_b - lq_f
            log_a = term_score + term_proposal

            # Repulsion bonus
            term_repulsion = 0.0
            d_prop, d_cur = float("nan"), float("nan")
            if collected_w:
                d_prop = _mean_repulsion_diversity(
                    center_w=w_prop,
                    center_idx=idx_p,
                    center_theta=theta_p,
                    center_proba=proba_p,
                    center_hessian=hessian_p,
                )
                d_cur = _mean_repulsion_diversity(
                    center_w=w_cur,
                    center_idx=idx_cur,
                    center_theta=theta_cur,
                    center_proba=proba_cur,
                    center_hessian=hessian_cur,
                )
                term_repulsion = repulsion_weight * self.n * (
                    np.log(d_prop + 1e-12) - np.log(d_cur + 1e-12)
                )
                log_a += term_repulsion

            u = np.random.random()
            accepted = np.log(u + 1e-300) < log_a
            if accepted:
                S, sc_cur = S_p, sc_p
                theta_cur, idx_cur = theta_p, idx_p
                proba_cur = proba_p
                if hessian_p is None:
                    hessian_cur = self._hessian_matrix(self.X[:, idx_cur], proba_cur, self.sigma2)
                else:
                    hessian_cur = hessian_p
                w_cur = w_prop
                n_acc += 1
                if t >= burn_in and (t - burn_in) % skip == 0:
                    key = tuple(idx_cur)
                    if key not in seen:
                        seen.add(key)
                        results.append(idx_cur.copy())
                        result_scores.append(sc_cur)
                        collected_w.append(w_cur.copy())
                        result_hessians[key] = hessian_cur.copy()

            log_alpha_history.append({
                "term_score": float(term_score),
                "term_proposal": float(term_proposal),
                "term_repulsion": float(term_repulsion),
                "log_alpha": float(log_a),
                "accepted": accepted,
                "d_prop": float(d_prop),
                "d_cur": float(d_cur),
            })

        # Summary of log_alpha terms (helps debug repulsion vs score scale)
        if log_alpha_history:
            hist = log_alpha_history
            ts = [h["term_score"] for h in hist]
            tp = [h["term_proposal"] for h in hist]
            tr = [h["term_repulsion"] for h in hist]
            la = [h["log_alpha"] for h in hist]
            print(
                "log_alpha terms (mean ± std): "
                f"term_score={np.mean(ts):.3f}±{np.std(ts):.3f}, "
                f"term_proposal={np.mean(tp):.3f}±{np.std(tp):.3f}, "
                f"term_repulsion={np.mean(tr):.3f}±{np.std(tr):.3f}, "
                f"log_alpha={np.mean(la):.3f}±{np.std(la):.3f}"
            )

        self._mh_hessian_cache = result_hessians
        self._print_hessian_timing_stats("MH repulsive")
        return results, result_scores, n_acc / max(n_steps, 1), collected_w, log_alpha_history, n_finetune, n_full

    def random_sample(self, support_size, n_samples=50):
        """Uniformly random support sets (baseline comparison)."""
        results, scores = [], []
        seen = set()
        for _ in range(n_samples * 20):
            if len(results) >= n_samples:
                break
            S = sorted(np.random.choice(self.p, support_size, replace=False).tolist())
            key = tuple(S)
            if key in seen:
                continue
            seen.add(key)
            try:
                sc, _, _ = self.score_support(S)
                results.append(S)
                scores.append(sc)
            except Exception:
                pass
        return results, scores

    def find_initial_support(self, l1_C=2):
        """Find a sparse starting support via L1-regularized logistic regression."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l1", C=l1_C, solver="liblinear",
                fit_intercept=True, max_iter=5000,
            )
            clf.fit(self.X, self.y)
        support = np.nonzero(clf.coef_.ravel())[0].tolist()
        if not support:
            corr = self.compute_correlation_scores(self.X, self.y)
            support = np.argsort(corr)[-5:].tolist()
        return support

    @staticmethod
    def _hessian_eig(X_S, proba, sigma2):
        """Compute Hessian eigendecomposition for ellipsoid sampling."""
        H = MCMCRashomonSampler._hessian_matrix(X_S, proba, sigma2)
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs

    @staticmethod
    def _hessian_eig_from_matrix(H):
        """Eigendecomposition from a precomputed Hessian matrix."""
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs

    @staticmethod
    def _sample_ellipsoid(theta_center, X_S, y, sigma2, n_samples, ub,
                         eigvals=None, eigvecs=None):
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
        if eigvals is not None and eigvecs is not None:
            pass  # use provided
        else:
            proba_center = 1.0 / (1.0 + np.exp(-np.clip(X_S @ theta_center, -20, 20)))
            eigvals, eigvecs = MCMCRashomonSampler._hessian_eig(
                X_S, proba_center, sigma2
            )
        a = np.sqrt(2 * ub / np.maximum(eigvals, 1e-12))

        u = np.random.randn(n_samples, d)
        u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
        r = np.random.random(n_samples) ** (1.0 / d)
        z = u * r[:, None]  # uniform in unit ball

        # Rotate then scale using eigvecs as columns
        deltas = (eigvecs * a) @ z.T  # shape (d, n_samples)
        theta_samples = theta_center[:, None] + deltas  # (d, n_samples)
        theta_samples = theta_samples.T  # (n_samples, d)

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

    def sample(
        self,
        n_steps: int = 5000,
        burn_in: int = 1000,
        target_models: int = 50,
        beta: float = 1.0,
        proposal: str = "swap",
        l1_C: float = 0.1,
        init_support: Optional[List[int]] = None,
        n_swaps: int = 2,
        eps: Optional[float] = None,
        mh_variant: str = "standard",
        ellipsoid_augment: bool = False,
        ellipsoid_n_samples: int = 10,
        repulsion_weight: float = 0.0,
        diversity_fn: Optional[Callable[[List[np.ndarray], np.ndarray], float]] = None,
        finetune_coordinate: bool = False,
        finetune_corr_threshold: float = 0.3,
        use_hessian_update: bool = True,
        rashomon_loss_bound: Optional[float] = None,
    ) -> RashomonResult:
        """
        Sample models from a Rashomon set.
        Args:
            n_steps:              Total MH steps.
            burn_in:              Steps to discard before collecting.
            target_models:        Target number of unique support sets.
            beta:                 Inverse temperature (lower = more exploration).
            proposal:             One of "swap", "multi_swap", "multi_swap_3",
                                  "add_remove", "swap_or_resize",
                                  "weighted_swap", "promise_score_mixture", "feature_group",
                                  "mixture", or "random".
            l1_C:                 L1 regularization strength for initial support.
            init_support:         Starting support set (auto if None).
            n_swaps:              Swaps per step for multi_swap.
            eps:                  Override self.eps if provided.
            mh_variant:           "standard" or "repulsive".
            ellipsoid_augment:    If True, sample additional models from a
                                  Hessian-defined ellipsoid around each support
                                  set's MAP estimate.
            ellipsoid_n_samples:  Candidates to draw per support set (default 20).
            repulsion_weight:     Strength of diversity bonus in repulsive MH
                                  (0 = standard MH). Only used when
                                  mh_variant="repulsive".
            diversity_fn:         For mh_variant="repulsive", callable
                                  (accepted_samples, candidate) -> float.
                                  Default: diversity_min_l1. See module-level
                                  diversity_* functions.
            finetune_coordinate:  If True, for 1-swap proposals where swapped
                                  features are correlated (|corr| >= threshold),
                                  use fast 1D linear regression to fit only the
                                  new feature's weight instead of full refit.
            finetune_corr_threshold: Min |correlation| to use finetune (default 0.3).
            use_hessian_update:  If False, disable Hessian update shortcuts in
                                  finetune paths by always passing
                                  ``existing_hessian=None``.
            rashomon_loss_bound:  Optional absolute NLL threshold. If provided,
                                  overrides eps-based filtering and returns all
                                  models with nll <= rashomon_loss_bound.

        Returns:
            RashomonResult
        """
        eps = eps if eps is not None else self.eps
        t0 = time.perf_counter()
        self._mh_hessian_cache = {}
        self._rashomon_hessian_cache = {}

        # 1. Initial support
        if init_support is None:
            init_support = self.find_initial_support(l1_C)
            # init_support = list(range(self.X.shape[1]))
        idx0 = sorted(init_support)
        theta0, proba0 = self.fit_logistic(self.X[:, idx0], self.y, self.sigma2)
        w_opt = np.zeros(self.p)
        w_opt[idx0] = theta0
        h0 = self._hessian_matrix(self.X[:, idx0], proba0, self.sigma2)
        best_err = self.misclassification_error(proba0, self.y)
        reference_nll = self.total_nll(proba0, self.y, theta0, self.sigma2)
        print(f"Initial support: {len(init_support)} features, error: {best_err:.4f}, nll: {reference_nll:.2f}")
        print(f"Time taken to find initial support and fit logistic model: {time.perf_counter() - t0} seconds")
        # 2. Build proposal
        corr = self.compute_correlation_scores(self.X, self.y)
        prop_map = {
            "swap": self.propose_swap,
            "multi_swap": partial(self.propose_multi_swap, n_swaps=n_swaps),
            "multi_swap_3": partial(self.propose_multi_swap, n_swaps=3),
            "add_remove": self.propose_add_or_remove,
            "swap_or_resize": self.propose_swap_or_resize,
            "weighted_swap": partial(self.propose_weighted_swap, corr_scores=corr),
            "promise_score_mixture": partial[tuple[Any, float, float] | tuple[Any, Any, Any]](
                self.propose_promise_score_mixture,
                feature_scores=corr,
                swap_prob=0.55,
                add_prob=0.30,
                remove_prob=0.15,
                temperature=0.2,
            ),
            "feature_group": self.propose_feature_group_swap,
            "mixture": self.make_mixture_proposal([
                self.propose_swap_or_resize,
                partial(self.propose_multi_swap, n_swaps=2),
                partial(self.propose_multi_swap, n_swaps=3),
            ]),
            "random": None,
        }
        if proposal not in prop_map:
            raise ValueError(
                f"Unknown proposal '{proposal}'. Options: {list(prop_map.keys())}"
            )

        # 3. Explore support sets (MH returns supports, scores, acc_rate, collected_w)
        support_to_w = {tuple(idx0): w_opt.copy()}
        support_to_hessian: Dict[tuple, np.ndarray] = {tuple(idx0): h0.copy()}
        log_alpha_history = None
        mh_skip = max(1, (n_steps - burn_in) // max(target_models, 1))
        if proposal == "random":
            supports, _ = self.random_sample(len(init_support), target_models)
            acc_rate = float("nan")
            mh_collected_w: Optional[List[np.ndarray]] = None
            n_finetune, n_full = 0, 0
        elif mh_variant == "repulsive":
            supports, _, acc_rate, mh_collected_w, log_alpha_history, n_finetune, n_full = self.mh_sample_repulsive(
                init_support, n_steps, burn_in, mh_skip, beta,
                prop_map[proposal], repulsion_weight,
                diversity_fn=diversity_fn,
                ellipsoid_augment=ellipsoid_augment,
                ellipsoid_n_samples=ellipsoid_n_samples,
                finetune_coordinate=finetune_coordinate,
                finetune_corr_threshold=finetune_corr_threshold,
                use_hessian_update=use_hessian_update,
            )
            for i, S in enumerate(supports):
                if i < len(mh_collected_w):
                    support_to_w[tuple(sorted(S))] = mh_collected_w[i]
            support_to_hessian.update(self._mh_hessian_cache)
        else:
            supports, _, acc_rate, mh_collected_w, n_finetune, n_full = self.mh_sample(
                init_support, n_steps, burn_in, mh_skip, beta,
                prop_map[proposal],
                finetune_coordinate=finetune_coordinate,
                finetune_corr_threshold=finetune_corr_threshold,
                use_hessian_update=use_hessian_update,
            )
            for i, S in enumerate(supports):
                if i < len(mh_collected_w):
                    support_to_w[tuple(sorted(S))] = mh_collected_w[i]
            support_to_hessian.update(self._mh_hessian_cache)
        supports.append(init_support)
        print(
            f"Explored {len(supports)} unique support sets "
            f"(accept rate: {acc_rate:.2%})"
        )

        # 4. Build models from stored MAPs (skip refit when we have theta from MH)
        all_w, all_err, all_nll, all_supp = [], [], [], []
        center_info = []  # (theta, idx, nll, eigvals, eigvecs) per support set
        best_nll = reference_nll
        for S in tqdm(supports, desc="Fitting models"):
            if not S:
                continue
            idx = sorted(S)
            key = tuple(idx)
            if key in support_to_w:
                w = support_to_w[key]
                theta = w[idx]
                proba = 1.0 / (1.0 + np.exp(-np.clip(self.X[:, idx] @ theta, -20, 20)))
            else:
                try:
                    theta, proba = self.fit_logistic(
                        self.X[:, idx], self.y, self.sigma2
                    )
                except Exception:
                    print(f"Error fitting logistic model for support set {idx}")
                    continue
                w = np.zeros(self.p)
                w[idx] = theta
            err = self.misclassification_error(proba, self.y)
            nll = self.total_nll(proba, self.y, theta, self.sigma2)
            if nll < best_nll:
                best_nll = nll

            all_w.append(w)
            all_err.append(err)
            all_nll.append(nll)
            all_supp.append(S)
            hessian_s = support_to_hessian.get(key)
            if hessian_s is None:
                hessian_s = self._hessian_matrix(self.X[:, idx], proba, self.sigma2)
                support_to_hessian[key] = hessian_s
            self._rashomon_hessian_cache[key] = hessian_s.copy()
            if ellipsoid_augment:
                eigvals, eigvecs = self._hessian_eig_from_matrix(hessian_s)
                center_info.append((theta, idx, nll, eigvals, eigvecs))

        ellipsoid_by_support: Dict[tuple, List[np.ndarray]] = {}
        if ellipsoid_augment and center_info:
            # Compute best NLL across all centers for the ellipsoid radii
            reference_nll = min(info[2] for info in center_info)
            n_ell_accepted = 0
            for theta_c, idx_c, nll_c, eigvals_c, eigvecs_c in tqdm(center_info,
                                               desc="Ellipsoid augmentation"):
                ub = max(nll_c - reference_nll, 1e-6)
                X_S = self.X[:, idx_c]
                theta_samples, errs, nlls_ell = self._sample_ellipsoid(
                    theta_c, X_S, self.y, self.sigma2,
                    ellipsoid_n_samples, ub,
                    eigvals=eigvals_c, eigvecs=eigvecs_c,
                )
                supp_key = tuple(sorted(idx_c))
                ellipsoid_by_support.setdefault(supp_key, [])
                for ts, te, nl in zip(theta_samples, errs, nlls_ell):
                    w = np.zeros(self.p)
                    w[idx_c] = ts
                    all_w.append(w)
                    all_err.append(te)
                    all_nll.append(nl)
                    all_supp.append(list(idx_c))
                    ellipsoid_by_support[supp_key].append(w.copy())
                    n_ell_accepted += 1
                    # if te < best_err:
                    #     best_err = te
                        # w_opt = w.copy()
            print(
                f"Ellipsoid augmentation: {n_ell_accepted} additional candidates "
                f"from {len(center_info)} support sets"
            )

        # 5. Filter by Rashomon bound.
        nlls_arr = np.array(all_nll, dtype=np.float64)
        if nlls_arr.size == 0:
            nll_bound = float(rashomon_loss_bound) if rashomon_loss_bound is not None else float(reference_nll * (1 + eps))
            mask = np.array([], dtype=bool)
            errs = np.array([], dtype=np.float64)
        else:
            best_collected_nll = float(np.min(nlls_arr))
            if rashomon_loss_bound is not None:
                nll_bound = float(rashomon_loss_bound)
            else:
                nll_bound = float(best_collected_nll * (1 + eps))
            mask = nlls_arr <= nll_bound
            errs = np.array(all_err)
        bound = float(np.max(errs[mask])) if mask.any() else best_err

        w_out = np.array(all_w)[mask] if mask.any() else np.zeros((0, self.p))
        e_out = errs[mask]
        s_out = [s for s, m in zip(all_supp, mask) if m]

        w_mcmc_out = None
        w_ellipsoid_out = None
        ellipsoid_by_support_filtered: Dict[tuple, List[np.ndarray]] = {}
        if ellipsoid_augment and center_info:
            n_mcmc = len(center_info)
            mask_mcmc = mask[:n_mcmc]
            mask_ell = mask[n_mcmc:]
            if mask_mcmc.any():
                w_mcmc_out = np.array(all_w[:n_mcmc])[mask_mcmc]
            if mask_ell.any():
                w_ellipsoid_out = np.array(all_w[n_mcmc:])[mask_ell]
            # Filter ellipsoid_by_support by Rashomon bound (samples at indices n_mcmc+)
            ell_idx = n_mcmc
            for supp_key, samples in ellipsoid_by_support.items():
                kept = []
                for _ in samples:
                    if ell_idx < len(mask) and mask[ell_idx]:
                        kept.append(all_w[ell_idx])
                    ell_idx += 1
                if kept:
                    ellipsoid_by_support_filtered[supp_key] = kept

        rt = time.perf_counter() - t0
        print(f"Best NLL: {best_nll:.2f}, NLL bound: {nll_bound:.2f}")
        print(f"Best error: {best_err:.4f}, Rashomon set: {mask.sum()}/{len(errs)} models, runtime: {rt:.1f}s")
        print(f"MH scoring: finetuned {n_finetune} times, full optimization {n_full} times")

        return RashomonResult(
            w_models=w_out,
            w_opt=w_opt,
            support_sets=s_out,
            errors=e_out,
            best_error=best_err,
            error_bound=bound,
            eps=eps,
            n_explored=len(errs),
            n_accepted=int(mask.sum()),
            runtime=rt,
            acceptance_rate=acc_rate,
            w_mcmc=w_mcmc_out,
            w_ellipsoid=w_ellipsoid_out,
            ellipsoid_by_support=ellipsoid_by_support_filtered if ellipsoid_by_support_filtered else None,
            log_alpha_history=log_alpha_history,
            best_nll=best_nll,
        )