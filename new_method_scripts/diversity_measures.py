"""
Diversity measures for Rashomon set models.

Provides pairwise diversity metrics between models in a RashomonResult:
  - Hamming distance: fraction of features where two models disagree on inclusion
  - Shape function distance: average L1 distance between GAM shape functions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from new_method_scripts.mcmc_rashomon import RashomonResult


def pairwise_hamming(result: RashomonResult) -> Tuple[np.ndarray, float]:
    """
    Pairwise Hamming distance on support sets (feature inclusion).

    For each pair of models, computes the fraction of features where one model
    includes it and the other does not.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix of Hamming distances.
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    active = (W != 0).astype(np.float64)  # (n_models, p)
    n = active.shape[0]
    p = active.shape[1]

    # |a XOR b| = |a| + |b| - 2|a AND b|  →  hamming = (XOR count) / p
    overlap = active @ active.T  # (n, n) — counts of shared active features
    counts = active.sum(axis=1)  # (n,)
    xor_counts = counts[:, None] + counts[None, :] - 2 * overlap
    dist_matrix = xor_counts / p

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dist = float(dist_matrix[mask].mean()) if mask.sum() > 0 else 0.0
    return dist_matrix, mean_dist


def pairwise_support_jaccard(
    result: RashomonResult,
    thresh: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    Pairwise Jaccard *distance* on support sets (binary feature inclusion).

    For each pair of models, supports are |w_j| > thresh. Jaccard similarity is
    |A ∩ B| / |A ∪ B|; distance is 1 - similarity. Empty–empty pairs have
    distance 0; empty vs non-empty have distance 1.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    n = W.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=np.float64), 0.0
    active = (np.abs(W) > thresh).astype(np.float64)
    overlap = active @ active.T
    counts = active.sum(axis=1)
    union = counts[:, None] + counts[None, :] - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc_sim = np.where(union > 0, overlap / union, 1.0)
    dist_matrix = 1.0 - jacc_sim
    np.fill_diagonal(dist_matrix, 0.0)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dist = float(dist_matrix[mask].mean()) if mask.sum() > 0 else 0.0
    return dist_matrix, mean_dist


def pairwise_shape_distance(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
) -> Tuple[np.ndarray, float]:
    """
    Pairwise shape function diversity (average L1 distance).

    For each original feature, the shape function at threshold v_j is the
    reverse cumulative sum of weights for that feature group.  The distance
    between two models is the mean (over features and thresholds) of the
    absolute difference in shape function values.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    n = W.shape[0]

    # Build shape-function matrix: for each model, concatenate the reverse
    # cumulative sums across all feature groups.
    shape_vecs = []
    for _, indices in feature_groups.items():
        cols = W[:, indices]  # (n_models, n_thresholds)
        shapes = np.cumsum(cols[:, ::-1], axis=1)[:, ::-1]
        shape_vecs.append(shapes)
    S = np.hstack(shape_vecs)  # (n_models, total_thresholds)

    # Pairwise L1 via broadcasting (memory-efficient chunked version)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    chunk = 200
    for i in range(0, n, chunk):
        ie = min(i + chunk, n)
        diff = np.abs(S[i:ie, None, :] - S[None, :, :])  # (chunk, n, d)
        dist_matrix[i:ie, :] = diff.mean(axis=2)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dist = float(dist_matrix[mask].mean()) if mask.sum() > 0 else 0.0
    return dist_matrix, mean_dist


def pairwise_prediction_hamming(
    result: RashomonResult,
    X: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Pairwise Hamming distance between model predictions.

    For each pair of models, computes the fraction of samples on which
    they predict different classes.

    Args:
        result: RashomonResult with w_models of shape (n_models, p).
        X:      (n_samples, p) design matrix used for prediction.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    preds = (X @ W.T >= 0).astype(np.float64)   # (n_samples, n_models)
    n_models = W.shape[0]
    n_samples = X.shape[0]

    # |pred_a XOR pred_b| = |a| + |b| - 2|a AND b|
    agree = preds.T @ preds                       # (n_models, n_models)
    ones = preds.sum(axis=0)                       # (n_models,)
    disagree = ones[:, None] + ones[None, :] - 2 * agree
    dist_matrix = disagree / n_samples

    mask = np.triu(np.ones((n_models, n_models), dtype=bool), k=1)
    mean_dist = float(dist_matrix[mask].mean()) if mask.sum() > 0 else 0.0
    return dist_matrix, mean_dist


def pairwise_weight_l1(result: RashomonResult) -> Tuple[np.ndarray, float]:
    """
    Pairwise average L1 distance between raw weight vectors.

    Uses chunked computation to avoid OOM when n_models is large
    (e.g. 20k+ models from ellipsoid augmentation).

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    n = W.shape[0]

    # Memory-efficient chunked version (avoids (n,n,d) allocation)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    chunk = 200
    for i in range(0, n, chunk):
        ie = min(i + chunk, n)
        diff = np.abs(W[i:ie, None, :] - W[None, :, :])  # (chunk, n, d)
        dist_matrix[i:ie, :] = diff.mean(axis=2)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dist = float(dist_matrix[mask].mean()) if mask.sum() > 0 else 0.0
    return dist_matrix, mean_dist


def _normalize_shape(shape):
    """Center and L2-normalize a shape vector."""
    shape = shape - shape.mean()
    norm = np.linalg.norm(shape)
    if norm > 1e-12:
        shape = shape / norm
    return shape


def gam_structural_distance(
    w1: np.ndarray,
    w2: np.ndarray,
    feature_groups: Dict[str, List[int]],
) -> float:
    """
    Structural diversity between two GAMs, invariant to per-feature
    translation and scale.

    Each feature's shape function is centered and L2-normalized before
    comparison via cosine distance.  Features are weighted by the average
    pre-normalization L2 norm of the two models so that irrelevant
    (near-zero) features don't dominate the metric.

    Returns a value in [0, 1]:  0 = identical structure, 1 = maximally different.
    """
    distances = []
    weights = []

    for _, indices in feature_groups.items():
        shape1 = np.cumsum(w1[indices][::-1])[::-1]
        shape2 = np.cumsum(w2[indices][::-1])[::-1]

        s1 = shape1 - shape1.mean()
        s2 = shape2 - shape2.mean()
        n1 = np.linalg.norm(s1)
        n2 = np.linalg.norm(s2)

        if n1 < 1e-12 and n2 < 1e-12:
            distances.append(0.0)
        elif n1 < 1e-12 or n2 < 1e-12:
            distances.append(1.0)
        else:
            s1 /= n1
            s2 /= n2
            cos_sim = np.clip(np.dot(s1, s2), -1.0, 1.0)
            distances.append((1.0 - cos_sim) / 2.0)

        weights.append((n1 + n2) / 2.0)

    distances = np.array(distances)
    weights = np.array(weights)
    total = weights.sum()
    if total > 1e-12:
        return float(distances @ weights / total)
    return 0.0


def pairwise_structural_distance(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
) -> Tuple[np.ndarray, float]:
    """
    Pairwise structural (cosine-based) distance between GAM shape functions.

    Each per-feature shape is centered and L2-normalized to factor out
    vertical translation and scale, then compared via cosine distance.
    Features are weighted by their average pre-normalization L2 norm
    so that near-zero features contribute less.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix in [0, 1].
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    n = W.shape[0]

    shapes_per_feat = []
    for _, indices in feature_groups.items():
        cols = W[:, indices]
        shapes = np.cumsum(cols[:, ::-1], axis=1)[:, ::-1]
        centered = shapes - shapes.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        shapes_per_feat.append((centered, norms))

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dists = []
            wts = []
            for centered, norms in shapes_per_feat:
                n1, n2 = norms[i], norms[j]
                if n1 < 1e-12 and n2 < 1e-12:
                    dists.append(0.0)
                elif n1 < 1e-12 or n2 < 1e-12:
                    dists.append(1.0)
                else:
                    cos_sim = np.dot(centered[i], centered[j]) / (n1 * n2)
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    dists.append((1.0 - cos_sim) / 2.0)
                # wts.append((n1 + n2) / 2.0)
                wts.append(1.0)

            dists = np.array(dists)
            wts = np.array(wts)
            total = wts.sum()
            d = float(dists @ wts / total) if total > 1e-12 else 0.0
            dist_matrix[i, j] = dist_matrix[j, i] = d

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dist = float(dist_matrix[mask].mean()) if mask.sum() > 0 else 0.0
    return dist_matrix, mean_dist


def monotonicity_diversity(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
    flat_tol: float = 1e-10,
) -> Dict[str, object]:
    """
    For each feature, classify every model's shape function as monotonically
    increasing, monotonically decreasing, non-monotonic, or flat (all weights
    effectively zero).  Report the fractions and an entropy-based diversity
    score.

    A shape is classified by looking at the successive differences of the
    step-function values (reverse cumulative sum of weights).  If all diffs
    are >= 0 it is increasing; if all are <= 0 it is decreasing; otherwise
    non-monotonic.  Shapes whose centered L2 norm is below ``flat_tol`` are
    labelled flat (feature unused).

    The per-feature diversity score is the Shannon entropy of the
    (increasing, decreasing, non-monotonic) distribution (excluding flat),
    normalized by log(3) so that 1.0 = perfectly uniform three-way split.

    Returns:
        Dict with keys:
            per_feature: dict mapping feature name -> {
                "frac_increasing", "frac_decreasing", "frac_nonmono",
                "frac_flat", "entropy"
            }
            mean_entropy: average normalized entropy across features
                          (only features with at least one non-flat model).
    """
    W = result.w_models
    n_models = W.shape[0]

    per_feature = {}
    entropies = []

    for feat, indices in feature_groups.items():
        cols = W[:, indices]
        shapes = np.cumsum(cols[:, ::-1], axis=1)[:, ::-1]  # (n_models, n_thresholds)
        centered = shapes - shapes.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)

        n_inc = 0
        n_dec = 0
        n_non = 0
        n_flat = 0

        for m in range(n_models):
            if norms[m] < flat_tol:
                n_flat += 1
                continue
            diffs = np.diff(shapes[m])
            if np.all(diffs >= -1e-14):
                n_inc += 1
            elif np.all(diffs <= 1e-14):
                n_dec += 1
            else:
                n_non += 1

        n_active = n_inc + n_dec + n_non
        if n_active > 0:
            p_inc = n_inc / n_active
            p_dec = n_dec / n_active
            p_non = n_non / n_active
            probs = np.array([p_inc, p_dec, p_non])
            probs = probs[probs > 0]
            ent = float(-np.sum(probs * np.log(probs)) / np.log(3))
            entropies.append(ent)
        else:
            p_inc = p_dec = p_non = 0.0
            ent = 0.0

        per_feature[feat] = {
            "frac_increasing": p_inc,
            "frac_decreasing": p_dec,
            "frac_nonmono": p_non,
            "frac_flat": n_flat / n_models,
            "entropy": ent,
        }

    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    return {
        "per_feature": per_feature,
        "mean_entropy": mean_entropy,
    }


def variable_importance(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Permutation importance for each original feature across the Rashomon set.

    For a single model, permutation importance of feature f is the increase
    in misclassification error when the columns belonging to f are randomly
    shuffled (averaged over ``n_repeats`` shuffles).

    This function computes min, max, mean, and std of that quantity across
    all models in the Rashomon set.

    Args:
        result:         RashomonResult.
        feature_groups: Dict mapping feature names to column index lists.
        X:              (n, p) design matrix.
        y:              (n,) binary labels in {0, 1}.
        n_repeats:      Number of random permutations per feature per model.

    Returns:
        Dict mapping feature name -> {"min", "max", "mean", "std", "values"}
        where "values" is the (n_models,) array of per-model importances.
    """
    W = result.w_models  # (n_models, p)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples, _ = X.shape
    n_models = W.shape[0]

    logits_base = X @ W.T  # (n_samples, n_models)
    base_errors = np.mean((logits_base >= 0).astype(float) != y[:, None], axis=0)

    out = {}
    for feat, indices in feature_groups.items():
        importance = np.zeros(n_models)
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm = np.random.permutation(n_samples)
            X_perm[:, indices] = X_perm[perm][:, indices]
            logits_perm = X_perm @ W.T
            perm_errors = np.mean(
                (logits_perm >= 0).astype(float) != y[:, None], axis=0
            )
            importance += perm_errors - base_errors
        importance /= n_repeats

        out[feat] = {
            "min": float(importance.min()),
            "max": float(importance.max()),
            "mean": float(importance.mean()),
            "std": float(importance.std()),
            "values": importance,
        }
    return out


def plot_variable_importance(
    vi_dict: Dict[str, Dict[str, float]],
    label: str = "",
    ax=None,
):
    """Plot permutation importance ranges (min/max bars with mean marker)."""
    import matplotlib.pyplot as plt

    feats = list(vi_dict.keys())
    means = [vi_dict[f]["mean"] for f in feats]
    mins = [vi_dict[f]["min"] for f in feats]
    maxs = [vi_dict[f]["max"] for f in feats]
    stds = [vi_dict[f]["std"] for f in feats]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(feats))))

    y = np.arange(len(feats))
    ax.barh(y, [mx - mn for mn, mx in zip(mins, maxs)],
            left=mins, color="steelblue", alpha=0.35, edgecolor="steelblue",
            label="range [min, max]")
    ax.errorbar(means, y, xerr=stds, fmt="o", color="black",
                capsize=3, markersize=4, label="mean ± std")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Permutation Importance (error increase)")
    title = "Permutation Importance"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.legend(fontsize=9)
    return ax


def diversity_summary(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
    X: Optional[np.ndarray] = None,
    label: str = "",
) -> Dict[str, float]:
    """
    Compute all diversity measures and print a summary.

    Args:
        result:         RashomonResult.
        feature_groups: Dict mapping feature names to column index lists.
        X:              Design matrix for prediction Hamming (skipped if None).
        label:          Optional label for the printout.

    Returns a dict with keys: hamming, pred_hamming, shape_l1, weight_l1, structural.
    """
    _, h = pairwise_hamming(result)
    _, s = pairwise_shape_distance(result, feature_groups)
    _, w = pairwise_weight_l1(result)
    _, st = pairwise_structural_distance(result, feature_groups)
    mono = monotonicity_diversity(result, feature_groups)

    out = {
        "hamming": h, "shape_l1": s, "weight_l1": w,
        "structural": st, "monotonicity": mono,
    }

    tag = f" ({label})" if label else ""
    print(f"Diversity summary{tag}  [{result.n_accepted} models]")
    print(f"  Pairwise Hamming (support)   : {h:.4f}")

    if X is not None:
        _, ph = pairwise_prediction_hamming(result, X)
        out["pred_hamming"] = ph
        print(f"  Pairwise Hamming (prediction): {ph:.4f}")

    print(f"  Pairwise shape L1 distance   : {s:.4f}")
    print(f"  Pairwise weight L1 distance  : {w:.4f}")
    print(f"  Pairwise structural distance : {st:.4f}")
    print(f"  Monotonicity entropy (avg)   : {mono['mean_entropy']:.4f}")
    for feat, info in mono["per_feature"].items():
        print(f"    {feat:>20s}:  inc {info['frac_increasing']:.0%}"
              f"  dec {info['frac_decreasing']:.0%}"
              f"  non {info['frac_nonmono']:.0%}"
              f"  flat {info['frac_flat']:.0%}"
              f"  H={info['entropy']:.3f}")
    return out
