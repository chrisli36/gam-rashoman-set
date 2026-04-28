"""
Directed nearest-neighbor coverage metrics between two model sets.

For sets A and B, the directed coverage distances are:
    d(a, B) = min_{b in B} dist(a, b)
for each a in A.

This module summarizes these distances with mean/min/max so callers can report:
    - A -> B coverage (small set covered by large set)
    - B -> A reverse coverage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


DistanceFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]


def _pairwise_euclidean(a: np.ndarray, b: np.ndarray, _: Optional[np.ndarray] = None) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _pairwise_l1(a: np.ndarray, b: np.ndarray, _: Optional[np.ndarray] = None) -> np.ndarray:
    diff = np.abs(a[:, None, :] - b[None, :, :])
    return np.sum(diff, axis=2)


def _pairwise_support_jaccard(
    a: np.ndarray,
    b: np.ndarray,
    _: Optional[np.ndarray] = None,
    support_threshold: float = 1e-8,
) -> np.ndarray:
    a_support = np.abs(a) > support_threshold
    b_support = np.abs(b) > support_threshold
    inter = np.logical_and(a_support[:, None, :], b_support[None, :, :]).sum(axis=2)
    union = np.logical_or(a_support[:, None, :], b_support[None, :, :]).sum(axis=2)
    sim = np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union > 0)
    return 1.0 - sim


def _pairwise_prediction_hamming(a: np.ndarray, b: np.ndarray, x_eval: Optional[np.ndarray]) -> np.ndarray:
    if x_eval is None:
        raise ValueError("Prediction Hamming distance requires x_eval.")
    a_pred = (x_eval @ a.T >= 0).astype(np.int8)
    b_pred = (x_eval @ b.T >= 0).astype(np.int8)
    return np.mean(a_pred[:, :, None] != b_pred[:, None, :], axis=0)


DISTANCE_REGISTRY: Dict[str, DistanceFn] = {
    "euclidean": _pairwise_euclidean,
    "l1": _pairwise_l1,
    "support_jaccard": _pairwise_support_jaccard,
    "prediction_hamming": _pairwise_prediction_hamming,
}


@dataclass
class DirectedCoverageSummary:
    mean_epsilon: float
    min_epsilon: float
    max_epsilon: float
    n_source: int
    n_target: int


def _validate_model_set(name: str, models: np.ndarray) -> np.ndarray:
    arr = np.asarray(models, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (n_models, n_features); got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty; cannot compute coverage distances.")
    return arr


def directed_nearest_distances(
    source_models: np.ndarray,
    target_models: np.ndarray,
    distance: str = "euclidean",
    x_eval: Optional[np.ndarray] = None,
) -> np.ndarray:
    source = _validate_model_set("source_models", source_models)
    target = _validate_model_set("target_models", target_models)
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            "Feature dimension mismatch: "
            f"source has {source.shape[1]}, target has {target.shape[1]}"
        )
    if distance not in DISTANCE_REGISTRY:
        valid = ", ".join(sorted(DISTANCE_REGISTRY))
        raise ValueError(f"Unknown distance '{distance}'. Expected one of: {valid}")

    pairwise = DISTANCE_REGISTRY[distance](source, target, x_eval)
    return np.min(pairwise, axis=1)


def summarize_directed_coverage(
    source_models: np.ndarray,
    target_models: np.ndarray,
    distance: str = "euclidean",
    x_eval: Optional[np.ndarray] = None,
) -> DirectedCoverageSummary:
    nn = directed_nearest_distances(
        source_models=source_models,
        target_models=target_models,
        distance=distance,
        x_eval=x_eval,
    )
    return DirectedCoverageSummary(
        mean_epsilon=float(np.mean(nn)),
        min_epsilon=float(np.min(nn)),
        max_epsilon=float(np.max(nn)),
        n_source=int(nn.shape[0]),
        n_target=int(np.asarray(target_models).shape[0]),
    )


def bidirectional_coverage_summary(
    set_a_models: np.ndarray,
    set_b_models: np.ndarray,
    distance: str = "euclidean",
    x_eval: Optional[np.ndarray] = None,
) -> Dict[str, DirectedCoverageSummary]:
    return {
        "a_to_b": summarize_directed_coverage(
            source_models=set_a_models,
            target_models=set_b_models,
            distance=distance,
            x_eval=x_eval,
        ),
        "b_to_a": summarize_directed_coverage(
            source_models=set_b_models,
            target_models=set_a_models,
            distance=distance,
            x_eval=x_eval,
        ),
    }
