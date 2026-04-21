"""
Short benchmark: Hessian update vs scratch for add/remove/swap proposals.

Uses the same add/remove/swap Hessian update logic from `run_compas.ipynb`,
implemented in `MCMCRashomonSampler._hessian_add_feature_update` and
`MCMCRashomonSampler._hessian_remove_feature_update`.

Example:
    python new_method_scripts/benchmark_hessian_add_remove_swap.py \
        --dataset /usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark/compas.csv \
        --trials 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, os.path.abspath(".."))

from new_method_scripts.mcmc_rashomon import MCMCRashomonSampler, binarize_data
from new_method_scripts.run_diversity_experiment import select_features_l1

def _hessian_matrix(X_S, proba, sigma2, jitter=0.0):
    p = np.clip(proba, 1e-8, 1 - 1e-8)
    W = p * (1 - p)
    d = X_S.shape[1]
    return (X_S.T * W) @ X_S + (1 / sigma2 + jitter) * np.eye(d)

def _hessian_add_feature_update(
    existing_hessian: np.ndarray,
    X_prev: np.ndarray,
    W_prev: np.ndarray,
    x_added: np.ndarray,
    proba_new: np.ndarray,
    sigma2: float,
    add_pos: int,
    tau: float = 0.01,
) -> np.ndarray:
    """Cheap Hessian update when proposal adds exactly one feature."""
    # p_prev = np.clip(proba_prev, 1e-8, 1 - 1e-8)
    p_new = np.clip(proba_new, 1e-8, 1 - 1e-8)
    # W_prev = p_prev * (1 - p_prev)
    W_new = p_new * (1 - p_new)
    delta_W = W_new - W_prev

    # percentile_tau = np.percentile(np.abs(delta_W), (1 - tau) * 100)
    
    # idx = np.where(np.abs(delta_W) > tau)[0]

    # X = X_prev[idx]          
    # w = delta_W[idx,None]      
    # hessian_update = (X * w).T @ X
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


def summarize(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"mean={arr.mean():.6f}s, std={arr.std(ddof=0):.6f}s"


def load_compas_like_data(dataset_path: str, max_features: int) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(dataset_path)
    X, y, feature_names, feature_groups = binarize_data(data[:10000], num_estimators=200)
    if X.shape[1] > max_features:
        X, _, _ = select_features_l1(
            X,
            y,
            feature_names,
            feature_groups,
            max_features=max_features,
        )
    return X.astype(np.float64, copy=False), y.astype(np.float64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hessian update vs scratch.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark/covertype.csv",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--max_features", type=int, default=50)
    parser.add_argument("--l1_c", type=float, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = load_compas_like_data(args.dataset, max_features=args.max_features)
    sampler = MCMCRashomonSampler(X, y, sigma2=args.sigma2)

    S = sampler.find_initial_support(l1_C=args.l1_c)
    if len(S) < 2:
        S = list(range(min(max(2, len(S) + 1), X.shape[1])))
    if len(S) >= X.shape[1]:
        raise ValueError("Support includes all features; no add/swap benchmark possible.")

    X_old = X[:, S]
    _, proba_old = sampler.fit_logistic(X_old, y, sigma2=args.sigma2)
    W_old = proba_old * (1 - proba_old)
    H_old = _hessian_matrix(X_old, proba_old, sigma2=args.sigma2)

    add_update_t, add_scratch_t, add_max_diff = [], [], []
    rem_update_t, rem_scratch_t, rem_max_diff = [], [], []
    swp_update_t, swp_scratch_t, swp_max_diff = [], [], []
    out_features = [j for j in range(X.shape[1]) if j not in S]
    add_feature = int(rng.choice(out_features))
    remove_feature = int(rng.choice(S))
    remove_pos = int(S.index(remove_feature))
    # ---------- add ----------
    S_add = S + [add_feature]
    X_add = X[:, S_add]
    _, proba_add = sampler.fit_logistic(X_add, y, sigma2=args.sigma2)

    # ---------- remove ----------
    S_remove = [j for j in S if j != remove_feature]
    X_remove = X[:, S_remove]
    _, proba_remove = sampler.fit_logistic(X_remove, y, sigma2=args.sigma2)
    
    for _ in tqdm(range(args.trials)):
        t0 = time.perf_counter()
        H_add_upd = _hessian_add_feature_update(
            existing_hessian=H_old,
            X_prev=X_old,
            W_prev=W_old,
            x_added=X[:, add_feature],
            proba_new=proba_add,
            sigma2=args.sigma2,
            add_pos=len(S),
            tau=args.tau,
        )
        add_update_t.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        H_add_scr = sampler._hessian_matrix(X_add, proba_add, sigma2=args.sigma2)
        add_scratch_t.append(time.perf_counter() - t1)
        add_max_diff.append(float(np.max(np.abs(H_add_upd - H_add_scr))))

        t2 = time.perf_counter()
        H_rem_upd = sampler._hessian_remove_feature_update(
            existing_hessian=H_old,
            X_new=X_remove,
            proba_prev=proba_old,
            proba_new=proba_remove,
            remove_pos=remove_pos,
            tau=args.tau,
        )
        rem_update_t.append(time.perf_counter() - t2)

        t3 = time.perf_counter()
        H_rem_scr = sampler._hessian_matrix(X_remove, proba_remove, sigma2=args.sigma2)
        rem_scratch_t.append(time.perf_counter() - t3)
        rem_max_diff.append(float(np.max(np.abs(H_rem_upd - H_rem_scr))))

        # ---------- swap (remove then add), starting from H_old ----------
        S_final = S_remove + [add_feature]
        X_final = X[:, S_final]
        _, proba_final = sampler.fit_logistic(X_final, y, sigma2=args.sigma2)

        t4 = time.perf_counter()
        H_swap_mid = sampler._hessian_remove_feature_update(
            existing_hessian=H_old,
            X_new=X_remove,
            proba_prev=proba_old,
            proba_new=proba_remove,
            remove_pos=remove_pos,
            tau=args.tau,
        )
        W_remove = proba_remove * (1 - proba_remove)
        H_swap_upd = _hessian_add_feature_update(
            existing_hessian=H_swap_mid,
            X_prev=X_remove,
            W_prev=W_remove,
            x_added=X[:, add_feature],
            proba_new=proba_final,
            sigma2=args.sigma2,
            add_pos=len(S_remove),
            tau=args.tau,
        )
        swp_update_t.append(time.perf_counter() - t4)

        t5 = time.perf_counter()
        H_swap_scr = _hessian_matrix(X_final, proba_final, sigma2=args.sigma2)
        swp_scratch_t.append(time.perf_counter() - t5)
        swp_max_diff.append(float(np.max(np.abs(H_swap_upd - H_swap_scr))))

    def print_block(name: str, upd: List[float], scr: List[float], diffs: List[float]) -> None:
        upd_mean = float(np.mean(upd))
        scr_mean = float(np.mean(scr))
        speedup = scr_mean / max(upd_mean, 1e-12)
        print(f"\n{name}")
        print(f"  Update : {summarize(upd)}")
        print(f"  Scratch: {summarize(scr)}")
        print(f"  Speedup (scratch/update): {speedup:.2f}x")
        print(f"  Max |H_update - H_scratch|: {np.max(diffs):.3e}")

    print(f"Data shape: X={X.shape}, y={y.shape}, |S|={len(S)}, trials={args.trials}")
    print_block("ADD", add_update_t, add_scratch_t, add_max_diff)
    print_block("REMOVE", rem_update_t, rem_scratch_t, rem_max_diff)
    print_block("SWAP (ADD+REMOVE)", swp_update_t, swp_scratch_t, swp_max_diff)


if __name__ == "__main__":
    main()
