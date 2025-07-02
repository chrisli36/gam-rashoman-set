import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prepare_gam import *
from gam_rs_utils.utils import *
from src.rset_app import RSetGAMs
from src.rset_opt import *
import pickle
import numpy as np
import cvxpy as cp
from time import time

# def l1_sparse_ellipsoid_solution(H, w_orig, rset_bound, l0, ep=None):
#     d = len(w_orig)
#     x = cp.Variable(d)
#     penalty = cp.norm1(x + ep) if ep is not None else cp.norm1(x)
#     obj = cp.Minimize(l0 * penalty)
#     constraint = [cp.quad_form(x - w_orig, H) <= rset_bound]
#     prob = cp.Problem(obj, constraint)
#     prob.solve()
#     return x.value

# def l1_relaxation(H, w_orig, rset_bound, l0, k, X, y, n):
#     samples = []
#     for _ in range(n):
#         ep = 0.1 * np.random.randn(len(w_orig))
#         solutions = {"w_orig": w_orig}
#         solutions['w_sol'] = l1_sparse_ellipsoid_solution(H, w_orig, rset_bound, l0, ep)
#         solutions['hard_thresholded_w'] = hard_threshold(solutions['w_sol'], k)

#         for k, v in solutions.items():
#             print(f"{k}: {v} {in_ellipsoid(v, w_orig, H, rset_bound)} {get_loss(X, y, np.zeros(1), np.array([v]))}")

#         samples.append(solutions['hard_thresholded_w'])
#         if not in_ellipsoid(solutions['hard_thresholded_w'], w_orig, H, rset_bound):
#             print("Warning: solution not in ellipsoid!")
#     return np.array(samples)

def in_ellipsoid(x, w_orig, H, eps):
    delta = x - w_orig
    val = delta.T @ H @ delta
    return val <= eps, val

def max_proj_direction_in_ellipsoid(H, w_orig, eps, v, l0):
    d = len(w_orig)
    x = cp.Variable(d)
    obj = cp.Maximize(cp.matmul(v, x - w_orig) - l0 * cp.norm1(x))
    constraint = [cp.quad_form(x - w_orig, H) <= eps]
    prob = cp.Problem(obj, constraint)
    prob.solve()
    return x.value

def hard_threshold(x, k2):
    x = x.copy()
    small_indices = np.argsort(np.abs(x))[:k2]
    x[small_indices] = 0
    return x

def extremal_sampling(H, w_orig, rset_bound, l0, k2, X, y, n, max_attempts=10_000):
    samples = []
    attempts = 0
    while len(samples) < n and attempts < max_attempts:
        v = 0.1 * np.random.randn(len(w_orig))
        solutions = {"w_orig": w_orig}
        solutions['w_sol'] = max_proj_direction_in_ellipsoid(H, w_orig, rset_bound, v, l0)
        solutions['hard_thresholded_w'] = hard_threshold(solutions['w_sol'], k2)

        if rset.in_rset(solutions['hard_thresholded_w']):
            samples.append(solutions['hard_thresholded_w'])
        attempts += 1
    print(f"out of {attempts} attempts, found {len(samples)} after hard thresholding")
    return np.array(samples)

# filepath = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"

results = []
for dname, settings in dataset_settings:
    print(f"Dataset: {dname}")
    l0 = settings["l0"]
    l2 = settings["l2"]
    m = settings["m"]
    ne = settings["num_estimators"]
    n_support_set = settings["n_support_set"]
    binned = True
    n_samples = 100
    max_attempts = 1000

    start = time()

    sparse_gam = prepare_sparse_gam(dname, l0, l2, m, ne, binned)
    model = RSetOPT(sparse_gam)
    model.finetune_ellipsoid()
    H_opt = model.get_normalized_H()
    model.update_file(H_opt, model.w_orig)

    with open(sparse_gam, "rb") as f:
        res = pickle.load(f)

    rset = RSetGAMs(sparse_gam)
    solutions = extremal_sampling(res['H_opt'], res['w_opt'], res['rset_bound'] * 0.1, 0.001, 6, res['X'], res['y'], 100, max_attempts=1000)

    end = time()

    # solutions
    betas = solutions
    beta0 = np.zeros(solutions.shape[0])
    # optimal solution
    opt_betas = res['w_opt']
    opt_beta0 = np.zeros(1)

    X = res['X']
    y = res['y']

    results.append({
        "dataset": dname,
        "l0": l0,
        "l2": l2,
        "m": m,
        "n_estimators": ne,
        "n_support_set": n_support_set,
        "beta0": beta0,
        "betas": betas,
        "opt_beta0": opt_beta0,
        "opt_betas": opt_betas,
        "predictions": get_predictions(X, beta0, betas),
        "runtime": end - start,
    })

with open(f"""analysis/results/methods/quadratic_programming.pkl""", "wb") as f:
    pickle.dump(results, f)

# get_loss(X, y, beta0, betas, verbose=False, loss_type="accuracy", plot=True, opt_beta0=opt_beta0, opt_betas=opt_betas)
# get_loss(X, y, beta0, betas, verbose=False, loss_type="logistic", plot=True, opt_beta0=opt_beta0, opt_betas=opt_betas, l2=l2)
# _ = plot_gam(np.array(res['header_new'][1:]), solutions[:, 1:])
