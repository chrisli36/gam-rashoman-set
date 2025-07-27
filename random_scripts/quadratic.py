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
from method_scripts.results_class import MethodType, create_results_object, save_results

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

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def extremal_sampling(H, w_orig, rset_bound, l0, k2, n, r_min, max_attempts=10_000):
    samples = []
    attempts = 0
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        v = 0.1 * np.random.randn(len(w_orig))
        solutions = {"w_orig": w_orig}
        solutions['w_sol'] = max_proj_direction_in_ellipsoid(H, w_orig, rset_bound, v, l0)
        solutions['hard_thresholded_w'] = hard_threshold(solutions['w_sol'], k2)

        if not rset.in_rset(solutions['hard_thresholded_w']):
            continue
        if not all(euclidean_distance(solutions['hard_thresholded_w'], prev) >= r_min for prev in samples):
            continue
        samples.append(solutions['hard_thresholded_w'])
    print(f"out of {attempts} attempts, found {len(samples)} after hard thresholding")
    return np.array(samples)

# filepath = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"

results = []
for dname, settings in dataset_settings:
    print(f"{BLUE}Dataset: {dname}{RESET}")
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
    solutions = extremal_sampling(res['H_opt'], res['w_opt'], res['rset_bound'] * 0.1, 0.001, 6, 100, 0.01, max_attempts=1000)

    end = time()

    X = res['X']
    y = res['y']
    w_opt = res['w_opt']
    sample_p = res['sample_proportion']

    print(f"{solutions.shape[0]} solutions found")
    print("Average logistic loss: ", get_loss(X, y, solutions, loss_type="logistic", l2=l2, sample_p=sample_p))
    print("Opt model logistic loss: ", get_loss_one_model(X, y, w_opt, loss_type="logistic", l2=l2, sample_p=sample_p))

    result_obj = create_results_object(
        method_type=MethodType.QUADRATIC,
        dataset=dname,
        l0=l0,
        l2=l2,
        m=m,
        n_estimators=ne,
        n_support_set=n_support_set,
        w_rset=solutions,
        w_opt=w_opt,
        rset_bound=res['rset_bound'],
        predictions=get_predictions(X, solutions),
        runtime=end - start,
    )
    results.append(result_obj)

save_results(results, MethodType.QUADRATIC)

# get_loss(X, y, solutions, verbose=False, loss_type="accuracy", plot=True, w_opt=res['w_opt'])
# get_loss(X, y, solutions, verbose=False, loss_type="logistic", plot=True, w_opt=res['w_opt'], l2=l2)
# _ = plot_gam(np.array(res['header_new'][1:]), solutions[:, 1:])
