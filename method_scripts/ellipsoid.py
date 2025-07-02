import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle as pkl
from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from gam_rs_utils.utils import *
from src.rset_opt import *
from time import time

def hard_threshold(x, k2):
    x = x.copy()
    small_indices = np.argsort(np.abs(x))[:k2]
    x[small_indices] = 0
    return x

methods = [
    {"method": "uniform"},
    {"method": "poisson", "max_attempts": 10_000},
]

for method in methods:
    results = []
    for dname, settings in dataset_settings:
        print(f"Dataset: {dname}, method: {method['method']}")
        l0 = settings["l0"]
        l2 = settings["l2"]
        m = settings["m"]
        ne = settings["num_estimators"]
        n_support_set = settings["n_support_set"]
        binned = True
        n_samples = 100
        max_attempts = 1000

        if method["method"] == "poisson":
            method["r_min"] = settings["r_min"]
        filepath = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"

        start = time()

        sparse_gam = prepare_sparse_gam(dname, l0, l2, m, num_estimators=ne, binned=binned)

        model = RSetOPT(sparse_gam)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)

        with open(sparse_gam, 'rb') as f:
            sparse_gam = pkl.load(f)
        X = sparse_gam["X"]
        y = sparse_gam["y"]
        header = sparse_gam["header_new"]
        w_opt = sparse_gam['w_opt']

        w_samples, rset = get_models_from_rset(filepath, n_samples=max_attempts, plot_shape=False, sample_from_surface=False, method=method)

        w_samples_zeroed = []
        n_support = w_samples.shape[1]
        k2 = n_support - 1 - n_support_set
        attempts = 0
        while len(w_samples_zeroed) < n_samples and attempts < max_attempts and attempts < w_samples.shape[0]:
            w_samp = w_samples[attempts]
            w_samp_zeroed = hard_threshold(w_samp, k2)
            if rset.in_rset(w_samp_zeroed):
                w_samples_zeroed.append(w_samp_zeroed)
            attempts += 1
        print(f"\tout of {attempts} attempts, kept {len(w_samples_zeroed)} after hard thresholding")
        w_samples_zeroed = np.vstack(w_samples_zeroed)

        end = time()

        betas = w_samples_zeroed
        beta0 = np.zeros(w_samples_zeroed.shape[0])

        print("Loss: ", get_loss(X, y, beta0, betas))
        results.append({
            "dataset": dname,
            "l0": l0,
            "l2": l2,
            "m": m,
            "n_estimators": ne,
            "n_support_set": n_support_set,
            "beta0": beta0,
            "betas": betas,
            "opt_beta0": np.zeros(1),
            "opt_betas": w_opt,
            "predictions": get_predictions(X, np.zeros(len(w_samples_zeroed)), w_samples_zeroed),
            "runtime": end - start,
        })

    with open(f"""analysis/results/methods/{method["method"]}_ellipsoid_sampling.pkl""", "wb") as f:
        pkl.dump(results, f)

