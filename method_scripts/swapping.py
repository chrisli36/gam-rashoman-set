import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from gam_rs_utils.binarize_dataset import binarize_dataset
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.utils import *
from time import time

results = []
for dname, settings in dataset_settings:
    ne = settings["num_estimators"]
    gt = settings['m'] - 1.0
    n_support_set = settings['n_support_set']

    path = f'datasets/{dname}.csv'
    print(f"{BLUE}Dataset: {dname}{RESET}")

    X_one_hot, y, header, sample_p = get_binned_dataset(path, ne)
    X_one_hot_no_intercept = X_one_hot[:, 1:] # remove intercept column

    start = time()

    rs = fasterrisk.RiskScoreOptimizer(X_one_hot_no_intercept, y, k=n_support_set, lb=-100, ub=100, gap_tolerance=gt, select_top_m=-1, maxAttempts=25)
    rs.optimize_with_swaps_beam_search(swaps=5, beam_size=100, verbose=True)

    end = time()

    # optimal model
    w_opt = np.concatenate([np.array([rs.opt_beta0]), rs.opt_betas])
    # models in the rashomomon set with swapped features
    w_rset = np.column_stack([rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas])

    l2 = rs.lambda2
    rset_bound = rs.rset_bound

    print(f"\t{rs.sparseDiversePool_betas.shape[0]} solutions, {end - start:.2f} seconds")
    print("Average logistic loss: ", get_loss(X_one_hot, y, w_rset, loss_type="logistic", l2=l2, sample_p=sample_p))
    print("Opt model logistic loss: ", get_loss_one_model(X_one_hot, y, w_opt, loss_type="logistic", l2=l2, sample_p=sample_p))
    results.append({
        "dataset": dname,
        'l2': l2,
        "n_estimators": ne,
        "n_support_set": n_support_set,
        "gap_tolerance": gt,
        "w_rset": w_rset,
        "w_opt": w_opt,
        "rset_bound": rset_bound,
        "predictions": get_predictions(X_one_hot, w_rset),
        "runtime": end - start,
    })

with open(f"analysis/results/methods/swapping.pkl", "wb") as f:
    pickle.dump(results, f)

# get_loss(X_one_hot, y, w_rset, loss_type="accuracy", verbose=True, plot=True, w_opt=w_opt)
# get_loss(X_one_hot, y, w_rset, loss_type="logistic", verbose=True, plot=True, w_opt=w_opt, l2=l2)
