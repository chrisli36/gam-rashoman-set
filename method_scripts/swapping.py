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
    gt = settings['gap_tolerance']
    n_support_set = settings['n_support_set']

    path = f'datasets/{dname}.csv'
    dataset = pd.read_csv(path)
    print(f"Dataset: {dname}")
    print(f"Binarized shape: {dataset.shape}")

    df, thresholds, header, threshold_guess_time = binarize_dataset(dataset, ne)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X, header = convert_cumulative_to_binned(X, header)
    header = pd.Index(["intercept"] + header).astype("object")
    X_one_hot, y = utils.get_X_y(X, y, is_df=False)

    start = time()

    rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=n_support_set, lb=-100, ub=100, gap_tolerance=gt, select_top_m=-1, maxAttempts=25)
    rs.optimize_with_swaps_beam_search(swaps=5, beam_size=100, verbose=True)

    end = time()

    # optimal model
    opt_beta0 = rs.opt_beta0
    opt_betas = rs.opt_betas

    # models in the rashomomon set with swapped features
    beta0 = rs.sparseDiversePool_beta0
    betas = rs.sparseDiversePool_betas

    lambda2 = rs.lambda2

    print(f"\t{rs.sparseDiversePool_betas.shape[0]} solutions, {end - start:.2f} seconds")
    results.append({
        "dataset": dname,
        "n_estimators": ne,
        "n_support_set": n_support_set,
        "gap_tolerance": gt,
        "beta0": beta0,
        "betas": betas,
        "opt_beta0": opt_beta0,
        "opt_betas": opt_betas,
        "predictions": get_predictions(X_one_hot, beta0, betas),
        "runtime": end - start,
    })

with open(f"analysis/results/methods/swapping.pkl", "wb") as f:
    pickle.dump(results, f)

# get_loss(X_one_hot, y, beta0, betas, loss_type="accuracy", verbose=True, plot=True, opt_beta0=opt_beta0, opt_betas=opt_betas)
# get_loss(X_one_hot, y, beta0, betas, loss_type="logistic", verbose=True, plot=True, opt_beta0=opt_beta0, opt_betas=opt_betas, l2=lambda2)
