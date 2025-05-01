import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from gam_rs_utils.binarize_dataset import binarize_dataset
from gam_rs_utils.utils import *
from FasterRisk.src.fasterrisk import fasterrisk
from time import time
import pickle

dataset_settings = {
    'bank': {
        'gap_tolerance': 0.013,
        'num_estimators': 50,
    },
    'compas': {
        'gap_tolerance': 0.003,
        'num_estimators': 50,
    },
    'diabetes': {
        'gap_tolerance': 0.0080,
        'num_estimators': 200,
    },
    # 'netherlands': {
    #     'gap_tolerance': 0.0045,
    #     'num_estimators': 50,
    # },
    'spambase': {
        'gap_tolerance': 0.008,
        'num_estimators': 50,
    },
}

results = []
num_swaps = 5
for dataset_name, settings in dataset_settings.items():
    ne = settings["num_estimators"]
    gt = settings['gap_tolerance']

    path = 'datasets/{}.csv'.format(dataset_name)
    dataset = pd.read_csv(path)
    print(f"Dataset: {dataset_name}")
    print(f"Binarized shape: {dataset.shape}")

    df, thresholds, header, threshold_guess_time = binarize_dataset(dataset, ne)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    header = pd.Index(["intercept"] + list(X.columns)).astype("object")
    X_one_hot, y = utils.get_X_y(X, y)

    finetuning_strategies = [
        {"strategy": "finetune all"}, 
        {"strategy": "no finetuning"}, 
        {"strategy": "every other"}, 
    ]
    finetuning_strategies += [{"strategy": f"finetune uncorrelated", "threshold": t} for t in np.arange(0.1, 1.0, 0.1)]

    for lf in finetuning_strategies:
        start = time()
        rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=10, lb=-100, ub=100, gap_tolerance=gt, select_top_m=-1, maxAttempts=25)
        rs.optimize_with_swaps_beam_search(swaps=num_swaps, beam_size=1000, limit_finetuning=lf)
        end = time()

        result = {
            "dataset": dataset_name,
            "dataset_shape": dataset.shape,
            "num_estimators": ne,
            "gap_tolerance": gt,
            "feature_selection": "top",
            "limit_finetuning": lf,
            "threshold_guess_time": threshold_guess_time,
            "swaps": num_swaps,
            "num_features": len(header),
            "runtime": end - start,
            "betas": rs.sparseDiversePool_betas,
            "beta0": rs.sparseDiversePool_beta0,
            "num_solutions": rs.sparseDiversePool_betas.shape[0],
            "loss": get_loss(X_one_hot, y, rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas)
        }
        results.append(result)
        print(f"\tfinetuning strategy: {lf}, {rs.sparseDiversePool_betas.shape[0]} solutions, {end - start:.2f} seconds")

        del rs

with open(f"analysis/results/finetuning.pkl", "wb") as f:
    pickle.dump(results, f)