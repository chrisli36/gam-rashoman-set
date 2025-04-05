from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from gam_rs_utils.binarize_dataset import binarize_dataset
from gam_rs_utils.utils import *
from FasterRisk.src.fasterrisk import fasterrisk

import pickle
import numpy as np
from time import time

# load the dataset
dataset_name = 'bank'
path = 'datasets/{}.csv'.format(dataset_name)
dataset = pd.read_csv(path)
print(f"Dataset: {dataset_name}")
print(f"shape: {dataset.shape}")

df, thresholds, header, threshold_guess_time = binarize_dataset(dataset, 50)
X, y = df.iloc[:, :-1], df.iloc[:, -1]

header = list(X.columns)
header = pd.Index(["intercept"] + header)
header = header.astype("object")

X_one_hot, y = utils.get_X_y(X, y)

start = time()
rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=10, lb=-100, ub=100, gap_tolerance=0.006, select_top_m=-1)
rs.optimize_with_swaps(swaps=3, fanout_decay=1, feature_selection="top")
beta0 = rs.sparseDiversePool_beta0
betas = rs.sparseDiversePool_betas
end = time()

print(len(betas), end - start, "seconds")
print(get_loss(X_one_hot, y, beta0, betas))


start = time()
rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=10, lb=-100, ub=100, gap_tolerance=0.006, select_top_m=-1)
rs.optimize_with_swaps_beam_search(swaps=3, beam_size=11100)
beta0 = rs.sparseDiversePool_beta0
betas = rs.sparseDiversePool_betas
end = time()

print(len(betas), end - start, "seconds")
print(get_loss(X_one_hot, y, beta0, betas))