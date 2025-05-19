from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from gam_rs_utils.binarize_dataset import binarize_dataset
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.utils import *

import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from tqdm import tqdm

# load the dataset
dataset_name = 'bank'
path = 'datasets/{}.csv'.format(dataset_name)
dataset = pd.read_csv(path)
print(f"Dataset: {dataset_name}")
print(f"Binarized shape: {dataset.shape}")

df, thresholds, header, threshold_guess_time = binarize_dataset(dataset, 50)
X, y = df.iloc[:, :-1], df.iloc[:, -1]

header = pd.Index(["intercept"] + list(X.columns)).astype("object")
X_one_hot, y = utils.get_X_y(X, y)

# rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=10, lb=-100, ub=100, gap_tolerance=0.006, select_top_m=-1)
# rs.optimize_with_swaps(swaps=3, fanout_decay=1, feature_selection="top")
# beta0 = rs.sparseDiversePool_beta0
# betas = rs.sparseDiversePool_betas

rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=10, lb=-100, ub=100, gap_tolerance=0.006, select_top_m=-1)
rs.optimize_with_swaps_beam_search(swaps=3, beam_size=1000)
beta0 = rs.sparseDiversePool_beta0
betas = rs.sparseDiversePool_betas

# # save betas and beta0 to a pickle file
# with open("compas_0.008_3.pkl", "wb") as f:
#     pkl.dump({"betas": betas, "beta0": beta0}, f)

print(f"found {betas.shape[0]} models")
print(get_loss(X_one_hot, y, beta0, betas, verbose=True))

# union_of_support_sets = plot_gam(header, betas[:100, :])
# sorted_data = dict(sorted(union_of_support_sets.items(), key=lambda x: x[1], reverse=True))

# # Plot histogram
# plt.figure(figsize=(15, 6))
# plt.bar(list(sorted_data.keys()), list(sorted_data.values()), color='blue', alpha=0.7)
# plt.xticks(rotation=90, fontsize=8)
# plt.ylabel('Count')
# plt.title('Count of Features across different Support Sets', fontsize=20)
# plt.tight_layout()
# plt.show()