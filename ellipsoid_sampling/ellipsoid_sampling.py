import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle as pkl
from src.run_app import *
from src.prepare_gam import *
from matplotlib import pyplot as plt
from src.utils import *
from gam_rs_utils.utils import *
from FasterRisk.src.fasterrisk import fasterrisk
from blocking_method.blocking import optimize_support
from src.rset_opt import *
from time import time

dname = "compas"
l0 = 0.001
l2 = 0.001
m = 1.01
num_estimators = 50
binned = True

ne = num_estimators
gt = 0.0075
n_support_set = 15

path = f'datasets/{dname}.csv'
dataset = pd.read_csv(path)
print(f"Dataset: {dname}")
print(f"Binarized shape: {dataset.shape}")

# df, thresholds, header, threshold_guess_time = binarize_dataset(dataset, ne)
# X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
# X, header = convert_cumulative_to_binned(X, header)
# header = pd.Index(["intercept"] + header).astype("object")
# X_one_hot, y = utils.get_X_y(X, y, is_df=False)

# start = time()
# rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=n_support_set, lb=-100, ub=100, gap_tolerance=gt, select_top_m=-1, maxAttempts=25)
# rs.optimize_with_swaps_beam_search(swaps=3, beam_size=100, verbose=True)
# end = time()

# losses = []
# for i in range(rs.sparseDiversePool_betas.shape[0]):
#     losses.append(get_loss(X_one_hot, y, np.array([rs.sparseDiversePool_beta0[i]]), np.array([rs.sparseDiversePool_betas[i]])))

# opt_loss = get_loss(X_one_hot, y, np.array([rs.opt_beta0]), np.array([rs.opt_betas]))
# print(opt_loss, get_loss(X_one_hot, y, rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas, verbose=True))

# plot_distribution(losses, opt_loss)

sparse_gam = prepare_sparse_gam(dname, l0, l2, m, num_estimators=num_estimators, binned=binned)
filepath = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"

model = RSetOPT(filepath)
model.finetune_ellipsoid()
H_opt = model.get_normalized_H()
model.update_file(H_opt, model.w_orig)

with open(filepath, 'rb') as f:
    sparse_gam = pkl.load(f)
X = sparse_gam["X"]
y = sparse_gam["y"]
header = sparse_gam["header_new"]
w_opt = sparse_gam['w_opt']

methods = [
    {"method": "uniform"},
]

losses = []
for m in methods:
    print("Sampling method: ", m["method"])
    w_samples, rset = get_models_from_rset(filepath, n_samples=100, plot_shape=True, sample_from_surface=False, method=m)
    for w_sample in w_samples:
        losses.append(rset.check_obj(w_sample))
opt_loss = rset.check_obj(w_opt)
print(len(losses))

plot_distribution(losses, opt_loss)