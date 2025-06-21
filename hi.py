from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from src.rset_opt import *
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.binarize_dataset import binarize_dataset
from gam_rs_utils.utils import *
from blocking_method.blocking import optimize_support

from time import time

# Baseline GAM with blocking
dname = "diabetes"
lamb0 = 0.001
lamb2 = 0.001
multiplier = 1.01
n_support_set = 15
n_estimators = 50

# sparse_gam = prepare_sparse_gam(dname, lamb0, lamb2, multiplier, n_estimators)
# baseline_gam = optimize_support(sparse_gam, n_support_set)

sparse_gam = f"{dname}_{lamb0}_{lamb2}_{multiplier}.p"
baseline_gam = f"{dname}_{lamb0}_{lamb2}_{multiplier}_merge_bins_{n_support_set}.p"

with open(sparse_gam, "rb") as f:
    sparse_gam = pickle.load(f)
with open(baseline_gam, 'rb') as f:
    baseline_gam = pickle.load(f)

X = sparse_gam["X"]
y = sparse_gam["y"]
indices = baseline_gam["indices"]
w_center_block = baseline_gam["w_center_block"]

predictions = np.zeros((X.shape[0], len(indices)))
for support in range(len(indices)):
    new_X = get_new_X(indices[support], X)
    w_sample = w_center_block[support]

    logit = new_X @ w_sample
    y_pred = np.exp(logit) / (1 + np.exp(logit))
    y_pred = np.where(y_pred > 0.5, 1, -1)
    predictions[:, support] = y_pred

y_broadcasted = np.repeat(y[:, np.newaxis], predictions.shape[1], axis=1)
loss_matrix = (predictions != y_broadcasted)
print(loss_matrix.mean())

w_orig = np.array([sparse_gam["w_orig"]])
print(get_loss(X, y, np.zeros(1), w_orig))

# Swapped GAM
dataset = pd.read_csv("datasets/diabetes.csv")
df, thresholds, header, threshold_guess_time = binarize_dataset(dataset, n_estimators)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
header = pd.Index(["intercept"] + list(X.columns)).astype("object")
X_one_hot, y = utils.get_X_y(X, y)

rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=15, lb=-100, ub=100, gap_tolerance=0.006, select_top_m=-1, maxAttempts=25)
rs.optimize_with_swaps_beam_search(swaps=5, beam_size=100, verbose=True)

print(get_loss(X_one_hot, y, rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas))

