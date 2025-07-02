import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from src.rset_opt import *
from gam_rs_utils.utils import *
from blocking_method.blocking import optimize_support
from time import time


# sparse_gam = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"
# baseline_gam = f"models/{dname}_{l0}_{l2}_{m}_{binned}_merge_bins_{n_support_set}.p"

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

    baseline_gam = optimize_support(sparse_gam, n_support_set)

    with open(sparse_gam, "rb") as f:
        sparse_gam = pickle.load(f)
    with open(baseline_gam, 'rb') as f:
        baseline_gam = pickle.load(f)

    X = sparse_gam["X"]
    y = sparse_gam["y"]
    w_opt = sparse_gam["w_opt"]

    indices = baseline_gam["indices"]
    w_center_block = baseline_gam["w_center_block"]

    end = time()

    predictions = np.zeros((X.shape[0], len(indices)))
    w_samples = np.zeros((len(indices), len(w_opt)))
    for support in range(len(indices)):
        new_X = get_new_X(indices[support], X)
        w_sample = w_center_block[support]

        logit = new_X @ w_sample
        y_pred = np.exp(logit) / (1 + np.exp(logit))
        y_pred = np.where(y_pred > 0.5, 1, -1)
        predictions[:, support] = y_pred

        true_w_sample = get_true_w_sample(indices[support], w_sample)
        w_samples[support] = true_w_sample

    beta0 = np.zeros(w_samples.shape[0])
    betas = w_samples

    try:
        assert np.allclose(predictions, get_predictions(X, beta0, betas))
    except AssertionError:
        print("predictions:")
        print(predictions)
        print("get_predictions output:")
        print(get_predictions(X, np.zeros(w_samples.shape[0]), w_samples))
        raise

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
        "predictions": predictions,
        "runtime": end - start,
    })

with open(f"""analysis/results/methods/blocking.pkl""", "wb") as f:
    pickle.dump(results, f)

# get_loss(X, y, np.zeros(w_samples.shape[0]), w_samples, loss_type="accuracy", verbose=True, plot=True, opt_beta0=0, opt_betas=w_opt)
# get_loss(X, y, np.zeros(w_samples.shape[0]), w_samples, loss_type="logistic", verbose=True, plot=True, opt_beta0=0, opt_betas=w_opt, l2=l2)
