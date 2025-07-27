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

def get_binned_dataset(dname, num_estimators):
    data = pd.read_csv(f"datasets/{dname}.csv")
    df, _, header, _ = binarize_dataset(data, num_estimators)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X_new, header_new = convert_cumulative_to_binned(X, header)
    header_new = ["intercept"] + header_new
    X_new, y = utils.get_X_y(X_new, y, is_df=False)
    return X_new, y, header_new

methods = [
    "poisson_ellipsoid_sampling", 
    "uniform_ellipsoid_sampling",
    "blocking", 
    "quadratic_programming", 
    "swapping"
]
dataset_names = ["bank", "compas", "diabetes", "spambase", "mimic2"]

all_datasets_results = {}

for dname in dataset_names:
    all_columns = set()
    method_dfs = {}
    for method in methods:
        with open(f"analysis/results/methods/{method}.pkl", "rb") as f:
            results = pickle.load(f)
        df = pd.DataFrame(results)
        df = df[df["dataset"] == dname]
        method_dfs[method] = df
        all_columns.update(df.columns)
    all_columns = list(all_columns)

    dfs = []
    for method, df in method_dfs.items():
        df = df.reindex(columns=all_columns)
        df["method"] = method
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    all_datasets_results[dname] = combined_df

for dname, df in all_datasets_results.items():
    accuracy, logistic, hamming, iou, euclidean, cosine = [], [], [], [], [], []
    model_count = []
    for i, row in df.iterrows():
        X, y, header = get_binned_dataset(dname, row["n_estimators"])
        betas = row["betas"]
        beta0 = row["beta0"]
        opt_betas = row["opt_betas"]
        opt_beta0 = row["opt_beta0"]
        predictions = row["predictions"]
        l2 = 0.001

        if betas.shape[0] == 0:
            accuracy.append(0)
            logistic.append(0)
            hamming.append(0)
            iou.append(0)
            euclidean.append(0)
            cosine.append(0)
            model_count.append(0)
            continue

        accuracy.append(get_loss(X, y, beta0, betas, loss_type="accuracy"))
        logistic.append(get_loss(X, y, beta0, betas, loss_type="logistic", l2=l2))
        hamming.append(average_pairwise_diversity(predictions.T, hamming_distance, limit=100))

        iou.append(average_pairwise_diversity(betas, inverse_IoU, limit=100))
        euclidean.append(average_pairwise_diversity(betas, euclidean_distance, limit=100))
        cosine.append(average_pairwise_diversity(betas, inverse_cosine_similarity, limit=100))
        model_count.append(betas.shape[0])

    df['accuracy'] = accuracy
    df['logistic'] = logistic
    df['hamming'] = hamming
    df['iou'] = iou
    df['euclidean'] = euclidean
    df['cosine'] = cosine
    df['model_count'] = model_count

for dname, settings in dataset_settings[:1]:
    l0 = settings["l0"]
    l2 = settings["l2"]
    m = settings["m"]
    ne = settings["num_estimators"]
    binned = True

    filepath = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"
    with open(filepath, "rb") as f:
        sparse_gam = pickle.load(f)
    print(dname, sparse_gam['rset_bound'])

dataset_results = all_datasets_results['bank']
X = sparse_gam['X']
y = sparse_gam['y']
sample_p = sparse_gam['sample_proportion']
opt_betas = dataset_results['opt_betas'].iloc[0]
opt_beta0 = dataset_results['opt_beta0'].iloc[0]
l2 = dataset_results['l2']

