import pandas as pd
import numpy as np
from gam_rs_utils.compute_thresholds import compute_thresholds, cut


def binarize_dataset(dataset, num_estimators, random_selection = False, max_thresholds_per_feat = 10, thresholds=None, header=None, csv_path=None):
    """Binarize dataset using GOSDT thresholds"""
    X, Y = pd.DataFrame(dataset.values[:, :-1], columns = dataset.columns[:-1]), pd.DataFrame(dataset.values[:, -1],columns = [dataset.columns[-1]])
    if thresholds is None:
        X_binary, thresholds, header, threshold_guess_time = compute_thresholds(
            X, Y, n_est = num_estimators, max_depth = 1, random_selection = random_selection, max_thresholds_per_feat = max_thresholds_per_feat)
        X_binary = X_binary[header]
        if csv_path is not None:
            pd.concat([X_binary, Y], axis=1).to_csv(csv_path, index=False)
        return pd.concat([X_binary, Y], axis=1), thresholds, header, threshold_guess_time
    else:
        # Both header and thresholds must be provided
        X_binary = cut(X.copy(), thresholds)
        X_binary = X_binary[header]
        if csv_path is not None:
            pd.concat([X_binary, Y], axis=1).to_csv(csv_path, index=False)
        return pd.concat([X_binary, Y], axis=1)
