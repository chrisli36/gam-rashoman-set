import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import math
import pandas as pd
from typing import List, Tuple, Dict, Callable, Optional
from src.prepare_gam import *
from gam_rs_utils.compute_thresholds import compute_thresholds, cut
from method_scripts.results import Results
from method_scripts.results import MethodType

BLACK   = '\033[30m'
RED     = '\033[31m'
GREEN   = '\033[32m'
YELLOW  = '\033[33m'
BLUE    = '\033[34m'
MAGENTA = '\033[35m'
CYAN    = '\033[36m'
WHITE   = '\033[37m'
RESET   = '\033[0m'

METHODS = [MethodType.ELLIPSOID, MethodType.BLOCKING, MethodType.HYBRID, MethodType.QUADRATIC, MethodType.MCMC]
DATASET_NAMES = ["bank", "compas", "diabetes"] #, "spambase", "mimic2"]

dataset_settings = [
    ('bank', {
        "l0": [0.001],
        "l2": [0.001],
        "m": [1.05],
        "eps": [0.28],
        "r_min": [1],
        'ne': [50],
        'n_support_set': [20],
        'beta': [0.3],
    }),
    ('compas', {
        "l0": [0.001],
        "l2": [0.001],
        "m": [1.025],
        "eps": [0.61],
        "r_min": [0.1],
        'ne': [50],
        'n_support_set': [15],
        'beta': [0.5],
    }),
    ("diabetes", {
        "l0": [0.001],
        "l2": [0.001],
        "m": [1.02],
        "eps": [0.45],
        "r_min": [0.1],
        'ne': [200],
        'n_support_set': [45],
        'beta': [0.6],
    }),
    # ('spambase', {
    #     "l0": [0.001],
    #     "l2": [0.001],
    #     "m": [1.01],
    #     "r_min": [0.1],
    #     'ne': [50],
    #     'n_support_set': [25],
    #     # 'n_samples': [100],
    # }),
    # ('mimic2', {
    #     "l0": [0.0005],
    #     "l2": [0.001],
    #     "m": [1.012],
    #     "r_min": [0.1],
    #     'ne': [50],
    #     'n_support_set': [25],
    #     # 'n_samples': [100],
    # }),
]
# 'netherlands': {},

# [(1, 2), (4, 5), (8, 9)]
# ["a", "b", "c", "d", "e", "f", "g"]
# ["a", "b", "b", "c", "d", "d", "e", "f", "g", "g"]

class DatasetUtils:
    @staticmethod
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

    @staticmethod
    def get_feature_thresholds(weights: np.ndarray, columns: np.ndarray) -> Dict[str, List[Tuple[List[float], float]]]:
        """
        Extracts feature thresholds and weights from column names and weights.
        Args:
            weights: 1D numpy array of weights.
            columns: 1D numpy array of column names.
        Returns:
            Dictionary mapping feature names to list of (thresholds, weight) tuples.
        """
        feature_thresholds = defaultdict(list)
        for col, weight in zip(columns, weights):
            match = re.search(r'([a-zA-Z]+)', col)
            feature = match.group(1)
            threshold = re.findall(r'[\d.]+', col)
            feature_thresholds[feature].append((list(map(float, threshold)), weight))
        return feature_thresholds

    @staticmethod
    def get_feature_ranges(columns: np.ndarray) -> Dict[str, List[float]]:
        """
        Extracts feature ranges from column names.
        Args:
            columns: 1D numpy array of column names.
        Returns:
            Dictionary mapping feature names to list of thresholds.
        """
        feature_ranges = defaultdict(list)
        for col in columns:
            match = re.search(r'([a-zA-Z]+)', col)
            feature = match.group(1)
            threshold = re.findall(r'[\d.]+', col)
            feature_ranges[feature].append(list(map(float, threshold)))
        return feature_ranges

    @staticmethod
    def convert_cumulative_to_binned(X: np.ndarray, header: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Converts cumulative binary features to binned features.
        Args:
            X: 2D numpy array of cumulative binary features.
            header: List of feature names in the format "feature<=threshold".
        Returns:
            Tuple of (new_X, new_header) where new_X is the binned feature array and new_header is the updated header.
        """
        # assumes that header is a list of strings with format "feature<=threshold"
        # build a dictionary of features to thresholds
        # e.g. {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}
        feature_to_thresholds = defaultdict(list)
        for h in header:
            if h == "intercept":
                feature_to_thresholds[h].append(0.0)
                continue
            feat, thres = h.split("<=")
            feature_to_thresholds[feat].append(float(thres))

        # build new header list
        # e.g. ["intercept", "f1<=1.0", "1.0<f1<=2.0"...]
        new_header = []
        for feat, thresholds in feature_to_thresholds.items():
            if feat == "intercept":
                new_header.append(feat)
                continue
            new_header.append(f"{feat}<={thresholds[0]}")
            for i in range(len(thresholds) - 1):
                new_header.append(f"{thresholds[i]}<{feat}<={thresholds[i + 1]}")

        # build new binned X
        new_X = np.zeros(X.shape)
        column_idx = 0
        for _, thresholds in feature_to_thresholds.items():
            prev = np.zeros(X.shape[0])
            for _ in range(len(thresholds)):
                new_X[:, column_idx] = X[:, column_idx] - prev
                prev = X[:, column_idx]
                column_idx += 1

        return new_X, new_header

    @staticmethod
    def get_binned_dataset(path: str, num_estimators: int) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Loads a dataset, binarizes it, and converts to binned format.
        Args:
            path: Path to the CSV file.
            num_estimators: Number of estimators for binarization.
        Returns:
            Tuple of (X_new, y, header, header_new, sample_p).
        """
        data = pd.read_csv(path)
        df, _, header, _ = DatasetUtils.binarize_dataset(data, num_estimators)
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        X_new, header_new = DatasetUtils.convert_cumulative_to_binned(X, header)
        header_new = ["intercept"] + header_new
        X_new, y = utils.get_X_y(X_new, y, is_df=False)
        return X_new, y, header, header_new

class ModelUtils:
    """Class containing methods for model evaluation and processing utilities."""
    @staticmethod
    def convert_cumulative_to_binned(w_rset, header):
        """
        Converts a set of cumulative weights to binned weights.
        Args:
            w_rset: 2D numpy array of cumulative weights.
            header: List of feature names in the format "feature<=threshold".
        Returns:
            w_rset_binned: 2D numpy array of binned weights.
        """
        w_rset_binned = []
        header_object = ModelUtils.get_header_object(header)
        for w in w_rset:
            new_w = np.zeros(len(header))
            i = w.shape[0] - 1
            for feat, thresholds in reversed(header_object.items()):
                if feat == 'intercept':
                    new_w[i] = w[i]
                    continue
                cumulative_weight = 0.0
                for _ in range(len(thresholds)):
                    cumulative_weight += w[i]
                    new_w[i] = cumulative_weight
                    i -= 1
            w_rset_binned.append(np.array(new_w))
        return np.vstack(w_rset_binned)

    @staticmethod
    def get_loss_one_model(X: np.ndarray, y: np.ndarray, w: np.ndarray, sample_p: Optional[np.ndarray] = None, loss_type: str = "accuracy", l2: Optional[float] = None) -> float:
        """
        Computes the loss for a single model.
        Args:
            X_one_hot: 2D numpy array of features.
            y: 1D numpy array of targets.
            w: 1D numpy array of weights.
            sample_p: Sample probabilities.
            loss_type: 'accuracy' or 'logistic'.
            l2: L2 regularization parameter (optional).
        Returns:
            Loss value as float.
        """
        logit = X @ w
        if loss_type == "accuracy":
            y_pred = np.exp(logit) / (1 + np.exp(logit))
            y_pred = np.where(y_pred > 0.5, 1, -1)
            loss = (y != y_pred).mean()
            return loss
        elif loss_type == "logistic":
            if sample_p is None:
                sample_p = Results.get_sample_proportion(X)
            loss = np.mean(np.log1p(np.exp(-y * logit))) + l2 * (sample_p[1:] * w[1:]**2).sum()
            return loss
        return

    @staticmethod
    def get_loss(X: np.ndarray, y: np.ndarray, w_rset: np.ndarray, loss_type: str = "accuracy", verbosity: int = 0, w_opt: Optional[np.ndarray] = None, l2: Optional[float] = None) -> float:
        """
        Computes the losses of a set of models.
        Args:
            X: 2D numpy array of features.
            y: 1D numpy array of targets.
            w_rset: 2D numpy array of model weights.
            loss_type: 'accuracy' or 'logistic'.
            verbosity: Verbosity level.
            w_opt: Optional optimal weights for comparison.
            l2: L2 regularization parameter (optional).
        Returns:
            Tuple of (losses, opt_loss) where losses is a list of loss values and opt_loss is the loss of the optimal model.
        """
        sample_p = Results.get_sample_proportion(X)
        if len(w_rset) == 0:
            return 0, None
        losses = []
        for i in range(len(w_rset)):
            wi = w_rset[i, :]
            loss = ModelUtils.get_loss_one_model(X, y, wi, sample_p, loss_type, l2)
            losses.append(loss)
        opt_loss = None if w_opt is None else ModelUtils.get_loss_one_model(X, y, w_opt, sample_p, loss_type, l2)
        if verbosity > 0:
            print(f"Optimal model {loss_type} loss: {opt_loss}")
            print(f"Average {loss_type} loss: {np.mean(losses)}")
        if verbosity > 1:
            Plotter.plot_distribution(losses, opt_loss)
        return losses, opt_loss

    @staticmethod
    def get_logits(X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Computes logits for a set of models.
        Args:
            X: 2D numpy array of features.
            w: 2D numpy array of model weights.
        """
        return X @ w

    @staticmethod
    def get_predictions(X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Computes predictions for a set of models.
        Args:
            X: 2D numpy array of features.
            w: 2D numpy array of model weights.
        Returns:
            2D numpy array of predictions.
        """
        if len(w) == 0:
            return np.array([])
        y_preds = np.zeros((X.shape[0], len(w)))
        for i in range(len(w)):
            wi = w[i, :]
            logit = X @ wi
            y_pred = np.exp(logit) / (1 + np.exp(logit))
            y_pred = np.where(y_pred > 0.5, 1, -1)
            y_preds[:, i] = y_pred
        return y_preds

    @staticmethod
    def hard_threshold(x: np.ndarray, k2: int) -> np.ndarray:
        """
        Apply hard thresholding to keep only the largest k2 elements.

        Args:
            x: numpy array of weights
            k2: number of elements to keep

        Returns:
            numpy array of weights with the smallest k2 elements set to 0
        """
        x = x.copy()
        small_indices = np.argsort(np.abs(x))[:k2]
        x[small_indices] = 0
        return x

    @staticmethod
    def hard_threshold_samples(w_samples: np.ndarray, rset, n_support_set: int) -> np.ndarray:
        """
        Apply hard thresholding to a set of model samples.
        
        Args:
            w_samples: 2D numpy array of model weights
            rset: Rashomon set object with in_rset method
            n_support_set: Number of support set elements to keep
            
        Returns:
            numpy array of thresholded samples that are still in the Rashomon set
        """
        w_samples_zeroed = []
        n_support = w_samples.shape[1]
        k2 = n_support - 1 - n_support_set
        
        for i in range(w_samples.shape[0]):
            w_samp = w_samples[i, 1:]
            w_samp_zeroed = np.concatenate([np.array([w_samples[i, 0]]), ModelUtils.hard_threshold(w_samp, k2)])
            if rset.in_rset(w_samp_zeroed):
                w_samples_zeroed.append(w_samp_zeroed)
        
        print(f"\tout of {w_samples.shape[0]} samples, kept {len(w_samples_zeroed)} after hard thresholding")
        
        return np.array(w_samples_zeroed) if len(w_samples_zeroed) > 0 else np.array([])

    @staticmethod
    def print_results_summary(w_rset: np.ndarray, w_opt: np.ndarray, 
                            X: np.ndarray, y: np.ndarray, l2: float, runtime: float) -> None:
        """
        Print a summary of results for a dataset.
        
        Args:
            w_rset: Rashomon set models
            w_opt: Optimal model
            X: Feature matrix
            y: Target vector
            l2: L2 regularization parameter
            runtime: Runtime in seconds
        """
        print(f"{RED}SUMMARY{RESET}")
        print(f"\t{w_rset.shape[0]} solutions, {runtime:.2f} seconds")
        print(f"\tAverage logistic loss: {np.mean(ModelUtils.get_loss(X, y, w_rset, loss_type='logistic', l2=l2)[0])}")
        sample_p = Results.get_sample_proportion(X)
        print(f"\tOpt model logistic loss: {ModelUtils.get_loss_one_model(X, y, w_opt, loss_type='logistic', l2=l2, sample_p=sample_p)}")

    @staticmethod
    def get_header_object(header):
        """
        Converts a header to a header object.
        Args:
            header: a list of strings with format "feature<=threshold" or "threshold<feature<=threshold".
            e.g. [
                  "intercept", 
                  "f1<=1.0", "1.0<f1<=2.0", "2.0<f1<=3.0", 
                  "f2<=4.0", "4.0<f2<=5.0", "5.0<f2<=6.0",
                  ...
                 ]
        Returns:
            a dictionary with feature names as keys and lists of thresholds as values.
            e.g. {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}
        """
        header_object = defaultdict(list)
        header_object['intercept']
        for h in header[1:]:
            feature = re.search(r'([a-zA-Z_=]+)', h).group(1)
            threshold = [float(t) for t in re.findall(r'-?[\d.]+', h)][-1]
            header_object[feature].append(threshold)
        return header_object

    @staticmethod
    def expand_w(w, sparse_header, header):
        """
        Expands a binned sparse weight vector to a weight vector over all features.
        Args:
            w: Binned sparse weight vector.
            sparse_header: Sparse header object.
            header: Full header object.
        Returns:
            The binned weight vector with all features.
        """
        new_w = [w[0]]
        wi = 1
        for h, thresholds in header.items():
            if h == 'intercept':
                continue
            if h not in sparse_header:
                new_w.extend([0.0] * len(thresholds))
                continue
            sparse_thresholds = sparse_header[h]
            si = 0
            for t in thresholds:
                if t > sparse_thresholds[si]:
                    wi += 1
                    si += 1
                new_w.append(w[wi])
            wi += 1
        return np.array(new_w)

    @staticmethod
    def expand_w_samples(w_samples, sparse_header, header):
        """
        Expands a set of binned sparse weight vectors to a set of weight vectors over all features.
        Args:
            w_samples: Set of binned sparse weight vectors.
            sparse_header: Sparse header object.
            header: Full header object.
        Returns:
            The set of weight vectors with all features.
        """
        expanded_w_samples = []
        for w in w_samples:
            expanded_w_samples.append(ModelUtils.expand_w(w, sparse_header, header))
        return np.array(expanded_w_samples)

class Plotter:
    @staticmethod
    def plot_distribution(losses: List[float], opt_loss: Optional[float] = None) -> None:
        """
        Plots a histogram of model losses, optionally marking the optimal loss.
        Args:
            losses: List of loss values.
            opt_loss: Optional optimal loss value to mark.
        Returns:
            None
        """
        plt.hist(losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        avg_loss = np.mean(losses)
        plt.axvline(avg_loss, color='green', linestyle='solid', linewidth=2, label=f'Average Loss = {avg_loss:.4f}')
        if opt_loss is not None:
            plt.axvline(opt_loss, color='red', linestyle='dashed', linewidth=2, label=f'Optimal Loss = {opt_loss:.4f}')
        
        plt.xlabel('Loss')
        plt.ylabel('Number of Models')
        plt.title('Loss Distribution in Rashomon Set')
        plt.legend()
        
        losses = losses + ([] if opt_loss == None else [opt_loss])
        plt.xlim(min(losses), max(losses))
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_two_var(results: pd.DataFrame, x: str, y: str) -> None:
        """
        Plots a line plot for two variables grouped by dataset.
        Args:
            results: DataFrame containing results.
            x: Name of x-axis variable.
            y: Name of y-axis variable.
        Returns:
            None
        """
        fig, ax = plt.subplots()
        for dataset_name, data_group in results.groupby("dataset"):
            xs = data_group[x]
            ys = data_group[y]
            ax.plot(xs, ys, label=dataset_name, marker="o")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.show()

    @staticmethod
    def plot_two_var_bar(results: pd.DataFrame, x: str, y: str) -> None:
        """
        Plots a grouped bar chart for two variables by dataset.
        Args:
            results: DataFrame containing results.
            x: Name of x-axis variable.
            y: Name of y-axis variable.
        Returns:
            None
        """
        fig, ax = plt.subplots()
        x_vals = results[x].unique()
        datasets = results["dataset"].unique()
        width = 0.8 / len(datasets)  # Adjust width for grouped bars
        x_indices = range(len(x_vals))

        for i, dataset_name in enumerate(datasets):
            data_group = results[results["dataset"] == dataset_name]
            ys = [data_group[data_group[x] == val][y].values[0] if not data_group[data_group[x] == val].empty else 0 for val in x_vals]
            offset = (i - len(datasets)/2) * width + width/2
            ax.bar([xi + offset for xi in x_indices], ys, width=width, label=dataset_name)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_vals, rotation=45, ha='right')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(title="Dataset")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_two_var_bar_2(results: pd.DataFrame, x: str, y: str) -> None:
        """
        Plots multiple bar charts for two variables, one per dataset.
        Args:
            results: DataFrame containing results.
            x: Name of x-axis variable.
            y: Name of y-axis variable.
        Returns:
            None
        """
        datasets = results["dataset"].unique()
        num_datasets = len(datasets)

        n_cols = 3
        n_rows = (num_datasets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
        axes = axes.flatten()
        colors = matplotlib.colormaps["tab10"]

        for i, dataset_name in enumerate(datasets):
            ax = axes[i]
            data_group = results[results["dataset"] == dataset_name]
            x_vals = data_group[x].unique()
            ys = [data_group[data_group[x] == val][y].values[0] if not data_group[data_group[x] == val].empty else 0 for val in x_vals]
            ax.bar(x_vals, ys, color=colors(i))
            ax.set_title(f"{dataset_name}", fontsize=16)
            ax.set_xlabel(x, fontsize=16)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, ha='center', fontsize=15)
            ax.tick_params(axis='y', labelsize=15)

        for j in range(len(datasets), len(axes)):
            fig.delaxes(axes[j])

        for i in range(n_rows):
            axes[i * n_cols].set_ylabel(y, fontsize=16)
        fig.suptitle(f"{y} by {x}", fontsize=20)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_variable_importance(X: np.ndarray, betas: np.ndarray, header: np.ndarray, bins: bool) -> Dict[str, List[float]]:
        """
        Computes variable importance for each feature across models.
        Args:
            X: 2D numpy array of features.
            betas: 2D numpy array of model weights.
            header: 1D numpy array of feature names.
            bins: Whether to use bin counts or not.
        Returns:
            Dictionary mapping feature names to lists of importance values.
        """
        feature_to_vi = defaultdict(list)
        for b in betas:
            b = np.abs(b) / np.sum(np.abs(b))
            nonzero_indices = b.nonzero()[0]
            columns = header[nonzero_indices]
            weights = b[nonzero_indices]
            feature_thresholds = DatasetUtils.get_feature_thresholds(weights, columns)

            num_bins = 0
            for feature, threshold_weights in feature_thresholds.items():
                cumulative = 0
                variable_importance = 0
                for _, weight in threshold_weights:
                    idx = nonzero_indices[num_bins]
                    bin_count = sum(X[:, idx]) - cumulative if not bins else sum(X[:, idx])
                    variable_importance += bin_count * np.abs(weight) / len(X)

                    cumulative += bin_count
                    num_bins += 1
                feature_to_vi[feature].append(variable_importance)
        return feature_to_vi

    @staticmethod
    def plot_variable_importance(feature_to_vi: Dict[str, List[float]]) -> None:
        """
        Plots histograms of variable importance for each feature.
        Args:
            feature_to_vi: Dictionary mapping feature names to lists of importance values.
        Returns:
            None
        """
        num_features = len(feature_to_vi)
        cols = 3
        rows = math.ceil(num_features / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        axes = axes.flatten()

        for ax, (feature, values) in zip(axes, feature_to_vi.items()):
            ax.hist(values)
            ax.set_title(feature)

        # Turn off any unused axes
        for i in range(len(feature_to_vi), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_gam(header: np.ndarray, list_of_weights: np.ndarray) -> Dict[str, int]:
        """
        Plots GAM step functions for each feature and returns support set counts.
        Args:
            header: 1D numpy array of feature names.
            list_of_weights: 2D numpy array of model weights.
        Returns:
            Dictionary mapping feature names to support set counts.
        """
        feature_to_data = defaultdict(list)
        union_of_support_sets = defaultdict(int)
        for i in tqdm(range(len(list_of_weights))):
            weights = list_of_weights[i, :]
            columns = header[np.nonzero(weights)[0]]
            weights = weights[np.nonzero(weights)[0]]
            for column in columns:
                union_of_support_sets[column] += 1

            feature_thresholds = DatasetUtils.get_feature_thresholds(weights, columns)
            for feature, thresholds_weights in feature_thresholds.items():
                if feature == 'sex' or feature == 'current':
                    continue
                thresholds, feature_weights = zip(*thresholds_weights)

                x_vals, y_vals = [], []
                x_vals.append(0)
                y_vals.append(feature_weights[0])
                for i in range(len(thresholds) - 1):
                    x_vals.append(thresholds[i][0])
                    y_vals.append(feature_weights[i])

                # For the last value of the interval, add it once more
                x_vals.append(thresholds[-1][-1])
                y_vals.append(feature_weights[-1])

                feature_to_data[feature].append((x_vals, y_vals))

        cols = 3
        rows = math.ceil(len(feature_to_data) / cols)
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
        axs = axs.flatten()
        for ax, (feature, data) in zip(axs, feature_to_data.items()):
            for x_vals, y_vals in data:
                ax.step(x_vals, y_vals, where="post", color='r', alpha=0.1)
            ax.set_ylabel("Predicted Logit")
            ax.set_title(feature)

        plt.tight_layout()
        plt.show()
        return union_of_support_sets

class Metrics:
    @staticmethod
    def get_mean_and_ci(values: List[float], ci: float = 95) -> Tuple[float, Tuple[float, float]]:
        """
        Computes the mean and confidence interval of a list of values.
        Args:
            values: List of values.
            ci: Confidence level (default 95).
        Returns:
            Mean and confidence interval as tuple.
        """
        return np.mean(values), np.percentile(values, [(100-ci)/2, 100-(100-ci)/2])

    @staticmethod
    def average_pairwise_diversity(
            betas: np.ndarray,
            diversity_metric: Callable[..., float],
            limit: int,
            X: Optional[np.ndarray] = None,
            ci: float = 95
        ):
        """
        Computes the average pairwise diversity among a set of models using a given metric.
        
        Args:
            betas: 2D numpy array of model weights.
            diversity_metric: Function to compute diversity between two models.
            limit: Number of models to sample for diversity calculation.
            X: Optional feature matrix for metrics that require it.
            return_ci: If True, also return a confidence interval.
            ci: Confidence level (default 95).
        
        Returns:
            - mean diversity (float)
            - (optionally) confidence interval as (low, high)
        """
        if len(betas) < 2:
            return 0.0, (0.0, 0.0)

        num_samples = 1
        if len(betas) > limit:
            num_samples = 1 + int(len(betas) / limit)
        else:
            limit = len(betas)
        
        all_diversities = []
        for _ in range(num_samples):
            diversity = []
            sampled_idx = np.random.choice(len(betas), limit, replace=False)
            sampled_betas = betas[sampled_idx]
            for i in range(len(sampled_betas)):
                for j in range(i + 1, len(sampled_betas)):
                    if X is not None:
                        diversity.append(diversity_metric(X, sampled_betas[i], sampled_betas[j]))
                    else:
                        diversity.append(diversity_metric(sampled_betas[i], sampled_betas[j]))
            all_diversities.append(np.mean(diversity))
        
        return Metrics.get_mean_and_ci(all_diversities, ci)

    @staticmethod
    def compute_logit_variance(logits: np.ndarray, ci: float = 95) -> float:
        """
        Computes the variance of a set of logits.
        Args:
            logits: 2D numpy array of logits.
            ci: Confidence level (default 95).
        Returns:
            Tuple of (mean logit variance, confidence interval) as (float, (float, float)).
        """
        if len(logits) < 2:
            return 0.0, (0.0, 0.0)
        logits_var = np.var(logits, axis=0)
        return Metrics.get_mean_and_ci(logits_var, ci)

    @staticmethod
    def compute_logit_pairwise_distance(logits: np.ndarray, ci: float = 95) -> float:
        """
        Computes the distance between a set of logits.
        Args:
            logits: 2D numpy array of logits.
            ci: Confidence level (default 95).
        Returns:
            Tuple of (mean logit distance, confidence interval) as (float, (float, float)).
        """
        if len(logits) < 2:
            return 0.0, (0.0, 0.0)
        from scipy.spatial.distance import pdist
        pair_dists = pdist(logits, metric="euclidean")
        return Metrics.get_mean_and_ci(pair_dists, ci)

    @staticmethod
    def hamming_distance(pred_1: np.ndarray, pred_2: np.ndarray) -> int:
        """
        Computes the Hamming distance between two prediction arrays.
        Args:
            pred_1: 1D numpy array of predictions.
            pred_2: 1D numpy array of predictions.
        Returns:
            Integer Hamming distance.
        """
        return np.sum(pred_1 != pred_2)

    @staticmethod
    def inverse_IoU(betas_1: np.ndarray, betas_2: np.ndarray) -> float:
        """
        Computes the inverse Intersection over Union (IoU) between two weight vectors.
        Args:
            betas_1: 1D numpy array of weights.
            betas_2: 1D numpy array of weights.
        Returns:
            1 - IoU as float.
        """
        indices_1 = betas_1.nonzero()[0]
        indices_2 = betas_2.nonzero()[0]

        intersection = len(set(indices_1).intersection(set(indices_2)))
        union = len(set(indices_1).union(set(indices_2)))

        return 1 - intersection / union

    @staticmethod
    def inverse_correlation(X: np.ndarray, betas_1: np.ndarray, betas_2: np.ndarray) -> float:
        """
        Computes the inverse correlation between two sets of selected features.
        Args:
            X: 2D numpy array of features.
            betas_1: 1D numpy array of weights.
            betas_2: 1D numpy array of weights.
        Returns:
            1 - correlation as float.
        """
        indices_1 = betas_1.nonzero()[0]
        indices_2 = betas_2.nonzero()[0]

        X_subset_1 = X[:, indices_1]
        X_subset_2 = X[:, indices_2]
            
        correlation = np.corrcoef(X_subset_1.T, X_subset_2.T)[0, 1]
        return 1 - correlation

    @staticmethod
    def euclidean_distance(betas_1: np.ndarray, betas_2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two weight vectors.
        Args:
            betas_1: 1D numpy array of weights.
            betas_2: 1D numpy array of weights.
        Returns:
            Euclidean distance as float.
        """
        return np.linalg.norm(betas_1 - betas_2)

    @staticmethod
    def inverse_cosine_similarity(betas_1: np.ndarray, betas_2: np.ndarray) -> float:
        """
        Computes the inverse cosine similarity between two weight vectors.
        Args:
            betas_1: 1D numpy array of weights.
            betas_2: 1D numpy array of weights.
        Returns:
            1 - cosine similarity as float.
        """
        dot_product = np.dot(betas_1, betas_2)
        norm_a = np.linalg.norm(betas_1)
        norm_b = np.linalg.norm(betas_2)
        if norm_a == 0 or norm_b == 0:
            return 0
        return 1 - dot_product / (norm_a * norm_b)

    @staticmethod
    def shape_diversity(X: np.ndarray, betas_1: np.ndarray, betas_2: np.ndarray) -> float:
        """
        Computes the average absolute difference between two sets of shape functions.
        Args:
            X: 2D numpy array of features.
            betas_1: 1D numpy array of weights.
            betas_2: 1D numpy array of weights.
        Returns:
            Average absolute difference between two sets of shape functions as float.
        """
        return np.mean(np.abs(X @ betas_1 - X @ betas_2))

    @staticmethod
    def shape_difference(X: np.ndarray, betas_1: np.ndarray, betas_2: np.ndarray) -> float:
        """
        Computes the average absolute difference between two sets of shape functions.
        Args:
            X: 2D numpy array of features.
            betas_1: 1D numpy array of weights.
            betas_2: 1D numpy array of weights.
        Returns:
            difference between two sets of shape functions as float.
        """
        return X @ np.abs(betas_1 - betas_2)

    @staticmethod
    def prediction_difference(predictions_1: np.ndarray, predictions_2: np.ndarray) -> float:
        """
        Computes the average absolute difference between two sets of predictions.
        Args:
            predictions_1: 1D numpy array of predictions.
            predictions_2: 1D numpy array of predictions.
        Returns:
            Average absolute difference between two sets of predictions as float.
        """
        return np.mean(np.abs(predictions_1 - predictions_2))


class MultiPlotter:
    """
    A class for plotting multiple distribution plots, bar charts, or GAM shape functions stacked vertically with unified x-axis.
    """
    def __init__(self, figsize=(10, 12), title: Optional[str] = None):
        self.distributions = []
        self.bar_charts = []
        self.shape_functions = []
        self.variable_importance_distributions = []
        self.shape_diversities = []
        self.figsize = figsize
        self.title = title
        self.x_min = float('inf')
        self.x_max = float('-inf')
        self.x_labels = None
        self.feature_names = None
        self.row_y_limits = None
        self.column_x_limits = None

    def add_distribution(self, losses: List[float], method_name: str, opt_loss: Optional[float] = None):
        """
        Add a distribution to be plotted.
        Args:
            losses: List of loss values.
            method_name: Name of the method for the subplot title.
            opt_loss: Optional optimal loss value to mark.
        """
        # Update global x-axis bounds
        current_min, current_max = min(losses), max(losses)
        if opt_loss is not None:
            current_min = min(current_min, opt_loss)
            current_max = max(current_max, opt_loss)
        
        self.x_min = min(self.x_min, current_min)
        self.x_max = max(self.x_max, current_max)
        
        # Store distribution data
        self.distributions.append({
            'losses': losses,
            'method_name': method_name,
            'opt_loss': opt_loss
        })

    def add_bar(self, means_and_cis: List[Tuple[float, Tuple[float, float]]], method_name: str, x_labels: Optional[List] = None):
        """
        Add a bar chart to be plotted.
        Args:
            means_and_cis: List of means and confidence intervals for the bars.
            method_name: Name of the method for the subplot title.
            x_labels: List of x-axis labels. Should be the same for all charts.
        """
        # Set x_labels on first call, verify consistency on subsequent calls
        if self.x_labels is None:
            self.x_labels = x_labels
        elif x_labels is not None and x_labels != self.x_labels:
            raise ValueError("x_labels must be the same for all bar charts")
        
        # Store bar chart data
        self.bar_charts.append({
            'y_values': [means_and_cis[0] for means_and_cis in means_and_cis],
            'y_errors': [means_and_cis[1] for means_and_cis in means_and_cis],
            'method_name': method_name
        })

    def add_shape_function(self, header: np.ndarray, w_rset: np.ndarray, method_name: str):
        """
        Add GAM shape functions to be plotted, grouped by feature.
        Args:
            header: Array of feature names.
            w_rset: Array of model weights.
            method_name: Name of the method for the legend.
        """
        # Extract feature names and thresholds from header
        feature_data = defaultdict(list)
        for i in range(w_rset.shape[0]):
            weights = w_rset[i, :]
            columns = header[np.nonzero(weights)[0]]
            weights = weights[np.nonzero(weights)[0]]

            feature_thresholds = DatasetUtils.get_feature_thresholds(weights, columns)
            for feature, thresholds_weights in feature_thresholds.items():
                if feature == 'sex' or feature == 'current':
                    continue
                thresholds, feature_weights = zip(*thresholds_weights)

                x_vals, y_vals = [], []
                x_vals.append(0)
                y_vals.append(feature_weights[0])
                for i in range(len(thresholds) - 1):
                    x_vals.append(thresholds[i][0])
                    y_vals.append(feature_weights[i])

                # For the last value of the interval, add it once more
                x_vals.append(thresholds[-1][-1])
                y_vals.append(feature_weights[-1])

                feature_data[feature].append((x_vals, y_vals))
        
        # Set feature names on first call, verify consistency on subsequent calls
        if self.feature_names is None:
            self.feature_names = list(feature_data.keys())
        elif set(feature_data.keys()) != set(self.feature_names):
            # raise ValueError("Feature names must be the same for all shape function calls")
            for feature in set(self.feature_names) - set(feature_data.keys()):
                feature_data[feature]
        
        # Update y-limits for each feature
        if self.row_y_limits is None:
            self.row_y_limits = defaultdict(lambda: [float('inf'), float('-inf')])
        
        for feature in self.feature_names:
            for x_vals, y_vals in feature_data[feature]:
                self.row_y_limits[feature] = [
                    min(self.row_y_limits[feature][0], np.min(y_vals)), 
                    max(self.row_y_limits[feature][1], np.max(y_vals))
                ]
        
        # Store shape function data
        self.shape_functions.append({
            'feature_data': feature_data,
            'method_name': method_name
        })

    def add_variable_importance_distribution(self, feature_to_vi: Dict[str, List[float]], method_name: str):
        """
        Add variable importance distribution to be plotted.
        Args:
            feature_to_vi: Dictionary mapping feature names to lists of importance values.
            method_name: Name of the method for the subplot title.
        """
        # Set feature names on first call, verify consistency on subsequent calls
        if self.feature_names is None:
            self.feature_names = list(feature_to_vi.keys())
        # elif set(feature_to_vi.keys()) != set(self.feature_names):
        #     raise ValueError("Feature names must be the same for all variable importance distributions")
        
        if self.row_y_limits is None:
            self.row_y_limits = defaultdict(lambda: [float('inf'), float('-inf')])
        if self.column_x_limits is None:
            self.column_x_limits = defaultdict(lambda: [float('inf'), float('-inf')])
        
        for feature in self.feature_names:
            for vi in feature_to_vi[feature]:
                self.column_x_limits[feature] = [
                    min(self.column_x_limits[feature][0], np.min(vi)),
                    max(self.column_x_limits[feature][1], np.max(vi))
                ]

        self.variable_importance_distributions.append({
            'feature_to_vi': feature_to_vi,
            'method_name': method_name
        })
    
    def add_model_reliance(self, feature_to_vi: Dict[str, List[float]], method_name: str):
        """
        Add model reliance to be plotted.
        Args:
            feature_to_vi: Dictionary mapping method names to lists of reliance values.
            method_name: Name of the method for the subplot title.
        """
        if self.feature_names is None:
            self.feature_names = list(feature_to_vi.keys())
        elif set(feature_to_vi.keys()) != set(self.feature_names):
            raise ValueError("Feature names must be the same for all model reliance")
        
        self.variable_importance_distributions.append({
            'feature_to_vi': feature_to_vi,
            'method_name': method_name
        })

    def add_shape_diversity(self, feature_to_diversity: Dict[str, List[float]], method_name: str):
        """
        Add shape diversity to be plotted.
        Args:
            feature_to_diversity: Dictionary mapping feature names to lists of diversity values.
            method_name: Name of the method for the subplot title.
        """
        if self.feature_names is None:
            self.feature_names = list(feature_to_diversity.keys())
        elif set(feature_to_diversity.keys()) != set(self.feature_names):
            raise ValueError("Feature names must be the same for all shape diversity")
        
        self.shape_diversities.append({
            'feature_to_diversity': feature_to_diversity,
            'method_name': method_name
        })

    def plot_distributions(self):
        """
        Plot distributions stacked vertically with unified x-axis.
        """

        if not self.distributions:
            print("No distributions to plot. Use add_distribution() first.")
            return
        
        n_plots = len(self.distributions)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)
            
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
            
        # Set overall title if provided
        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        for i, dist_data in enumerate(self.distributions):
            ax = axes[i]
            losses = dist_data['losses']
            method_name = dist_data['method_name']
            opt_loss = dist_data['opt_loss']

            # Plot histogram
            ax.hist(losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

            # Add average loss line
            avg_loss = np.mean(losses)
            ax.axvline(avg_loss, color='green', linestyle='solid', linewidth=2, 
                        label=f'Average Loss = {avg_loss:.4f}')
            
            # Add optimal loss line if provided
            if opt_loss is not None:
                ax.axvline(opt_loss, color='red', linestyle='dashed', linewidth=2, 
                            label=f'Optimal Loss = {opt_loss:.4f}')
                
            ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Models')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Set x-axis limits to be consistent across all plots
            ax.set_xlim(self.x_min, self.x_max)

        # Set x-axis label for the bottom subplot only
        axes[-1].set_xlabel('Loss')

        plt.tight_layout()
        return fig

    def plot_bar_charts(self):
        """
        Plot bar charts stacked vertically with unified x-axis.
        """

        if not self.bar_charts:
            print("No bar charts to plot. Use add_bar() first.")
            return
        
        n_plots = len(self.bar_charts)
        fig, axes = plt.subplots(n_plots, 1, figsize=self.figsize, sharex=True)
            
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
            
        # Set overall title if provided
        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')
            
        x_indices = range(len(self.x_labels))
        # Create colors for each bar position (shared across subplots)
        n_bars = len(self.x_labels)
        bar_colors = plt.cm.Set3(np.linspace(0, 1, n_bars))
            
        for i, chart_data in enumerate(self.bar_charts):
            ax = axes[i]
            y_values = chart_data['y_values']
            y_errors = chart_data['y_errors']
            method_name = chart_data['method_name']
            
            # Plot bar chart with different colors for each bar
            yerr = np.array([[m - low, high - m] for m, (low, high) in zip(y_values, y_errors)]).T
            bars = ax.bar(x_indices, y_values, yerr=yerr, alpha=0.7, edgecolor='black')
            
            # Set different colors for each bar
            for bar, color in zip(bars, bar_colors):
                bar.set_color(color)
            
            # Set x-axis labels and formatting
            ax.set_xticks(x_indices)
            ax.set_xticklabels(self.x_labels, rotation=45, ha='right')
                
            ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
        # Set x-axis label for the bottom subplot only
        axes[-1].set_xlabel('X Values')

        plt.tight_layout()
        return fig

    def plot_shape_functions(self):
        """
        Plot GAM shape functions stacked vertically with unified x-axis.
        """

        if not self.shape_functions:
            print("No shape functions to plot. Use add_shape_function() first.")
            return

        n_features = len(self.feature_names)
        n_methods = len(self.shape_functions)
        fig, axes = plt.subplots(n_features, n_methods, figsize=(6 * n_methods, 4 * n_features), sharex=False)

        # Ensure axes is 2D for consistent indexing
        if n_features == 1:
            axes = np.array([axes])

        # Set overall title if provided
        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Colors for different methods
        colors = plt.cm.Set3(np.linspace(0, 1, n_methods))

        for feature_idx, feature in enumerate(self.feature_names):
            for method_idx, shape_data in enumerate(self.shape_functions):
                ax = axes[feature_idx, method_idx]

                feature_data = shape_data['feature_data']
                method_name = shape_data['method_name']

                for x_vals, y_vals in feature_data[feature]:
                    ax.step(
                        x_vals,
                        y_vals,
                        where="post",
                        color=colors[method_idx],
                        alpha=0.8,
                        linewidth=2,
                    )

                # Apply consistent y-lims per row
                y_min_row, y_max_row = self.row_y_limits[feature]
                ax.set_ylim(y_min_row, y_max_row)
                ax.grid(True, alpha=0.3)

                # Bottom row: put method names as x-axis labels
                if feature_idx == n_features - 1:
                    ax.set_xlabel(self.shape_functions[method_idx]['method_name'])

            # Left label for the row with feature name
            left_ax = axes[feature_idx, 0]
            left_ax.set_ylabel(
                f"{feature}", rotation=45, labelpad=35, va='center', fontsize=12, ha='right'
            )

        plt.tight_layout()
        return fig

    def plot_variable_importance_distributions(self):
        """
        Plot variable importance distributions stacked vertically with unified x-axis.
        """

        if not self.variable_importance_distributions:
            print("No variable importance distributions to plot. Use add_variable_importance_distribution() first.")
            return

        n_rows = len(self.variable_importance_distributions)
        n_cols = len(self.feature_names)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(8 * n_cols, 3 * n_rows),
            sharex='col'
        )

        # Ensure axes is 2D for consistent indexing
        if n_rows == 1:
            axes = np.array([axes])

        # Set overall title if provided
        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Colors for different methods
        colors = plt.cm.Set3(np.linspace(0, 1, n_rows))

        for i, dist_data in enumerate(self.variable_importance_distributions):
            feature_to_vi = dist_data['feature_to_vi']
            method_name = dist_data['method_name']
            for j, feature_name in enumerate(self.feature_names):
                ax = axes[i, j]
                ax.hist(feature_to_vi[feature_name], bins=30, alpha=0.7, color=colors[i], edgecolor='black')
                x_min, x_max = self.column_x_limits[feature_name]
                span = x_max - x_min
                if span == 0:
                    # Add small padding when all values are identical
                    pad = max(1e-12, abs(x_min) * 0.05)
                else:
                    pad = 0.05 * span
                ax.set_xlim(x_min - pad, x_max + pad)
                ax.grid(True, alpha=0.3)

                if i == n_rows - 1:
                    ax.set_xlabel(f'Variable Importance ({feature_name})')

            left_ax = axes[i, 0]
            left_ax.set_ylabel(
                f'Number of Models ({method_name})', rotation=45, labelpad=35, va='center', fontsize=12, ha='right'
            )

        plt.tight_layout()
        return fig

    def plot_model_reliance_violin_plots(self):
        """
        Plot model reliance violin plots stacked vertically with unified x-axis.
        """
        if not self.variable_importance_distributions:
            print("No model reliance to plot. Use add_model_reliance() first.")
            return
        
        n_rows = len(self.feature_names)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)

        if n_rows == 1:
            axes = [axes]

        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Get method labels once (they should be the same for all features)
        method_labels = [dist_data['method_name'] for dist_data in self.variable_importance_distributions]
        
        for row, feature in enumerate(self.feature_names):
            method_data = []
            for dist_data in self.variable_importance_distributions:
                feature_to_vi = dist_data['feature_to_vi']
                method_data.append(feature_to_vi[feature])

            ax = axes[row]
            # Only set labels on the last subplot to avoid tick location conflicts
            labels_to_use = method_labels if row == n_rows - 1 else None
            violin_parts = ax.violinplot(method_data, positions=range(1, len(method_data) + 1), showmeans=True, showmedians=True)

            colors = plt.cm.Set3(np.linspace(0, 1, len(method_data)))
            for i, (patch, color) in enumerate(zip(violin_parts['bodies'], colors)):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Set x-axis labels
            if labels_to_use:
                ax.set_xticks(range(1, len(method_data) + 1))
                ax.set_xticklabels(labels_to_use)

            ax.set_title(f'{feature}')
            ax.set_ylabel('Variable Importance')
            ax.grid(True, alpha=0.3)
            
            # Hide x-axis labels for all subplots except the last one
            if row < n_rows - 1:
                ax.set_xticklabels([])

        # Set x-axis label only on the last subplot
        axes[-1].set_xlabel('Method')

        plt.tight_layout()
        return fig

    def plot_shape_diversity(self):
        """
        Plot shape diversity box plots stacked vertically with unified x-axis.
        """
        if not self.shape_diversities:
            print("No shape diversities to plot. Use add_shape_diversity() first.")
            return
        
        n_rows = len(self.feature_names)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)

        if n_rows == 1:
            axes = [axes]

        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Get method labels once (they should be the same for all features)
        method_labels = [dist_data['method_name'] for dist_data in self.shape_diversities]

        for row, feature in enumerate(self.feature_names):
            collated_diversities = []
            collated_errors = []
            for dist_data in self.shape_diversities:
                feature_to_diversity = dist_data['feature_to_diversity']
                collated_diversities.append(feature_to_diversity[feature][0][0])
                collated_errors.append(feature_to_diversity[feature][0][1])

            ax = axes[row]
            # Create bar positions
            x_positions = range(len(method_labels))
            yerr = np.array([[m - low, high - m] for m, (low, high) in zip(collated_diversities, collated_errors)]).T
            ax.bar(x_positions, collated_diversities, yerr=yerr)
            
            ax.set_title(f'{feature}')
            ax.set_ylabel('Shape Diversity')
            ax.grid(True, alpha=0.3)

            # Set x-axis labels only on the last subplot
            if row == n_rows - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(method_labels, rotation=45, ha='right')
                ax.set_xlabel('Method')
            else:
                # Hide x-axis labels for upper subplots
                ax.set_xticklabels([])

        plt.tight_layout()
        return fig