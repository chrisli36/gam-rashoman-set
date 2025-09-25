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
from typing import List, Tuple, Dict, Callable, Optional, Any, Union
from gam_rs_utils.binarize_dataset import binarize_dataset
from src.prepare_gam import *

BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'

dataset_settings = [
    ('bank', {
        "l0": 0.001,
        "l2": 0.001,
        "m": 1.05,
        "r_min": 1,
        'num_estimators': 50,
        'n_support_set': 20,
    }),
    ('compas', {
        "l0": 0.001,
        "l2": 0.001,
        "m": 1.025,
        "r_min": 0.1,
        'num_estimators': 50,
        'n_support_set': 15,
    }),
    ("diabetes", {
        "l0": 0.001,
        "l2": 0.001,
        "m": 1.02,
        "r_min": 0.1,
        'num_estimators': 200,
        'n_support_set': 45,
    }),
    ('spambase', {
        "l0": 0.001,
        "l2": 0.001,
        "m": 1.01,
        "r_min": 0.1,
        'num_estimators': 50,
        'n_support_set': 25,
    }),
    ('mimic2', {
        "l0": 0.0005,
        "l2": 0.001,
        "m": 1.012,
        "r_min": 0.1,
        'num_estimators': 50,
        'n_support_set': 25,
    }),
]
# 'netherlands': {},

# dataset manipulation
def get_y(dname: str) -> np.ndarray:
    """
    Loads the target variable y from a dataset and rescales it to [-1, 1].
    Args:
        dname: Dataset name (without .csv extension).
    Returns:
        1D numpy array of rescaled target values.
    """
    data = pd.read_csv("datasets/{}.csv".format(dname))
    y = data.iloc[:, -1].values
    y_max, y_min = np.max(y), np.min(y)
    y = -1 + 2 * (y-y_min)/(y_max-y_min)
    return y

# [(1, 2), (4, 5), (8, 9)]
# ["a", "b", "c", "d", "e", "f", "g"]
# ["a", "b", "b", "c", "d", "d", "e", "f", "g", "g"]

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
        if match:
            feature = match.group(1)
            threshold = re.findall(r'[\d.]+', col)
            # if feature == 'juv':
            #     feature = 'juv_misd_count'
            # if feature == 'juvenile':
            #     feature = 'juvenile_crimes'
            feature_thresholds[feature].append((list(map(float, threshold)), weight))
    return feature_thresholds

def get_feature_ranges(columns: np.ndarray) -> Dict[str, List[float]]:
    feature_ranges = defaultdict(list)
    for col in columns:
        match = re.search(r'([a-zA-Z]+)', col)
        if match:
            feature = match.group(1)
            threshold = re.findall(r'[\d.]+', col)
            feature_ranges[feature].append(list(map(float, threshold)))
    return feature_ranges

def count_support_sets(header: np.ndarray, list_of_weights: np.ndarray) -> Dict[str, int]:
    """
    Counts the number of times each feature appears in the support sets of models.
    Args:
        header: 1D numpy array of feature names.
        list_of_weights: 2D numpy array of model weights.
    Returns:
        Dictionary mapping feature names to counts.
    """
    union_of_support_sets = defaultdict(int)
    for i in range(len(list_of_weights)):
        weights = list_of_weights[i, :]
        columns = header[np.nonzero(weights)[0]]
        weights = weights[np.nonzero(weights)[0]]
        for column in columns:
            union_of_support_sets[column] += 1
    return union_of_support_sets

# plotting utilities
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

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalizes weights to have positive magnitude, and sum to 1.
    """
    abs_weights = np.abs(weights)
    return abs_weights / np.sum(abs_weights)

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
        b = normalize_weights(b)
        nonzero_indices = b.nonzero()[0]
        columns = header[nonzero_indices]
        weights = b[nonzero_indices]
        feature_thresholds = get_feature_thresholds(weights, columns)

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

        feature_thresholds = get_feature_thresholds(weights, columns)
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
    
    return get_mean_and_ci(all_diversities, ci)

def compute_predictive_diversity(logits: np.ndarray, ci: float = 95) -> float:
    """
    Computes the predictive diversity of a set of logits.
    Args:
        logits: 2D numpy array of logits.
        ci: Confidence level (default 95).
    Returns:
        Predictive diversity as float.
    """
    
    logits_var = np.var(logits, axis=0)
    return get_mean_and_ci(logits_var, ci)

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

def shape_diversity(X: np.ndarray, betas_1: np.ndarray, betas_2: np.ndarray) -> float:
    return np.mean(np.abs(X @ betas_1 - X @ betas_2))

def shape_difference(X: np.ndarray, betas_1: np.ndarray, betas_2: np.ndarray) -> float:
    return X @ np.abs(betas_1 - betas_2)

def prediction_diversity(predictions_1: np.ndarray, predictions_2: np.ndarray) -> float:
    return np.mean(np.abs(predictions_1 - predictions_2))