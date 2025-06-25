import numpy as np
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import math
import pandas as pd

# dataset manipulation
def convert_cumulative_to_binned(X, header):
    # assumes that header is a list of strings with format "feature<=threshold"
    feature_to_thresholds = defaultdict(list)
    for h in header:
        feat, thres = h.split("<=")
        feature_to_thresholds[feat].append(float(thres))

    new_header = []
    for feat, thresholds in feature_to_thresholds.items():
        new_header.append(f"{feat}<={thresholds[0]}")
        for i in range(len(thresholds) - 1):
            new_header.append(f"{thresholds[i]}<{feat}<={thresholds[i + 1]}")

    new_X = np.zeros(X.shape)
    column_idx = 0
    for _, thresholds in feature_to_thresholds.items():
        prev = np.zeros(X.shape[0])
        for i in range(len(thresholds)):
            new_X[:, column_idx] = X[:, column_idx] - prev
            prev = X[:, column_idx]
            column_idx += 1

    return new_X, new_header

def get_y(dname):
    data = pd.read_csv("datasets/{}.csv".format(dname))
    y = data.iloc[:, -1].values
    y_max, y_min = np.max(y), np.min(y)
    y = -1 + 2 * (y-y_min)/(y_max-y_min)
    return y

# [(1, 2), (4, 5), (8, 9)]
# ["a", "b", "c", "d", "e", "f", "g"]
# ["a", "b", "b", "c", "d", "d", "e", "f", "g", "g"]

def get_true_w_sample(indices, w_sample):
    new_w_sample = []
    indices_ptr = 0
    w_ptr = 0
    while indices_ptr < len(indices):
        i, j = indices[indices_ptr]
        curr_len = len(new_w_sample)
        if curr_len >= i and curr_len <= j:
            new_w_sample.append(w_sample[w_ptr])
            if curr_len == j:
                indices_ptr += 1
                w_ptr += 1
        else:
            new_w_sample.append(w_sample[w_ptr])
            w_ptr += 1
    while w_ptr < len(w_sample):
        new_w_sample.append(w_sample[w_ptr])
        w_ptr += 1

    assert len(new_w_sample) == len(w_sample) + sum([j - i for i, j in indices])
    return np.array(new_w_sample)

def get_new_X(indices, X):
    new_X = []
    col_pointer = 0
    for i, j in indices:
        if i > col_pointer:
            new_X.append(X[:, col_pointer:i])
        merged = np.max(X[:, i:j+1], axis=1, keepdims=True)
        new_X.append(merged)
        col_pointer = j + 1
    if col_pointer < X.shape[1]:
        new_X.append(X[:, col_pointer:])
    new_X = np.hstack(new_X)
    return new_X

def get_loss(X_one_hot, y, beta0, betas, verbose=False):
    if len(beta0) == 0:
        return 0
    mean_loss = 0
    for i in range(len(betas)):
        wi = betas[i, :]
        intercepti = beta0[i]
        logit = X_one_hot @ wi + intercepti
        y_pred = np.exp(logit) / (1 + np.exp(logit))
        y_pred = np.where(y_pred > 0.5, 1, -1)
        mean_loss += (y != y_pred).mean()
        if verbose:
            print(np.nonzero(wi)[0], (y != y_pred).mean())
    return mean_loss / len(betas)

def get_predictions(X_one_hot, beta0, betas):
    if len(beta0) == 0:
        return None
    y_preds = np.zeros((X_one_hot.shape[0], len(betas)))
    for i in range(len(betas)):
        wi = betas[i, :]
        intercepti = beta0[i]
        logit = X_one_hot @ wi + intercepti
        y_pred = np.exp(logit) / (1 + np.exp(logit))
        y_pred = np.where(y_pred > 0.5, 1, -1)
        y_preds[:, i] = y_pred
    return y_preds

def get_feature_thresholds(weights, columns):
    feature_thresholds = defaultdict(list)
    for col, weight in zip(columns, weights):
        match = re.search(r'([a-zA-Z]+)', col)
        if match:
            feature = match.group(1)
            threshold = re.findall(r'[\d.]+', col)
            if feature == 'juv':
                feature = 'juv_misd_count'
            if feature == 'juvenile':
                feature = 'juvenile_crimes'
            feature_thresholds[feature].append((list(map(float, threshold)), weight))
    return feature_thresholds

def count_support_sets(header, list_of_weights):
    union_of_support_sets = defaultdict(int)
    for i in range(len(list_of_weights)):
        weights = list_of_weights[i, :]
        columns = header[np.nonzero(weights)[0]]
        weights = weights[np.nonzero(weights)[0]]
        for column in columns:
            union_of_support_sets[column] += 1
    return union_of_support_sets

# plotting utilities
def plot_distribution(losses, opt_loss):
    plt.hist(losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(opt_loss, color='red', linestyle='dashed', linewidth=2, label=f'Optimal Loss = {opt_loss:.4f}')
    plt.xlabel('Loss')
    plt.ylabel('Number of Models')
    plt.title('Loss Distribution in Rashomon Set')
    plt.legend()
    plt.xlim(min(losses), max(losses))
    plt.grid(True)
    plt.show()

def plot_two_var(results, x, y):
    fig, ax = plt.subplots()
    for dataset_name, data_group in results.groupby("dataset"):
        xs = data_group[x]
        ys = data_group[y]
        ax.plot(xs, ys, label=dataset_name, marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    plt.show()

def plot_two_var_bar(results, x, y):
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

def plot_two_var_bar_2(results, x, y):
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

def get_variable_importance(X, betas, header, bins):
    feature_to_vi = defaultdict(list)
    for b in betas:
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

def plot_variable_importance(feature_to_vi):
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

def plot_gam(header, list_of_weights):
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

# diversity functions
def average_pairwise_diversity(betas, diversity_metric, limit, X=None):
    if len(betas) < 2:
        return 0.0
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
        all_diversities.append(sum(diversity) / len(diversity))
    return sum(all_diversities) / len(all_diversities)

def hamming_distance(pred_1, pred_2):
    return np.sum(pred_1 != pred_2)

def inverse_IoU(betas_1, betas_2):
    indices_1 = betas_1.nonzero()[0]
    indices_2 = betas_2.nonzero()[0]

    intersection = len(set(indices_1).intersection(set(indices_2)))
    union = len(set(indices_1).union(set(indices_2)))

    return 1 - intersection / union

def inverse_correlation(X, betas_1, betas_2):
    indices_1 = betas_1.nonzero()[0]
    indices_2 = betas_2.nonzero()[0]

    X_subset_1 = X[:, indices_1]
    X_subset_2 = X[:, indices_2]
        
    correlation = np.corrcoef(X_subset_1.T, X_subset_2.T)[0, 1]
    return 1 - correlation

def euclidean_distance(betas_1, betas_2):
    return np.linalg.norm(betas_1 - betas_2)

def inverse_cosine_similarity(betas_1, betas_2):
    dot_product = np.dot(betas_1, betas_2)
    norm_a = np.linalg.norm(betas_1)
    norm_b = np.linalg.norm(betas_2)
    if norm_a == 0 or norm_b == 0:
        return 0
    return 1 - dot_product / (norm_a * norm_b)