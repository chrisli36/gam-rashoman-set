import numpy as np
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

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

def plot_gam(header, list_of_weights):
    rows = 3; cols = 4
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
    axs = axs.flatten()
    ax_dict = {}
    counter = 0
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
            if feature not in ax_dict:
                ax = axs[counter]
                ax_dict[feature] = ax
                counter += 1
            else:
                ax = ax_dict[feature]
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

            ax.step(x_vals, y_vals, where="post", color='r', alpha=0.1)
            ax.set_ylabel("Predicted Logit")
            ax.set_title(feature)
    plt.tight_layout()
    plt.show()
    return union_of_support_sets

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