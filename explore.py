from src.run_app import *
from src.prepare_gam import *
from src.utils import *
import pickle as pkl
from FasterRisk.src.fasterrisk import fasterrisk

import re
from collections import defaultdict
from tqdm import tqdm

# load the dataset
dataset_name = 'compas'
path = 'datasets/{}.csv'.format(dataset_name)
dataset = pd.read_csv(path)
dataset.iloc[:, -1][dataset.iloc[:, -1] == 0] = -1
X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

X_one_hot, count = one_hot_encoding(X, one_hot=True)
y = pd.DataFrame(y)  # {0,1}

header = list(X_one_hot.columns)
header = pd.Index(["intercept"] + header)
header = header.astype("object")

# add a column of one to X_orig and make y in {1,-1}
X_one_hot, y = utils.get_X_y(X_one_hot, y)
y = y[:, -1]

# using extended get sparse diverse pool
rs = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=10, lb=-100, ub=100, gap_tolerance=0.006, select_top_m=100)
rs.optimize(generate_non_integer_solution=True, test=True, swaps=3)
beta0, betas = rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas

mean_loss = 0
for i in range(len(betas)):
    wi = betas[i, :]
    intercepti = beta0[i]
    logit = X_one_hot @ wi + intercepti
    y_pred = np.exp(logit) / (1 + np.exp(logit))
    y_pred = np.where(y_pred > 0.5, 1, -1)
    mean_loss += (y != y_pred).mean()
    print(np.nonzero(wi)[0])
    print((y != y_pred).mean())

# Input data
def get_feature_thresholds(weights, columns):
    # Extract features and thresholds
    feature_thresholds = {}
    for col, weight in zip(columns, weights):
        match = re.search(r'([a-zA-Z]+)', col)
        if match:
            feature = match.group(1)
            # threshold = col
            threshold = re.findall(r'[\d.]+', col)
            if feature == 'juv':
                feature = 'juv_misd_count'
            if feature == 'juvenile':
                feature = 'juvenile_crimes'
            if feature not in feature_thresholds:
                feature_thresholds[feature] = []
            # if feature == 'current':
            #     print(threshold,list(map(float, threshold)), weight)
            feature_thresholds[feature].append((list(map(float, threshold)), weight))

    return feature_thresholds

def plot_gam(X, X_one_hot, header, list_of_weights):
    # Create subplots
    num_features = X.shape[1]
    # fig, axes = plt.subplots(num_features-2, 1, figsize=(10, 3 * num_features), sharex=False)
    fig = plt.figure(figsize=(13, 3))
    fig.set_dpi(200)
    ax_dict = defaultdict(int)
    if num_features == 1:
        axes = [axes]  # Ensure axes is iterable for a single feature
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
                ax = fig.add_axes([0.05, 0.1+counter, 0.2, 0.8])
                ax_dict[feature] = ax
                counter += 1
            else:
                ax = ax_dict[feature]
            thresholds, feature_weights = zip(*thresholds_weights)

            x = range(len(thresholds))  # X-axis positions for thresholds
            x_vals, y_vals = [], []
            x_vals.append(0)
            y_vals.append(feature_weights[0])
            flag = False
            for i in range(len(thresholds) - 1):
                x_vals.append(thresholds[i][0])
                y_vals.append(feature_weights[i])
            if flag:
                continue

            # For the last value of the interval, add it once more
            x_vals.append(thresholds[-1][0])
            y_vals.append(feature_weights[-1])

            ax.step(x_vals, y_vals, where="post", color='r', alpha=0.1)
            ax.set_ylabel("Predicted Logit")
            ax.set_title(feature)
    plt.tight_layout()
    plt.show()
    return union_of_support_sets

union_of_support_sets = plot_gam(X, X_one_hot, header, betas[:100, :])

# Given dictionary
sorted_data = dict(sorted(union_of_support_sets.items(), key=lambda x: x[1], reverse=True))

# Prepare data for histogram
keys = list(sorted_data.keys())
values = list(sorted_data.values())

# Plot histogram
plt.figure(figsize=(15, 6))
plt.bar(keys, values, color='blue', alpha=0.7)
plt.xticks(rotation=90, fontsize=8)
plt.ylabel('Count')
plt.title('Count of Features across different Support Sets', fontsize=20)
plt.tight_layout()
plt.show()

# original: [  4   5   6   7   8  89 103 104 105 106]

# [  4   5   6  12  41  89 103 104 105 141]
# 0.33386419574344867
# [  4   5   6  12  41  89 103 104 105 140]
# 0.33386419574344867
# [  4   5   6   7   9  12  90 103 104 105]
# 0.33111336325466917
# [  2   4   5   6  41  89 103 104 105 141]
# 0.3327059504850152
# [  2   4   5   6  41  89 103 104 105 140]
# 0.3327059504850152
# [  2   4   5   6  12  89 103 104 105 141]
# 0.3345880990299696
# [  2   4   5   6  12  89 103 104 105 140]
# 0.3345880990299696
# [  2   4   5   6  12  41  89 103 104 105]
# 0.33386419574344867
# [  1   4   5   6  41  89 103 104 105 141]
# 0.3327059504850152
# [  1   4   5   6  41  89 103 104 105 140]
# 0.3327059504850152
# [  1   4   5   6  12  89 103 104 105 141]
# 0.3345880990299696
# [  1   4   5   6  12  89 103 104 105 140]
# 0.3345880990299696
# [  1   4   5   6  12  41  89 103 104 105]
# 0.33386419574344867

# # using the original get sparse diverse pool
# rs_orig = fasterrisk.RiskScoreOptimizer(X_one_hot, y, k=10, lb=-100, ub=100, gap_tolerance=0.002, select_top_m=10)
# rs_orig.optimize(generate_non_integer_solution=True, test=False)
# beta0_orig, betas_orig = rs_orig.sparseDiversePool_beta0, rs_orig.sparseDiversePool_betas

# mean_loss = 0
# for i in range(len(betas)):
#     wi = betas[i, :]
#     intercepti = beta0[i]
#     logit = X_one_hot @ wi + intercepti
#     y_pred = np.exp(logit) / (1 + np.exp(logit))
#     y_pred = np.where(y_pred > 0.5, 1, -1)
#     mean_loss += (y != y_pred).mean()
#     print((y != y_pred).mean())

# # check if the two solutions are the same
# assert np.allclose(beta0, beta0_orig), "beta0 is not equal to beta0_orig"
# assert np.allclose(betas, betas_orig), "betas is not equal to betas_orig"