import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gam_rs_utils.utils import *
from method_scripts.results import Results

SUPPORT_SET_METRICS = [Metrics.inverse_IoU, Metrics.euclidean_distance, Metrics.inverse_cosine_similarity]
LOSS_METRICS = ['accuracy', 'logistic']
LOGIT_METRICS = [Metrics.compute_logit_variance, Metrics.compute_logit_pairwise_distance]
PREDICTION_METRICS = [Metrics.hamming_distance]

def collate_datasets_results():
    all_datasets_results = {}
    for dn in DATASET_NAMES:
        runs = list()
        runs_paths = os.listdir(f"results/{dn}/method_results")
        runs_paths.sort()
        for fn in runs_paths:
            result = Results.load_result(f"results/{dn}/method_results/{fn}")
            runs.append(result.to_dataframe_row())
        all_datasets_results[dn] = pd.concat(runs, axis=0, ignore_index=True)
    return all_datasets_results

def compute_feature_specific_metrics(new_cols, bin_X, bin_header, w_rset, n_samples):
    # compute feature-specific metrics
    new_cols['feature_to_variable_importance'] = Plotter.get_variable_importance(
        bin_X,
        w_rset[:, 1:],
        np.array(bin_header[1:]),
        True
    )

    # compute variable importance for each feature
    feature_indices = ModelUtils.get_feature_indices(bin_header)
    feature_widths = ModelUtils.get_feature_widths(bin_header)

    feature_logit_difference = defaultdict(list)
    feature_shape_difference = defaultdict(list)
    for feature, indices in feature_indices.items():
        indices = np.array(indices)
        feature_logit_difference[feature].append(Metrics.average_pairwise_diversity(
            w_rset[:, indices], 
            Metrics.logit_difference, 
            limit=n_samples, 
            X=bin_X[:, indices]
        ))
        feature_shape_difference[feature].append(Metrics.average_pairwise_diversity(
            w_rset[:, indices], 
            Metrics.shape_difference, 
            limit=n_samples, 
            X=feature_widths[feature]
        ))
    new_cols['feature_logit_difference'].append(feature_logit_difference)
    new_cols['feature_shape_difference'].append(feature_shape_difference)

def get_label(row):
    if row['method_type'] == MethodType.SWAPPING:
        return f"{row['method_type'].value} {row['k']}"
    elif row['method_type'] == MethodType.ELLIPSOID:
        label = f"{row['method_type'].value} {row['sampling']}"
        if row['distance_metric']:
            label += f" ({row['distance_metric']})"
        return label
    elif row['method_type'] == MethodType.MCMC:
        return f"{row['method_type'].value} {row['proposal_function']}, sample {row['sample_from_rset']}"
    else:
        return row['method_type'].value

def populate_datasets_results(all_datasets_results, n_samples):
    # do hamming, model count, 
    for dn, df in all_datasets_results.items():
        new_cols = defaultdict(list)
        for i, row in df.iterrows():
            # load rashomon set for this run
            w_rset = row['w_rset']
            idx = np.random.choice(w_rset.shape[0], size=min(n_samples, w_rset.shape[0]), replace=False)
            w_rset = w_rset[idx]
            w_opt = row['w_opt']
            l0 = row['l0']
            l2 = row['l2']
            method_type = row['method_type']

            # load dataset used for this run
            dataset = Results.load_dataset(f"results/{dn}", {'l0': l0, 'l2': l2})
            bin_X = dataset['bin_X']
            y = dataset['y']
            bin_header = dataset['bin_header']
            cum_header = dataset['cum_header']
            print(f"\t{get_label(row)}")

            # compute diversity metrics on the support set
            for metric in SUPPORT_SET_METRICS:
                average_diversity = Metrics.average_pairwise_diversity(w_rset, metric, limit=n_samples)
                new_cols[metric.__name__].append(average_diversity)
            
            # compute loss values
            for metric in LOSS_METRICS:
                losses, opt_loss = ModelUtils.get_loss(bin_X, y, w_rset, loss_type=metric, w_opt=w_opt, l2=l2)
                new_cols[metric].append(losses)
                new_cols[metric + '_opt'].append(opt_loss)

            # compute logit metrics
            logits = bin_X @ w_rset.T
            for metric in LOGIT_METRICS:
                new_cols[metric.__name__].append(metric(logits))

            # compute prediction metrics
            predictions = ModelUtils.get_predictions(bin_X, w_rset)
            for metric in PREDICTION_METRICS:
                average_diversity = Metrics.average_pairwise_diversity(predictions.T, metric, limit=n_samples)
                new_cols[metric.__name__].append(average_diversity)

            # compute feature-specific metrics
            # compute_feature_specific_metrics(new_cols, bin_X, bin_header, w_rset, n_samples)

        # add new_cols to df
        for key, value in new_cols.items():
            df[key] = value

n_samples = 100
all_datasets_results = collate_datasets_results()

populate_datasets_results(all_datasets_results, n_samples)

with open(f'all_datasets_results.pkl', 'wb') as f:
    pickle.dump(all_datasets_results, f)


# for dname, df in all_datasets_results.items():
#     print(dname)
#     accuracy, logistic, hamming, iou, euclidean, cosine, feature_to_vi = [], [], [], [], [], [], []
#     model_count, headers = [], []
#     feature_to_shape_diversities = []
#     feature_to_shape_differences = []
#     logit_variances = []
#     logit_distances = []

#     path = f"../datasets/{dname}.csv"
#     X, y, header, new_header, sample_p = DatasetUtils.get_binned_dataset(path, df.iloc[0]['n_estimators'])
#     for i, row in df.iterrows():
#         w_rset = row["w_rset"]
#         w_opt = row["w_opt"]
#         predictions = row["predictions"]
#         l2 = 0.001
        
#         model_count.append((w_rset.shape[0], (w_rset.shape[0], w_rset.shape[0])))
#         if w_rset.shape[0] == 0:
#             accuracy.append((0, (0, 0)))
#             logistic.append((0, (0, 0)))
#             hamming.append((0, (0, 0)))
#             iou.append((0, (0, 0)))
#             euclidean.append((0, (0, 0)))
#             cosine.append((0, (0, 0)))
#             logit_variances.append((0, (0, 0)))
#             logit_distances.append((0, (0, 0)))
#             feature_to_vi.append([])
#             feature_to_shape_diversities.append({feature: [] for feature in new_header[1:]})
#             feature_to_shape_differences.append({feature: [] for feature in new_header[1:]})
#             headers.append(new_header)
#             continue

#         print(f"\t{i}: {w_rset.shape[0]}")
#         if w_rset.shape[0] > n_samples: 
#             w_rset = w_rset[:n_samples]
#             predictions = predictions[:n_samples]
#             print(f"\ttruncated to {n_samples}")

#         accuracy.append(get_mean_and_ci(ModelUtils.get_loss(X, y, w_rset, loss_type="accuracy")[0]))
#         logistic.append(get_mean_and_ci(ModelUtils.get_loss(X, y, w_rset, loss_type="logistic", l2=l2, sample_p=sample_p)[0]))
#         hamming.append(average_pairwise_diversity(predictions.T, hamming_distance, limit=n_samples))

#         logits = X @ w_rset.T
#         logit_variances.append(compute_logit_variance(logits))
#         logit_distances.append(average_pairwise_diversity(logits, euclidean_distance, limit=n_samples))

#         iou.append(average_pairwise_diversity(w_rset, inverse_IoU, limit=n_samples))
#         euclidean.append(average_pairwise_diversity(w_rset, euclidean_distance, limit=n_samples))
#         cosine.append(average_pairwise_diversity(w_rset, inverse_cosine_similarity, limit=n_samples))
#         headers.append(new_header)

#         feature_to_vi.append(get_variable_importance(X, w_rset[:, 1:], np.array(new_header[1:]), True))
        
#         # get feature to indices
#         feature_to_indices = defaultdict(list)
#         feature_to_widths = defaultdict(list)
#         feature_ranges = get_feature_ranges(np.array(new_header[1:]))
#         for feature, ranges in feature_ranges.items():
#             for r in ranges:
#                 if len(r) == 1:
#                     feature_to_widths[feature].append(r[0])
#                 else:
#                     feature_to_widths[feature].append(r[1] - r[0])
#         for i, col in enumerate(new_header[1:]):
#             match = re.search(r'([a-zA-Z]+)', col)
#             if match:
#                 feature = match.group(1)
#                 feature_to_indices[feature].append(i)
        
#         feature_to_shape_diversity = defaultdict(list)
#         feature_to_shape_difference = defaultdict(list)
#         for feature, indices in feature_to_indices.items():
#             indices = np.array(indices)
#             average_X_row = X[:, indices]
#             feature_to_shape_diversity[feature].append(average_pairwise_diversity(w_rset[:, indices], shape_diversity, limit=n_samples, X=average_X_row))
#             feature_to_shape_difference[feature].append(average_pairwise_diversity(w_rset[:, indices], shape_difference, limit=n_samples, X=feature_to_widths[feature]))
#         feature_to_shape_diversities.append(feature_to_shape_diversity)
#         feature_to_shape_differences.append(feature_to_shape_difference)
    
#     df['accuracy'] = accuracy
#     df['logistic'] = logistic
#     df['hamming'] = hamming
#     df['iou'] = iou
#     df['euclidean'] = euclidean
#     df['cosine'] = cosine
#     df['model_count'] = model_count
#     df['header'] = headers
#     df['feature_to_vi'] = feature_to_vi
#     df['feature_to_shape_diversities'] = feature_to_shape_diversities
#     df['feature_to_shape_differences'] = feature_to_shape_differences
#     df['logit_variances'] = logit_variances
#     df['logit_distances'] = logit_distances