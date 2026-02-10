import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gam_rs_utils.utils import *
from method_scripts.results import Results

SUPPORT_SET_METRICS = [Metrics.inverse_IoU, Metrics.euclidean_distance, Metrics.inverse_cosine_similarity]
LOSS_METRICS = ['accuracy', 'logistic']
LOGIT_METRICS = [Metrics.compute_logit_variance, Metrics.compute_logit_pairwise_distance]
PREDICTION_METRICS = [Metrics.hamming_distance]

WANT_TO_POPULATE = [
    'ellipsoid surface (euclidean)',
    'swapping 3.0',
]
proposal_functions = ["swap", "correlation_swap", "multi_swap", "same_feature_swap", "feature_swap"]
sample_from_rset = [0, 20]
mh_variant = ["incremental_cd", "standard"]
for pf in proposal_functions:
    for sr in sample_from_rset:
        for mv in mh_variant:
            WANT_TO_POPULATE.append(f"mcmc {pf}, sample {sr}, mh_variant {mv}")

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

def get_label(row):
    if row['method_type'] == MethodType.SWAPPING:
        return f"{row['method_type'].value} {row['k']}"
    elif row['method_type'] == MethodType.ELLIPSOID:
        label = f"{row['method_type'].value} {row['sampling']}"
        if row['distance_metric']:
            label += f" ({row['distance_metric']})"
        return label
    elif row['method_type'] == MethodType.MCMC:
        return f"{row['method_type'].value} {row['proposal_function']}, sample {row['sample_from_rset']}, mh_variant {row['mh_variant']}"
    else:
        return row['method_type'].value

def populate_datasets_results(all_datasets_results):
    # do hamming, model count, 
    for dn, df in all_datasets_results.items():
        new_cols = defaultdict(list)
        for i, row in df.iterrows():
            # load rashomon set for this run
            w_rset = row['w_rset']
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

            label = get_label(row)
            if label not in WANT_TO_POPULATE:
                print(f"\tskipping {label}")
                df.drop(index=i, inplace=True)
                continue
            print(f"\t{label}: {w_rset.shape[0]} models")
            new_cols['label'].append(label)

            # compute diversity metrics on the support set
            for metric in SUPPORT_SET_METRICS:
                diversity = Metrics.average_pairwise_diversity(w_rset, metric)
                new_cols[metric.__name__].append(diversity)
                print(f"\t\tcomputed {metric.__name__}")
            
            # compute loss values
            for metric in LOSS_METRICS:
                losses, opt_loss = ModelUtils.get_loss(bin_X, y, w_rset, loss_type=metric, w_opt=w_opt, l2=l2)
                new_cols[metric].append(losses)
                new_cols[metric + '_opt'].append(opt_loss)
                print(f"\t\tcomputed {metric}")

            # compute logit metrics
            logits = bin_X @ w_rset.T
            for metric in LOGIT_METRICS:
                new_cols[metric.__name__].append(metric(logits))
                print(f"\t\tcomputed {metric.__name__}")

            # compute prediction metrics
            predictions = ModelUtils.get_predictions(bin_X, w_rset)
            for metric in PREDICTION_METRICS:
                diversity = Metrics.average_pairwise_diversity(predictions.T, metric)
                new_cols[metric.__name__].append(diversity)
                print(f"\t\tcomputed {metric.__name__}")

                # also get the distribution of diversities
                diversities = Metrics.sample_diversities(predictions.T, metric)
                new_cols[metric.__name__ + '_distribution'].append(diversities)
                print(f"\t\tcomputed {metric.__name__} distribution")
            
            # compute feature-specific metrics
            new_cols['feature_to_variable_importance'].append(Plotter.get_variable_importance(
                bin_X,
                w_rset[:, 1:],
                np.array(bin_header[1:])
            ))
            print(f"\t\tcomputed variable importance by feature")

            # compute variable importance for each feature
            feature_indices = ModelUtils.get_feature_indices(bin_header)
            feature_widths = ModelUtils.get_feature_widths(bin_header)

            feature_logit_difference = defaultdict(list)
            feature_shape_difference = defaultdict(list)
            for feature, indices in feature_indices.items():
                print(f"\t\t\tcomputing {feature} logit difference")
                indices = np.array(indices)
                feature_logit_difference[feature].append(Metrics.average_pairwise_diversity(
                    w_rset[:, indices], 
                    Metrics.logit_difference, 
                    X=bin_X[:, indices]
                ))
                print(f"\t\t\tcomputing {feature} shape difference")
                feature_shape_difference[feature].append(Metrics.average_pairwise_diversity(
                    w_rset[:, indices], 
                    Metrics.shape_difference, 
                    X=feature_widths[feature]
                ))
            new_cols['feature_logit_difference'].append(feature_logit_difference)
            new_cols['feature_shape_difference'].append(feature_shape_difference)
            print(f"\t\tcomputed logit difference and shape difference by feature")

        # add new_cols to df
        for key, value in new_cols.items():
            df[key] = value

all_datasets_results = collate_datasets_results()
populate_datasets_results(all_datasets_results)

# sort rows by get_label(row)
for dn, df in all_datasets_results.items():
    all_datasets_results[dn] = df.sort_values(by='label')

with open(f'all_datasets_results.pkl', 'wb') as f:
    pickle.dump(all_datasets_results, f)
