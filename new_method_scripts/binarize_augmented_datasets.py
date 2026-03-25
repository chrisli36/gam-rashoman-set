#!/usr/bin/env python3
"""
Script to binarize augmented datasets (with prototype similarities) into three separate datasets:
1. Tabular features only (excluding prototype similarities)
2. Prototype similarities only
3. Tabular + Prototype similarities combined

For each binarization level (num_estimators), trains TreeFARMS Rashomon sets and reports optimal tree accuracy.
Creates a CSV table summarizing results.
"""

import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

try:
    from treefarms import TREEFARMS
    TREEFARMS_AVAILABLE = True
except ImportError:
    print("Warning: TreeFARMS not available. Results will be computed but TreeFARMS training will be skipped.")
    TREEFARMS_AVAILABLE = False

# fit the tree using gradient boosted classifier
seed = 3


def fit_boosted_tree(X, y, n_est=10, lr=0.1, d=1):
    clf = GradientBoostingClassifier(loss='log_loss', learning_rate=lr, n_estimators=n_est, max_depth=d, random_state=seed)
    clf.fit(X, y)
    out = clf.score(X, y)
    return clf, out

# perform cut on the dataset


def cut(X, ts):
    df = X.copy()
    colnames = X.columns
    used_thresholds = []
    for j in range(len(ts)):
        for s in range(len(ts[j])):
            X[colnames[j]+'<='+str(ts[j][s])] = 1
            k = df[colnames[j]] > ts[j][s]
            X.loc[k, colnames[j]+'<='+str(ts[j][s])] = 0
        X = X.drop(colnames[j], axis=1)
    return X

# compute the thresholds


def get_thresholds(X, y, n_est, lr, d, backselect=False, random_selection=False,
                   max_thresholds_per_feat=10):

    if not random_selection:
        # got a complaint here...
        y = np.ravel(y)
        # X is a dataframe
        clf, out = fit_boosted_tree(X, y, n_est, lr, d)
        # print('acc:', out, 'acc cv:', score.mean())
        thresholds = []

        for j in range(X.shape[1]):
            tj = np.array([])
            for i in range(len(clf.estimators_)):
                f = clf.estimators_[i, 0].tree_.feature
                t = clf.estimators_[i, 0].tree_.threshold
                tj = np.append(tj, t[f == j])
            tj = np.unique(tj)
            thresholds.append(tj.tolist())

        X_new = cut(X, thresholds)
        clf1, out1 = fit_boosted_tree(X_new, y, n_est, lr, d)
        # print('acc','1:', out1, 'acc1 cv:', scorep.mean())

        outp = 1
        Xp = X_new.copy()
        clfp = clf1
        itr = 0
        if backselect:
            while outp >= out1 and itr < X_new.shape[1]-1:
                vi = clfp.feature_importances_
                if vi.size > 0:
                    c = Xp.columns
                    i = np.argmin(vi)
                    Xp = Xp.drop(c[i], axis=1)
                    clfp, outp = fit_boosted_tree(Xp, y, n_est, lr, d)
                    # print(outp,out1)
                    itr += 1
                else:
                    break
            Xp[c[i]] = X_new[c[i]]
            # _, _ = fit_boosted_tree(Xp, y, n_est, lr, d)

        h = Xp.columns
        # print('features:', h)
        return Xp, thresholds, h
    else:
        thresholds = []
        for j in range(X.shape[1]):
            tj = []
            num_features = np.random.choice(max_thresholds_per_feat)
            min_val = min(X.iloc[:, j])
            max_val = max(X.iloc[:, j])
            for i in range(num_features):
                t = np.random.uniform(min_val, max_val)
                tj.append(t)
            tj = sorted(tj)
            thresholds.append(tj)
        X_new = cut(X, thresholds)
        return X_new, thresholds, X_new.columns

# compute the thresholds


def compute_thresholds(X, y, n_est, max_depth, random_selection=False, max_thresholds_per_feat=10):
    # n_est, max_depth: GBDT parameters
    # set LR to 0.1
    lr = 0.1
    # GradientBoostingClassifier does not accept NaN; drop rows with missing values
    if isinstance(X, pd.DataFrame):
        valid = ~X.isna().any(axis=1)
    else:
        valid = ~np.isnan(np.asarray(X, dtype=float)).any(axis=1)
    if not valid.all():
        n_drop = int((~valid).sum())
        X = X.loc[valid] if isinstance(X, pd.DataFrame) else X[valid]
        if hasattr(y, "loc"):
            y = y.loc[valid]
        else:
            y = np.asarray(y)[np.asarray(valid)]
        if n_drop > 0:
            print(f"  Dropped {n_drop} rows with missing values for GBDT threshold fitting")
    start = time.perf_counter()
    X, thresholds, header = get_thresholds(
        X, y, n_est, lr, max_depth, backselect=False, random_selection=random_selection, max_thresholds_per_feat=max_thresholds_per_feat)
    guess_time = time.perf_counter()-start

    return X, thresholds, header, guess_time


def binarize_dataset_multiclass_one_vs_all(dataset, num_estimators, max_thresholds_per_feat=10, max_features=50, random_seed=42, continuous_cols=None, binary_cols=None):
    """Binarize multi-class dataset using one-vs-all approach. Combines thresholds from all classes."""
    X, Y = pd.DataFrame(dataset.values[:, :-1], columns=dataset.columns[:-1]), pd.DataFrame(
        dataset.values[:, -1], columns=[dataset.columns[-1]])
    
    # Get unique classes
    unique_classes = sorted(Y.iloc[:, 0].unique())
    num_classes = len(unique_classes)
    
    print(f"  Multi-class binarization: {num_classes} classes ({unique_classes})")
    
    # Identify already-binary features
    # Use provided binary_cols/continuous_cols if available, otherwise detect automatically
    if binary_cols is None or continuous_cols is None:
        binary_cols = []
        continuous_cols = []
        for col in X.columns:
            unique_vals = X[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
            else:
                continuous_cols.append(col)
    else:
        # Use provided lists, but filter to only include columns that exist in X
        binary_cols = [col for col in binary_cols if col in X.columns]
        continuous_cols = [col for col in continuous_cols if col in X.columns]
    
    X_binary_keep = X[binary_cols].copy() if binary_cols else pd.DataFrame(index=X.index)
    
    # For each class, do one-vs-all binarization
    all_thresholds_dict = {}  # Store thresholds for each continuous column from all classes
    all_binary_features = set(binary_cols)  # Start with binary features
    
    for class_label in unique_classes:
        print(f"    Processing class {class_label} vs all...")
        # Create binary label: 1 for this class, 0 for others
        Y_binary = (Y.iloc[:, 0] == class_label).astype(int)
        Y_binary_df = pd.DataFrame(Y_binary, columns=[Y.columns[0]])
        
        if continuous_cols:
            X_continuous = X[continuous_cols]
            # Binarize continuous features for this one-vs-all task
            X_binary_cont, thresholds_cont, header_cont, _ = compute_thresholds(
                X_continuous, Y_binary_df, n_est=num_estimators, max_depth=1, 
                random_selection=False, max_thresholds_per_feat=max_thresholds_per_feat)
            
            # Store thresholds for each continuous column
            for i, col in enumerate(continuous_cols):
                if col not in all_thresholds_dict:
                    all_thresholds_dict[col] = []
                if i < len(thresholds_cont) and thresholds_cont[i] is not None:
                    all_thresholds_dict[col].extend(thresholds_cont[i])
            
            # Collect all binary features from this class
            all_binary_features.update(header_cont)
    
    # Combine thresholds: for each continuous column, take unique thresholds
    combined_thresholds = []
    for col in X.columns:
        if col in binary_cols:
            combined_thresholds.append(None)
        elif col in continuous_cols:
            if col in all_thresholds_dict:
                # Get unique thresholds and sort
                unique_thresh = sorted(set(all_thresholds_dict[col]))
                # Limit to max_thresholds_per_feat
                if len(unique_thresh) > max_thresholds_per_feat:
                    unique_thresh = unique_thresh[:max_thresholds_per_feat]
                combined_thresholds.append(unique_thresh)
            else:
                combined_thresholds.append([])
        else:
            combined_thresholds.append([])
    
    # Apply combined thresholds to create final binary features
    if continuous_cols:
        cont_thresholds = [combined_thresholds[list(X.columns).index(col)] for col in continuous_cols]
        X_continuous = X[continuous_cols]
        X_binary_cont = cut(X_continuous.copy(), cont_thresholds)
        # Combine with binary features
        X_binary = pd.concat([X_binary_keep.reset_index(drop=True), X_binary_cont.reset_index(drop=True)], axis=1)
        header = list(binary_cols) + list(X_binary_cont.columns)
    else:
        X_binary = X_binary_keep
        header = list(binary_cols)
    
    # Random subsample if number of features > max_features
    if len(header) > max_features:
        print(f"    Randomly subsampling from {len(header)} to {max_features} features...")
        np.random.seed(random_seed)
        selected_features = np.random.choice(header, size=max_features, replace=False).tolist()
        X_binary = X_binary[selected_features]
        header = selected_features
    
    # Combine with original label
    result = pd.concat([X_binary.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
    
    return result, combined_thresholds, header, 0.0


def binarize_dataset(dataset, num_estimators, random_selection=False, max_thresholds_per_feat=10, continuous_cols = None, binary_cols = None, thresholds=None, header=None, csv_path=None, multiclass=False, max_features=50):
    """Binarize dataset using GOSDT thresholds. Keeps already-binary features as-is.
    
    Args:
        multiclass: If True, use one-vs-all approach for multi-class labels (only when thresholds=None)
        max_features: Maximum number of features to keep (for multiclass, random subsample if exceeded)
    """
    if multiclass and thresholds is None:
        # Use one-vs-all approach for multi-class (training phase)
        result, thresholds_full, header, threshold_guess_time = binarize_dataset_multiclass_one_vs_all(
            dataset, num_estimators, max_thresholds_per_feat=max_thresholds_per_feat, max_features=max_features,
            continuous_cols=continuous_cols, binary_cols=binary_cols
        )
        if csv_path is not None:
            result.to_csv(csv_path, index=False)
        return result, thresholds_full, header, threshold_guess_time
    
    if thresholds is None:
        X, Y = pd.DataFrame(dataset.values[:, :-1], columns=dataset.columns[:-1]), pd.DataFrame(
            dataset.values[:, -1], columns=[dataset.columns[-1]])
        
        # Identify already-binary features (one-hot encoded)
        # Use provided binary_cols/continuous_cols if available, otherwise detect automatically
        if binary_cols is None or continuous_cols is None:
            binary_cols = []
            continuous_cols = []
            for col in X.columns:
                unique_vals = X[col].dropna().unique()
                # Check if feature is already binary (only 0 and 1, or only 0, or only 1)
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    binary_cols.append(col)
                else:
                    continuous_cols.append(col)
        else:
            # Use provided lists, but filter to only include columns that exist in X
            binary_cols = [col for col in binary_cols if col in X.columns]
            continuous_cols = [col for col in continuous_cols if col in X.columns]
        # Keep binary features as-is
        X_binary_keep = X[binary_cols].copy() if binary_cols else pd.DataFrame(index=X.index)
        
        # Binarize only continuous features
        if continuous_cols:
            X_continuous = X[continuous_cols]
            X_binary_cont, thresholds_cont, header_cont, threshold_guess_time = compute_thresholds(
                X_continuous, Y, n_est=num_estimators, max_depth=1, random_selection=random_selection, max_thresholds_per_feat=max_thresholds_per_feat)
            X_binary_cont = X_binary_cont[header_cont]
            # Combine binary and binarized continuous features
            X_binary = pd.concat([X_binary_keep.reset_index(drop=True), X_binary_cont.reset_index(drop=True)], axis=1)
            # Store thresholds as list matching original column order (None for binary cols)
            thresholds_full = []
            for col in X.columns:
                if col in binary_cols:
                    thresholds_full.append(None)
                else:
                    idx = continuous_cols.index(col)
                    thresholds_full.append(thresholds_cont[idx])
            header = list(binary_cols) + list(header_cont)
        else:
            # All features are already binary
            X_binary = X_binary_keep
            thresholds_full = [None] * len(binary_cols)
            header = list(binary_cols)
            threshold_guess_time = 0.0
        
        if csv_path is not None:
            pd.concat([X_binary, Y], axis=1).to_csv(csv_path, index=False)
        
        return pd.concat([X_binary, Y], axis=1), thresholds_full, header, threshold_guess_time
    else:
        # Both header and thresholds must be provided
        X, Y = pd.DataFrame(dataset.values[:, :-1], columns=dataset.columns[:-1]), pd.DataFrame(
            dataset.values[:, -1], columns=[dataset.columns[-1]])
        
        # Identify binary vs continuous features
        # Use provided binary_cols/continuous_cols if available, otherwise detect automatically
        if binary_cols is None or continuous_cols is None:
            binary_cols = []
            continuous_cols = []
            for col in X.columns:
                unique_vals = X[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    binary_cols.append(col)
                else:
                    continuous_cols.append(col)
        else:
            # Use provided lists, but filter to only include columns that exist in X
            binary_cols = [col for col in binary_cols if col in X.columns]
            continuous_cols = [col for col in continuous_cols if col in X.columns]
        # Keep binary features as-is
        X_binary_keep = X[binary_cols].copy() if binary_cols else pd.DataFrame(index=X.index)
        
        # Apply binarization to continuous features
        if continuous_cols and thresholds:
            # thresholds is a list matching original column order (None for binary cols)
            # Extract thresholds for continuous columns only
            cont_thresholds = []
            for col in continuous_cols:
                if col in X.columns:
                    col_idx = list(X.columns).index(col)
                    if col_idx < len(thresholds):
                        cont_thresholds.append(thresholds[col_idx] if thresholds[col_idx] is not None else [])
                    else:
                        cont_thresholds.append([])
                else:
                    cont_thresholds.append([])
            
            X_continuous = X[continuous_cols]
            X_binary_cont = cut(X_continuous.copy(), cont_thresholds)
            # Get header for continuous binarized features (from header, excluding binary cols)
            cont_header = [col for col in header if col not in binary_cols]
            X_binary_cont = X_binary_cont[cont_header]
            # Combine
            X_binary = pd.concat([X_binary_keep.reset_index(drop=True), X_binary_cont.reset_index(drop=True)], axis=1)
        else:
            X_binary = X_binary_keep
        
        # Ensure columns are in the correct order according to header
        X_binary = X_binary[header]
        if csv_path is not None:
            pd.concat([X_binary, Y], axis=1).to_csv(csv_path, index=False)
        return pd.concat([X_binary, Y], axis=1)


def train_random_forest_and_get_metrics(X_train, y_train, X_val, y_val, random_seed=42):
    """Train Random Forest and return accuracy and F1 score on validation set."""
    try:
        # Determine average type for F1 score
        unique_labels = np.unique(y_val)
        if len(unique_labels) == 2:
            average_type = 'binary'
        else:
            average_type = 'macro'
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=10,
            random_state=random_seed,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_val)
        
        accuracy = (predictions == y_val).mean()
        f1 = f1_score(y_val, predictions, average=average_type, zero_division=0)
        
        return accuracy, f1
    except Exception as e:
        print(f"  Error training Random Forest: {e}")
        return None, None


def train_treefarms_and_get_best_tree_metrics(X_train, y_train, X_val, y_val, treefarms_config):
    """Train TreeFARMS on training data and return accuracy and F1 score of the best tree (first tree) on validation set."""
    if not TREEFARMS_AVAILABLE:
        return None, None
    
    try:
        tf_model = TREEFARMS(treefarms_config)
        tf_model.fit(X_train, y_train)
        
        if tf_model.get_tree_count() == 0:
            return None, None
        
        # Get the best tree (first tree in Rashomon set)
        best_tree = tf_model[0]
        predictions = best_tree.predict(X_val)
        accuracy = (predictions == y_val).mean()
        
        # Determine average type for F1 score
        unique_labels = np.unique(y_val)
        if len(unique_labels) == 2:
            average_type = 'binary'
        else:
            average_type = 'macro'
        
        f1 = f1_score(y_val, predictions, average=average_type, zero_division=0)
        
        return accuracy, f1
    except Exception as e:
        print(f"  Error training TreeFARMS: {e}")
        return None, None


def process_dataset_with_treefarms(train_csv, val_csv, output_dir, dataset_name, label_column, 
                                   num_estimators_list, treefarms_config, multiclass=False, 
                                   include_random_forest_baseline=True, n_splits=5, random_seed=42):
    """Process datasets with 5-fold cross-validation: combine train/val, create splits, binarize, train TreeFARMS/RF."""
    if dataset_name == "HAM":
        print("Setting regularization to 0.001 for HAM")
        treefarms_config['regularization'] = 0.001
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")
    
    # Load and combine datasets
    print(f"Loading {train_csv}...")
    df_train = pd.read_csv(train_csv)
    print(f"Loaded {len(df_train)} rows, {len(df_train.columns)} columns")
    
    print(f"Loading {val_csv}...")
    df_val = pd.read_csv(val_csv)
    print(f"Loaded {len(df_val)} rows, {len(df_val.columns)} columns")
    
    # Combine train and validation datasets
    print(f"\nCombining train and validation datasets...")
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    print(f"Combined dataset: {len(df_combined)} rows")
    
    # Identify columns
    proto_cols = [c for c in df_combined.columns if c.startswith('sim_proto_')]
    non_proto_cols = [c for c in df_combined.columns if not c.startswith('sim_proto_') and c != 'Image_name']
    
    # Remove label from non_proto_cols if it's there
    if label_column in non_proto_cols:
        non_proto_cols.remove(label_column)
    
    # Also remove isic_id if present (for HAM)
    if 'isic_id' in non_proto_cols:
        non_proto_cols.remove('isic_id')
    
    # Filter to include numeric and categorical columns
    numeric_non_proto_cols = []
    binary_non_proto_cols = []
    for col in non_proto_cols:
        dtype = df_combined[col].dtype
        unique_vals = df_combined[col].unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            binary_non_proto_cols.append(col)
        else:
            numeric_non_proto_cols.append(col)
    
    print(f"\nPrototype similarity columns: {len(proto_cols)}")
    print(f"Tabular feature columns (including categorical): {len(numeric_non_proto_cols)}")
    print(f"Binary non-prototype columns: {len(binary_non_proto_cols)}")
    print(f"Label column: {label_column}")
    
    # Prepare combined dataset with all features
    df_tabular = df_combined[numeric_non_proto_cols + binary_non_proto_cols + [label_column]].copy()
    df_proto = df_combined[proto_cols + [label_column]].copy()
    
    # Convert categorical and object columns to numeric codes
    for col in numeric_non_proto_cols:
        if df_tabular[col].dtype.name == 'category':
            df_tabular[col] = df_tabular[col].cat.codes
        elif df_tabular[col].dtype == 'object':
            all_categories = df_tabular[col].unique()
            df_tabular[col] = pd.Categorical(df_tabular[col], categories=all_categories).codes
    
    # Fill NaN values
    df_tabular = df_tabular.fillna(0)
    df_proto = df_proto.fillna(0)
    
    # Ensure label is last column
    cols_tabular = [c for c in df_tabular.columns if c != label_column] + [label_column]
    df_tabular = df_tabular[cols_tabular]
    cols_proto = [c for c in df_proto.columns if c != label_column] + [label_column]
    df_proto = df_proto[cols_proto]
    
    # Create combined dataset (tabular + prototype)
    df_combined_features = pd.concat([
        df_tabular.iloc[:, :-1].reset_index(drop=True),
        df_proto.iloc[:, :-1].reset_index(drop=True)
    ], axis=1)
    df_combined_features[label_column] = df_tabular[label_column].reset_index(drop=True)
    cols_combined = [c for c in df_combined_features.columns if c != label_column] + [label_column]
    df_combined_features = df_combined_features[cols_combined]
    
    # Get labels for stratified splitting
    y_all = df_tabular[label_column].astype(int).values
    
    # Create 5-fold stratified splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    splits = list(skf.split(df_tabular, y_all))
    
    print(f"\nCreated {n_splits} stratified train-validation splits")
    
    # Results storage
    treefarms_results = []
    rf_results = []
    
    # Process each num_estimators value
    for num_est in tqdm(num_estimators_list, desc=f"Processing {dataset_name}"):
        print(f"\n--- Processing num_estimators={num_est} ---")
        
        # Storage for metrics across splits
        tabular_accs, tabular_f1s = [], []
        proto_accs, proto_f1s = [], []
        combined_accs, combined_f1s = [], []
        num_binary_features_list = []
        
        # Random Forest metrics across splits
        rf_tab_accs, rf_tab_f1s = [], []
        rf_proto_accs, rf_proto_f1s = [], []
        rf_comb_accs, rf_comb_f1s = [], []
        
        # Process each split
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{n_splits}")
            
            # Split datasets
            df_train_tab_split = df_tabular.iloc[train_idx].reset_index(drop=True)
            df_val_tab_split = df_tabular.iloc[val_idx].reset_index(drop=True)
            df_train_proto_split = df_proto.iloc[train_idx].reset_index(drop=True)
            df_val_proto_split = df_proto.iloc[val_idx].reset_index(drop=True)
            df_train_comb_split = df_combined_features.iloc[train_idx].reset_index(drop=True)
            df_val_comb_split = df_combined_features.iloc[val_idx].reset_index(drop=True)
            
            # ========== TreeFARMS Experiments ==========
            # 1. Tabular
            # First, identify binary vs continuous columns from training set
            train_tab_binary_cols = []
            train_tab_continuous_cols = []
            for col in df_train_tab_split.columns[:-1]:  # Exclude label column
                unique_vals = df_train_tab_split[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    train_tab_binary_cols.append(col)
                else:
                    train_tab_continuous_cols.append(col)
            
            df_train_tab_binary, thresholds_tab, header_tab, _ = binarize_dataset(
                df_train_tab_split, num_estimators=num_est, max_thresholds_per_feat=10, 
                multiclass=multiclass, max_features=50,
                continuous_cols=train_tab_continuous_cols, binary_cols=train_tab_binary_cols
            )
            
            df_val_tab_binary = binarize_dataset(
                df_val_tab_split, num_estimators=num_est, thresholds=thresholds_tab, header=header_tab,
                continuous_cols=train_tab_continuous_cols, binary_cols=train_tab_binary_cols
            )
            
            X_train_tab = df_train_tab_binary.iloc[:, :-1]
            y_train_tab = df_train_tab_binary.iloc[:, -1].astype(int)
            X_val_tab = df_val_tab_binary.iloc[:, :-1]
            y_val_tab = df_val_tab_binary.iloc[:, -1].astype(int)
            
            acc_tab, f1_tab = train_treefarms_and_get_best_tree_metrics(
                X_train_tab, y_train_tab, X_val_tab, y_val_tab, treefarms_config
            )
            if acc_tab is not None:
                tabular_accs.append(acc_tab)
                tabular_f1s.append(f1_tab)
            
            # 2. Prototype
            # Identify binary vs continuous columns from training set
            train_proto_binary_cols = []
            train_proto_continuous_cols = []
            for col in df_train_proto_split.columns[:-1]:  # Exclude label column
                unique_vals = df_train_proto_split[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    train_proto_binary_cols.append(col)
                else:
                    train_proto_continuous_cols.append(col)
            
            df_train_proto_binary, thresholds_proto, header_proto, _ = binarize_dataset(
                df_train_proto_split, num_estimators=num_est, max_thresholds_per_feat=10,
                multiclass=multiclass, max_features=50,
                continuous_cols=train_proto_continuous_cols, binary_cols=train_proto_binary_cols
            )
            df_val_proto_binary = binarize_dataset(
                df_val_proto_split, num_estimators=num_est, thresholds=thresholds_proto, header=header_proto,
                continuous_cols=train_proto_continuous_cols, binary_cols=train_proto_binary_cols
            )
            
            X_train_proto = df_train_proto_binary.iloc[:, :-1]
            y_train_proto = df_train_proto_binary.iloc[:, -1].astype(int)
            X_val_proto = df_val_proto_binary.iloc[:, :-1]
            y_val_proto = df_val_proto_binary.iloc[:, -1].astype(int)
            
            acc_proto, f1_proto = train_treefarms_and_get_best_tree_metrics(
                X_train_proto, y_train_proto, X_val_proto, y_val_proto, treefarms_config
            )
            if acc_proto is not None:
                proto_accs.append(acc_proto)
                proto_f1s.append(f1_proto)
            
            # 3. Combined
            # Identify binary vs continuous columns from training set
            train_comb_binary_cols = []
            train_comb_continuous_cols = []
            for col in df_train_comb_split.columns[:-1]:  # Exclude label column
                unique_vals = df_train_comb_split[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    train_comb_binary_cols.append(col)
                else:
                    train_comb_continuous_cols.append(col)
            
            df_train_comb_binary, thresholds_comb, header_comb, _ = binarize_dataset(
                df_train_comb_split, num_estimators=num_est, max_thresholds_per_feat=10,
                multiclass=multiclass, max_features=50,
                continuous_cols=train_comb_continuous_cols, binary_cols=train_comb_binary_cols
            )
            df_val_comb_binary = binarize_dataset(
                df_val_comb_split, num_estimators=num_est, thresholds=thresholds_comb, header=header_comb,
                continuous_cols=train_comb_continuous_cols, binary_cols=train_comb_binary_cols
            )
            
            X_train_comb = df_train_comb_binary.iloc[:, :-1]
            y_train_comb = df_train_comb_binary.iloc[:, -1].astype(int)
            X_val_comb = df_val_comb_binary.iloc[:, :-1]
            y_val_comb = df_val_comb_binary.iloc[:, -1].astype(int)
            
            acc_comb, f1_comb = train_treefarms_and_get_best_tree_metrics(
                X_train_comb, y_train_comb, X_val_comb, y_val_comb, treefarms_config
            )
            if acc_comb is not None:
                combined_accs.append(acc_comb)
                combined_f1s.append(f1_comb)
            
            num_binary_features_list.append(int(np.mean([len(header_tab), len(header_proto), len(header_comb)])))
            
            # ========== Random Forest Baseline ==========
            if include_random_forest_baseline:
                # Tabular (non-binarized)
                X_train_tab_raw = df_train_tab_split.iloc[:, :-1]
                y_train_tab_raw = df_train_tab_split.iloc[:, -1].astype(int)
                X_val_tab_raw = df_val_tab_split.iloc[:, :-1]
                y_val_tab_raw = df_val_tab_split.iloc[:, -1].astype(int)
                
                acc_rf_tab, f1_rf_tab = train_random_forest_and_get_metrics(
                    X_train_tab_raw, y_train_tab_raw, X_val_tab_raw, y_val_tab_raw, 
                    random_seed=random_seed + fold_idx
                )
                if acc_rf_tab is not None:
                    rf_tab_accs.append(acc_rf_tab)
                    rf_tab_f1s.append(f1_rf_tab)
                
                # Prototype (non-binarized)
                X_train_proto_raw = df_train_proto_split.iloc[:, :-1]
                y_train_proto_raw = df_train_proto_split.iloc[:, -1].astype(int)
                X_val_proto_raw = df_val_proto_split.iloc[:, :-1]
                y_val_proto_raw = df_val_proto_split.iloc[:, -1].astype(int)
                
                acc_rf_proto, f1_rf_proto = train_random_forest_and_get_metrics(
                    X_train_proto_raw, y_train_proto_raw, X_val_proto_raw, y_val_proto_raw,
                    random_seed=random_seed + fold_idx
                )
                if acc_rf_proto is not None:
                    rf_proto_accs.append(acc_rf_proto)
                    rf_proto_f1s.append(f1_rf_proto)
                
                # Combined (non-binarized)
                X_train_comb_raw = df_train_comb_split.iloc[:, :-1]
                y_train_comb_raw = df_train_comb_split.iloc[:, -1].astype(int)
                X_val_comb_raw = df_val_comb_split.iloc[:, :-1]
                y_val_comb_raw = df_val_comb_split.iloc[:, -1].astype(int)
                
                acc_rf_comb, f1_rf_comb = train_random_forest_and_get_metrics(
                    X_train_comb_raw, y_train_comb_raw, X_val_comb_raw, y_val_comb_raw,
                    random_seed=random_seed + fold_idx
                )
                if acc_rf_comb is not None:
                    rf_comb_accs.append(acc_rf_comb)
                    rf_comb_f1s.append(f1_rf_comb)
        
        # Aggregate results across splits
        avg_num_binary_features = int(np.mean(num_binary_features_list)) if num_binary_features_list else 0
        
        treefarms_results.append({
            'num_estimators': num_est,
            'num_binary_features': avg_num_binary_features,
            'tabular_accuracy_mean': np.mean(tabular_accs) if tabular_accs else None,
            'tabular_accuracy_std': np.std(tabular_accs) if tabular_accs else None,
            'tabular_f1_mean': np.mean(tabular_f1s) if tabular_f1s else None,
            'tabular_f1_std': np.std(tabular_f1s) if tabular_f1s else None,
            'prototype_accuracy_mean': np.mean(proto_accs) if proto_accs else None,
            'prototype_accuracy_std': np.std(proto_accs) if proto_accs else None,
            'prototype_f1_mean': np.mean(proto_f1s) if proto_f1s else None,
            'prototype_f1_std': np.std(proto_f1s) if proto_f1s else None,
            'combined_accuracy_mean': np.mean(combined_accs) if combined_accs else None,
            'combined_accuracy_std': np.std(combined_accs) if combined_accs else None,
            'combined_f1_mean': np.mean(combined_f1s) if combined_f1s else None,
            'combined_f1_std': np.std(combined_f1s) if combined_f1s else None
        })
        
        print(f"    TreeFARMS Results (across {n_splits} splits):")
        if tabular_accs:
            print(f"      Tabular: Acc={np.mean(tabular_accs):.4f}±{np.std(tabular_accs):.4f}, F1={np.mean(tabular_f1s):.4f}±{np.std(tabular_f1s):.4f}")
        if proto_accs:
            print(f"      Prototype: Acc={np.mean(proto_accs):.4f}±{np.std(proto_accs):.4f}, F1={np.mean(proto_f1s):.4f}±{np.std(proto_f1s):.4f}")
        if combined_accs:
            print(f"      Combined: Acc={np.mean(combined_accs):.4f}±{np.std(combined_accs):.4f}, F1={np.mean(combined_f1s):.4f}±{np.std(combined_f1s):.4f}")
    
    treefarms_results_df = pd.DataFrame(treefarms_results)
    
    # Aggregate Random Forest results across splits
    if include_random_forest_baseline and rf_tab_accs:
        rf_results.append({
            'num_estimators': 'N/A (baseline)',
            'num_binary_features': 'N/A (non-binarized)',
            'tabular_accuracy_mean': np.mean(rf_tab_accs),
            'tabular_accuracy_std': np.std(rf_tab_accs),
            'tabular_f1_mean': np.mean(rf_tab_f1s),
            'tabular_f1_std': np.std(rf_tab_f1s),
            'prototype_accuracy_mean': np.mean(rf_proto_accs),
            'prototype_accuracy_std': np.std(rf_proto_accs),
            'prototype_f1_mean': np.mean(rf_proto_f1s),
            'prototype_f1_std': np.std(rf_proto_f1s),
            'combined_accuracy_mean': np.mean(rf_comb_accs),
            'combined_accuracy_std': np.std(rf_comb_accs),
            'combined_f1_mean': np.mean(rf_comb_f1s),
            'combined_f1_std': np.std(rf_comb_f1s)
        })
        
        print(f"\n    Random Forest Results (across {n_splits} splits):")
        print(f"      Tabular: Acc={np.mean(rf_tab_accs):.4f}±{np.std(rf_tab_accs):.4f}, F1={np.mean(rf_tab_f1s):.4f}±{np.std(rf_tab_f1s):.4f}")
        print(f"      Prototype: Acc={np.mean(rf_proto_accs):.4f}±{np.std(rf_proto_accs):.4f}, F1={np.mean(rf_proto_f1s):.4f}±{np.std(rf_proto_f1s):.4f}")
        print(f"      Combined: Acc={np.mean(rf_comb_accs):.4f}±{np.std(rf_comb_accs):.4f}, F1={np.mean(rf_comb_f1s):.4f}±{np.std(rf_comb_f1s):.4f}")
    
    rf_results_df = pd.DataFrame(rf_results) if rf_results else pd.DataFrame()
    
    return treefarms_results_df, rf_results_df


def main():
    """Main function to process DVM and HAM datasets."""
    
    base_dir = "/usr/xtmp/vb97/Multimodal_Rashomon_Sets"
    
    # TreeFARMS configuration
    treefarms_config = {
        "regularization": 0.01,
        "rashomon_bound_multiplier": 0.02,
        "depth_budget": 4,
        "verbose": False
    }
    
    # List of num_estimators values to test
    num_estimators_list = [10,20,50,100]
    
    all_results = {}
    all_rf_results = {}
    
    # ========== DVM Dataset ==========
    print("\n" + "="*60)
    print("PROCESSING DVM DATASET")
    print("="*60)
    
    dvm_output_dir = os.path.join(base_dir, "DVM_augmented_df")
    dvm_train_csv = os.path.join(dvm_output_dir, "13_last_only_0.9632.csv")
    dvm_val_csv = os.path.join(dvm_output_dir, "13_last_only_0.9632_val.csv")
    
    if os.path.exists(dvm_train_csv) and os.path.exists(dvm_val_csv):
        dvm_results, dvm_rf_results = process_dataset_with_treefarms(
            dvm_train_csv,
            dvm_val_csv,
            dvm_output_dir,
            "DVM",
            label_column="expensive_car",
            num_estimators_list=num_estimators_list,
            treefarms_config=treefarms_config
        )
        all_results['DVM'] = dvm_results
        all_rf_results['DVM'] = dvm_rf_results
        
        # Save DVM TreeFARMS results
        dvm_output_path = os.path.join(dvm_output_dir, "rashomon_set_accuracies_dvm.csv")
        dvm_results.to_csv(dvm_output_path, index=False)
        print(f"\nDVM TreeFARMS results saved to: {dvm_output_path}")
        
        # Save DVM Random Forest baseline results
        if not dvm_rf_results.empty:
            dvm_rf_output_path = os.path.join(dvm_output_dir, "random_forest_baseline_dvm.csv")
            dvm_rf_results.to_csv(dvm_rf_output_path, index=False)
            print(f"DVM Random Forest baseline results saved to: {dvm_rf_output_path}")
    else:
        print(f"Warning: DVM files not found")
    
    # ========== HAM Dataset ==========
    print("\n" + "="*60)
    print("PROCESSING HAM DATASET")
    print("="*60)
    
    ham_output_dir = os.path.join(base_dir, "HAM_augmented_df")
    ham_train_csv = os.path.join(ham_output_dir, "13_last_only_decorr_1e-05_0.8121.csv")
    ham_val_csv = os.path.join(ham_output_dir, "13_last_only_decorr_1e-05_0.8121_val.csv")
    
    if os.path.exists(ham_train_csv) and os.path.exists(ham_val_csv):
        ham_results, ham_rf_results = process_dataset_with_treefarms(
            ham_train_csv,
            ham_val_csv,
            ham_output_dir,
            "HAM",
            label_column="diagnosis_1",
            num_estimators_list=num_estimators_list,
            treefarms_config=treefarms_config
        )
        all_results['HAM'] = ham_results
        all_rf_results['HAM'] = ham_rf_results
        
        # Save HAM TreeFARMS results
        ham_output_path = os.path.join(ham_output_dir, "rashomon_set_accuracies_ham.csv")
        ham_results.to_csv(ham_output_path, index=False)
        print(f"\nHAM TreeFARMS results saved to: {ham_output_path}")
        
        # Save HAM Random Forest baseline results
        if not ham_rf_results.empty:
            ham_rf_output_path = os.path.join(ham_output_dir, "random_forest_baseline_ham.csv")
            ham_rf_results.to_csv(ham_rf_output_path, index=False)
            print(f"HAM Random Forest baseline results saved to: {ham_rf_output_path}")
    else:
        print(f"Warning: HAM files not found")
    
    # ========== HAM 7-Class Dataset ==========
    print("\n" + "="*60)
    print("PROCESSING HAM 7-CLASS DATASET")
    print("="*60)
    
    ham7_output_dir = os.path.join(base_dir, "HAM_7class_augmented_df")
    ham7_train_csv = os.path.join(ham7_output_dir, "13_last_only_0.7635.csv")
    ham7_val_csv = os.path.join(ham7_output_dir, "13_last_only_0.7635_val.csv")
    
    if os.path.exists(ham7_train_csv) and os.path.exists(ham7_val_csv):
        ham7_results, ham7_rf_results = process_dataset_with_treefarms(
            ham7_train_csv,
            ham7_val_csv,
            ham7_output_dir,
            "HAM_7class",
            label_column="diagnosis_2",
            num_estimators_list=num_estimators_list,
            treefarms_config=treefarms_config,
            multiclass=True
        )
        all_results['HAM_7class'] = ham7_results
        all_rf_results['HAM_7class'] = ham7_rf_results
        
        # Save HAM 7-class TreeFARMS results
        ham7_output_path = os.path.join(ham7_output_dir, "rashomon_set_accuracies_ham_7class.csv")
        ham7_results.to_csv(ham7_output_path, index=False)
        print(f"\nHAM 7-class TreeFARMS results saved to: {ham7_output_path}")
        
        # Save HAM 7-class Random Forest baseline results
        if not ham7_rf_results.empty:
            ham7_rf_output_path = os.path.join(ham7_output_dir, "random_forest_baseline_ham_7class.csv")
            ham7_rf_results.to_csv(ham7_rf_output_path, index=False)
            print(f"HAM 7-class Random Forest baseline results saved to: {ham7_rf_output_path}")
    else:
        print(f"Warning: HAM 7-class files not found")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF TREEFARMS RESULTS")
    print("="*60)
    
    for dataset_name, results_df in all_results.items():
        print(f"\n{dataset_name} TreeFARMS Results:")
        print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("SUMMARY OF RANDOM FOREST BASELINE RESULTS")
    print("="*60)
    
    for dataset_name, rf_results_df in all_rf_results.items():
        if not rf_results_df.empty:
            print(f"\n{dataset_name} Random Forest Baseline Results:")
            print(rf_results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("ALL PROCESSING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
