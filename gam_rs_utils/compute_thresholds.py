import numpy as np
import pandas as pd
import time
from sklearn.ensemble import GradientBoostingClassifier

# fit the tree using gradient boosted classifier
seed = 3
def fit_boosted_tree(X, y, n_est=10, lr=0.1, d=1):
    clf = GradientBoostingClassifier(loss='exponential', learning_rate=lr, n_estimators=n_est, max_depth=d,
                                     random_state=seed)
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

def get_thresholds(X, y, n_est, lr, d, backselect=False,random_selection = False,\
                   max_thresholds_per_feat = 10):

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
            min_val = min(X.iloc[:,j])
            max_val = max(X.iloc[:,j])
            for i in range(num_features):
                t = np.random.uniform(min_val,max_val)
                tj.append(t)
            tj = sorted(tj)
            thresholds.append(tj)
        X_new = cut(X, thresholds)
        return X_new, thresholds, X_new.columns

# compute the thresholds

def compute_thresholds(X, y, n_est, max_depth,random_selection = False,max_thresholds_per_feat=10):
    # n_est, max_depth: GBDT parameters
    # set LR to 0.1
    lr = 0.1
    start = time.perf_counter()
    X, thresholds, header = get_thresholds(
        X, y, n_est, lr, max_depth, backselect=False,random_selection = random_selection, max_thresholds_per_feat = max_thresholds_per_feat)
    guess_time = time.perf_counter()-start

    return X, thresholds, header, guess_time

