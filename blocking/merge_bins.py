import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pickle
import os
from src.rset_app import *
from src.rset_opt import *
from sklearn.linear_model import LogisticRegression
import torch
import random
import time

with open("blocking_method/compas_0.0005_0.001_1.01_merge_bins_15.p", "rb") as f:
    out = pickle.load(f)

with open("blocking_method/compas_0.0005_0.001_1.01.p", "rb") as f:
    res = pickle.load(f)

print("number of different support sets:", len(out["indices"]))

# find the first support set

# out["indices"][i] stores bins that need to be merged for support i
print(out["indices"][0])

for i in out["indices"][0]:
    print([res["header_new"][k] for k in i])

idx = 1
for i in out["indices"][0]:
    print(res["header_new"][idx:i[0]])
    print("merge bins", res["header_new"][i[0]:i[1]+1])
    idx = i[1]+1
print(res["header_new"][i[1]+1:])

# Given this support set, its Rashomon set parameters are 
out["hessian_block"][0].shape

print(out["w_center_block"][0].shape)
out["w_center_block"][0]

# following sampling function is slightly different from the one in rset_opt.py
def sample_in_subset_ellipsoid(H, w, ub, n_samples):
    d = w.shape[0]
    u = np.random.normal(size=(n_samples,d)) # randomly sample iid gaussian
    u = u/(np.linalg.norm(u,axis=1).reshape(-1,1)) # normalize to get uniformly random unit vectors
    r = (np.random.random(size=n_samples))**(1/d) # sample radius (uniformly in a sphere)
    x_ = u * r.reshape(-1,1) # x_ is a uniformly random point in a sphere
    
    lamb, V = np.linalg.eigh(H) # eigen decomposition
    a = np.sqrt(ub/lamb) # scaling factor
    dw_samples = ((a*V) @ x_.T).T # transformation to a ellipsoid
    w_samples = dw_samples + w

    return w_samples

from gam_rs_utils.utils import *
import re

support = 1
X = res['X']

w_samples = sample_in_subset_ellipsoid(out["hessian_block"][support], 
                                       out["w_center_block"][support], 
                                       out["ub_block"][support], 
                                       n_samples=100)

new_X = []
col_pointer = 0
for i, j in out['indices'][support]:
    if i > col_pointer:
        new_X.append(X[:, col_pointer:i])
    merged = np.max(X[:, i:j+1], axis=1, keepdims=True)
    new_X.append(merged)
    col_pointer = j + 1
if col_pointer < X.shape[1]:
    new_X.append(X[:, col_pointer:])
new_X = np.hstack(new_X)

regex = r"(?:(.*)<)?(.*)<=(.*)"
header = res['header_new']
new_header = []
col_pointer = 0
for i, j in out['indices'][support]:
    if i > col_pointer:
        new_header.extend(header[col_pointer:i])
    start, feature, _ = re.match(regex, header[i]).groups()
    _, _, end = re.match(regex, header[j]).groups()
    new_header.append(f"{feature}<={end}" if start is not None else f"{feature}<={end}")
    col_pointer = j + 1
if col_pointer < len(header):
    new_header.extend(header[col_pointer:])
new_header = np.array(new_header)
print(new_header)

feature_to_vi = get_variable_importance(new_X, w_samples, new_header, True)
plot_variable_importance(feature_to_vi)

plot_gam(new_header[1:], w_samples[:, 1:])