import numpy as np
import pandas as pd
import pickle as pkl
from src.run_app import *
from src.prepare_gam import *
from matplotlib import pyplot as plt
from src.utils import *

dname = "compas"
l0 = 0.001
l2 = 0.5
m = 1.01
betas_fastSparse = prepare_sparse_gam(dname, l0, l2, m)
filepath = "{}_{}_{}_{}.p".format(dname, l0, l2, m)

methods = [
    {"method": "uniform"},
    {"method": "poisson", "r_min": 0.1, "max_attempts": 1000},
]

for m in methods:
    print("Sampling method: ", m["method"])
    w_samples = get_models_from_rset(filepath, n_samples=100, plot_shape=True, sample_from_surface=False, method=m)
