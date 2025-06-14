from src.run_app import *
from src.prepare_gam import *
from src.utils import *
from src.rset_opt import *
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.binarize_dataset import binarize_dataset
from gam_rs_utils.utils import *

from time import time

dataset = pd.read_csv("datasets/diabetes.csv")
df, thresholds, header, threshold_guess_time = binarize_dataset(dataset, 50)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

binned_X, binned_header = convert_cumulative_to_binned(X, header)
print(binned_X.shape)
print(binned_header)
print(X[0, :])
print(binned_X[0, :])