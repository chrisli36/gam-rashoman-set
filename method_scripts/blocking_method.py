import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
from typing import Dict, Any
from src.prepare_gam import *
from src.rset_opt import *
from gam_rs_utils.utils import *
from blocking.blocking import optimize_support
from base_method import BaseGAMRSetMethod
from results import MethodType, Results
from time import time

class BlockingMethod(BaseGAMRSetMethod):
    """
    Blocking method for finding models in the GAM Rashomon set.
    
    This method uses blocking optimization to find diverse models
    in the Rashomon set.
    """
    
    def __init__(self):
        """Initialize the blocking method."""
        super().__init__(MethodType.BLOCKING)
    
    def run_dataset(self, dname: str, n_samples: int = 100, l0: float = None, l2: float = None, 
                          m: float = None, num_estimators: int = None, n_support_set: int = None, **kwargs) -> Any:
        """
        Run the blocking method on a single dataset.
        
        Args:
            dname: Dataset name
            n_samples: Number of samples to generate (used as n_combs_max)
            l0: L0 regularization parameter
            l2: L2 regularization parameter
            m: Margin parameter
            num_estimators: Number of estimators
            n_support_set: Number of support features
            
        Returns:
            Results object for this dataset
        """
        # Use parameters directly
        ne = num_estimators
        
        # Prepare sparse GAM
        start = time()
        
        # Create or load binarized dataset
        binarized_data = Results.create_binarized_dataset(dname, ne)
        X_one_hot = binarized_data['X']
        y = binarized_data['y']
        header = binarized_data['header']
        header_new = binarized_data['header_new']
        sparse_gam = prepare_sparse_gam(dname, l0, l2, m, X_one_hot, y, header, header_new)
        
        model = RSetOPT(sparse_gam)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        baseline_gam = optimize_support(sparse_gam, n_support_set, n_combs_max=n_samples)
        
        # Load data
        with open(sparse_gam, "rb") as f:
            sparse_gam_data = pickle.load(f)
        with open(baseline_gam, 'rb') as f:
            baseline_gam_data = pickle.load(f)
        
        X = sparse_gam_data["X"]
        y = sparse_gam_data["y"]
        w_opt = sparse_gam_data["w_opt"]
        sample_p = sparse_gam_data["sample_proportion"]
        rset_bound = sparse_gam_data["rset_bound"]
        
        indices = baseline_gam_data["indices"]
        w_center_block = baseline_gam_data["w_center_block"]
        
        end = time()
        
        # Process results
        predictions = np.zeros((X.shape[0], len(indices)))
        w_samples = np.zeros((len(indices), len(w_opt)))
        
        for support in range(len(indices)):
            new_X = self.get_new_X(indices[support], X)
            w_sample = w_center_block[support]
            
            logit = new_X @ w_sample
            y_pred = np.exp(logit) / (1 + np.exp(logit))
            y_pred = np.where(y_pred > 0.5, 1, -1)
            predictions[:, support] = y_pred
            
            true_w_sample = self.get_true_w_sample(indices[support], w_sample)
            w_samples[support] = true_w_sample
        
        # Verify predictions match
        try:
            assert np.allclose(predictions, ModelUtils.get_predictions(X, w_samples))
        except AssertionError:
            print("predictions:")
            print(predictions)
            print("get_predictions output:")
            print(ModelUtils.get_predictions(X, w_samples))
            raise
        
        # Print results summary
        ModelUtils.print_results_summary(w_samples, w_opt, X, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            method_type=MethodType.BLOCKING,
            dataset=dname,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            n_samples=n_samples,
            w_rset=w_samples,
            w_opt=w_opt,
            rset_bound=rset_bound,
            runtime=end - start,
        )

    def get_true_w_sample(self, indices: List[Tuple[int, int]], w_sample: List[float]) -> np.ndarray:
        """
        Adjusts a weight sample vector based on merged feature indices.
        Args:
            indices: List of (start, end) tuples for merged features.
            w_sample: List of weights.
        Returns:
            Adjusted numpy array of weights.
        """
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

    def get_new_X(self, indices: List[Tuple[int, int]], X: np.ndarray) -> np.ndarray:
        """
        Merges columns in X according to provided indices.
        Args:
            indices: List of (start, end) tuples for columns to merge.
            X: 2D numpy array of features.
        Returns:
            2D numpy array with merged columns.
        """
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

if __name__ == "__main__":
    # Run the blocking method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = BlockingMethod()
    method.run_all_datasets(dataset_settings) 