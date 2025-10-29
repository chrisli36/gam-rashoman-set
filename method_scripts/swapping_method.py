import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Dict, Any
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.utils import *
from src.rset_opt import *
from base_method import BaseGAMRSetMethod
from results_class import MethodType, Results
from time import time
import pickle as pkl


class SwappingMethod(BaseGAMRSetMethod):
    """
    Swapping method for finding models in the GAM Rashomon set.
    
    This method uses FasterRisk's swapping algorithm to find diverse models
    in the Rashomon set.
    """

    def __init__(self):
        """Initialize the swapping method."""
        super().__init__(MethodType.SWAPPING)
        self.extra_settings = [
            {"k": 3},
            {"k": 4},
            {"k": 5}
        ]

    def run_dataset(self, dname: str, n_samples: int = 100, l0: float = None, l2: float = None, 
                          m: float = None, num_estimators: int = None, n_support_set: int = None, 
                          k: int = 3, **kwargs) -> Any:
        """
        Run the swapping method on a single dataset.
        
        Args:
            dname: Dataset name
            n_samples: Number of samples to generate (used as beam_size)
            l0: L0 regularization parameter
            l2: L2 regularization parameter
            m: Margin parameter
            num_estimators: Number of estimators
            n_support_set: Number of support features
            k: Number of swaps

        Returns:
            Results object for this dataset
        """
        ne = num_estimators

        # Create or load binarized dataset
        binarized_data = Results.create_binarized_dataset(dname, ne)
        X_one_hot = binarized_data['X']
        y = binarized_data['y']
        header = binarized_data['header']
        header_new = binarized_data['header_new']
        sample_p = binarized_data['sample_proportion']
        X_one_hot_no_intercept = X_one_hot[:, 1:]  # remove intercept column

        # Get starting solution
        sparse_gam_file = prepare_sparse_gam(dname, l0, l2, m, X_one_hot, y, header, header_new)
        
        model = RSetOPT(sparse_gam_file)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        with open(sparse_gam_file, 'rb') as f:
            sparse_gam_data = pkl.load(f)
        w_orig = sparse_gam_data["w_opt"]
        
        k2 = w_orig.shape[0] - 1 - n_support_set
        w_orig_zeroed = np.concatenate([np.array([w_orig[0]]), self.hard_threshold(w_orig[1:], k2)])
        
        # Run swapping algorithm
        start = time()
        
        rs = fasterrisk.RiskScoreOptimizer(
            X_one_hot_no_intercept, y, 
            k=n_support_set, 
            lb=-100, ub=100, 
            gap_tolerance=m - 1.0, 
            select_top_m=-1, 
            maxAttempts=25
        )
        rs.optimize_with_swaps_beam_search(
            swaps=k, 
            beam_size=n_samples, 
            verbose=True, 
            beta0=w_orig_zeroed[0], 
            betas=w_orig_zeroed[1:]
        )
        
        end = time()
        
        # Extract results
        w_opt = np.concatenate([np.array([rs.opt_beta0]), rs.opt_betas])
        w_rset = np.column_stack([rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas])
        
        l2 = rs.lambda2
        rset_bound = rs.rset_bound
        
        # Print results summary
        ModelUtils.print_results_summary(w_rset, w_opt, X_one_hot, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            method_type=MethodType.SWAPPING,
            dataset=dname,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            n_samples=n_samples,
            w_rset=w_rset,
            w_opt=w_opt,
            rset_bound=rset_bound,
            runtime=end - start,
        )


if __name__ == "__main__":
    # Run the swapping method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = SwappingMethod()
    method.run_all_datasets(dataset_settings) 