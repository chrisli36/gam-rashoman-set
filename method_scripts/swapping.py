import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Any
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.utils import *
from method_scripts.base_method import BaseGAMRSetMethod
from method_scripts.results import MethodType, Results
from time import time


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
            {"k": 1},
            {"k": 2},
            {"k": 3},
            {"k": 4},
            {"k": 5}
        ]

    def run_dataset(self, dn: str, eps: float, n_samples: int = 100, l0: float = None, l2: float = None, 
                          m: float = None, ne: int = None, k: int = 3, **kwargs) -> Any:
        """
        Run the swapping method on a single dataset.
        
        Args:
            dn: Dataset name
            eps: Epsilon parameter for the rset bound
            n_samples: Number of samples to generate (used as beam_size)
            l0: L0 regularization parameter
            l2: L2 regularization parameter
            m: Multiplier parameter
            ne: Number of estimators
            k: Number of swaps

        Returns:
            Results object for this dataset
        """
        # Create or load dataset
        fastsparse_data = Results.create_fastsparse_dataset(dn, l0, l2)
        bin_X = fastsparse_data['bin_X']
        bin_header = fastsparse_data['bin_header']
        cum_X = fastsparse_data['cum_X']
        cum_header = fastsparse_data['cum_header']
        y = fastsparse_data['y']
        w = fastsparse_data['w']
        
        # Run swapping algorithm
        start = time()
        
        rs = fasterrisk.RiskScoreOptimizer(
            cum_X[:,1:], y, 
            k=w.nonzero()[0].shape[0]-1, 
            lb=-100, ub=100, 
            gap_tolerance=m - 1.0, 
            select_top_m=-1, 
            maxAttempts=25
        )
        rs.optimize_with_swaps_beam_search(
            swaps=k, 
            beam_size=n_samples, 
            verbose=True, 
            beta0=w[0], 
            betas=w[1:]
        )
        
        end = time()
        
        # Extract results
        w_opt = np.concatenate([np.array([rs.opt_beta0]), rs.opt_betas])
        w_rset = np.column_stack([rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas])

        w_opt_binned = ModelUtils.convert_cumulative_to_binned(np.array([w_opt]), cum_header)[0]
        w_rset_binned = ModelUtils.convert_cumulative_to_binned(w_rset, cum_header)

        for i in range(w_rset.shape[0]):
            if not np.allclose(cum_X @ w_rset[i], bin_X @ w_rset_binned[i]):
                print(f"{RED}not equal: model {i}{RESET}")
        if not np.allclose(cum_X @ w_opt, bin_X @ w_opt_binned):
            print(f"{RED}not equal: opt model{RESET}")
        
        # filter w_rset_binned to only include models that are in the rset
        sample_p = Results.get_sample_proportion(bin_X)
        rset_indices = []
        for i in range(w_rset_binned.shape[0]):
            log_loss = ModelUtils.get_loss_one_model(bin_X, y, w_rset_binned[i], sample_p=sample_p, loss_type="logistic", l2=l2)
            if log_loss > eps:
                continue
            rset_indices.append(i)
        w_rset_binned = w_rset_binned[rset_indices]
        
        # Print results summary
        ModelUtils.print_results_summary(
            w_rset_binned, w_opt_binned, 
            bin_X, y, l2, 
            end - start
        )
        
        # Create and return result object
        return self.create_result_object(
            method_type=MethodType.SWAPPING,
            dataset=dn,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            k=k,
            n_support_set=w.nonzero()[0].shape[0]-1,
            n_samples=w_rset.shape[0],
            w_rset=w_rset_binned,
            w_opt=w_opt_binned,
            rset_bound=eps,
            runtime=end - start,
        )


if __name__ == "__main__":
    # Run the swapping method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = SwappingMethod()
    method.run_all_datasets(dataset_settings) 