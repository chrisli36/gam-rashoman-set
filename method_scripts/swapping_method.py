import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Dict, Any
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.utils import *
from base_method import BaseGAMRSetMethod
from results_class import MethodType
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
    
    def run_single_dataset(self, dname: str, settings: Dict[str, Any]) -> Any:
        """
        Run the swapping method on a single dataset.
        
        Args:
            dname: Dataset name
            settings: Dataset-specific settings
            
        Returns:
            Results object for this dataset
        """
        # Extract settings
        ne = settings["num_estimators"]
        gt = settings['m'] - 1.0
        n_support_set = settings['n_support_set']
        
        # Load and prepare data
        path = f'datasets/{dname}.csv'
        X_one_hot, y, header, header_new, sample_p = self.get_binned_dataset(path, ne)
        X_one_hot_no_intercept = X_one_hot[:, 1:]  # remove intercept column
        
        # Run swapping algorithm
        start = time()
        
        rs = fasterrisk.RiskScoreOptimizer(
            X_one_hot_no_intercept, y, 
            k=n_support_set, 
            lb=-100, ub=100, 
            gap_tolerance=gt, 
            select_top_m=-1, 
            maxAttempts=25
        )
        rs.optimize_with_swaps_beam_search(swaps=5, beam_size=100, verbose=True)
        
        end = time()
        
        # Extract results
        w_opt = np.concatenate([np.array([rs.opt_beta0]), rs.opt_betas])
        w_rset = np.column_stack([rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas])
        
        l2 = rs.lambda2
        rset_bound = rs.rset_bound
        
        # Print results summary
        self.print_results_summary(w_rset, w_opt, X_one_hot, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            dataset=dname,
            l2=l2,
            n_estimators=ne,
            n_support_set=n_support_set,
            gap_tolerance=gt,
            w_rset=w_rset,
            w_opt=w_opt,
            rset_bound=rset_bound,
            predictions=self.get_predictions(X_one_hot, w_rset),
            runtime=end - start,
        )


if __name__ == "__main__":
    # Run the swapping method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = SwappingMethod()
    method.run_all_datasets(dataset_settings) 