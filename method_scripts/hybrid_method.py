import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle as pkl
from typing import Dict, Any
from gam_rs_utils.binarize_dataset import binarize_dataset
from FasterRisk.src.fasterrisk import fasterrisk
from gam_rs_utils.utils import *
from src.rset_opt import *
from src.run_app import *
from base_method import BaseGAMRSetMethod
from results_class import MethodType
from time import time

class HybridMethod(BaseGAMRSetMethod):
    """
    Swapping method for finding models in the GAM Rashomon set.
    
    This method uses FasterRisk's swapping algorithm to find diverse models
    in the Rashomon set.
    """
    
    def __init__(self):
        """Initialize the swapping method."""
        super().__init__(MethodType.HYBRID)
    
    def run_single_dataset(self, dname: str, settings: Dict[str, Any]) -> Any:
        """
        Run the hybrid method on a single dataset.
        
        Args:
            dname: Dataset name
            settings: Dataset-specific settings
            
        Returns:
            Results object for this dataset
        """
        # Extract settings
        l0 = settings["l0"]
        l2 = settings["l2"]
        ne = settings["num_estimators"]
        gt = settings['m'] - 1.0
        n_support_set = settings['n_support_set']
        m = settings["m"]
        uniform_method = {"method": "uniform", "sample_from_surface": False}
        n_samples = 1000
        n_samples_to_keep = 10

        # Load and prepare data
        path = f'datasets/{dname}.csv'
        X_one_hot, y, header, header_new, sample_p = BaseGAMRSetMethod.get_binned_dataset(path, ne)
        X_one_hot_no_intercept = X_one_hot[:, 1:]  # remove intercept column

        # Run ellipsoid method to get starting solutions
        start = time()

        sparse_gam_file = prepare_sparse_gam(dname, l0, l2, m, X_one_hot, y, header, header_new)

        model = RSetOPT(sparse_gam_file)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        # Load data
        with open(sparse_gam_file, 'rb') as f:
            sparse_gam_data = pkl.load(f)
        
        # Get models from Rashomon set
        w_samples, rset = get_models_from_rset(
            sparse_gam_file, n_samples=n_samples, plot_shape=False, 
            sample_from_surface=uniform_method.get("sample_from_surface", False), 
            method=uniform_method
        )

        # Apply hard thresholding
        w_samples_zeroed = self.hard_threshold_samples(w_samples, rset, n_support_set)
        w_samples_zeroed = w_samples_zeroed[:n_samples_to_keep]

        # run swapping method on each model
        w_rset = []
        rs = fasterrisk.RiskScoreOptimizer(
            X_one_hot_no_intercept, y, 
            k=n_support_set, 
            lb=-100, ub=100, 
            gap_tolerance=gt, 
            select_top_m=-1, 
            maxAttempts=25
        )
        
        for w_sample in w_samples_zeroed:
            beta0 = w_sample[0]
            betas = w_sample[1:]
            rs.optimize_with_swaps_beam_search(swaps=5, beam_size=10, verbose=True, beta0=beta0, betas=betas)

            w_rset.append(np.concatenate([np.array([rs.opt_beta0]), rs.opt_betas]))
            
            # Add diverse pool solutions
            if rs.sparseDiversePool_beta0.size > 0:
                diverse_solutions = np.column_stack([rs.sparseDiversePool_beta0, rs.sparseDiversePool_betas])
                w_rset.extend([diverse_solutions[i] for i in range(diverse_solutions.shape[0])])

        end = time()

        # Extract results
        w_rset = np.vstack(w_rset)
        
        l2 = rs.lambda2
        rset_bound = rs.rset_bound
        
        # Print results summary
        self.print_results_summary(w_rset, sparse_gam_data['w_opt'], X_one_hot, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            dataset=dname,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            gap_tolerance=gt,
            w_rset=w_rset,
            w_opt=sparse_gam_data['w_opt'],
            rset_bound=rset_bound,
            predictions=self.get_predictions(X_one_hot, w_rset),
            runtime=end - start,
        )


if __name__ == "__main__":
    # Run the swapping method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = HybridMethod()
    method.run_all_datasets(dataset_settings) 