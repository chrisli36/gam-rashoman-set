import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
from time import time
from typing import Dict, Any
from src.prepare_gam import *
from src.rset_opt import *
from gam_rs_utils.utils import *
from blocking_method.blocking import optimize_support
from base_method import BaseGAMRSetMethod
from results_class import MethodType


class BlockingMethod(BaseGAMRSetMethod):
    """
    Blocking method for finding models in the GAM Rashomon set.
    
    This method uses blocking optimization to find diverse models
    in the Rashomon set.
    """
    
    def __init__(self):
        """Initialize the blocking method."""
        super().__init__(MethodType.BLOCKING)
    
    def run_single_dataset(self, dname: str, settings: Dict[str, Any]) -> Any:
        """
        Run the blocking method on a single dataset.
        
        Args:
            dname: Dataset name
            settings: Dataset-specific settings
            
        Returns:
            Results object for this dataset
        """
        # Extract settings
        l0 = settings["l0"]
        l2 = settings["l2"]
        m = settings["m"]
        ne = settings["num_estimators"]
        n_support_set = settings["n_support_set"]
        binned = True
        
        # Prepare sparse GAM
        start = time()
        
        sparse_gam = prepare_sparse_gam(dname, l0, l2, m, ne, binned)
        
        model = RSetOPT(sparse_gam)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        baseline_gam = optimize_support(sparse_gam, n_support_set)
        
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
            new_X = get_new_X(indices[support], X)
            w_sample = w_center_block[support]
            
            logit = new_X @ w_sample
            y_pred = np.exp(logit) / (1 + np.exp(logit))
            y_pred = np.where(y_pred > 0.5, 1, -1)
            predictions[:, support] = y_pred
            
            true_w_sample = get_true_w_sample(indices[support], w_sample)
            w_samples[support] = true_w_sample
        
        # Verify predictions match
        try:
            assert np.allclose(predictions, get_predictions(X, w_samples))
        except AssertionError:
            print("predictions:")
            print(predictions)
            print("get_predictions output:")
            print(get_predictions(X, w_samples))
            raise
        
        # Print results summary
        self.print_results_summary(w_samples, w_opt, X, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            dataset=dname,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            w_rset=w_samples,
            w_opt=w_opt,
            rset_bound=rset_bound,
            predictions=predictions,
            runtime=end - start,
        )


if __name__ == "__main__":
    # Run the blocking method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = BlockingMethod()
    method.run_all_datasets(dataset_settings) 