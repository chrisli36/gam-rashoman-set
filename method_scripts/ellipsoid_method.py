import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle as pkl
from time import time
from typing import Dict, Any, List
from src.prepare_gam import *
from src.rset_opt import *
from src.run_app import *
from gam_rs_utils.utils import *
from base_method import BaseGAMRSetMethod
from results_class import MethodType


def hard_threshold(x: np.ndarray, k2: int) -> np.ndarray:
    """Apply hard thresholding to keep only the largest k2 elements."""
    x = x.copy()
    small_indices = np.argsort(np.abs(x))[:k2]
    x[small_indices] = 0
    return x


class EllipsoidMethod(BaseGAMRSetMethod):
    """
    Ellipsoid sampling method for finding models in the GAM Rashomon set.
    
    This method uses various sampling strategies on the ellipsoid surface
    to find diverse models in the Rashomon set.
    """
    
    def __init__(self):
        """Initialize the ellipsoid method."""
        super().__init__(MethodType.ELLIPSOID)
        self.methods = [
            {"method": "uniform"},
            {"method": "poisson", "max_attempts": 100_000, "euclidean": False},
            {"method": "poisson", "max_attempts": 100_000, "euclidean": True},
            {"method": "permutation", "n_base_points": 10, "n_sign_samples": 100, "poisson": False},
            {"method": "permutation", "n_base_points": 10, "n_sign_samples": 100, "poisson": True},
        ]
    
    def run_all_datasets(self, dataset_settings: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Run the ellipsoid method on all datasets with multiple sampling strategies.
        
        Args:
            dataset_settings: List of (dataset_name, settings) tuples
        """
        for method in self.methods:
            self.results = []  # Reset results for each method
            
            for dname, settings in dataset_settings:
                print(f"{BLUE}Dataset: {dname}, method: {method['method']}{RESET}")
                
                # Extract common settings
                ne = settings["num_estimators"]
                n_support_set = settings["n_support_set"]
                
                # Run the method-specific implementation
                result_obj = self.run_single_dataset(dname, settings, method)
                self.results.append(result_obj)
            
            # Save results for this method
            extra = ''
            if method['method'] == 'poisson' and method['euclidean']:
                extra = 'euclidean'
            elif method['method'] == 'permutation' and method['poisson']:
                extra = 'poisson'
            filename = f"analysis/results/methods/{method['method']}_{extra}_ellipsoid_sampling.pkl"
            self.save_results(filename)
    
    def run_single_dataset(self, dname: str, settings: Dict[str, Any], method: Dict[str, Any]) -> Any:
        """
        Run the ellipsoid method on a single dataset with a specific sampling strategy.
        
        Args:
            dname: Dataset name
            settings: Dataset-specific settings
            method: Sampling method configuration
            
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
        n_samples = 100
        max_attempts = 10_000
        
        if method["method"] == "poisson":
            method["r_min"] = settings["r_min"]
        
        filepath = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"
        
        # Prepare sparse GAM
        start = time()
        
        sparse_gam = prepare_sparse_gam(dname, l0, l2, m, num_estimators=ne, binned=binned)
        
        model = RSetOPT(sparse_gam)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        # Load data
        sparse_gam_file = f"models/{dname}_{l0}_{l2}_{m}_{binned}.p"
        with open(sparse_gam_file, 'rb') as f:
            sparse_gam_data = pkl.load(f)
        
        X = sparse_gam_data["X"]
        y = sparse_gam_data["y"]
        header = sparse_gam_data["header_new"]
        sample_p = sparse_gam_data["sample_proportion"]
        
        # Get models from Rashomon set
        w_samples, rset = get_models_from_rset(
            filepath, n_samples=max_attempts, plot_shape=False, 
            sample_from_surface=False, method=method
        )
        
        # Apply hard thresholding
        w_samples_zeroed = []
        n_support = w_samples.shape[1]
        k2 = n_support - 1 - n_support_set
        attempts = 0
        
        while len(w_samples_zeroed) < n_samples and attempts < max_attempts and attempts < w_samples.shape[0]:
            w_samp = w_samples[attempts]
            w_samp_zeroed = hard_threshold(w_samp, k2)
            if rset.in_rset(w_samp_zeroed):
                w_samples_zeroed.append(w_samp_zeroed)
            attempts += 1
        
        print(f"\tout of {attempts} attempts, kept {len(w_samples_zeroed)} after hard thresholding")
        
        if len(w_samples_zeroed) == 0:
            w_samples_zeroed = np.array([])
        else:
            w_samples_zeroed = np.vstack(w_samples_zeroed)
        
        end = time()
        
        # Print results summary
        self.print_results_summary(w_samples_zeroed, sparse_gam_data['w_opt'], X, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            dataset=dname,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            w_rset=w_samples_zeroed,
            w_opt=sparse_gam_data['w_opt'],
            rset_bound=rset.rset_bound,
            predictions=get_predictions(X, w_samples_zeroed),
            runtime=end - start,
        )


if __name__ == "__main__":
    # Run the ellipsoid method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = EllipsoidMethod()
    method.run_all_datasets(dataset_settings) 