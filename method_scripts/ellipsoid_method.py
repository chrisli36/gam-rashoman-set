import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle as pkl
from typing import Dict, Any, List
from src.prepare_gam import *
from src.rset_opt import *
from src.run_app import *
from gam_rs_utils.utils import *
from base_method import BaseGAMRSetMethod
from results_class import MethodType
from time import time

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
            {"method": "uniform", "sample_from_surface": False},
            {"method": "uniform", "sample_from_surface": True},
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
                
                # Run the method-specific implementation
                result_obj = self.run_single_dataset(dname, settings, method)
                self.results.append(result_obj)
            
            # Save results for this method
            extra = ''
            if method['method'] == 'poisson' and method['euclidean']:
                extra = 'euclidean'
            elif method['method'] == 'permutation' and method['poisson']:
                extra = 'poisson'
            elif method['method'] == 'uniform' and method['sample_from_surface']:
                extra = 'surface'
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
        n_samples = 1000
        
        if method["method"] == "poisson":
            method["r_min"] = settings["r_min"]
        
        # load and prepare data
        path = f'datasets/{dname}.csv'
        X_one_hot, y, header, header_new, _ = BaseGAMRSetMethod.get_binned_dataset(path, ne)
        
        # Prepare sparse GAM
        start = time()

        sparse_gam_file = prepare_sparse_gam(dname, l0, l2, m, X_one_hot, y, header, header_new)
        
        model = RSetOPT(sparse_gam_file)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        # Load data
        with open(sparse_gam_file, 'rb') as f:
            sparse_gam_data = pkl.load(f)
        
        X = sparse_gam_data["X"]
        y = sparse_gam_data["y"]
        header = sparse_gam_data["header_new"]
        sample_p = sparse_gam_data["sample_proportion"]
        
        # Get models from Rashomon set
        w_samples, rset = get_models_from_rset(
            sparse_gam_file, n_samples=n_samples, plot_shape=False, 
            sample_from_surface=method.get("sample_from_surface", False), 
            method=method
        )
        
        # Apply hard thresholding
        w_samples_zeroed = self.hard_threshold_samples(w_samples, rset, n_support_set)
        
        end = time()
        
        # Print results summary
        self.print_results_summary(w_samples_zeroed, sparse_gam_data['w_opt'], X, y, l2, sample_p, end - start)
        
        print(sparse_gam_data['w_opt'].shape, sparse_gam_data['w_opt'])
        print(len(header_new), header_new)
        print(X_one_hot.shape)
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
            predictions=self.get_predictions(X, w_samples_zeroed),
            runtime=end - start,
        )

if __name__ == "__main__":
    # Run the ellipsoid method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = EllipsoidMethod()
    method.run_all_datasets(dataset_settings) 