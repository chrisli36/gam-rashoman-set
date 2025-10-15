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
from results_class import MethodType, Results
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
            # {"method": "uniform", "sample_from_surface": False},
            # {"method": "uniform", "sample_from_surface": True},
            {"method": "poisson", "max_attempts": 100_000, "rejection": "predictive_diversity"},
            # {"method": "poisson", "max_attempts": 100_000, "rejection": "mahalanobis"},
            # {"method": "poisson", "max_attempts": 100_000, "rejection": "euclidean"},
            # {"method": "permutation", "n_base_points": 10, "n_sign_samples": 100, "poisson": False},
            # {"method": "permutation", "n_base_points": 10, "n_sign_samples": 100, "poisson": True},
        ]
    
    def run_all_datasets(self, dataset_settings: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Run the ellipsoid method on all datasets with multiple sampling strategies.
        
        Args:
            dataset_settings: List of (dataset_name, settings) tuples
        """
        for method in self.methods:
            self.results = []  # Reset results for each method

            if method['method'] == 'poisson':
                # for r_min_multiplier in [0.01, 0.1, 0.2, 0.3, 0.4]:
                for r_min_multiplier in [0.5, 0.6, 0.7, 0.8]:
                    self.results = []
                    for dname, settings in dataset_settings:
                        print(f"{BLUE}Dataset: {dname}, method: {method['method']} - {method['rejection']}, r_min: {r_min_multiplier}{RESET}")
                        result_obj = self.run_dataset(dname, method=method, r_min=r_min_multiplier, **settings)
                        self.results.append(result_obj)

                        filename = f"analysis/results/methods/poisson_r_min_{r_min_multiplier}_ellipsoid_sampling.pkl"
                        self.save_results(filename)
                continue
            
            for dname, settings in dataset_settings:
                print(f"{BLUE}Dataset: {dname}, method: {method['method']}{RESET}")
                
                # Run the method-specific implementation
                result_obj = self.run_dataset(dname, settings, method)
                self.results.append(result_obj)
            
                # Save results for this method
                extra = ''
                if method['method'] == 'poisson':
                    extra = f"{method['rejection']}_r_min_{r_min_multiplier}"
                elif method['method'] == 'permutation' and method['poisson']:
                    extra = 'poisson'
                elif method['method'] == 'uniform' and method['sample_from_surface']:
                    extra = 'surface'
                filename = f"analysis/results/methods/{method['method']}_{extra}_ellipsoid_sampling.pkl"
                self.save_results(filename)
    
    def run_dataset(self, dname: str, n_samples: int = 1000, method: Dict[str, Any] = None, 
                          r_min: float = None, l0: float = None, l2: float = None, m: float = None, 
                          num_estimators: int = None, n_support_set: int = None, **kwargs) -> Any:
        """
        Run the ellipsoid method on a single dataset with a specific sampling strategy.
        
        Args:
            dname: Dataset name
            n_samples: Number of samples to generate
            method: Sampling method configuration
            r_min: Optional r_min multiplier for poisson method
            l0: L0 regularization parameter
            l2: L2 regularization parameter
            m: Margin parameter
            num_estimators: Number of estimators
            n_support_set: Number of support features
            **kwargs: Additional unused parameters
            
        Returns:
            Results object for this dataset
        """
        ne = num_estimators
        
        # if method is None:
        #     method = {"method": "poisson", "max_attempts": 100_000, "rejection": "predictive_diversity"}
        
        # if method["method"] == "poisson" and "r_min" in kwargs:
        #     method["r_min"] = kwargs["r_min"]

        # if r_min is not None:
        #     method["r_min"] = r_min
        
        # Create or load binarized dataset
        binarized_data = Results.create_binarized_dataset(dname, ne)
        X_one_hot = binarized_data['X']
        y = binarized_data['y']
        header = binarized_data['header']
        header_new = binarized_data['header_new']
        
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
        w_samples_zeroed = ModelUtils.hard_threshold_samples(w_samples, rset, n_support_set)
        
        end = time()
        
        # Print results summary
        ModelUtils.print_results_summary(w_samples_zeroed, sparse_gam_data['w_opt'], X, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            dataset=dname,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            n_samples=n_samples,
            w_rset=w_samples_zeroed,
            w_opt=sparse_gam_data['w_opt'],
            rset_bound=rset.rset_bound,
            predictions=ModelUtils.get_predictions(X, w_samples_zeroed),
            runtime=end - start,
        )

if __name__ == "__main__":
    # Run the ellipsoid method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = EllipsoidMethod()
    method.run_all_datasets(dataset_settings) 