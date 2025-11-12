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
from results import MethodType, Results
from time import time
import itertools

class EllipsoidMethod(BaseGAMRSetMethod):
    """
    Ellipsoid sampling method for finding models in the GAM Rashomon set.
    
    This method uses various sampling strategies on the ellipsoid surface
    to find diverse models in the Rashomon set.
    """
    
    def __init__(self):
        """Initialize the ellipsoid method."""
        super().__init__(MethodType.ELLIPSOID)
        extra = {
            "sampling": ["uniform", "surface"], # , "permutation"],
            "distance_metric": ['euclidean', 'mahalanobis'], # , 'predictive'],
        }
        extra_settings = []
        for settings in itertools.product(*extra.values()):
            extra_settings.append(dict(zip(extra.keys(), settings)))
        self.extra_settings = extra_settings
    
    def run_dataset(self, dn: str, n_samples: int = 1_000, l0: float = None, l2: float = None, 
                      m: float = None, num_estimators: int = None, n_support_set: int = None,
                      sampling: str = "uniform", distance_metric: Optional[str] = None, 
                      r_min: float = None, **kwargs) -> Any:
        """
        Run the ellipsoid method on a single dataset with a specific sampling strategy.
        
        Args:
            dname: Dataset name
            n_samples: Number of samples to generate
            l0: L0 regularization parameter
            l2: L2 regularization parameter
            m: Margin parameter
            num_estimators: Number of estimators
            n_support_set: Number of support features
            sampling: Sampling method
            distance_metric: Distance metric
            r_min: Minimum distance from the original model for rejection sampling
            **kwargs: Additional unused parameters
            
        Returns:
            Results object for this dataset
        """
        ne = num_estimators
        
        # Create or load binarized dataset
        data = pd.read_csv(f"datasets/{dn}.csv")
        fastsparse_data = Results.create_fastsparse_dataset(dn, l0, l2)
        # X = fastsparse_data['X']
        y = fastsparse_data['y']
        header = fastsparse_data['header']
        w = fastsparse_data['w']

        X_new, header_new = utils.binary_to_one_hot(data.iloc[:,:-1], w, header)
        
        # Prepare sparse GAM
        start = time()

        sparse_gam_file = prepare_sparse_gam(dn, l0, l2, m, X_new, y, header, header_new)
        
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
            sampling=sampling, distance_metric=distance_metric, r_min=r_min,
        )
        
        # # Apply hard thresholding
        # w_samples_zeroed = ModelUtils.hard_threshold_samples(w_samples, rset, n_support_set)
        
        end = time()
        
        # Print results summary
        # ModelUtils.print_results_summary(w_samples_zeroed, sparse_gam_data['w_opt'], X, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            method_type=MethodType.ELLIPSOID,
            dataset=dn,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            n_samples=n_samples,
            w_rset=w_samples,
            w_opt=sparse_gam_data['w_opt'],
            rset_bound=rset.rset_bound,
            runtime=end - start,
            sampling=sampling,
            distance_metric=distance_metric,
        )

if __name__ == "__main__":
    # Run the ellipsoid method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = EllipsoidMethod()
    method.run_all_datasets(dataset_settings) 