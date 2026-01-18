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
from method_scripts.base_method import BaseGAMRSetMethod
from method_scripts.results import MethodType, Results
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
            "sampling": ["uniform", "surface", "permutation"],
            "distance_metric": [None, 'euclidean', 'mahalanobis', 'predictive'],
        }
        extra_settings = []
        for settings in itertools.product(*extra.values()):
            extra_settings.append(dict(zip(extra.keys(), settings)))
        self.extra_settings = extra_settings
    
    def run_dataset(self, dn: str, n_samples: int = 1_000, l0: float = None, l2: float = None, 
                      eps: float = None, ne: int = None, n_support_set: int = None,
                      sampling: str = "uniform", distance_metric: Optional[str] = None, 
                      r_min: float = None, **kwargs) -> Any:
        """
        Run the ellipsoid method on a single dataset with a specific sampling strategy.
        
        Args:
            dname: Dataset name
            n_samples: Number of samples to generate
            l0: L0 regularization parameter
            l2: L2 regularization parameter
            eps: Epsilon parameter for the rset bound
            num_estimators: Number of estimators
            n_support_set: Number of support features
            sampling: Sampling method
            distance_metric: Distance metric
            r_min: Minimum distance from the original model for rejection sampling
            **kwargs: Additional unused parameters
            
        Returns:
            Results object for this dataset
        """
        # Create or load dataset
        data = pd.read_csv(f"datasets/{dn}.csv")
        fastsparse_data = Results.create_fastsparse_dataset(dn, l0, l2)
        bin_X = fastsparse_data['bin_X']
        cum_header = fastsparse_data['cum_header']
        y = fastsparse_data['y']
        w = fastsparse_data['w']

        # Prepare sparse GAM
        start = time()

        # extract sparse X and header from fastsparse w
        sparse_X, sparse_header = utils.binary_to_one_hot(data.iloc[:,:-1], w, cum_header)

        # fit sparse GAM
        sparse_gam_file = prepare_sparse_gam(dn, l0, l2, eps, sparse_X, y, cum_header, sparse_header)
        
        model = RSetOPT(sparse_gam_file)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        # Load data
        with open(sparse_gam_file, 'rb') as f:
            sparse_gam_data = pkl.load(f)
        m = sparse_gam_data['multiplier']
        
        # Get models from Rashomon set
        w_samples, rset = get_models_from_rset(
            sparse_gam_file, n_samples=n_samples, plot_shape=False, 
            sampling=sampling, distance_metric=distance_metric, r_min=r_min, eps=eps
        )
        assert eps == rset.rset_bound

        # expand w_samples to match the full header
        header_object = ModelUtils.get_header_object(cum_header)
        sparse_header_object = ModelUtils.get_header_object(sparse_header)
        w_samples = ModelUtils.expand_w_samples(w_samples, sparse_header_object, header_object)

        # expand w_opt to match the full header
        w_opt = ModelUtils.expand_w(sparse_gam_data['w_opt'], sparse_header_object, header_object)
        
        # # Apply hard thresholding
        # w_samples_zeroed = ModelUtils.hard_threshold_samples(w_samples, rset, n_support_set)
        
        end = time()
        
        # Print results summary
        ModelUtils.print_results_summary(w_samples, w_opt, bin_X, y, l2, end - start)
        
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
            w_opt=w_opt,
            rset_bound=eps,
            runtime=end - start,
            sampling=sampling,
            distance_metric=distance_metric,
        )

if __name__ == "__main__":
    # Run the ellipsoid method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = EllipsoidMethod()
    method.run_all_datasets(dataset_settings) 