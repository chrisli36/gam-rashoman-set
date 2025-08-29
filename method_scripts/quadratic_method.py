import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import cvxpy as cp
from typing import Dict, Any, Tuple
from src.prepare_gam import *
from src.rset_opt import *
from src.rset_app import RSetGAMs
from gam_rs_utils.utils import *
from base_method import BaseGAMRSetMethod
from results_class import MethodType
from time import time


def in_ellipsoid(x: np.ndarray, w_orig: np.ndarray, H: np.ndarray, eps: float) -> Tuple[bool, float]:
    """Check if a point is in the ellipsoid."""
    delta = x - w_orig
    val = delta.T @ H @ delta
    return val <= eps, val


def max_proj_direction_in_ellipsoid(H: np.ndarray, w_orig: np.ndarray, eps: float, v: np.ndarray, l0: float) -> np.ndarray:
    """Find the maximum projection direction in the ellipsoid."""
    d = len(w_orig)
    x = cp.Variable(d)
    obj = cp.Maximize(cp.matmul(v, x - w_orig) - l0 * cp.norm1(x))
    constraint = [cp.quad_form(x - w_orig, H) <= eps]
    # try adding constraints to encourage poisson distribution
    prob = cp.Problem(obj, constraint)
    prob.solve()
    return x.value


def hard_threshold(x: np.ndarray, k2: int) -> np.ndarray:
    """Apply hard thresholding to keep only the largest k2 elements."""
    x = x.copy()
    small_indices = np.argsort(np.abs(x))[:k2]
    x[small_indices] = 0
    return x


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(x - y)


def extremal_sampling(H: np.ndarray, w_orig: np.ndarray, rset_bound: float, l0: float, 
                     k2: int, r_min: float, rset: RSetGAMs, attempts: int = 10_000) -> np.ndarray:
    """Perform extremal sampling to find diverse models."""
    samples = []
    for _ in range(attempts):
        v = 0.1 * np.random.randn(len(w_orig))
        solutions = {"w_orig": w_orig}
        solutions['w_sol'] = max_proj_direction_in_ellipsoid(H, w_orig, rset_bound, v, l0)
        solutions['hard_thresholded_w'] = hard_threshold(solutions['w_sol'], k2)
        
        if not rset.in_rset(solutions['hard_thresholded_w']):
            continue
        if not all(euclidean_distance(solutions['hard_thresholded_w'], prev) >= r_min for prev in samples):
            continue
        samples.append(solutions['hard_thresholded_w'])
    
    print(f"out of {attempts} attempts, found {len(samples)} after hard thresholding")
    return np.array(samples)


class QuadraticMethod(BaseGAMRSetMethod):
    """
    Quadratic programming method for finding models in the GAM Rashomon set.
    
    This method uses extremal sampling with quadratic programming to find
    diverse models in the Rashomon set.
    """
    
    def __init__(self):
        """Initialize the quadratic method."""
        super().__init__(MethodType.QUADRATIC)
    
    def run_single_dataset(self, dname: str, settings: Dict[str, Any]) -> Any:
        """
        Run the quadratic method on a single dataset.
        
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
        attempts = 10_000
        
        # Prepare sparse GAM
        start = time()
        
        path = f'datasets/{dname}.csv'
        X_one_hot, y, header, header_new, _ = BaseGAMRSetMethod.get_binned_dataset(path, ne)
        sparse_gam_file = prepare_sparse_gam(dname, l0, l2, m, X_one_hot, y, header, header_new)
        model = RSetOPT(sparse_gam_file)
        model.finetune_ellipsoid()
        H_opt = model.get_normalized_H()
        model.update_file(H_opt, model.w_orig)
        
        # Load data
        with open(sparse_gam_file, "rb") as f:
            res = pickle.load(f)
        
        rset = RSetGAMs(sparse_gam_file)
        solutions = extremal_sampling(
            res['H_opt'], res['w_opt'], res['rset_bound'] * 0.1, 
            0.001, 6, 0.01, rset, attempts=attempts
        )
        
        end = time()
        
        # Extract data
        X = res['X']
        y = res['y']
        w_opt = res['w_opt']
        sample_p = res['sample_proportion']
        
        # Print results summary
        self.print_results_summary(solutions, w_opt, X, y, l2, sample_p, end - start)
        
        # Create and return result object
        return self.create_result_object(
            dataset=dname,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            w_rset=solutions,
            w_opt=w_opt,
            rset_bound=res['rset_bound'],
            predictions=self.get_predictions(X, solutions),
            runtime=end - start,
        )


if __name__ == "__main__":
    # Run te quadratic method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = QuadraticMethod()
    method.run_all_datasets(dataset_settings) 