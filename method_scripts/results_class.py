import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
from enum import Enum

class MethodType(Enum):
    ELLIPSOID = "ellipsoid"
    BLOCKING = "blocking"
    QUADRATIC = "quadratic"
    SWAPPING = "swapping"
    HYBRID = "hybrid"

@dataclass
class MethodResults:
    """Base class for method results with common fields"""
    dataset: str
    n_estimators: int
    n_support_set: int
    n_samples: int
    w_rset: np.ndarray
    w_opt: np.ndarray
    rset_bound: float
    predictions: np.ndarray
    runtime: float
    l0: float
    l2: float
    m: float
    
    def __post_init__(self):
        """Validate the data types and shapes"""
        if not isinstance(self.dataset, str):
            raise TypeError(f"dataset must be str, got {type(self.dataset)}")
        if not isinstance(self.n_estimators, int):
            raise TypeError(f"n_estimators must be int, got {type(self.n_estimators)}")
        if not isinstance(self.n_support_set, int):
            raise TypeError(f"n_support_set must be int, got {type(self.n_support_set)}")
        if not isinstance(self.n_samples, int):
            raise TypeError(f"n_samples must be int, got {type(self.n_samples)}")
        if not isinstance(self.w_rset, np.ndarray):
            raise TypeError(f"w_rset must be np.ndarray, got {type(self.w_rset)}")
        if not isinstance(self.w_opt, np.ndarray):
            raise TypeError(f"w_opt must be np.ndarray, got {type(self.w_opt)}")
        if not isinstance(self.rset_bound, (int, float)):
            raise TypeError(f"rset_bound must be numeric, got {type(self.rset_bound)}")
        if not isinstance(self.predictions, np.ndarray):
            raise TypeError(f"predictions must be np.ndarray, got {type(self.predictions)}")
        if not isinstance(self.runtime, (int, float)):
            raise TypeError(f"runtime must be numeric, got {type(self.runtime)}")
        if not isinstance(self.l0, (int, float)):
            raise TypeError(f"l0 must be numeric, got {type(self.l0)}")
        if not isinstance(self.l2, (int, float)):
            raise TypeError(f"l2 must be numeric, got {type(self.l2)}")
        if not isinstance(self.m, (int, float)):
            raise TypeError(f"m must be numeric, got {type(self.m)}")

@dataclass
class EllipsoidMethodResults(MethodResults):
    """Results for ellipsoid method"""
    
    def __post_init__(self):
        super().__post_init__()

@dataclass
class SwappingMethodResults(MethodResults):
    """Results specifically for swapping method"""
    
    def __post_init__(self):
        super().__post_init__()

@dataclass
class HybridMethodResults(MethodResults):
    """Results specifically for hybrid method"""
    
    def __post_init__(self):
        super().__post_init__()

class Results:
    @staticmethod
    def create_results_object(method_type: MethodType, **kwargs) -> Union[EllipsoidMethodResults, SwappingMethodResults, HybridMethodResults]:
        """Factory function to create the appropriate results object based on method type"""
        if method_type == MethodType.SWAPPING:
            return SwappingMethodResults(**kwargs)
        elif method_type == MethodType.HYBRID:
            return HybridMethodResults(**kwargs)
        else:
            return EllipsoidMethodResults(**kwargs)

    @staticmethod
    def save_result(result: Union[EllipsoidMethodResults, SwappingMethodResults, HybridMethodResults], 
                          method_type: MethodType, dataset_name: str) -> str:
        """Save a single result to dataset-specific directory"""
        # Create directory structure
        results_dir = f"results/{dataset_name}"
        method_results_dir = f"{results_dir}/method_results"
        os.makedirs(method_results_dir, exist_ok=True)
        
        # Generate filename based on method type and parameters
        if method_type == MethodType.SWAPPING:
            filename = f"{method_type.value}_l2_{result.l2}_gap_tolerance_{result.gap_tolerance}_samples_{result.n_samples}.pkl"
        elif method_type == MethodType.HYBRID:
            filename = f"{method_type.value}_l0_{result.l0}_l2_{result.l2}_m_{result.m}_gap_tolerance_{result.gap_tolerance}_samples_{result.n_samples}.pkl"
        else:  # Standard methods (ELLIPSOID, BLOCKING, QUADRATIC)
            filename = f"{method_type.value}_l0_{result.l0}_l2_{result.l2}_m_{result.m}_samples_{result.n_samples}.pkl"
        
        filepath = f"{method_results_dir}/{filename}"
        
        with open(filepath, "wb") as f:
            pickle.dump(result, f)
        
        return filepath

    @staticmethod
    def save_binarized_dataset(dataset_name: str, num_estimators: int, X_binarized: np.ndarray, 
                              y: np.ndarray, header: List[str], sample_proportion: np.ndarray = None) -> str:
        """Save binarized dataset for reuse"""
        results_dir = f"results/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"{results_dir}/binarized_dataset_estimators_{num_estimators}.pkl"
        
        binarized_data = {
            'X': X_binarized,
            'y': y,
            'header': header,
            'num_estimators': num_estimators
        }
        
        if sample_proportion is not None:
            binarized_data['sample_proportion'] = sample_proportion
        
        with open(filename, "wb") as f:
            pickle.dump(binarized_data, f)
        
        return filename

    @staticmethod
    def load_binarized_dataset(dataset_name: str, num_estimators: int) -> Optional[Dict[str, Any]]:
        """Load binarized dataset if it exists"""
        filename = f"results/{dataset_name}/binarized_dataset_estimators_{num_estimators}.pkl"
        
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
                return data
        return None

    @staticmethod
    def create_binarized_dataset(dataset_name: str, num_estimators: int) -> Dict[str, Any]:
        """
        Create or load binarized dataset. If it doesn't exist, create it and save it.
        If it exists, load and return it.
        
        Args:
            dataset_name: Name of the dataset
            num_estimators: Number of estimators for binarization
            
        Returns:
            Dictionary containing X, y, header, and num_estimators
        """
        # Try to load existing dataset first
        binarized_data = Results.load_binarized_dataset(dataset_name, num_estimators)
        
        if binarized_data is not None:
            print(f"Loading cached binarized dataset for {dataset_name} with {num_estimators} estimators")
            return binarized_data
        
        # Create new binarized dataset
        print(f"Generating new binarized dataset for {dataset_name} with {num_estimators} estimators")
        
        # Import here to avoid circular imports
        from gam_rs_utils.utils import DatasetUtils
        
        path = f'datasets/{dataset_name}.csv'
        X_one_hot, y, header, header_new, sample_p = DatasetUtils.get_binned_dataset(path, num_estimators)
        
        # Save binarized dataset for future use
        Results.save_binarized_dataset(dataset_name, num_estimators, X_one_hot, y, header_new, sample_p)
        
        return {
            'X': X_one_hot,
            'y': y,
            'header': header,
            'header_new': header_new,
            'sample_proportion': sample_p,
            'num_estimators': num_estimators
        }

    @staticmethod
    def save_results(results: List[Union[EllipsoidMethodResults, SwappingMethodResults, HybridMethodResults]], method_type: MethodType, filename: Optional[str] = None) -> str:
        """Save results to pickle file with proper validation (legacy method)"""
        if not results:
            raise ValueError("Results list cannot be empty")
        
        # Validate all results are of the same type
        if method_type == MethodType.SWAPPING:
            expected_type = SwappingMethodResults
        elif method_type == MethodType.HYBRID:
            expected_type = HybridMethodResults
        else:
            expected_type = StandardMethodResults
        
        for i, result in enumerate(results):
            if not isinstance(result, expected_type):
                raise TypeError(f"Result {i} must be {expected_type.__name__}, got {type(result).__name__}")
        
        if filename is None:
            filename = f"analysis/results/methods/{method_type.value}.pkl"
        
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        
        return filename

    @staticmethod
    def load_results(filename: str) -> List[Union[EllipsoidMethodResults, SwappingMethodResults, HybridMethodResults]]:
        """Load results from pickle file"""
        with open(filename, "rb") as f:
            results = pickle.load(f)
        
        # Validate loaded results
        if not isinstance(results, list):
            raise TypeError(f"Loaded results must be a list, got {type(results)}")
        
        return results 