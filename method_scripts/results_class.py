import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np
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

@dataclass
class StandardMethodResults(MethodResults):
    """Results for ellipsoid, blocking, and quadratic methods"""
    l0: float
    l2: float
    m: float
    
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.l0, (int, float)):
            raise TypeError(f"l0 must be numeric, got {type(self.l0)}")
        if not isinstance(self.l2, (int, float)):
            raise TypeError(f"l2 must be numeric, got {type(self.l2)}")
        if not isinstance(self.m, (int, float)):
            raise TypeError(f"m must be numeric, got {type(self.m)}")

@dataclass
class SwappingMethodResults(MethodResults):
    """Results specifically for swapping method"""
    l2: float
    gap_tolerance: float
    swapping_percentages: List[float]
    
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.l2, (int, float)):
            raise TypeError(f"l2 must be numeric, got {type(self.l2)}")
        if not isinstance(self.gap_tolerance, (int, float)):
            raise TypeError(f"gap_tolerance must be numeric, got {type(self.gap_tolerance)}")
        if not isinstance(self.swapping_percentages, list):
            raise TypeError(f"swapping_percentages must be list, got {type(self.swapping_percentages)}")

@dataclass
class HybridMethodResults(StandardMethodResults):
    """Results specifically for hybrid method - extends StandardMethodResults with gap_tolerance"""
    gap_tolerance: float
    
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.gap_tolerance, (int, float)):
            raise TypeError(f"gap_tolerance must be numeric, got {type(self.gap_tolerance)}")

def create_results_object(method_type: MethodType, **kwargs) -> Union[StandardMethodResults, SwappingMethodResults, HybridMethodResults]:
    """Factory function to create the appropriate results object based on method type"""
    if method_type == MethodType.SWAPPING:
        return SwappingMethodResults(**kwargs)
    elif method_type == MethodType.HYBRID:
        return HybridMethodResults(**kwargs)
    else:
        return StandardMethodResults(**kwargs)

def save_results(results: List[Union[StandardMethodResults, SwappingMethodResults, HybridMethodResults]], method_type: MethodType, filename: Optional[str] = None) -> str:
    """Save results to pickle file with proper validation"""
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

def load_results(filename: str) -> List[Union[StandardMethodResults, SwappingMethodResults, HybridMethodResults]]:
    """Load results from pickle file"""
    with open(filename, "rb") as f:
        results = pickle.load(f)
    
    # Validate loaded results
    if not isinstance(results, list):
        raise TypeError(f"Loaded results must be a list, got {type(results)}")
    
    return results 