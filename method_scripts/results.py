import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
from enum import Enum
import pandas as pd

class MethodType(Enum):
    ELLIPSOID = "ellipsoid"
    BLOCKING = "blocking"
    QUADRATIC = "quadratic"
    SWAPPING = "swapping"
    HYBRID = "hybrid"

@dataclass
class MethodResults:
    """Base class for method results with common fields"""
    method_type: MethodType
    dataset: str
    n_estimators: int
    n_support_set: int
    n_samples: int
    w_rset: np.ndarray
    w_opt: np.ndarray
    rset_bound: float
    runtime: float
    l0: float
    l2: float
    m: float
    
    def __post_init__(self):
        """Validate the data types and shapes"""
        if not isinstance(self.method_type, MethodType):
            raise TypeError(f"method_type must be MethodType, got {type(self.method_type)}")
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
        if not isinstance(self.runtime, (int, float)):
            raise TypeError(f"runtime must be numeric, got {type(self.runtime)}")
        if not isinstance(self.l0, (int, float)):
            raise TypeError(f"l0 must be numeric, got {type(self.l0)}")
        if not isinstance(self.l2, (int, float)):
            raise TypeError(f"l2 must be numeric, got {type(self.l2)}")
        if not isinstance(self.m, (int, float)):
            raise TypeError(f"m must be numeric, got {type(self.m)}")
    
    def get_filename(self) -> str:
        return f"{self.method_type.value}_l0_{self.l0}_l2_{self.l2}_m_{self.m}_samples_{self.n_samples}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary"""
        return asdict(self)
    
    def to_dataframe_row(self) -> pd.DataFrame:
        """Convert the dataclass to a DataFrame row (1-row DataFrame)"""
        return pd.DataFrame([self.to_dict()])

@dataclass
class BlockingMethodResults(MethodResults):
    """Results for blocking method"""
    method_type = MethodType.BLOCKING
    
    def __post_init__(self):
        super().__post_init__()

@dataclass
class QuadraticMethodResults(MethodResults):
    """Results for quadratic method"""
    method_type = MethodType.QUADRATIC

    def __post_init__(self):
        super().__post_init__()

@dataclass
class SwapppingMethodResults(MethodResults):
    """Results for swapping method"""
    method_type = MethodType.SWAPPING
    
    def __post_init__(self):
        super().__post_init__()

@dataclass
class HybridMethodResults(MethodResults):
    """Results for hybrid method"""
    method_type = MethodType.HYBRID
    
    def __post_init__(self):
        super().__post_init__()

@dataclass
class EllipsoidMethodResults(MethodResults):
    """Results for ellipsoid method"""
    method_type = MethodType.ELLIPSOID
    sampling: str
    distance_metric: Optional[str]
    
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.sampling, str):
            raise TypeError(f"sampling must be str, got {type(self.sampling)}")
        if self.distance_metric is not None and not isinstance(self.distance_metric, str):
            raise TypeError(f"distance_metric must be str or None, got {type(self.distance_metric)}")
    
    def get_filename(self, method_type: MethodType) -> str:
        return super().get_filename(method_type) + f"_sampling_{self.sampling}_distance_metric_{self.distance_metric}"

class Results:
    @staticmethod
    def create_results_object(method_type: MethodType, **kwargs) -> Union[EllipsoidMethodResults, BlockingMethodResults, QuadraticMethodResults, SwapppingMethodResults, HybridMethodResults]:
        """Factory function to create the appropriate results object based on method type"""
        if method_type == MethodType.ELLIPSOID:
            return EllipsoidMethodResults(method_type=method_type, **kwargs)
        elif method_type == MethodType.BLOCKING:
            return BlockingMethodResults(method_type=method_type, **kwargs)
        elif method_type == MethodType.QUADRATIC:
            return QuadraticMethodResults(method_type=method_type, **kwargs)
        elif method_type == MethodType.SWAPPING:
            return SwapppingMethodResults(method_type=method_type, **kwargs)
        elif method_type == MethodType.HYBRID:
            return HybridMethodResults(method_type=method_type, **kwargs)
        else:
            raise ValueError(f"Invalid method type: {method_type}")

    @staticmethod
    def save_result(result: Union[MethodResults, EllipsoidMethodResults], dataset_name: str) -> str:
        """Save a single result to dataset-specific directory"""
        # Create directory structure
        results_dir = f"results/{dataset_name}"
        method_results_dir = f"{results_dir}/method_results"
        os.makedirs(method_results_dir, exist_ok=True)
        
        filename = result.get_filename()
        filepath = f"{method_results_dir}/{filename}.pkl"
        
        with open(filepath, "wb") as f:
            pickle.dump(result, f)
        
        return filepath

    @staticmethod
    def load_result(filepath: str) -> Union[EllipsoidMethodResults, BlockingMethodResults, QuadraticMethodResults, SwapppingMethodResults, HybridMethodResults]:
        """Load result from pickle file"""
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        return result 

    @staticmethod
    def save_binarized_dataset(dataset_name: str, num_estimators: int, X_binarized: np.ndarray, 
                              y: np.ndarray, header: List[str], header_new: List[str], sample_proportion: np.ndarray = None) -> str:
        """Save binarized dataset for reuse"""
        results_dir = f"results/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"{results_dir}/estimators_{num_estimators}.pkl"
        
        binarized_data = {
            'X': X_binarized,
            'y': y,
            'header': header,
            'header_new': header_new,
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
        filename = f"results/{dataset_name}/estimators_{num_estimators}.pkl"
        
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
        Results.save_binarized_dataset(dataset_name, num_estimators, X_one_hot, y, header, header_new, sample_p)
        
        return {
            'X': X_one_hot,
            'y': y,
            'header': header,
            'header_new': header_new,
            'sample_proportion': sample_p,
            'num_estimators': num_estimators
        }
