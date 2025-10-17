import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from gam_rs_utils.utils import *
from results_class import MethodType, Results

class BaseGAMRSetMethod(ABC):
    """
    Base class for GAM Rashomon Set methods. Provides a common interface and shared 
    functionality for different methods of finding models in the Rashomon set.
    """
    
    def __init__(self, method_type: MethodType):
        """
        Initialize the base method.
        
        Args:
            method_type: The type of method (ELLIPSOID, BLOCKING, QUADRATIC, SWAPPING)
        """
        self.method_type = method_type
        self.extra_settings = []
    
    def run_all_datasets(self, dataset_settings: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Run the method on all datasets and save results individually.
        
        Args:
            dataset_settings: List of (dataset_name, settings) tuples
        """
        for dname, settings in dataset_settings:
            # Extract parameter lists and their keys
            param_keys = []
            param_values = []
            
            for key, value in settings.items():
                if isinstance(value, list):
                    param_keys.append(key)
                    param_values.append(value)
            
            # Generate all combinations of parameters
            param_combinations = list(itertools.product(*param_values))
            
            for combination in param_combinations:
                combination_settings = dict(zip(param_keys, combination))
                
                n_samples = combination_settings.get('n_samples', 100)
                
                for extra_settings in self.extra_settings:
                    print(f"{BLUE}Dataset: {dname}, params: {combination_settings}, extra: {extra_settings}{RESET}")
                    
                    combination_settings.update(extra_settings)
                    result_obj = self.run_dataset(dname, n_samples, **combination_settings)
                    
                    saved_path = Results.save_result(result_obj, self.method_type, dname)
                    print(f"{GREEN}Saved result to: {saved_path}{RESET}")
    
    @abstractmethod
    def run_dataset(self, dname: str, n_samples: int = 100, **kwargs) -> Any:
        """
        Run the method on a single dataset.
        
        Args:
            dname: Dataset name
            n_samples: Number of samples to generate
            **kwargs: Additional method-specific parameters
            
        Returns:
            Results object for this dataset
        """
        pass
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save results to file (legacy method for backward compatibility).
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        return Results.save_results(self.results, self.method_type, filename)
    
    def create_result_object(self, **kwargs) -> Any:
        """
        Create a result object with the appropriate method type.
        
        Args:
            **kwargs: Result object parameters
            
        Returns:
            Result object
        """
        return Results.create_results_object(method_type=self.method_type, **kwargs)
