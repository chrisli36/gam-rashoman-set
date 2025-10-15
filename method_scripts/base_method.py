import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from gam_rs_utils.utils import *
from gam_rs_utils.utils import DatasetUtils, ModelUtils
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
        self.results = []
    
    def run_all_datasets(self, dataset_settings: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Run the method on all datasets and save results individually.
        
        Args:
            dataset_settings: List of (dataset_name, settings) tuples
        """
        for dname, settings in dataset_settings:
            for n_samples in settings['n_samples']:
                print(f"{BLUE}Dataset: {dname}, n_samples: {n_samples}{RESET}")
                
                # Run the method-specific implementation
                result_obj = self.run_single_dataset(dname, settings, n_samples)
                
                # Save result immediately to dataset-specific directory
                saved_path = Results.save_single_result(result_obj, self.method_type, dname)
                print(f"{GREEN}Saved result to: {saved_path}{RESET}")
                
                self.results.append(result_obj)
    
    @abstractmethod
    def run_single_dataset(self, dname: str, settings: Dict[str, Any], n_samples: int = 100) -> Any:
        """
        Run the method on a single dataset.
        
        Args:
            dname: Dataset name
            settings: Dataset-specific settings
            n_samples: Number of samples to generate
            
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
