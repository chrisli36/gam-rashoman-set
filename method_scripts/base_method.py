import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from time import time
from gam_rs_utils.utils import *
from results_class import MethodType, create_results_object, save_results


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
        Run the method on all datasets and save results.
        
        Args:
            dataset_settings: List of (dataset_name, settings) tuples
        """
        for dname, settings in dataset_settings:
            print(f"{BLUE}Dataset: {dname}{RESET}")
            
            # Extract common settings
            ne = settings["num_estimators"]
            n_support_set = settings["n_support_set"]
            
            # Run the method-specific implementation
            result_obj = self.run_single_dataset(dname, settings)
            self.results.append(result_obj)
        
        # Save results
        self.save_results()
    
    @abstractmethod
    def run_single_dataset(self, dname: str, settings: Dict[str, Any]) -> Any:
        """
        Run the method on a single dataset.
        
        Args:
            dname: Dataset name
            settings: Dataset-specific settings
            
        Returns:
            Results object for this dataset
        """
        pass
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save results to file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        return save_results(self.results, self.method_type, filename)
    
    def get_common_settings(self, settings: Dict[str, Any]) -> Tuple[int, int]:
        """
        Extract common settings from dataset settings.
        
        Args:
            settings: Dataset settings dictionary
            
        Returns:
            Tuple of (num_estimators, n_support_set)
        """
        ne = settings["num_estimators"]
        n_support_set = settings["n_support_set"]
        return ne, n_support_set
    
    def create_result_object(self, **kwargs) -> Any:
        """
        Create a result object with the appropriate method type.
        
        Args:
            **kwargs: Result object parameters
            
        Returns:
            Result object
        """
        return create_results_object(method_type=self.method_type, **kwargs)
    
    def print_results_summary(self, w_rset: np.ndarray, w_opt: np.ndarray, 
                            X: np.ndarray, y: np.ndarray, l2: float, 
                            sample_p: np.ndarray, runtime: float) -> None:
        """
        Print a summary of results for a dataset.
        
        Args:
            w_rset: Rashomon set models
            w_opt: Optimal model
            X: Feature matrix
            y: Target vector
            l2: L2 regularization parameter
            sample_p: Sample proportions
            runtime: Runtime in seconds
        """
        print(f"\t{w_rset.shape[0]} solutions, {runtime:.2f} seconds")
        print("Average logistic loss: ", get_loss(X, y, w_rset, loss_type="logistic", l2=l2, sample_p=sample_p))
        print("Opt model logistic loss: ", get_loss_one_model(X, y, w_opt, loss_type="logistic", l2=l2, sample_p=sample_p)) 