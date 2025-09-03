import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
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
        print("Average logistic loss: ", BaseGAMRSetMethod.get_loss(X, y, w_rset, loss_type="logistic", l2=l2, sample_p=sample_p))
        print("Opt model logistic loss: ", BaseGAMRSetMethod.get_loss_one_model(X, y, w_opt, loss_type="logistic", l2=l2, sample_p=sample_p)) 

    def get_binned_dataset(path: str, num_estimators: int) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Loads a dataset, binarizes it, and converts to binned format.
        Args:
            path: Path to the CSV file.
            num_estimators: Number of estimators for binarization.
        Returns:
            Tuple of (X_new, y, header, header_new, sample_p).
        """
        data = pd.read_csv(path)
        df, _, header, _ = binarize_dataset(data, num_estimators)
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        X_new, header_new = BaseGAMRSetMethod.convert_cumulative_to_binned(X, header)
        header_new = ["intercept"] + header_new
        X_new, y = utils.get_X_y(X_new, y, is_df=False)
        sample_p = X_new.sum(0) / X_new.shape[0]
        return X_new, y, header, header_new, sample_p

    def get_loss_one_model(X_one_hot: np.ndarray, y: np.ndarray, w: np.ndarray, loss_type: str = "accuracy", l2: Optional[float] = None, sample_p: Optional[np.ndarray] = None) -> float:
        
        """
        Computes the loss for a single model.
        Args:
            X_one_hot: 2D numpy array of features.
            y: 1D numpy array of targets.
            w: 1D numpy array of weights.
            loss_type: 'accuracy' or 'logistic'.
            l2: L2 regularization parameter (optional).
            sample_p: Sample probabilities (optional).
        Returns:
            Loss value as float.
        """
        logit = X_one_hot @ w
        if loss_type == "accuracy":
            y_pred = np.exp(logit) / (1 + np.exp(logit))
            y_pred = np.where(y_pred > 0.5, 1, -1)
            loss = (y != y_pred).mean()
            return loss
        elif loss_type == "logistic":
            loss = np.mean(np.log1p(np.exp(-y * logit))) + l2 * (sample_p[1:] * w[1:]**2).sum()
            return loss
        return

    def get_loss(X_one_hot: np.ndarray, y: np.ndarray, w_rset: np.ndarray, loss_type: str = "accuracy", verbosity: int = 0, w_opt: Optional[np.ndarray] = None, l2: Optional[float] = None, sample_p: Optional[np.ndarray] = None) -> float:
        """
        Computes the average loss over a set of models.
        Args:
            X_one_hot: 2D numpy array of features.
            y: 1D numpy array of targets.
            w_rset: 2D numpy array of model weights.
            loss_type: 'accuracy' or 'logistic'.
            verbosity: Verbosity level.
            w_opt: Optional optimal weights for comparison.
            l2: L2 regularization parameter (optional).
            sample_p: Sample probabilities (optional).
        Returns:
            Mean loss value as float.
        """
        if len(w_rset) == 0:
            return 0
        losses = []
        for i in range(len(w_rset)):
            wi = w_rset[i, :]
            loss = BaseGAMRSetMethod.get_loss_one_model(X_one_hot, y, wi, loss_type, l2, sample_p)
            losses.append(loss)
        opt_loss = None if w_opt is None else BaseGAMRSetMethod.get_loss_one_model(X_one_hot, y, w_opt, loss_type, l2, sample_p)
        if verbosity > 0:
            print(f"Optimal model {loss_type} loss: {opt_loss}")
            print(f"Average {loss_type} loss: {np.mean(losses)}")
        if verbosity > 1:
            plot_distribution(losses, opt_loss)
        return losses, opt_loss

    def get_logits(self, X_one_hot: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Computes logits for a set of models.
        Args:
            X_one_hot: 2D numpy array of features.
            w: 2D numpy array of model weights.
        """
        return X_one_hot @ w

    def get_predictions(self, X_one_hot: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Computes predictions for a set of models.
        Args:
            X_one_hot: 2D numpy array of features.
            w: 2D numpy array of model weights.
        Returns:
            2D numpy array of predictions.
        """
        if len(w) == 0:
            return np.array([])
        y_preds = np.zeros((X_one_hot.shape[0], len(w)))
        for i in range(len(w)):
            wi = w[i, :]
            logit = X_one_hot @ wi
            y_pred = np.exp(logit) / (1 + np.exp(logit))
            y_pred = np.where(y_pred > 0.5, 1, -1)
            y_preds[:, i] = y_pred
        return y_preds

    def convert_cumulative_to_binned(X: np.ndarray, header: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Converts cumulative binary features to binned features.
        Args:
            X: 2D numpy array of cumulative binary features.
            header: List of feature names in the format "feature<=threshold".
        Returns:
            Tuple of (new_X, new_header) where new_X is the binned feature array and new_header is the updated header.
        """
        # assumes that header is a list of strings with format "feature<=threshold"
        feature_to_thresholds = defaultdict(list)
        for h in header:
            feat, thres = h.split("<=")
            feature_to_thresholds[feat].append(float(thres))

        new_header = []
        for feat, thresholds in feature_to_thresholds.items():
            new_header.append(f"{feat}<={thresholds[0]}")
            for i in range(len(thresholds) - 1):
                new_header.append(f"{thresholds[i]}<{feat}<={thresholds[i + 1]}")

        new_X = np.zeros(X.shape)
        column_idx = 0
        for _, thresholds in feature_to_thresholds.items():
            prev = np.zeros(X.shape[0])
            for i in range(len(thresholds)):
                new_X[:, column_idx] = X[:, column_idx] - prev
                prev = X[:, column_idx]
                column_idx += 1

        return new_X, new_header

    def hard_threshold(self, x: np.ndarray, k2: int) -> np.ndarray:
        """
        Apply hard thresholding to keep only the largest k2 elements.

        Args:
            x: numpy array of weights
            k2: number of elements to keep

        Returns:
            numpy array of weights with the smallest k2 elements set to 0
        """
        x = x.copy()
        small_indices = np.argsort(np.abs(x))[:k2]
        x[small_indices] = 0
        return x

    def hard_threshold_samples(self, w_samples: np.ndarray, rset, n_support_set: int) -> np.ndarray:
        w_samples_zeroed = []
        n_support = w_samples.shape[1]
        k2 = n_support - 1 - n_support_set
        
        for i in range(w_samples.shape[0]):
            w_samp = w_samples[i]
            w_samp_zeroed = self.hard_threshold(w_samp, k2)
            if rset.in_rset(w_samp_zeroed):
                w_samples_zeroed.append(w_samp_zeroed)
        
        print(f"\tout of {w_samples.shape[0]} samples, kept {len(w_samples_zeroed)} after hard thresholding")
        
        return np.array(w_samples_zeroed) if len(w_samples_zeroed) > 0 else np.array([])
