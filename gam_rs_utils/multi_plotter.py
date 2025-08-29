import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
from typing import List, Optional
from .utils import get_feature_thresholds

class MultiPlotter:
    """
    A class for plotting multiple distribution plots, bar charts, or GAM shape functions stacked vertically with unified x-axis.
    """
    
    def __init__(self, figsize=(10, 12), title: Optional[str] = None):
        self.distributions = []
        self.bar_charts = []
        self.shape_functions = []
        self.figsize = figsize
        self.title = title
        self.x_min = float('inf')
        self.x_max = float('-inf')
        self.x_labels = None
        self.feature_names = None
    
    def add_distribution(self, losses: List[float], method_name: str, opt_loss: Optional[float] = None):
        """
        Add a distribution to be plotted.
        Args:
            losses: List of loss values.
            method_name: Name of the method for the subplot title.
            opt_loss: Optional optimal loss value to mark.
        """
        # Update global x-axis bounds
        current_min, current_max = min(losses), max(losses)
        if opt_loss is not None:
            current_min = min(current_min, opt_loss)
            current_max = max(current_max, opt_loss)
        
        self.x_min = min(self.x_min, current_min)
        self.x_max = max(self.x_max, current_max)
        
        # Store distribution data
        self.distributions.append({
            'losses': losses,
            'method_name': method_name,
            'opt_loss': opt_loss
        })
        
    def add_bar(self, y_values: List[float], method_name: str, x_labels: Optional[List] = None):
        """
        Add a bar chart to be plotted.
        Args:
            y_values: List of y values for the bars.
            method_name: Name of the method for the subplot title.
            x_labels: List of x-axis labels. Should be the same for all charts.
        """
        # Set x_labels on first call, verify consistency on subsequent calls
        if self.x_labels is None:
            self.x_labels = x_labels
        elif x_labels is not None and x_labels != self.x_labels:
            raise ValueError("x_labels must be the same for all bar charts")
        
        # Store bar chart data
        self.bar_charts.append({
            'y_values': y_values,
            'method_name': method_name
        })
        
    def add_shape_function(self, header: np.ndarray, w_rset: np.ndarray, method_name: str):
        """
        Add GAM shape functions to be plotted, grouped by feature.
        Args:
            header: Array of feature names.
            w_rset: Array of model weights.
            method_name: Name of the method for the legend.
        """
        # Extract feature names and thresholds from header
        feature_data = defaultdict(list)
        for i in range(w_rset.shape[0]):
            weights = w_rset[i, :]
            columns = header[np.nonzero(weights)[0]]
            weights = weights[np.nonzero(weights)[0]]

            feature_thresholds = get_feature_thresholds(weights, columns)
            for feature, thresholds_weights in feature_thresholds.items():
                if feature == 'sex' or feature == 'current':
                    continue
                thresholds, feature_weights = zip(*thresholds_weights)

                x_vals, y_vals = [], []
                x_vals.append(0)
                y_vals.append(feature_weights[0])
                for i in range(len(thresholds) - 1):
                    x_vals.append(thresholds[i][0])
                    y_vals.append(feature_weights[i])

                # For the last value of the interval, add it once more
                x_vals.append(thresholds[-1][-1])
                y_vals.append(feature_weights[-1])

                feature_data[feature].append((x_vals, y_vals))
        
        # Set feature names on first call, verify consistency on subsequent calls
        if self.feature_names is None:
            self.feature_names = list(feature_data.keys())
        elif set(feature_data.keys()) != set(self.feature_names):
            raise ValueError("Feature names must be the same for all shape function calls")
        
        # Store shape function data
        self.shape_functions.append({
            'feature_data': feature_data,
            'method_name': method_name
        })
        
    def plot_everything(self):
        """
        Plot all distributions, bar charts, or GAM shape functions stacked vertically with unified x-axis.
        """
        # Check for mixing plot types
        plot_types = [bool(self.distributions), bool(self.bar_charts), bool(self.shape_functions)]
        if sum(plot_types) > 1:
            print("Error: Cannot mix different plot types. Use only one type.")
            return
        
        if not any(plot_types):
            print("No plots to display. Use add_distribution(), add_bar(), or add_shape_function() first.")
            return
        
        # Plot distributions
        if self.distributions:
            n_plots = len(self.distributions)
            fig, axes = plt.subplots(n_plots, 1, figsize=self.figsize, sharex=True)
            
            # Handle single subplot case
            if n_plots == 1:
                axes = [axes]
            
            # Set overall title if provided
            if self.title:
                fig.suptitle(self.title, fontsize=16, fontweight='bold')
            
            for i, dist_data in enumerate(self.distributions):
                ax = axes[i]
                losses = dist_data['losses']
                method_name = dist_data['method_name']
                opt_loss = dist_data['opt_loss']
                
                # Plot histogram
                ax.hist(losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add average loss line
                avg_loss = np.mean(losses)
                ax.axvline(avg_loss, color='green', linestyle='solid', linewidth=2, 
                           label=f'Average Loss = {avg_loss:.4f}')
                
                # Add optimal loss line if provided
                if opt_loss is not None:
                    ax.axvline(opt_loss, color='red', linestyle='dashed', linewidth=2, 
                               label=f'Optimal Loss = {opt_loss:.4f}')
                
                ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
                ax.set_ylabel('Number of Models')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Set x-axis limits to be consistent across all plots
                ax.set_xlim(self.x_min, self.x_max)
            
            # Set x-axis label for the bottom subplot only
            axes[-1].set_xlabel('Loss')
        
        # Plot bar charts
        elif self.bar_charts:
            n_plots = len(self.bar_charts)
            fig, axes = plt.subplots(n_plots, 1, figsize=self.figsize, sharex=True)
            
            # Handle single subplot case
            if n_plots == 1:
                axes = [axes]
            
            # Set overall title if provided
            if self.title:
                fig.suptitle(self.title, fontsize=16, fontweight='bold')
            
            x_indices = range(len(self.x_labels))
            
            for i, chart_data in enumerate(self.bar_charts):
                ax = axes[i]
                y_values = chart_data['y_values']
                method_name = chart_data['method_name']
                
                # Plot bar chart
                ax.bar(x_indices, y_values, color='orange', alpha=0.7, edgecolor='black')
                
                # Set x-axis labels and formatting
                ax.set_xticks(x_indices)
                ax.set_xticklabels(self.x_labels, rotation=45, ha='right')
                
                ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
            
            # Set x-axis label for the bottom subplot only
            axes[-1].set_xlabel('X Values')
        
        # Plot GAM shape functions
        elif self.shape_functions:
            n_features = len(self.feature_names)
            n_methods = len(self.shape_functions)
            fig, axes = plt.subplots(n_features, n_methods, figsize=(6 * n_methods, 4 * n_features), sharex=False)

            # Ensure axes is 2D for consistent indexing
            if n_features == 1:
                axes = np.array([axes])

            # Set overall title if provided
            if self.title:
                fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # Colors for different methods
            colors = plt.cm.Set3(np.linspace(0, 1, n_methods))

            # Compute consistent y-limits per row (feature)
            row_y_limits = []
            for feature_name in self.feature_names:
                y_min_row = np.inf
                y_max_row = -np.inf
                for shape_data in self.shape_functions:
                    feature_data = shape_data['feature_data']
                    if feature_name in feature_data:
                        for _, y_vals in feature_data[feature_name]:
                            y_min_row = min(y_min_row, np.min(y_vals))
                            y_max_row = max(y_max_row, np.max(y_vals))
                if not np.isfinite(y_min_row) or not np.isfinite(y_max_row):
                    y_min_row, y_max_row = -1.0, 1.0
                row_y_limits.append((y_min_row, y_max_row))

            for feature_idx, feature_name in enumerate(self.feature_names):
                for method_idx, shape_data in enumerate(self.shape_functions):
                    ax = axes[feature_idx, method_idx]

                    feature_data = shape_data['feature_data']
                    method_name = shape_data['method_name']

                    if feature_name in feature_data:
                        for x_vals, y_vals in feature_data[feature_name]:
                            ax.step(
                                x_vals,
                                y_vals,
                                where="post",
                                color=colors[method_idx],
                                alpha=0.8,
                                linewidth=2,
                            )

                    # Apply consistent y-lims per row
                    y_min_row, y_max_row = row_y_limits[feature_idx]
                    ax.set_ylim(y_min_row, y_max_row)
                    ax.grid(True, alpha=0.3)

                    # Bottom row: put method names as x-axis labels
                    if feature_idx == n_features - 1:
                        ax.set_xlabel(self.shape_functions[method_idx]['method_name'])

                # Left label for the row with feature name
                left_ax = axes[feature_idx, 0]
                left_ax.set_ylabel(
                    f"{feature_name}", rotation=45, labelpad=35, va='center', fontsize=12, ha='right'
                )
        
        plt.tight_layout()
        plt.show()
