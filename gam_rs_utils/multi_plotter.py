import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
from typing import List, Optional, Dict
from .utils import get_feature_thresholds

class MultiPlotter:
    """
    A class for plotting multiple distribution plots, bar charts, or GAM shape functions stacked vertically with unified x-axis.
    """
    
    def __init__(self, figsize=(10, 12), title: Optional[str] = None):
        self.distributions = []
        self.bar_charts = []
        self.shape_functions = []
        self.variable_importance_distributions = []
        self.shape_diversities = []
        self.figsize = figsize
        self.title = title
        self.x_min = float('inf')
        self.x_max = float('-inf')
        self.x_labels = None
        self.feature_names = None
        self.row_y_limits = None
        self.column_x_limits = None

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
        
        # Update y-limits for each feature
        if self.row_y_limits is None:
            self.row_y_limits = defaultdict(lambda: [float('inf'), float('-inf')])
        
        for feature in self.feature_names:
            for x_vals, y_vals in feature_data[feature]:
                self.row_y_limits[feature] = [
                    min(self.row_y_limits[feature][0], np.min(y_vals)), 
                    max(self.row_y_limits[feature][1], np.max(y_vals))
                ]
        
        # Store shape function data
        self.shape_functions.append({
            'feature_data': feature_data,
            'method_name': method_name
        })

    def add_variable_importance_distribution(self, feature_to_vi: Dict[str, List[float]], method_name: str):
        """
        Add variable importance distribution to be plotted.
        Args:
            feature_to_vi: Dictionary mapping feature names to lists of importance values.
            method_name: Name of the method for the subplot title.
        """
        # Set feature names on first call, verify consistency on subsequent calls
        if self.feature_names is None:
            self.feature_names = list(feature_to_vi.keys())
        elif set(feature_to_vi.keys()) != set(self.feature_names):
            raise ValueError("Feature names must be the same for all variable importance distributions")
        
        if self.row_y_limits is None:
            self.row_y_limits = defaultdict(lambda: [float('inf'), float('-inf')])
        if self.column_x_limits is None:
            self.column_x_limits = defaultdict(lambda: [float('inf'), float('-inf')])
        
        for feature in self.feature_names:
            for vi in feature_to_vi[feature]:
                self.column_x_limits[feature] = [
                    min(self.column_x_limits[feature][0], np.min(vi)),
                    max(self.column_x_limits[feature][1], np.max(vi))
                ]

        self.variable_importance_distributions.append({
            'feature_to_vi': feature_to_vi,
            'method_name': method_name
        })
    
    def add_model_reliance(self, feature_to_vi: Dict[str, List[float]], method_name: str):
        """
        Add model reliance to be plotted.
        Args:
            feature_to_vi: Dictionary mapping method names to lists of reliance values.
            method_name: Name of the method for the subplot title.
        """
        if self.feature_names is None:
            self.feature_names = list(feature_to_vi.keys())
        elif set(feature_to_vi.keys()) != set(self.feature_names):
            raise ValueError("Feature names must be the same for all model reliance")
        
        self.variable_importance_distributions.append({
            'feature_to_vi': feature_to_vi,
            'method_name': method_name
        })

    def add_shape_diversity(self, feature_to_diversity: Dict[str, List[float]], method_name: str):
        """
        Add shape diversity to be plotted.
        Args:
            feature_to_diversity: Dictionary mapping feature names to lists of diversity values.
            method_name: Name of the method for the subplot title.
        """
        if self.feature_names is None:
            self.feature_names = list(feature_to_diversity.keys())
        elif set(feature_to_diversity.keys()) != set(self.feature_names):
            raise ValueError("Feature names must be the same for all shape diversity")
        
        self.shape_diversities.append({
            'feature_to_diversity': feature_to_diversity,
            'method_name': method_name
        })

    def plot_distributions(self):
        """
        Plot distributions stacked vertically with unified x-axis.
        """

        if not self.distributions:
            print("No distributions to plot. Use add_distribution() first.")
            return
        
        n_plots = len(self.distributions)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)
            
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

        plt.tight_layout()
        plt.show()

    def plot_bar_charts(self):
        """
        Plot bar charts stacked vertically with unified x-axis.
        """

        if not self.bar_charts:
            print("No bar charts to plot. Use add_bar() first.")
            return
        
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

        plt.tight_layout()
        plt.show()

    def plot_shape_functions(self):
        """
        Plot GAM shape functions stacked vertically with unified x-axis.
        """

        if not self.shape_functions:
            print("No shape functions to plot. Use add_shape_function() first.")
            return

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

        for feature_idx, feature in enumerate(self.feature_names):
            for method_idx, shape_data in enumerate(self.shape_functions):
                ax = axes[feature_idx, method_idx]

                feature_data = shape_data['feature_data']
                method_name = shape_data['method_name']

                for x_vals, y_vals in feature_data[feature]:
                    ax.step(
                        x_vals,
                        y_vals,
                        where="post",
                        color=colors[method_idx],
                        alpha=0.8,
                        linewidth=2,
                    )

                # Apply consistent y-lims per row
                y_min_row, y_max_row = self.row_y_limits[feature]
                ax.set_ylim(y_min_row, y_max_row)
                ax.grid(True, alpha=0.3)

                # Bottom row: put method names as x-axis labels
                if feature_idx == n_features - 1:
                    ax.set_xlabel(self.shape_functions[method_idx]['method_name'])

            # Left label for the row with feature name
            left_ax = axes[feature_idx, 0]
            left_ax.set_ylabel(
                f"{feature}", rotation=45, labelpad=35, va='center', fontsize=12, ha='right'
            )

        plt.tight_layout()
        plt.show()

    def plot_variable_importance_distributions(self):
        """
        Plot variable importance distributions stacked vertically with unified x-axis.
        """

        if not self.variable_importance_distributions:
            print("No variable importance distributions to plot. Use add_variable_importance_distribution() first.")
            return

        n_rows = len(self.variable_importance_distributions)
        n_cols = len(self.feature_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), sharex=True)

        # Ensure axes is 2D for consistent indexing
        if n_rows == 1:
            axes = np.array([axes])

        # Set overall title if provided
        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Colors for different methods
        colors = plt.cm.Set3(np.linspace(0, 1, n_rows))

        for i, dist_data in enumerate(self.variable_importance_distributions):
            feature_to_vi = dist_data['feature_to_vi']
            method_name = dist_data['method_name']
            for j, feature_name in enumerate(self.feature_names):
                ax = axes[i, j]
                ax.hist(feature_to_vi[feature_name], bins=30, alpha=0.7, color=colors[i], edgecolor='black')
                ax.set_title(f'{method_name}')
                ax.set_xlabel('Variable Importance')
                ax.set_ylabel('Number of Models')
                x_min, x_max = self.column_x_limits[feature_name]
                ax.set_xlim(x_min, x_max)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_model_reliance_box_plots(self):
        """
        Plot model reliance box plots stacked vertically with unified x-axis.
        """
        if not self.variable_importance_distributions:
            print("No model reliance to plot. Use add_model_reliance() first.")
            return
        
        n_rows = len(self.feature_names)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)

        if n_rows == 1:
            axes = [axes]

        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Get method labels once (they should be the same for all features)
        method_labels = [dist_data['method_name'] for dist_data in self.variable_importance_distributions]
        
        for row, feature in enumerate(self.feature_names):
            method_data = []
            for dist_data in self.variable_importance_distributions:
                feature_to_vi = dist_data['feature_to_vi']
                method_data.append(feature_to_vi[feature])

            ax = axes[row]
            # Only set labels on the last subplot to avoid tick location conflicts
            labels_to_use = method_labels if row == n_rows - 1 else None
            box_plot = ax.boxplot(method_data, labels=labels_to_use, patch_artist=True)

            colors = plt.cm.Set3(np.linspace(0, 1, len(method_data)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f'{feature}')
            ax.set_ylabel('Variable Importance')
            ax.grid(True, alpha=0.3)
            
            # Hide x-axis labels for all subplots except the last one
            if row < n_rows - 1:
                ax.set_xticklabels([])

        # Set x-axis label only on the last subplot
        axes[-1].set_xlabel('Method')

        plt.tight_layout()
        plt.show()

    def plot_shape_diversity(self):
        """
        Plot shape diversity box plots stacked vertically with unified x-axis.
        """
        if not self.shape_diversities:
            print("No shape diversities to plot. Use add_shape_diversity() first.")
            return
        
        n_rows = len(self.feature_names)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)

        if n_rows == 1:
            axes = [axes]

        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Get method labels once (they should be the same for all features)
        method_labels = [dist_data['method_name'] for dist_data in self.shape_diversities]

        for row, feature in enumerate(self.feature_names):
            collated_diversities = []
            for dist_data in self.shape_diversities:
                feature_to_diversity = dist_data['feature_to_diversity']
                collated_diversities.append(feature_to_diversity[feature][0])

            ax = axes[row]
            # Create bar positions
            x_positions = range(len(method_labels))
            ax.bar(x_positions, collated_diversities)
            
            ax.set_title(f'{feature}')
            ax.set_ylabel('Shape Diversity')
            ax.grid(True, alpha=0.3)

            # Set x-axis labels only on the last subplot
            if row == n_rows - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(method_labels, rotation=45, ha='right')
                ax.set_xlabel('Method')
            else:
                # Hide x-axis labels for upper subplots
                ax.set_xticklabels([])

        plt.tight_layout()
        plt.show()