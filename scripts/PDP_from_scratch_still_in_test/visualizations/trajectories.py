"""
Trajectory visualization for PDP configurations.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .base import BaseVisualizer, PlotStyle
from ..config import PDPConfig
from ..data.loader import Dataset
from ..core.pdp import PDPResult


class TrajectoryVisualizer(BaseVisualizer):
    """
    Creates static trajectory visualizations for each configuration.
    
    Shows point trajectories with arrows indicating movement direction,
    with optional class-based coloring.
    """
    
    def __init__(
        self,
        config: PDPConfig,
        style: Optional[PlotStyle] = None,
        mode: str = "absolute"
    ):
        """
        Initialize trajectory visualizer.
        
        Args:
            config: PDP configuration
            style: Plot styling
            mode: Visualization mode ('absolute', 'relative', 'finetuned')
        """
        super().__init__(config, style)
        self.mode = mode
    
    @property
    def output_subfolder(self) -> str:
        return "trajectories"
    
    @property
    def name(self) -> str:
        return f"Trajectory ({self.mode})"
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset
    ) -> List[Path]:
        """
        Generate trajectory visualizations for all configurations.
        
        Args:
            result: PDP analysis result (used for variant name)
            dataset: Dataset containing trajectory data
            
        Returns:
            List of paths to generated images
        """
        generated_files = []
        
        # Get class colors if available
        class_colors, legend_handles = self.get_class_colors(dataset)
        point_colors = self.get_point_colors(dataset.num_points, dataset)
        
        for config_id in range(dataset.num_configurations):
            filepath = self._visualize_single(
                config_id,
                dataset,
                result.variant_name,
                point_colors,
                class_colors,
                legend_handles
            )
            generated_files.append(filepath)
        
        return generated_files
    
    def _get_bounds(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Calculate plot boundaries based on mode."""
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if self.mode == "absolute":
            # Use fixed boundaries from config
            return (
                self.config.boundaries.min_x,
                self.config.boundaries.max_x,
                self.config.boundaries.min_y,
                self.config.boundaries.max_y
            )
        elif self.mode == "relative":
            # Scale to data range with padding
            max_range = max(x_range, y_range)
            padding = max_range * 0.05
            
            if x_range >= y_range:
                return (
                    x_min - padding,
                    x_max + padding,
                    y_min - (max_range - y_range) / 2 - padding,
                    y_max + (max_range - y_range) / 2 + padding
                )
            else:
                return (
                    x_min - (max_range - x_range) / 2 - padding,
                    x_max + (max_range - x_range) / 2 + padding,
                    y_min - padding,
                    y_max + padding
                )
        else:  # finetuned
            # Custom bounds for specific use case (e.g., tennis court)
            x_center = (x_min + x_max) / 2
            return (
                x_center - 100,
                x_center + 100,
                7.25,
                18.0
            )
    
    def _visualize_single(
        self,
        config_id: int,
        dataset: Dataset,
        variant_name: str,
        point_colors: List,
        class_colors: dict,
        legend_handles: List
    ) -> Path:
        """
        Generate trajectory visualization for a single configuration.
        """
        # Get data for this configuration
        df_config = dataset.get_configuration(config_id)
        
        x_vals = df_config['x'].to_numpy()
        y_vals = df_config['y'].to_numpy()
        
        # Calculate bounds
        x_min, x_max, y_min, y_max = self._get_bounds(x_vals, y_vals)
        
        # Create figure
        fig_height = 10 if self.mode == "absolute" else 14
        fig, ax = plt.subplots(figsize=(12, fig_height), dpi=100)
        
        # Track coordinates for final bounds
        all_x, all_y = [], []
        
        if dataset.num_timestamps == 1:
            # Single timestamp: just plot points
            for point_id in range(dataset.num_points):
                mask = df_config['poiID'] == point_id
                point_data = df_config[mask]
                
                if len(point_data) == 0:
                    continue
                
                x = point_data['x'].iloc[0]
                y = point_data['y'].iloc[0]
                all_x.append(x)
                all_y.append(y)
                
                # Get color
                color = self._get_point_color(
                    config_id, 0, point_id, dataset,
                    point_colors, class_colors
                )
                
                ax.scatter(x, y, color=color, s=150, zorder=5)
                ax.text(x, y, f'p{point_id}', fontsize=10, ha='right')
        else:
            # Multiple timestamps: draw arrows
            for point_id in range(dataset.num_points):
                for tst_id in range(dataset.num_timestamps - 1):
                    # Get start and end positions
                    start = dataset.get_timestamp(config_id, tst_id)
                    start = start[start['poiID'] == point_id]
                    
                    end = dataset.get_timestamp(config_id, tst_id + 1)
                    end = end[end['poiID'] == point_id]
                    
                    if len(start) == 0 or len(end) == 0:
                        continue
                    
                    x1, y1 = start['x'].iloc[0], start['y'].iloc[0]
                    x2, y2 = end['x'].iloc[0], end['y'].iloc[0]
                    
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
                    
                    # Get color
                    color = self._get_point_color(
                        config_id, tst_id, point_id, dataset,
                        point_colors, class_colors
                    )
                    
                    # Draw arrow
                    dx, dy = x2 - x1, y2 - y1
                    head_width = (x_max - x_min) / 50
                    head_length = head_width * 1.5
                    
                    ax.arrow(
                        x1, y1, dx, dy,
                        head_width=head_width,
                        head_length=head_length,
                        length_includes_head=True,
                        linewidth=2,
                        color=color,
                        alpha=0.8
                    )
                    
                    # Label first point
                    if tst_id == 0:
                        ax.text(x1, y1, f'p{point_id}', fontsize=10, ha='right')
        
        # Set bounds
        if all_x and all_y:
            padding_x = (max(all_x) - min(all_x)) * 0.1
            padding_y = (max(all_y) - min(all_y)) * 0.1
            ax.set_xlim(min(all_x) - padding_x - 1, max(all_x) + padding_x + 1)
            ax.set_ylim(min(all_y) - padding_y - 1, max(all_y) + padding_y + 1)
        
        # Style
        ax.set_xlabel('X-Axis (m)', fontsize=14)
        ax.set_ylabel('Y-Axis (m)', fontsize=14)
        ax.set_title(f'Configuration {config_id}', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_facecolor('white')
        
        # Add legend if classes present
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title='Class',
                loc='upper right',
                bbox_to_anchor=(1.15, 1)
            )
        
        # Save
        mode_suffix = {'absolute': 'sa', 'relative': 'sr', 'finetuned': 'sf'}
        suffix = f"{mode_suffix.get(self.mode, 'sa')}{config_id}"
        filepath = self.save_figure(fig, variant_name, suffix=suffix)
        return filepath
    
    def _get_point_color(
        self,
        config_id: int,
        tst_id: int,
        point_id: int,
        dataset: Dataset,
        point_colors: List,
        class_colors: dict
    ):
        """Get color for a point, preferring class color if available."""
        if dataset.has_classes and class_colors:
            cls = dataset.get_class_for_point(config_id, tst_id, point_id)
            if cls is not None and cls in class_colors:
                return class_colors[cls]
        
        return point_colors[point_id % len(point_colors)]
