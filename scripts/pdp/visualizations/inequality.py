"""
Inequality Matrix visualization for PDP Analysis.

Visualizes the inequality matrices that compare positional relationships
between points, which form the basis of the PDP distance calculations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .base import BaseVisualizer, PlotStyle
from ..config import PDPConfig
from ..data.loader import Dataset
from ..core.pdp import PDPResult, PDPAnalyzer


class InequalityVisualizer(BaseVisualizer):
    """
    Visualizer for inequality matrices.
    
    Creates heatmap visualizations of the inequality matrices showing
    positional relationships between points across timestamps.
    
    Values in the matrix:
    - 0 (blue): point_i < point_j
    - 1 (white): point_i ≈ point_j (equal within roughness)
    - 2 (red): point_i > point_j
    """
    
    @property
    def output_subfolder(self) -> str:
        return "inequality_matrices"
    
    @property
    def name(self) -> str:
        return "Inequality Matrices"
    
    def __init__(
        self,
        config: PDPConfig,
        style: Optional[PlotStyle] = None
    ):
        super().__init__(config, style)
        self.output_dir = Path(config.output_path) / "inequality_matrices"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset,
        configs_to_show: Optional[List[int]] = None,
        max_configs: int = 8
    ) -> List[str]:
        """
        Generate inequality matrix visualizations.
        
        Args:
            result: PDP analysis result
            dataset: The input dataset
            configs_to_show: Specific configuration IDs to visualize 
                             (None = auto-select up to max_configs)
            max_configs: Maximum number of configurations to visualize
            
        Returns:
            List of generated file paths
        """
        return self.generate(dataset, result, configs_to_show, max_configs)
    
    def generate(
        self,
        dataset: Dataset,
        result: PDPResult,
        configs_to_show: Optional[List[int]] = None,
        max_configs: int = 8
    ) -> List[str]:
        """
        Generate inequality matrix visualizations.
        
        Args:
            dataset: The input dataset
            result: PDP analysis result
            configs_to_show: Specific configuration IDs to visualize 
                             (None = auto-select up to max_configs)
            max_configs: Maximum number of configurations to visualize
            
        Returns:
            List of generated file paths
        """
        generated_files = []
        
        # Determine which configurations to visualize
        if configs_to_show is None:
            n_configs = min(dataset.num_configurations, max_configs)
            configs_to_show = list(range(n_configs))
        
        # Recompute inequality matrices (they may not be stored in result)
        analyzer = PDPAnalyzer(
            dataset,
            window_length=self.config.window_length,
            rough_x=self.config.rough_x,
            rough_y=self.config.rough_y
        )
        inequality_matrices = analyzer.compute_all_inequality_matrices()
        
        # Number of windows
        num_windows = dataset.num_timestamps - self.config.window_length + 1
        
        # Generate visualizations for each configuration
        for config_id in configs_to_show:
            files = self._visualize_config(
                config_id,
                inequality_matrices,
                num_windows,
                dataset.num_points,
                result.variant_name
            )
            generated_files.extend(files)
        
        # Also save matrices as CSV
        csv_files = self._save_matrices_csv(
            inequality_matrices,
            configs_to_show,
            num_windows,
            result.variant_name
        )
        generated_files.extend(csv_files)
        
        return generated_files
    
    def _visualize_config(
        self,
        config_id: int,
        inequality_matrices: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
        num_windows: int,
        num_points: int,
        variant_name: str
    ) -> List[str]:
        """Generate visualization for a single configuration."""
        generated_files = []
        
        # Create figure with subplots for each window
        # 2 rows (X and Y), num_windows columns
        fig, axes = plt.subplots(
            2, num_windows,
            figsize=(4 * num_windows, 8),
            squeeze=False
        )
        
        # Custom colormap: blue (0) -> white (1) -> red (2)
        colors = ['#2166ac', '#f7f7f7', '#b2182b']  # Blue, White, Red
        cmap = mcolors.LinearSegmentedColormap.from_list('inequality', colors, N=3)
        
        for window_idx in range(num_windows):
            x_matrix, y_matrix = inequality_matrices[(config_id, window_idx)]
            
            # X dimension
            ax_x = axes[0, window_idx]
            im_x = ax_x.imshow(x_matrix, cmap=cmap, vmin=0, vmax=2, aspect='equal')
            ax_x.set_title(f'X - Window {window_idx}', fontsize=10)
            ax_x.set_xlabel('Point j')
            ax_x.set_ylabel('Point i')
            self._add_grid(ax_x, x_matrix.shape[0])
            
            # Y dimension
            ax_y = axes[1, window_idx]
            im_y = ax_y.imshow(y_matrix, cmap=cmap, vmin=0, vmax=2, aspect='equal')
            ax_y.set_title(f'Y - Window {window_idx}', fontsize=10)
            ax_y.set_xlabel('Point j')
            ax_y.set_ylabel('Point i')
            self._add_grid(ax_y, y_matrix.shape[0])
        
        # Add colorbar
        cbar = fig.colorbar(im_x, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['< (less)', '≈ (equal)', '> (greater)'])
        
        # Title
        fig.suptitle(
            f'Inequality Matrices - Config {config_id} ({variant_name})\n'
            f'Comparing positional relationships between points',
            fontsize=12, fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save
        filename = f'N_C_PDPg_{variant_name}_inequality_config{config_id}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.style.figure_dpi, bbox_inches='tight')
        plt.close(fig)
        
        generated_files.append(str(filepath))
        return generated_files
    
    def _add_grid(self, ax, size: int):
        """Add grid lines to matrix visualization."""
        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(
            which='minor',
            color=self.style.grid_color,
            linestyle='-',
            linewidth=0.5,
            alpha=0.3
        )
        ax.tick_params(which='minor', size=0)
    
    def _save_matrices_csv(
        self,
        inequality_matrices: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
        configs: List[int],
        num_windows: int,
        variant_name: str
    ) -> List[str]:
        """Save inequality matrices to CSV files."""
        generated_files = []
        
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        for config_id in configs:
            for window_idx in range(num_windows):
                x_matrix, y_matrix = inequality_matrices[(config_id, window_idx)]
                
                # Save X matrix
                x_filename = f'inequality_x_config{config_id}_window{window_idx}.csv'
                x_filepath = csv_dir / x_filename
                self._save_matrix_csv(x_matrix, x_filepath)
                generated_files.append(str(x_filepath))
                
                # Save Y matrix
                y_filename = f'inequality_y_config{config_id}_window{window_idx}.csv'
                y_filepath = csv_dir / y_filename
                self._save_matrix_csv(y_matrix, y_filepath)
                generated_files.append(str(y_filepath))
        
        return generated_files
    
    def _save_matrix_csv(self, matrix: np.ndarray, filepath: Path):
        """Save a single matrix to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in matrix:
                writer.writerow(row.tolist())


def visualize_inequality_matrices(
    config: PDPConfig,
    dataset: Dataset,
    result: PDPResult,
    configs_to_show: Optional[List[int]] = None,
    max_configs: int = 8
) -> List[str]:
    """
    Convenience function to generate inequality matrix visualizations.
    
    Args:
        config: PDP configuration
        dataset: Input dataset
        result: PDP analysis result
        configs_to_show: Specific configurations to visualize
        max_configs: Maximum number of configurations to visualize
        
    Returns:
        List of generated file paths
    """
    visualizer = InequalityVisualizer(config)
    return visualizer.generate(dataset, result, configs_to_show, max_configs)
