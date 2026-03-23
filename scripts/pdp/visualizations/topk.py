"""
Top-K nearest neighbor visualization for PDP distance matrices.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseVisualizer
from ..data.loader import Dataset
from ..core.pdp import PDPResult


class TopKVisualizer(BaseVisualizer):
    """
    Creates Top-K nearest neighbor bar charts for each configuration.
    
    For each configuration, shows all other configurations sorted by
    their PDP distance, making it easy to identify nearest neighbors.
    """
    
    @property
    def output_subfolder(self) -> str:
        return "topk"
    
    @property
    def name(self) -> str:
        return "Top-K"
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset
    ) -> List[Path]:
        """
        Generate Top-K visualizations for all configurations.
        
        Args:
            result: PDP analysis result containing distance matrix
            dataset: Original dataset
            
        Returns:
            List of paths to generated bar chart images
        """
        distance_matrix = result.distance_matrix
        n = len(distance_matrix)
        generated_files = []
        
        for config_id in range(n):
            filepath = self._visualize_single(
                config_id,
                distance_matrix[config_id],
                result.variant_name
            )
            generated_files.append(filepath)
        
        return generated_files
    
    def _visualize_single(
        self,
        config_id: int,
        distances: np.ndarray,
        variant_name: str
    ) -> Path:
        """
        Generate Top-K visualization for a single configuration.
        
        Args:
            config_id: Configuration ID
            distances: Distance values to all configurations
            variant_name: PDP variant name
            
        Returns:
            Path to generated image
        """
        n = len(distances)
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(8, n * 0.4), 6), dpi=100)
        
        # Create bar chart
        labels = [str(idx) for idx in sorted_indices]
        bars = ax.bar(
            range(n),
            sorted_distances,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Highlight the reference configuration (distance = 0)
        for i, idx in enumerate(sorted_indices):
            if idx == config_id:
                bars[i].set_color('coral')
                break
        
        # Style axes
        self.setup_axes(ax)
        
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=max(6, 10 - n // 5))
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 110, 10))
        
        ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Configuration ID (sorted by distance)', fontsize=10)
        ax.set_ylabel('Distance', fontsize=10)
        ax.set_title(f'Top-K for Configuration {config_id} ({variant_name})', fontsize=12)
        
        # Save
        filepath = self.save_figure(fig, variant_name, suffix=f"c{config_id}")
        return filepath
