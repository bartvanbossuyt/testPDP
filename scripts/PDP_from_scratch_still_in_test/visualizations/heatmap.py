"""
Heatmap visualization for PDP distance matrices.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseVisualizer
from ..data.loader import Dataset
from ..core.pdp import PDPResult


class HeatmapVisualizer(BaseVisualizer):
    """
    Creates heatmap visualizations of PDP distance matrices.
    
    The heatmap shows pairwise distances between all configurations,
    with colors indicating similarity (low distance = similar).
    """
    
    @property
    def output_subfolder(self) -> str:
        return "heatmap"
    
    @property
    def name(self) -> str:
        return "Heatmap"
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset
    ) -> List[Path]:
        """
        Generate heatmap visualization.
        
        Args:
            result: PDP analysis result containing distance matrix
            dataset: Original dataset (used for labels)
            
        Returns:
            List containing path to generated heatmap image
        """
        distance_matrix = result.distance_matrix
        n = len(distance_matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, n * 0.5), max(8, n * 0.4)), dpi=300)
        
        # Create heatmap using matshow
        cax = ax.matshow(distance_matrix, cmap='OrRd', vmin=0, vmax=100)
        
        # Add colorbar
        cbar = fig.colorbar(cax, ax=ax, shrink=0.8)
        cbar.set_label('Distance (%)', rotation=270, labelpad=15)
        
        # Set tick marks for grid lines
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        
        # Major ticks for labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(n))
        ax.set_yticklabels(range(n))
        
        # Grid lines
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
        
        # Annotate cells with values (only if matrix is small enough)
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    value = distance_matrix[i, j]
                    text_color = 'white' if value > 50 else 'black'
                    ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                            color=text_color, fontsize=max(6, 10 - n // 4))
        
        ax.set_xlabel('Configuration ID')
        ax.set_ylabel('Configuration ID')
        ax.set_title(f'PDP Distance Matrix Heatmap ({result.variant_name})')
        
        # Move x-axis labels to bottom
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        
        # Save and return
        filepath = self.save_figure(fig, result.variant_name)
        return [filepath]
