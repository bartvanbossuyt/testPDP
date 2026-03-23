"""
Hierarchical clustering visualization for PDP distance matrices.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from .base import BaseVisualizer
from ..data.loader import Dataset
from ..core.pdp import PDPResult


class HierarchicalClusteringVisualizer(BaseVisualizer):
    """
    Creates hierarchical clustering dendrograms from PDP distance matrices.
    
    Uses Ward's method for linkage and displays the resulting dendrogram
    showing how configurations cluster based on their PDP distances.
    """
    
    @property
    def output_subfolder(self) -> str:
        return "hclust"
    
    @property
    def name(self) -> str:
        return "Hierarchical Clustering"
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset,
        linkage_method: str = 'ward'
    ) -> List[Path]:
        """
        Generate hierarchical clustering dendrogram.
        
        Args:
            result: PDP analysis result containing distance matrix
            dataset: Original dataset
            linkage_method: Linkage method ('ward', 'single', 'complete', 'average')
            
        Returns:
            List containing path to generated dendrogram image
        """
        distance_matrix = result.distance_matrix
        n = len(distance_matrix)
        
        # Convert to condensed distance matrix
        condensed = squareform(distance_matrix.astype(float))
        
        # Compute hierarchical clustering
        Z = linkage(condensed, method=linkage_method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(11, n * 0.4), 8), dpi=100)
        
        # Generate labels
        labels = [str(i) for i in range(n)]
        
        # Set dendrogram line width
        matplotlib.rcParams['lines.linewidth'] = 2
        
        # Create dendrogram
        dend = dendrogram(
            Z,
            labels=labels,
            ax=ax,
            color_threshold=0,
            above_threshold_color='steelblue',
            leaf_rotation=45,
            leaf_font_size=max(8, 12 - n // 5)
        )
        
        # Style the plot
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        
        ax.set_xlabel('Configuration ID', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title(f'Hierarchical Clustering ({result.variant_name}, {linkage_method} linkage)')
        
        # Save and return
        filepath = self.save_figure(fig, result.variant_name)
        return [filepath]
