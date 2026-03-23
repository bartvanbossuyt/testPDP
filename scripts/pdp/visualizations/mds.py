"""
MDS (Multidimensional Scaling) visualization for PDP distance matrices.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from .base import BaseVisualizer
from ..data.loader import Dataset
from ..core.pdp import PDPResult


class MDSVisualizer(BaseVisualizer):
    """
    Creates MDS (Multidimensional Scaling) visualizations of PDP distance matrices.
    
    Projects the high-dimensional distance relationships into 2D space
    while preserving pairwise distances as much as possible.
    """
    
    @property
    def output_subfolder(self) -> str:
        return "mds"
    
    @property
    def name(self) -> str:
        return "MDS"
    
    def transform(
        self,
        distance_matrix: np.ndarray,
        n_components: int = 2,
        random_state: int = 1
    ) -> pd.DataFrame:
        """
        Perform MDS transformation.
        
        Args:
            distance_matrix: Square distance matrix
            n_components: Number of output dimensions
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with dimension columns
        """
        mds = MDS(
            n_components=n_components,
            dissimilarity='precomputed',
            random_state=random_state,
            normalized_stress='auto'
        )
        
        embedding = mds.fit_transform(distance_matrix.astype(float))
        
        columns = [f'Dimension {i+1}' for i in range(n_components)]
        return pd.DataFrame(embedding, columns=columns)
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset,
        random_state: Optional[int] = None
    ) -> List[Path]:
        """
        Generate MDS visualization.
        
        Args:
            result: PDP analysis result containing distance matrix
            dataset: Original dataset (used for potential coloring)
            random_state: Random seed (uses config value if None)
            
        Returns:
            List containing path to generated MDS plot
        """
        if random_state is None:
            random_state = self.config.random_seed
        
        distance_matrix = result.distance_matrix
        n = len(distance_matrix)
        
        # Perform MDS transformation
        embedding = self.transform(distance_matrix, random_state=random_state)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 8), dpi=100)
        
        # Plot points
        ax.scatter(
            embedding['Dimension 1'],
            embedding['Dimension 2'],
            s=80,
            c='steelblue',
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8
        )
        
        # Annotate points with configuration IDs
        for i in range(n):
            ax.annotate(
                str(i),
                xy=(embedding.iloc[i, 0], embedding.iloc[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
        
        # Style axes
        self.setup_axes(ax)
        
        # Add grid
        ax.xaxis.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'MDS Projection ({result.variant_name})')
        
        # Save and return
        filepath = self.save_figure(fig, result.variant_name)
        return [filepath]
