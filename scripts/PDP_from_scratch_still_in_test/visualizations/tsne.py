"""
t-SNE visualization for PDP distance matrices.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from .base import BaseVisualizer
from ..data.loader import Dataset
from ..core.pdp import PDPResult


class TSNEVisualizer(BaseVisualizer):
    """
    Creates t-SNE visualizations of PDP distance matrices.
    
    Uses t-distributed Stochastic Neighbor Embedding to project
    the distance relationships into 2D space, emphasizing local structure.
    """
    
    @property
    def output_subfolder(self) -> str:
        return "tsne"
    
    @property
    def name(self) -> str:
        return "t-SNE"
    
    def transform(
        self,
        distance_matrix: np.ndarray,
        perplexity: Optional[int] = None,
        random_state: int = 0
    ) -> pd.DataFrame:
        """
        Perform t-SNE transformation.
        
        Args:
            distance_matrix: Square distance matrix
            perplexity: t-SNE perplexity (auto-computed if None)
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with dimension columns
        """
        n = len(distance_matrix)
        
        # Auto-compute perplexity if not specified
        if perplexity is None:
            perplexity = min(30, max(5, n // 3))
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='random',
            random_state=random_state,
            metric='precomputed'
        )
        
        embedding = tsne.fit_transform(distance_matrix.astype(float))
        
        return pd.DataFrame(embedding, columns=['Dimension 1', 'Dimension 2'])
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset,
        perplexity: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> List[Path]:
        """
        Generate t-SNE visualization.
        
        Args:
            result: PDP analysis result containing distance matrix
            dataset: Original dataset
            perplexity: t-SNE perplexity (auto if None)
            random_state: Random seed (uses config value if None)
            
        Returns:
            List containing path to generated t-SNE plot
        """
        if random_state is None:
            random_state = self.config.random_seed
        
        distance_matrix = result.distance_matrix
        n = len(distance_matrix)
        
        # Perform t-SNE transformation
        embedding = self.transform(
            distance_matrix,
            perplexity=perplexity,
            random_state=random_state
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 8), dpi=100)
        
        # Plot points
        ax.scatter(
            embedding['Dimension 1'],
            embedding['Dimension 2'],
            s=80,
            c='coral',
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8
        )
        
        # Annotate points
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
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f't-SNE Projection ({result.variant_name})')
        
        # Save and return
        filepath = self.save_figure(fig, result.variant_name)
        return [filepath]
