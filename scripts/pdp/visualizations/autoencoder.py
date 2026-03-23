"""
Autoencoder-based dimensionality reduction visualization.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import BaseVisualizer
from ..data.loader import Dataset
from ..core.pdp import PDPResult


class AutoencoderVisualizer(BaseVisualizer):
    """
    Creates autoencoder-based dimensionality reduction visualizations.
    
    Uses a neural network autoencoder to learn a 2D embedding of the
    distance matrix, providing an alternative to MDS and t-SNE.
    """
    
    @property
    def output_subfolder(self) -> str:
        return "autoencoder"
    
    @property
    def name(self) -> str:
        return "Autoencoder"
    
    def transform(
        self,
        distance_matrix: np.ndarray,
        encoding_dim: int = 2,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: int = 0
    ) -> tuple:
        """
        Perform autoencoder transformation.
        
        Args:
            distance_matrix: Square distance matrix
            encoding_dim: Bottleneck layer size (embedding dimensions)
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Keras verbosity
            
        Returns:
            Tuple of (embedding DataFrame, final MSE loss)
        """
        try:
            from tensorflow.keras.layers import Input, Dense
            from tensorflow.keras.models import Model
        except ImportError:
            raise ImportError(
                "TensorFlow is required for autoencoder visualization. "
                "Install with: pip install tensorflow"
            )
        
        input_dim = distance_matrix.shape[1]
        
        # Build autoencoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        # Train
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        history = autoencoder.fit(
            distance_matrix.astype(float),
            distance_matrix.astype(float),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=verbose
        )
        
        # Get embedding
        embedding = encoder.predict(distance_matrix.astype(float), verbose=0)
        df_embedding = pd.DataFrame(
            embedding,
            columns=[f'Dimension {i+1}' for i in range(encoding_dim)]
        )
        
        final_mse = history.history['loss'][-1]
        
        return df_embedding, final_mse
    
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset,
        epochs: int = 100
    ) -> List[Path]:
        """
        Generate autoencoder visualization.
        
        Args:
            result: PDP analysis result containing distance matrix
            dataset: Original dataset
            epochs: Training epochs for autoencoder
            
        Returns:
            List containing path to generated plot
        """
        distance_matrix = result.distance_matrix
        n = len(distance_matrix)
        
        # Perform transformation
        embedding, final_mse = self.transform(distance_matrix, epochs=epochs)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # Plot points
        ax.scatter(
            embedding['Dimension 1'],
            embedding['Dimension 2'],
            s=80,
            c='purple',
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
        
        # Style
        self.setup_axes(ax)
        
        ax.xaxis.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(
            f'Autoencoder Projection ({result.variant_name})\n'
            f'Final MSE: {final_mse:.5f}',
            fontsize=12
        )
        
        # Save
        filepath = self.save_figure(fig, result.variant_name)
        return [filepath]
