"""
Base visualization classes and utilities for PDP Analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from ..config import PDPConfig
from ..data.loader import Dataset
from ..core.pdp import PDPResult


@dataclass
class PlotStyle:
    """Common plot styling configuration."""
    font_family: str = 'monospace'
    font_size: int = 12
    figure_dpi: int = 300
    colormap: str = 'cividis'
    background_color: str = 'white'
    grid_color: str = 'black'
    grid_alpha: float = 0.5
    grid_style: str = 'dotted'
    
    def apply(self):
        """Apply style settings to matplotlib."""
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.size'] = self.font_size
        mpl.rcParams['figure.dpi'] = self.figure_dpi


class BaseVisualizer(ABC):
    """
    Abstract base class for all PDP visualizations.
    
    Provides common functionality for saving plots, styling, and
    handling different PDP variants.
    """
    
    def __init__(
        self,
        config: PDPConfig,
        style: Optional[PlotStyle] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            config: PDP configuration
            style: Plot styling (uses default if None)
        """
        self.config = config
        self.style = style or PlotStyle()
        self.style.apply()
    
    @property
    @abstractmethod
    def output_subfolder(self) -> str:
        """Subfolder name for this visualizer's output."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this visualizer."""
        pass
    
    def get_output_path(self, variant_name: str) -> Path:
        """Get output directory for this variant."""
        return self.config.get_output_path(self.output_subfolder)
    
    def get_filename(self, variant_name: str, suffix: str = "") -> str:
        """Generate standardized filename."""
        base = f"N_C_PDPg_{variant_name}_{self.output_subfolder}"
        if suffix:
            base = f"{base}_{suffix}"
        return f"{base}.png"
    
    def save_figure(
        self,
        fig: plt.Figure,
        variant_name: str,
        suffix: str = "",
        close: bool = True
    ) -> Path:
        """
        Save figure to appropriate output location.
        
        Args:
            fig: Matplotlib figure to save
            variant_name: PDP variant name
            suffix: Optional suffix for filename
            close: Whether to close figure after saving
            
        Returns:
            Path to saved file
        """
        output_dir = self.get_output_path(variant_name)
        filename = self.get_filename(variant_name, suffix)
        filepath = output_dir / filename
        
        fig.savefig(filepath, dpi=self.style.figure_dpi, bbox_inches='tight')
        
        if close:
            plt.close(fig)
        
        return filepath
    
    @abstractmethod
    def visualize(
        self,
        result: PDPResult,
        dataset: Dataset
    ) -> List[Path]:
        """
        Generate visualization(s).
        
        Args:
            result: PDP analysis result
            dataset: Original dataset
            
        Returns:
            List of paths to generated files
        """
        pass
    
    def get_point_colors(
        self,
        num_points: int,
        dataset: Optional[Dataset] = None
    ) -> List[str]:
        """
        Get colors for points.
        
        Args:
            num_points: Number of points
            dataset: Optional dataset for class-based coloring
            
        Returns:
            List of color values
        """
        if num_points == 3:
            return ['black', 'blue', 'magenta']
        
        cmap = plt.cm.get_cmap(self.style.colormap)
        return [cmap(i / num_points) for i in range(num_points)]
    
    def get_class_colors(
        self,
        dataset: Dataset
    ) -> Tuple[dict, list]:
        """
        Get color mapping for classes.
        
        Args:
            dataset: Dataset with class information
            
        Returns:
            Tuple of (class_to_color dict, legend_handles list)
        """
        import matplotlib.patches as patches
        
        if not dataset.has_classes:
            return {}, []
        
        classes = dataset.get_unique_classes()
        cmap = plt.cm.get_cmap('tab20')
        
        color_map = {}
        handles = []
        
        for idx, cls in enumerate(classes):
            color = cmap(idx % cmap.N)
            color_map[cls] = color
            handles.append(patches.Patch(color=color, label=str(cls)))
        
        return color_map, handles
    
    def setup_axes(
        self,
        ax: plt.Axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Apply common axes styling.
        
        Args:
            ax: Axes to style
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
            title: Optional title
        """
        ax.set_facecolor(self.style.background_color)
        ax.spines['bottom'].set_color(self.style.grid_color)
        ax.spines['left'].set_color(self.style.grid_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
