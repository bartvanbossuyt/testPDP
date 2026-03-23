"""
Configuration management for PDP Analysis.

Uses dataclasses for type-safe, immutable configuration objects.
Supports loading from YAML/JSON files or creating programmatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import os


@dataclass(frozen=True)
class PDPVariant:
    """Configuration for a specific PDP variant."""
    enabled: bool = False
    buffer_x: float = 15.0
    buffer_y: float = 1.0
    rough_x: float = 30.0
    rough_y: float = 3.0


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for visualization outputs."""
    static_absolute: bool = True
    static_relative: bool = False
    static_finetuned: bool = False
    dynamic_absolute: bool = False
    heatmap: bool = True
    hclust: bool = True
    mds: bool = True
    tsne: bool = True
    topk: bool = True
    autoencoder: bool = False
    inequality_matrices: bool = False


@dataclass(frozen=True)
class BoundaryConfig:
    """Spatial boundary configuration."""
    min_x: float = -150.0
    max_x: float = 150.0
    min_y: float = -150.0
    max_y: float = 150.0


@dataclass
class PDPConfig:
    """
    Main configuration class for PDP Analysis.
    
    Attributes:
        dataset_path: Path to the input CSV dataset
        output_folder: Directory for all outputs
        window_length: Number of timestamps in sliding window
        visualizations: Which visualizations to generate
        boundaries: Spatial boundaries for plots
        fundamental: Configuration for fundamental PDP
        buffer: Configuration for buffer PDP
        rough: Configuration for rough PDP
        buffer_rough: Configuration for buffer+rough PDP
        num_frames: Number of frames for animations
        random_seed: Random seed for reproducibility
    """
    dataset_path: str
    output_folder: str
    window_length: int = 3
    visualizations: VisualizationConfig = field(default_factory=VisualizationConfig)
    boundaries: BoundaryConfig = field(default_factory=BoundaryConfig)
    
    # PDP variants
    run_fundamental: bool = True
    run_buffer: bool = False
    run_rough: bool = False
    run_buffer_rough: bool = False
    
    # Buffer/rough parameters
    buffer_x: float = 15.0
    buffer_y: float = 1.0
    rough_x: float = 30.0
    rough_y: float = 3.0
    
    # Animation settings
    num_frames: int = 20
    
    # Reproducibility
    random_seed: int = 42
    
    # Derived attributes (computed after loading data)
    num_configurations: int = 0
    num_timestamps: int = 0
    num_points: int = 0
    has_classes: bool = False
    
    def __post_init__(self):
        """Ensure paths are absolute and output folder exists."""
        self.dataset_path = os.path.abspath(os.path.expanduser(self.dataset_path))
        self.output_folder = os.path.abspath(os.path.expanduser(self.output_folder))
        
    def get_output_path(self, *subfolders: str) -> Path:
        """Get an output path, creating subdirectories as needed."""
        path = Path(self.output_folder)
        for subfolder in subfolders:
            path = path / subfolder
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_active_variants(self) -> List[str]:
        """Return list of enabled PDP variants."""
        variants = []
        if self.run_fundamental:
            variants.append("fundamental")
        if self.run_buffer:
            variants.append("buffer")
        if self.run_rough:
            variants.append("rough")
        if self.run_buffer_rough:
            variants.append("buffer_rough")
        return variants
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "dataset_path": self.dataset_path,
            "output_folder": self.output_folder,
            "window_length": self.window_length,
            "run_fundamental": self.run_fundamental,
            "run_buffer": self.run_buffer,
            "run_rough": self.run_rough,
            "run_buffer_rough": self.run_buffer_rough,
            "buffer_x": self.buffer_x,
            "buffer_y": self.buffer_y,
            "rough_x": self.rough_x,
            "rough_y": self.rough_y,
            "num_frames": self.num_frames,
            "random_seed": self.random_seed,
            "visualizations": {
                "static_absolute": self.visualizations.static_absolute,
                "static_relative": self.visualizations.static_relative,
                "static_finetuned": self.visualizations.static_finetuned,
                "dynamic_absolute": self.visualizations.dynamic_absolute,
                "heatmap": self.visualizations.heatmap,
                "hclust": self.visualizations.hclust,
                "mds": self.visualizations.mds,
                "tsne": self.visualizations.tsne,
                "topk": self.visualizations.topk,
                "autoencoder": self.visualizations.autoencoder,
            },
            "boundaries": {
                "min_x": self.boundaries.min_x,
                "max_x": self.boundaries.max_x,
                "min_y": self.boundaries.min_y,
                "max_y": self.boundaries.max_y,
            }
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(filepath: str) -> PDPConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to JSON configuration file
        
    Returns:
        PDPConfig instance
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract nested configs
    viz_data = data.pop("visualizations", {})
    boundary_data = data.pop("boundaries", {})
    
    visualizations = VisualizationConfig(**viz_data) if viz_data else VisualizationConfig()
    boundaries = BoundaryConfig(**boundary_data) if boundary_data else BoundaryConfig()
    
    return PDPConfig(
        visualizations=visualizations,
        boundaries=boundaries,
        **data
    )


def create_default_config(dataset_path: str, output_folder: str) -> PDPConfig:
    """
    Create a default configuration with sensible defaults.
    
    Args:
        dataset_path: Path to input dataset
        output_folder: Path for outputs
        
    Returns:
        PDPConfig with default settings
    """
    return PDPConfig(
        dataset_path=dataset_path,
        output_folder=output_folder,
    )
