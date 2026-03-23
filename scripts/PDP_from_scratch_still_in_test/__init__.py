"""
PDP Analysis Package - Pairwise Distance Pattern Analysis for Moving Objects

This package provides tools for analyzing trajectory configurations using
the Pairwise Distance Pattern (PDP) methodology.

Main components:
- config: Configuration management via dataclasses
- data: Dataset loading and transformations
- core: Core PDP algorithm and distance calculations
- visualizations: Various visualization methods
"""

__version__ = "2.0.0"
__author__ = "PDP Analysis Team"

from .config import PDPConfig, load_config
from .main import run_analysis

__all__ = ["PDPConfig", "load_config", "run_analysis", "__version__"]
