# -*- coding: utf-8 -*-
"""
Inverse Modules Package

Modular components for the PDP inverse analysis Streamlit application.
This package breaks down the large inverse.py monolith into maintainable modules.

Module Structure:
- authentication.py: Password protection and user access control
- styling.py: CSS styling and page configuration
- coordinates.py: Coordinate system management and point flattening
- latex_generation.py: LaTeX formula generation for PDP orderings
- lane_drawing.py: Road lane visualization logic
- visualization_matplotlib.py: Static matplotlib plot generation
- visualization_plotly.py: Interactive plotly graph generation
- matrix_heatmaps.py: PDP inequality matrix heatmap visualization
- animation_logic.py: Point generation animation and iteration logic
- pdp_matching.py: PDP order matching and comparison logic
- ui_components.py: Streamlit UI components and controls
"""

__version__ = "1.0.0"
__author__ = "Bart V"

# Import authentication functions
from .authentication import check_password

# Import styling functions
from .styling import setup_page_config, apply_custom_css, show_header

__all__ = [
    # Authentication
    'check_password',
    # Styling
    'setup_page_config',
    'apply_custom_css',
    'show_header',
]
