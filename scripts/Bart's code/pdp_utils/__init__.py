# -*- coding: utf-8 -*-
"""
PDP Utilities Package

This package contains utility modules for the PDP inverse application:
- core: Core PDP functions (inequality matrices, transformations)
- config: Lane configurations and settings
- data_loading: CSV loading and DataFrame processing
- order_comparison: PDP order matching functions
- drawing: Matplotlib/Plotly drawing utilities
"""

from .core import (
    COORD_DISPLAY_PRECISION,
    COORD_CSV_PRECISION,
    OBJECT_LABELS,
    SuccessfulPoint,
    compute_inequality_matrix,
    compare_inequality_matrices,
    compare_inequality_matrices_with_threshold,
    apply_buffer_transformation,
)

from .config import LANE_CONFIGURATIONS, DEFAULT_LANE_SETUP

from .data_loading import (
    to_numeric_series,
    read_clean_df,
    load_points_from_df,
    extract_points_from_df,
    get_available_configs,
    get_available_objects,
    get_time_range,
    get_coordinate_bounds,
)

from .order_comparison import (
    strip_primes,
    extract_order_string,
    check_pdp_match,
    check_pdp_match_detailed,
)

from .drawing import (
    BLUE,
    ORANGE,
    LABEL_FS,
    OBJECT_COLORS,
    OBJECT_COLORS_PLOTLY,
    setup_square_axes_basic,
    render_square_matplotlib_figure_basic,
    remove_duplicate_points,
    extract_longest_object_path,
    compute_perpendicular_offset,
)

__all__ = [
    # core
    "COORD_DISPLAY_PRECISION",
    "COORD_CSV_PRECISION", 
    "OBJECT_LABELS",
    "SuccessfulPoint",
    "compute_inequality_matrix",
    "compare_inequality_matrices",
    "compare_inequality_matrices_with_threshold",
    "apply_buffer_transformation",
    # config
    "LANE_CONFIGURATIONS",
    "DEFAULT_LANE_SETUP",
    # data_loading
    "to_numeric_series",
    "read_clean_df",
    "load_points_from_df",
    "extract_points_from_df",
    "get_available_configs",
    "get_available_objects",
    "get_time_range",
    "get_coordinate_bounds",
    # order_comparison
    "strip_primes",
    "extract_order_string",
    "check_pdp_match",
    "check_pdp_match_detailed",
    # drawing
    "BLUE",
    "ORANGE",
    "LABEL_FS",
    "OBJECT_COLORS",
    "OBJECT_COLORS_PLOTLY",
    "setup_square_axes_basic",
    "render_square_matplotlib_figure_basic",
    "remove_duplicate_points",
    "extract_longest_object_path",
    "compute_perpendicular_offset",
]
