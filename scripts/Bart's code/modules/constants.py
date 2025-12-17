# -*- coding: utf-8 -*-
"""
Constants and configuration defaults for PDP Inverse application.
"""

# ============= Coordinate Precision Settings =============
COORD_DISPLAY_PRECISION = 2   # Decimal places for UI display (hover text, status messages)
COORD_CSV_PRECISION = 3       # Decimal places for CSV export

# ============= Object Labels for Display =============
OBJECT_LABELS = ["k", "l", "m", "n", "p", "q", "r", "s", "u", "v"]

# ============= Default Animation Settings =============
DEFAULT_MAXDIST = 10.0
DEFAULT_WAIT_INTERVAL = 0.5
DEFAULT_NUM_ITERATIONS = 3
DEFAULT_NUM_CONFIGS = 1

# ============= Search Strategy Constants =============
MAX_SEARCH_STEPS = 7
MIN_DISTANCE_THRESHOLD = 1e-5

# ============= Color Palette =============
COLORS = {
    "original": "#1f77b4",      # Blue for original points
    "generated": "#ff7f0e",     # Orange for generated points
    "success": "#2ca02c",       # Green for successful match
    "failure": "#d62728",       # Red for failed match
    "neutral": "#7f7f7f",       # Gray for neutral states
    "highlight": "#9467bd",     # Purple for highlights
}

# ============= Heatmap Colors =============
HEATMAP_COLORS = {
    0: "#00AA00",  # Green (greater precedence)
    1: "#FFFF00",  # Yellow (equal)
    2: "#FF0000",  # Red (less precedence)
}
