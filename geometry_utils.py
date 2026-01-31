"""
Geometry utilities for PDP inverse problem.

This module contains pure geometry and math helper functions used across
the inverse analysis application. All functions are stateless and don't
depend on Streamlit session state.

Functions:
    - max_consecutive_dist: Maximum distance between consecutive points
    - square_limits_with_margin: Compute square axis limits with margin
    - format_t_subscript: Format timestamp as integer or float string
    - to_numeric_safe: Safely convert pandas Series to numeric

Constants:
    - DEFAULT_MARGIN: Default margin percentage for axis limits
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============= Constants =============
DEFAULT_MARGIN = 0.10  # 10% margin for axis limits


# ============= Pandas Utilities =============
def to_numeric_safe(s: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric, coercing bad values to NaN.
    
    This is a Pylance-friendly wrapper around pd.to_numeric.
    
    Args:
        s: Input pandas Series
        
    Returns:
        Numeric Series with non-numeric values converted to NaN
    """
    out = pd.to_numeric(s, errors="coerce")  # type: ignore[call-overload]
    return out


# ============= Geometry Functions =============
def max_consecutive_dist(pts: np.ndarray) -> float:
    """
    Return the maximum distance between consecutive points in an array.
    
    This is useful for determining appropriate step sizes in search
    algorithms based on the spacing of existing data points.
    
    Args:
        pts: (N, 2) array of 2D points
        
    Returns:
        Maximum Euclidean distance between consecutive points,
        or 0.0 if fewer than 2 points
    """
    n = pts.shape[0]
    if n < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    return float(np.max(dists))


def square_limits_with_margin(
    pts: np.ndarray,
    margin: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute square axis limits around points with a given margin.
    
    Ensures a square window and at least 'margin' distance from points
    to borders. The margin is adjusted based on data range.
    
    Args:
        pts: (N, 2) array of 2D points
        margin: Minimum margin distance from points to borders
        
    Returns:
        Tuple of (xlim, ylim) where each is a (min, max) tuple
    """
    xmin = float(np.min(pts[:, 0]))
    xmax = float(np.max(pts[:, 0]))
    ymin = float(np.min(pts[:, 1]))
    ymax = float(np.max(pts[:, 1]))
    
    # Calculate data range
    data_w = xmax - xmin
    data_h = ymax - ymin
    data_range = max(data_w, data_h, 1.0)  # Ensure minimum range of 1.0
    
    # Use at least 10% of data range as margin, or the provided margin, whichever is larger
    effective_margin = max(margin, data_range * DEFAULT_MARGIN, 5.0)  # At least 5 units margin
    
    xmin -= effective_margin
    xmax += effective_margin
    ymin -= effective_margin
    ymax += effective_margin

    w = xmax - xmin
    h = ymax - ymin
    side = max(w, h)
    if side <= 0:
        side = 1.0

    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)

    xlim = (cx - side / 2.0, cx + side / 2.0)
    ylim = (cy - side / 2.0, cy + side / 2.0)
    return xlim, ylim


def compute_square_limits_from_bounds(
    coord_min_x: float,
    coord_max_x: float,
    coord_min_y: float,
    coord_max_y: float,
    margin_percent: float = DEFAULT_MARGIN,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute square axis limits from coordinate bounds with percentage-based margin.
    
    Args:
        coord_min_x: Minimum x coordinate
        coord_max_x: Maximum x coordinate
        coord_min_y: Minimum y coordinate
        coord_max_y: Maximum y coordinate
        margin_percent: Margin as fraction of coordinate range (default 0.10 = 10%)
        
    Returns:
        Tuple of (xlim, ylim) where each is a (min, max) tuple
    """
    coord_width = coord_max_x - coord_min_x
    coord_height = coord_max_y - coord_min_y
    
    # Use percentage of each range as margin
    margin_x = coord_width * margin_percent
    margin_y = coord_height * margin_percent
    
    # Make it square by using the larger dimension (including margins)
    total_width = coord_width + 2 * margin_x
    total_height = coord_height + 2 * margin_y
    viz_side = max(total_width, total_height)
    
    # Center of coordinate bounds
    coord_cx = 0.5 * (coord_min_x + coord_max_x)
    coord_cy = 0.5 * (coord_min_y + coord_max_y)
    
    xlim = (coord_cx - viz_side / 2.0, coord_cx + viz_side / 2.0)
    ylim = (coord_cy - viz_side / 2.0, coord_cy + viz_side / 2.0)
    return xlim, ylim


# ============= Formatting Utilities =============
def format_t_subscript(tval: float) -> str:
    """
    Format a timestamp value as an integer subscript if possible, otherwise as a float.
    
    Examples:
        format_t_subscript(3.0) -> "3"
        format_t_subscript(3.5) -> "3.5"
        format_t_subscript(0.0) -> "0"
    
    Args:
        tval: Timestamp value to format
        
    Returns:
        String representation suitable for use in LaTeX subscripts
    """
    try:
        tnum = float(tval)
    except Exception:
        tnum = float(np.array(tval, dtype=float))
    return str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"


def compute_maxdist_for_points(all_points_plot: dict[int, np.ndarray]) -> float:
    """
    Calculate the maximum distance for movement vectors from point data.
    
    Uses consecutive point distances if available, otherwise pairwise distances.
    Falls back to 10.0 as default.
    
    Args:
        all_points_plot: Dict mapping object ID to (N, 2) point arrays
        
    Returns:
        Maximum distance value suitable for search step size
    """
    # Calculate maxdist from all objects using consecutive distances
    all_max_dists = [max_consecutive_dist(pts) for pts in all_points_plot.values() if pts.shape[0] > 0]
    maxdist_consecutive = max(all_max_dists) if all_max_dists else 0.0
    
    if maxdist_consecutive > 0:
        return maxdist_consecutive
    
    # For single timestamp: use distance between all pairs of points
    all_point_arrays: list[np.ndarray] = [pts for pts in all_points_plot.values() if pts.shape[0] > 0]
    
    if len(all_point_arrays) > 1:
        all_pts = np.vstack(all_point_arrays)
        pairwise_dists: list[float] = []
        for i in range(all_pts.shape[0]):
            for j in range(i + 1, all_pts.shape[0]):
                d = float(np.hypot(
                    all_pts[i, 0] - all_pts[j, 0],
                    all_pts[i, 1] - all_pts[j, 1]
                ))
                pairwise_dists.append(d)
        return float(max(pairwise_dists)) if pairwise_dists else 10.0
    
    return 10.0  # Default fallback
