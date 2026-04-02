# -*- coding: utf-8 -*-
"""
Drawing utilities for matplotlib and plotly.
Contains color definitions and basic plotting helpers.
"""

from typing import Tuple, Callable
import numpy as np
import matplotlib.axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Color scheme
BLUE = "C0"
ORANGE = "C1"
LABEL_FS = 9

# Colors for all objects (matplotlib style)
OBJECT_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

# Plotly-compatible colors (hex equivalents)
OBJECT_COLORS_PLOTLY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def setup_square_axes_basic(
    ax: matplotlib.axes.Axes, 
    xlim: Tuple[float, float], 
    ylim: Tuple[float, float]
) -> None:
    """Configure axes to be square with d₁/d₂ labels (without lane drawing)."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)
        sp.set_color("#222")
    ax.tick_params(axis="both", labelsize=9, width=0.8, color="#222")
    ax.set_xlabel("d₁", fontsize=11, labelpad=8)
    ax.set_ylabel("d₂", fontsize=11, labelpad=8)


def render_square_matplotlib_figure_basic(
    draw_fn: Callable[[matplotlib.axes.Axes], None],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    size_inches: float = 5.5,
    dpi: int = 160
) -> Figure:
    """Create a square Matplotlib figure and call draw_fn(ax) inside it."""
    fig = Figure(figsize=(size_inches, size_inches), dpi=dpi)
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    setup_square_axes_basic(ax, xlim, ylim)
    draw_fn(ax)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
    return fig


def remove_duplicate_points(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Remove consecutive duplicate points within tolerance."""
    if points.size == 0:
        return points
    filtered = [points[0]]
    for pt in points[1:]:
        if np.linalg.norm(pt - filtered[-1]) > tolerance:
            filtered.append(pt)
    return np.array(filtered, dtype=float)


def extract_longest_object_path(config_df) -> np.ndarray | None:
    """Extract the longest trajectory path from a configuration DataFrame."""
    best_pts = None
    best_score = -np.inf
    for obj_id, obj_df in config_df.groupby("o"):
        obj_sorted = obj_df.sort_values("t")
        pts = obj_sorted[["x", "y"]].to_numpy(dtype=float)
        pts = remove_duplicate_points(pts)
        if pts.shape[0] < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        score = total_length + (10.0 if obj_id == 0 else 0.0)
        if score > best_score:
            best_score = score
            best_pts = pts
    return best_pts


def compute_perpendicular_offset(centerline: np.ndarray, offset: float) -> np.ndarray:
    """
    Compute offset polyline perpendicular to the centerline.
    Positive offset goes to the "left" (counter-clockwise).
    """
    if centerline.shape[0] < 2:
        return centerline.copy()
    
    offset_pts = []
    n = len(centerline)
    
    for i in range(n):
        if i == 0:
            direction = centerline[1] - centerline[0]
        elif i == n - 1:
            direction = centerline[-1] - centerline[-2]
        else:
            direction = centerline[i + 1] - centerline[i - 1]
        
        length = np.linalg.norm(direction)
        if length < 1e-9:
            perpendicular = np.array([0.0, 1.0])
        else:
            direction = direction / length
            perpendicular = np.array([-direction[1], direction[0]])
        
        offset_pts.append(centerline[i] + perpendicular * offset)
    
    return np.array(offset_pts, dtype=float)
