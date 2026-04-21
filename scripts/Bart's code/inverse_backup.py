# -*- coding: utf-8 -*- 
# inverse.py
# 
# PDP Inverse Problem Solver - Streamlit Application
# ===================================================
# This application solves the inverse problem of finding point configurations that match
# a given ordering constraint in two dimensions (d₁ and d₂).
#
# Layout: Two-column visualization with academic styling
# - Left panel: Original configuration from CSV data
# - Right panel: Generated configuration that attempts to match the original ordering
#
# The app supports two search strategies:
# 1. Exponential: Halves the search radius iteratively until convergence
# 2. Binary: Uses binary search with fixed steps to find optimal placement
#
# CSV format: columns (c, t, o, x, y) where:
#   c = configuration ID
#   t = time index
#   o = object type (0 for k-points, 1 for l-points)
#   x, y = 2D coordinates
#
# The algorithm maintains ordering constraints: points must satisfy d₁ and d₂ orderings
# matching the original configuration.

from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure

from pathlib import Path
from typing import Tuple, Callable, IO, TypedDict
import io
import re
import time

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.axes
import matplotlib.spines
import matplotlib.patches

# Type definition for successful point data in the search process
# This tracks each successfully placed point during the iterative generation process
class SuccessfulPoint(TypedDict):
    point: np.ndarray              # Coordinates of the generated point in 2D space
    parent_idx: int                # Index in all_pts array (may reference another generated point)
    parent_point: np.ndarray       # Actual coordinates of the parent point used for generation
    original_parent_idx: int       # Index of the ORIGINAL seed point (k0, k1, k2, l0, l1, l2)
    iteration: int                 # Iteration number when this point was successfully accepted

# ============= Page configuration =============
st.set_page_config(
    page_title="pdp inverse",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============= Styles (academic look) ============
st.markdown(
    """
<style>
.block-container { padding: 1rem 1.2rem; max-width: 1800px; }
html, body, [class*='css'] { font-family: "Georgia","Times New Roman",serif; color:#111; }
.figure-title { font-size:1.00rem; font-weight:600; letter-spacing:.2px; margin-bottom:.4rem; }
h1, .headline { font-weight:700; letter-spacing:.5px; margin-bottom:.6rem; }
hr { border:none; border-top:1px solid #ddd; margin:.4rem 0 1rem 0; }
/* settings card */
.settings-card {
    background: #fafafa;
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    padding: 0.6rem 0.8rem 0.2rem 0.8rem;
    margin: 0.3rem 0 0.8rem 0;
}
.settings-card h3 { font-size: 1.0rem; margin: 0 0 0.3rem 0; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# ============= Main title =============
st.markdown("<h1 class='headline'>pdp inverse</h1>", unsafe_allow_html=True)
st.markdown("<hr />", unsafe_allow_html=True)

# ---------- Helper: Convert Series to Numeric ----------
def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, coercing bad values to NaN.
    
    This wrapper ensures type safety for Pylance/Pyright static type checking
    while handling pandas' dynamic typing.
    """
    out = pd.to_numeric(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        s, errors="coerce"
    )
    return out

# ============= Load CSV Data with Custom Header Support ============
def load_points(csv_name: str = "voorbeeld.csv", o_val: int = 0, c_val: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and filter point data from a CSV file.
    
    This function handles a custom CSV format where the first line may be a literal
    'header: c,t,o,x,y' string (which gets skipped), or a standard CSV header.
    
    Parameters:
        csv_name: Name of the CSV file (must be in the same directory as this script)
        o_val: Object type filter (0 = k-points, 1 = l-points)
        c_val: Configuration ID to filter on

    Filters applied:
      - c == c_val (specific configuration)
      - o == o_val (specific object type)

    Returns:
      - pts: (N,2) numpy array of [x,y] coordinates, sorted by time t
      - ts:  (N,) numpy array of corresponding t-values (sorted)
    """
    csv_path = Path(__file__).with_name(csv_name)
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        st.stop()

    # Peek at the first line to detect our simple header format
    with csv_path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()

    names = ["c", "t", "o", "x", "y"]

    if first.lower().startswith("header:"):
        # Custom header format: skip first line, use fixed names
        df = pd.read_csv(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            csv_path, header=None, names=names, skiprows=1
        )
    else:
        # Try normal CSV; if columns are not present, fall back to fixed names
        df = pd.read_csv(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            csv_path
        )
        if not set(names).issubset(df.columns):
            df = pd.read_csv(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                csv_path, header=None, names=names
            )

    # Force numeric columns and drop invalid rows
    for col in names:
        df[col] = to_numeric_series(df[col])
    df = df.dropna(subset=names)  # type: ignore
    df = df.reset_index(drop=True)

    # Filter on configuration c and object flag o
    sel = df[(df["c"] == c_val) & (df["o"] == o_val)].sort_values("t").reset_index(drop=True)  # type: ignore
    if sel.empty:
        st.error(f"No rows found for c={c_val}, o={o_val}.")
        st.stop()

    pts = sel[["x", "y"]].to_numpy(dtype=float)  # type: ignore
    ts = sel["t"].to_numpy(dtype=float)  # type: ignore
    return pts, ts

# ============= Settings Helper: Read and Clean DataFrame ============
def _read_clean_df(csv_name: str) -> pd.DataFrame:
    """Read the CSV file once and return a clean DataFrame for UI settings.
    
    This is called once at app initialization to populate the configuration
    selector and time window controls. All numeric columns are coerced and
    invalid rows are dropped.
    """
    csv_path = Path(__file__).with_name(csv_name)
    with csv_path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    names = ["c", "t", "o", "x", "y"]
    if first.lower().startswith("header:"):
        df = pd.read_csv(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            csv_path, header=None, names=names, skiprows=1
        )
    else:
        df = pd.read_csv(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            csv_path
        )
        if not set(names).issubset(df.columns):
            df = pd.read_csv(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                csv_path, header=None, names=names
            )
    for col in names:
        df[col] = to_numeric_series(df[col])
    df = df.dropna(subset=names)  # type: ignore
    df = df.reset_index(drop=True)
    return df  # type: ignore[return-value]

# Load the full dataset once for UI population
_df_all = _read_clean_df("voorbeeld.csv")
# Filter to only valid time values (removes any duplicates or invalid entries)
_mask_t = _df_all["t"].isin(_df_all["t"].unique())  # type: ignore
_df_all = _df_all[_mask_t]

# Extract all available configuration IDs for the dropdown selector
available_configs = sorted(_df_all["c"].dropna().unique().astype(int).tolist())  # type: ignore
if not available_configs:
    st.error("No configurations found (column 'c' is empty).")
    st.stop()


# ============= Settings Card (UI) ============
# This section creates the main control panel for selecting data and algorithm parameters
st.markdown("""
<div class='settings-card'>
    <h3>Settings</h3>
""", unsafe_allow_html=True)

# Three-column layout for basic settings: config selection, time window size, time window start
sc1, sc2, sc3 = st.columns([1,1,2], gap="small")
with sc1:
    # Configuration selector: choose which configuration ID (c value) to work with
    selected_c = st.selectbox(
        "Configuration (c)",
        options=available_configs,
        index=available_configs.index(11) if 11 in available_configs else 0,
        key="cfg_c"
    )
# Convert to integer for filtering operations
selected_c_int: int = int(selected_c) if selected_c is not None else int(available_configs[0])


# Extract time values for k-points (o=0) and l-points (o=1) in the selected configuration
# We need overlapping time values to ensure both object types have data at the same time points
_t_k = sorted(_df_all[(_df_all["c"] == selected_c_int) & (_df_all["o"] == 0)]["t"].unique().tolist())  # type: ignore
_t_l = sorted(_df_all[(_df_all["c"] == selected_c_int) & (_df_all["o"] == 1)]["t"].unique().tolist())  # type: ignore
# Find common time points between k and l (required for valid comparison)
_t_common = [t for t in _t_k if t in _t_l]
if not _t_common:
    st.error(f"No overlapping t-values for c={selected_c} between o=0 and o=1.")
    st.stop()


# Calculate total available time points and set default window size
n_timepoints = len(_t_common)
default_window = min(3, n_timepoints)  # Use 3 time points by default, or fewer if not available
with sc2:
    # Slider to select how many consecutive time points to include in the analysis
    # This allows focusing on a subset of the temporal data
    num_timestamps = st.slider(
        "Number of timestamps",
        min_value=1,
        max_value=n_timepoints,
        value=default_window,
        step=1,
        key="cfg_k",
    )

with sc3:
    # Select which time point to start the sliding window from
    # Valid starts are constrained so the window doesn't extend beyond available data
    valid_start_count = max(1, n_timepoints - num_timestamps + 1)
    valid_starts = _t_common[:valid_start_count]
    start_t = st.select_slider(
        "Starting time (t)",
        options=valid_starts,
        value=valid_starts[0],
        key="cfg_start_t",
    )


# ============= Algorithm Parameters ============
# These controls determine how the generation algorithm operates
st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
sc4, sc5, sc6 = st.columns([1,1,1], gap="small")
with sc4:
    # Search strategy selection:
    # - Exponential: Iteratively halves the search radius until convergence
    # - Binary: Uses binary search with 7 fixed steps to find optimal distance
    strategy = st.radio(
        "Strategy",
        options=["exponential", "binary"],
        index=0,
        key="cfg_strategy",
        help="Choose the search strategy for configuration generation."
    )
with sc5:
    # Number of points to generate per configuration
    # Each iteration picks a random parent point and tries to find a valid child location
    num_iterations = st.radio(
        "Number of iterations",
        options=[3, 4, 5],
        index=0,
        key="cfg_iterations",
        help="How many iterations to run for each configuration."
    )
with sc6:
    # Total number of distinct configurations to generate
    # Each configuration is an independent attempt to match the original ordering
    num_configs = st.radio(
        "Number of configurations",
        options=[1, 3, 10, 100],
        index=0,
        key="cfg_num_configs",
        help="How many configurations to generate."
    )


# ============= Animation Speed Control ============
# Controls how fast the animated visualization updates (does not affect generation mode)
sc_wait, _, _ = st.columns([1, 1, 1], gap="small")
with sc_wait:
    wait_interval_ms = st.selectbox(
        "Animation wait interval (ms)",
        options=[100, 200, 500, 1000, 2000, 5000],
        index=4,  # 2000 ms as default (moderate speed)
        key="cfg_wait_ms",
        help="Time between animation updates."
    )


# ============= Action Buttons ============
# Three modes of operation:
# 1. Animate 1: Step-by-step visualization of a single configuration
# 2. Animate 5: Sequential animation of 5 configurations (legacy, uses num_configs setting)
# 3. Generate: Batch mode - generates all configurations instantly without animation
st.markdown("<div style='display:flex;gap:1.2rem;margin-top:0.7rem;'>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1,1,1], gap="small")
with col_btn1:
    animate_btn = st.button("Animate 1 configuration", key="btn_animate")
with col_btn2:
    animate_5_btn = st.button("Animate 5 configurations", key="btn_animate_5")
with col_btn3:
    generate_btn = st.button("Generate configurations", key="btn_generate")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ============= Load and Filter Data for Selected Time Window ============
# Load all points for the selected configuration
# k_points are object type 0 (blue), l_points are object type 1 (orange)
k_points, k_vals = load_points("voorbeeld.csv", o_val=0, c_val=selected_c_int)
l_points, l_vals = load_points("voorbeeld.csv", o_val=1, c_val=selected_c_int)

# Determine which time indices fall within the user-selected sliding window
try:
    start_idx = _t_common.index(start_t)  # type: ignore[arg-type]
except ValueError:
    start_idx = 0  # Default to first time point if invalid
end_idx = start_idx + int(num_timestamps)
selected_ts_window = _t_common[start_idx:end_idx]
selected_ts_set = set(selected_ts_window)

# Filter both k and l points to only include those in the selected time window
mask_k_win = np.isin(k_vals, list(selected_ts_set))
k_points_plot = k_points[mask_k_win]
k_vals_plot = k_vals[mask_k_win]
mask_l_win = np.isin(l_vals, list(selected_ts_set))
l_points_plot = l_points[mask_l_win]
l_vals_plot = l_vals[mask_l_win]


# ============= Calculate Maximum Distance & Axis Limits ============
def max_consecutive_dist(pts: np.ndarray) -> float:
    """Calculate the maximum Euclidean distance between consecutive points.
    
    This is used to set the initial search radius and axis margins.
    """
    n = pts.shape[0]
    if n < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)  # Differences between consecutive points
    dists = np.hypot(diffs[:, 0], diffs[:, 1])  # Euclidean distances
    return float(np.max(dists))

# maxdist determines the initial search radius and axis margins
# It's the larger of the max consecutive distances in k-points or l-points
maxdist: float = max(max_consecutive_dist(k_points_plot), max_consecutive_dist(l_points_plot))

def square_limits_with_margin(
    pts: np.ndarray, margin: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute square axis limits around a set of points with specified margin.
    
    This ensures:
    1. The plot window is perfectly square (equal width and height)
    2. All points have at least 'margin' distance from the plot borders
    3. Both axes use the same scale for accurate distance visualization
    
    Returns: (xlim, ylim) tuples defining the axis ranges
    """
    xmin = float(np.min(pts[:, 0])) - margin
    xmax = float(np.max(pts[:, 0])) + margin
    ymin = float(np.min(pts[:, 1])) - margin
    ymax = float(np.max(pts[:, 1])) + margin

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

# Calculate the axis limits for both plots using all points (k and l combined)
# The margin is set to maxdist to ensure adequate space for point generation
XLIM, YLIM = square_limits_with_margin(
    np.vstack([k_points_plot, l_points_plot]),
    maxdist
)

# ============= Order String Generation (LaTeX Format) ============
# These functions create LaTeX strings showing the ordering of points along d₁ and d₂
# For example: "d₁: k_0 < l_0 < k_1" indicates k_0 is leftmost, then l_0, then k_1
def _format_t_subscript(tval: float) -> str:
    """Format time value for LaTeX subscripts.
    
    Converts to integer string if the value is a whole number (e.g., 2.0 → "2"),
    otherwise formats as a compact float (e.g., 2.5 → "2.5").
    """
    try:
        tnum = float(tval)
    except Exception:
        tnum = float(np.array(tval, dtype=float))
    return str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"

def make_d1_order_latex() -> str:
    """Generate LaTeX string showing the ordering of points along the d₁ axis (x-coordinate).
    
    Collects all k and l points, sorts them by x-coordinate, and creates a string like:
    "d₁: k_0 < l_0 = k_1 < l_1"
    
    Points with nearly identical coordinates (within tolerance) are shown with '=' instead of '<'.
    """
    entries: list[tuple[float, str]] = []
    for x, t in zip(k_points_plot[:, 0].tolist(), k_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(x), rf"k_{lbl}"))
    for x, t in zip(l_points_plot[:, 0].tolist(), l_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(x), rf"l_{lbl}"))

    if not entries:
        return r"d_1:"

    entries.sort(key=lambda it: it[0])
    tol = 1e-9
    out = [entries[0][1]]
    for i in range(1, len(entries)):
        prev_x = entries[i - 1][0]
        cur_x = entries[i][0]
        connector = " = " if abs(cur_x - prev_x) <= tol else " < "
        out.append(connector + entries[i][1])
    return r"d_1: " + "".join(out)

def make_d2_order_latex() -> str:
    """Generate LaTeX string showing the ordering of points along the d₂ axis (y-coordinate).
    
    Same as make_d1_order_latex but for the vertical axis.
    """
    entries: list[tuple[float, str]] = []
    for y, t in zip(k_points_plot[:, 1].tolist(), k_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(y), rf"k_{lbl}"))
    for y, t in zip(l_points_plot[:, 1].tolist(), l_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(y), rf"l_{lbl}"))

    if not entries:
        return r"d_2:"

    entries.sort(key=lambda it: it[0])
    tol = 1e-9
    out = [entries[0][1]]
    for i in range(1, len(entries)):
        prev_y = entries[i - 1][0]
        cur_y = entries[i][0]
        connector = " = " if abs(cur_y - prev_y) <= tol else " < "
        out.append(connector + entries[i][1])
    return r"d_2: " + "".join(out)

def make_d1_order_latex_generated() -> str:
    """
    Generate LaTeX order string for d₁ including dynamically generated points.
    
    This function shows the current state during animation/generation:
    - Uses primes (', '') and * markers to indicate generation count
    - Shows generated points in place of their parent points
    - Handles both original points and points generated from other generated points
    
    Generation markers:
    - No marker: original point
    - ': first generation (child of original)
    - '': second generation (grandchild)
    - *: third or higher generation
    """
    if not st.session_state.get("show_anim_circle", False) and not st.session_state.get("anim_running", False):
        if "anim_generated_point" not in st.session_state:
            return r"d_1:"
    gen_pt = st.session_state.get("anim_generated_point", None)
    parent_idx = st.session_state.get("anim_parent_idx", 0)
    all_pts = st.session_state.get("anim_all_pts", np.array([]))
    if gen_pt is None or all_pts.shape[0] == 0:
        return r"d_1:"
    entries: list[tuple[float, str]] = []
    n_k = k_points_plot.shape[0]
    n_l = l_points_plot.shape[0]
    total_original = n_k + n_l

    base_idx = int(parent_idx)
    if parent_idx >= total_original:
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - total_original)
        if 0 <= sidx < len(succ_list):
            base_idx = int(succ_list[sidx]["original_parent_idx"])
    parent_is_k = base_idx < n_k

    in_search = st.session_state.get("anim_in_search", False)
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    generation_counts: dict[int, int] = {}
    for sp in successful_points:
        oi = int(sp["original_parent_idx"])
        generation_counts[oi] = generation_counts.get(oi, 0) + 1

    def _prime_str(gen: int) -> str:
        """Return the prime marker for a given generation count."""
        if gen <= 0:
            return ""
        if gen == 1:
            return "'"
        if gen == 2:
            return "''"
        return "*"  # no superscript star anymore

    # Determine parent label
    if base_idx < n_k:
        parent_t = k_vals_plot[base_idx]
    else:
        parent_t = l_vals_plot[base_idx - n_k]
    lbl_parent = _format_t_subscript(float(parent_t))
    current_gen_count = generation_counts.get(base_idx, 0)
    label_gen_count = current_gen_count + (0 if in_search else 1)
    parent_primes = _prime_str(label_gen_count)
    if parent_is_k:
        entries.append((float(gen_pt[0]), rf"k{parent_primes}_{lbl_parent}"))
    else:
        entries.append((float(gen_pt[0]), rf"l{parent_primes}_{lbl_parent}"))

    # Track the latest generated point for each original index
    latest_generated: dict[int, np.ndarray] = {}
    for sp in successful_points:
        orig_idx = int(sp["original_parent_idx"])
        latest_generated[orig_idx] = sp["point"]

    # Original k points with possible generated replacements
    for i, (x, t) in enumerate(zip(k_points_plot[:, 0].tolist(), k_vals_plot.tolist())):
        if i == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(i, 0)
        primes_i = _prime_str(gen_cnt)
        if i in latest_generated:
            entries.append((float(latest_generated[i][0]), rf"k{primes_i}_{lbl}"))
        else:
            entries.append((float(x), rf"k{primes_i}_{lbl}"))

    # Original l points with possible generated replacements
    for j, (x, t) in enumerate(zip(l_points_plot[:, 0].tolist(), l_vals_plot.tolist())):
        glob_idx = n_k + j
        if glob_idx == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(glob_idx, 0)
        primes_j = _prime_str(gen_cnt)
        if glob_idx in latest_generated:
            entries.append((float(latest_generated[glob_idx][0]), rf"l{primes_j}_{lbl}"))
        else:
            entries.append((float(x), rf"l{primes_j}_{lbl}"))

    entries.sort(key=lambda it: it[0])
    tol = 1e-9
    out = [entries[0][1]]
    for i in range(1, len(entries)):
        prev_x = entries[i - 1][0]
        cur_x = entries[i][0]
        connector = " = " if abs(cur_x - prev_x) <= tol else " < "
        out.append(connector + entries[i][1])
    return r"d_1: " + "".join(out)

def make_d2_order_latex_generated() -> str:
    """
    Generate LaTeX order string for d₂ including dynamically generated points.
    
    Same as make_d1_order_latex_generated but for the y-coordinate (d₂ axis).
    Shows the vertical ordering with generation markers.
    """
    if not st.session_state.get("show_anim_circle", False) and not st.session_state.get("anim_running", False):
        if "anim_generated_point" not in st.session_state:
            return r"d_2:"
    gen_pt = st.session_state.get("anim_generated_point", None)
    parent_idx = st.session_state.get("anim_parent_idx", 0)
    all_pts = st.session_state.get("anim_all_pts", np.array([]))
    if gen_pt is None or all_pts.shape[0] == 0:
        return r"d_2:"
    entries: list[tuple[float, str]] = []
    n_k = k_points_plot.shape[0]
    n_l = l_points_plot.shape[0]
    total_original = n_k + n_l

    base_idx = int(parent_idx)
    if parent_idx >= total_original:
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - total_original)
        if 0 <= sidx < len(succ_list):
            base_idx = int(succ_list[sidx]["original_parent_idx"])
    parent_is_k = base_idx < n_k

    in_search = st.session_state.get("anim_in_search", False)
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    generation_counts: dict[int, int] = {}
    for sp in successful_points:
        oi = int(sp["original_parent_idx"])
        generation_counts[oi] = generation_counts.get(oi, 0) + 1

    def _prime_str(gen: int) -> str:
        if gen <= 0:
            return ""
        if gen == 1:
            return "'"
        if gen == 2:
            return "''"
        return "*"

    if base_idx < n_k:
        parent_t = k_vals_plot[base_idx]
    else:
        parent_t = l_vals_plot[base_idx - n_k]
    lbl_parent = _format_t_subscript(float(parent_t))
    current_gen_count = generation_counts.get(base_idx, 0)
    label_gen_count = current_gen_count + (0 if in_search else 1)
    parent_primes = _prime_str(label_gen_count)
    if parent_is_k:
        entries.append((float(gen_pt[1]), rf"k{parent_primes}_{lbl_parent}"))
    else:
        entries.append((float(gen_pt[1]), rf"l{parent_primes}_{lbl_parent}"))

    latest_generated: dict[int, np.ndarray] = {}
    for sp in successful_points:
        orig_idx = int(sp["original_parent_idx"])
        latest_generated[orig_idx] = sp["point"]

    for i, (y, t) in enumerate(zip(k_points_plot[:, 1].tolist(), k_vals_plot.tolist())):
        if i == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(i, 0)
        primes_i = _prime_str(gen_cnt)
        if i in latest_generated:
            entries.append((float(latest_generated[i][1]), rf"k{primes_i}_{lbl}"))
        else:
            entries.append((float(y), rf"k{primes_i}_{lbl}"))

    for j, (y, t) in enumerate(zip(l_points_plot[:, 1].tolist(), l_vals_plot.tolist())):
        glob_idx = n_k + j
        if glob_idx == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(glob_idx, 0)
        primes_j = _prime_str(gen_cnt)
        if glob_idx in latest_generated:
            entries.append((float(latest_generated[glob_idx][1]), rf"l{primes_j}_{lbl}"))
        else:
            entries.append((float(y), rf"l{primes_j}_{lbl}"))

    entries.sort(key=lambda it: it[0])
    tol = 1e-9
    out = [entries[0][1]]
    for i in range(1, len(entries)):
        prev_y = entries[i - 1][0]
        cur_y = entries[i][0]
        connector = " = " if abs(cur_y - prev_y) <= tol else " < "
        out.append(connector + entries[i][1])
    return r"d_2: " + "".join(out)

# ============= Order Comparison Helpers ============
# These functions normalize order strings for comparison by removing decorations

def _strip_primes(text: str) -> str:
    """Remove prime markers (', '', *) from a LaTeX string.
    
    This allows comparing order strings without caring about generation markers.
    For example, "k'_0 < l''_1" becomes "k_0 < l_1".
    """
    text = re.sub(r"\^\{\*\}", "", text)  # legacy, now not used but harmless
    text = re.sub(r"[']+", "", text)
    text = text.replace("*", "")
    return text

def _extract_order_string(latex_str: str) -> str:
    """Extract bare ordering from a LaTeX order string for comparison.
    
    Removes:
    - Axis prefixes (d_1:, d_2:)
    - Prime decorations (', '', *)
    - LaTeX braces ({, })
    
    Example: "d_1: k_{0}'' < l_{1}'" → "k_0 < l_1"
    
    This normalization allows us to check if two configurations have the same
    relative ordering, regardless of which points are original vs generated.
    """
    core = latex_str.replace("d_1:", "").replace("d_2:", "").strip()
    core_no_primes = _strip_primes(core)
    # remove {…} but keep inside, so k_{0} → k_0
    core_no_braces = re.sub(r"\{([^{}]+)\}", r"\1", core_no_primes)
    return core_no_braces

# ============= Order Match Tracking ============
# This function computes whether the generated configuration matches the original ordering

def update_order_match_flags() -> None:
    """Compute and store whether d₁ and d₂ orderings match between original and generated.
    
    Stores boolean flags in session_state:
    - order_match_d1: True if x-coordinate ordering matches
    - order_match_d2: True if y-coordinate ordering matches
    
    This is called after each successful point placement to track progress.
    """
    left_d1 = make_d1_order_latex()
    right_d1 = make_d1_order_latex_generated()
    st.session_state["order_match_d1"] = (
        _extract_order_string(left_d1) == _extract_order_string(right_d1)
    )

    left_d2 = make_d2_order_latex()
    right_d2 = make_d2_order_latex_generated()
    st.session_state["order_match_d2"] = (
        _extract_order_string(left_d2) == _extract_order_string(right_d2)
    )

# ============= Non-Animated Generation (Exponential Strategy) ============
def generate_exp() -> None:
    """
    Batch generation using the exponential search strategy (no animation).
    
    This is the fast, non-visual version of the exponential algorithm:
    1. Randomly selects a parent point from k or l (or previously generated points)
    2. Places a candidate point at distance 'maxdist' from the parent
    3. Checks if the resulting ordering matches the original
    4. If not matching: halves the distance and adjusts angle slightly, then retries
    5. If matching (or distance reaches ~0): accepts the point and moves to next iteration
    
    Process continues until:
    - All iterations for all configurations are complete, or
    - Safety limit (100,000 loops) is reached
    
    Uses parameters from session_state set by the UI radio buttons.
    No time.sleep() calls - runs as fast as possible.
    """
    max_loops = 100000  # Safety limit to prevent infinite loops
    loops = 0

    # Use radio button values as defaults if state variables are missing
    default_iterations = int(st.session_state.get("cfg_iterations", 3))
    default_num_configs = int(st.session_state.get("cfg_num_configs", 1))

    # Main generation loop - continues until all configurations are complete
    while st.session_state.get("anim_running", False) and loops < max_loops:
        loops += 1

        # Check current ordering match status
        left_d1 = make_d1_order_latex()  # Original d₁ ordering
        left_d2 = make_d2_order_latex()  # Original d₂ ordering
        right_d1 = make_d1_order_latex_generated()  # Current generated d₁ ordering
        right_d2 = make_d2_order_latex_generated()  # Current generated d₂ ordering

        same_d1 = _extract_order_string(left_d1) == _extract_order_string(right_d1)
        same_d2 = _extract_order_string(left_d2) == _extract_order_string(right_d2)

        # Extract current state variables
        completed_iterations = int(st.session_state.get("anim_completed_iterations", 0))
        max_iterations = int(st.session_state.get("anim_max_iterations", default_iterations))
        search_steps = int(st.session_state.get("anim_search_steps", 0))
        max_search_steps = 7  # Maximum binary search refinement steps

        distance = float(st.session_state.get("anim_distance", maxdist))
        angle = float(st.session_state.get("anim_angle", 0.0))
        gen_pt = st.session_state.get("anim_generated_point", None)
        parent_idx = int(st.session_state.get("anim_parent_idx", 0))
        all_pts = st.session_state.get("anim_all_pts", np.array([]))
        successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        in_search = bool(st.session_state.get("anim_in_search", True))

        # === Case 1: Success - orders match OR distance has collapsed to zero ===
        # Accept the current candidate point and prepare for the next iteration
        if (same_d1 and same_d2 and gen_pt is not None) or (distance <= 0.0 and gen_pt is not None):
            n_k = k_points_plot.shape[0]
            n_l = l_points_plot.shape[0]
            total_original = n_k + n_l
            if all_pts.size > 0 and parent_idx < total_original:
                parent_point_val = all_pts[parent_idx]
                original_parent_idx_val = parent_idx
            else:
                succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                sidx = int(parent_idx - total_original)
                if 0 <= sidx < len(succ_list):
                    parent_point_val = succ_list[sidx]["point"]
                    original_parent_idx_val = succ_list[sidx]["original_parent_idx"]
                else:
                    parent_point_val = np.array([0.0, 0.0])
                    original_parent_idx_val = 0
            sp: SuccessfulPoint = {
                "point": np.array(gen_pt, dtype=float),
                "parent_idx": parent_idx,
                "parent_point": parent_point_val,
                "original_parent_idx": original_parent_idx_val,
                "iteration": completed_iterations,
            }
            successful_points.append(sp)
            st.session_state["anim_successful_points"] = successful_points
            st.session_state["anim_completed_iterations"] = completed_iterations + 1
            st.session_state["anim_search_steps"] = 0
            st.session_state["anim_in_search"] = True
            st.session_state["anim_delta"] = None

            # <<< here: update order match for this placement >>>
            update_order_match_flags()

            # Check if we finished all iterations for this configuration
            if completed_iterations + 1 >= max_iterations:
                current_config = int(st.session_state.get("anim_current_config", 1))
                num_configs = int(st.session_state.get("anim_num_configs", default_num_configs))

                # Store this finished configuration
                all_configs: list = st.session_state.get("anim_all_configs", [])
                all_configs.append({
                    "config_num": current_config,
                    "points": list(successful_points)
                })
                st.session_state["anim_all_configs"] = all_configs

                # Attach config number to each successful point
                for sp in successful_points:
                    sp["config_num"] = current_config  # type: ignore

                # Decide whether to move on to the next configuration or stop
                if current_config < num_configs:
                    # Prepare next configuration
                    st.session_state["anim_current_config"] = current_config + 1
                    st.session_state["anim_completed_iterations"] = 0
                    st.session_state["anim_search_steps"] = 0
                    st.session_state["anim_running"] = True

                    n_k_reset = k_points_plot.shape[0]
                    n_l_reset = l_points_plot.shape[0]
                    total_original_reset = n_k_reset + n_l_reset
                    all_pts_reset = np.vstack([k_points_plot, l_points_plot])
                    all_indices_reset = list(range(total_original_reset))
                    if all_indices_reset:
                        chosen_idx_reset = int(np.random.choice(all_indices_reset))
                    else:
                        chosen_idx_reset = 0

                    youngest_point_reset = None
                    youngest_success_idx_reset = None
                    for idx, s in reversed(list(enumerate(successful_points))):
                        oi = s.get("original_parent_idx", None)
                        if oi is not None and int(oi) == chosen_idx_reset:
                            youngest_point_reset = s["point"]
                            youngest_success_idx_reset = idx
                            break

                    if youngest_point_reset is not None and youngest_success_idx_reset is not None:
                        parent_pt_reset = youngest_point_reset
                        parent_idx_reset = total_original_reset + youngest_success_idx_reset
                    else:
                        parent_idx_reset = chosen_idx_reset
                        parent_pt_reset = all_pts_reset[parent_idx_reset]

                    distance_new = maxdist
                    max_attempts = 20
                    for _ in range(max_attempts):
                        angle_local = float(np.random.uniform(0, 2 * np.pi))
                        new_x = parent_pt_reset[0] + distance_new * np.cos(angle_local)
                        new_y = parent_pt_reset[1] + distance_new * np.sin(angle_local)
                        if XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]:
                            break
                    else:
                        new_x = np.clip(new_x, XLIM[0], XLIM[1])
                        new_y = np.clip(new_y, YLIM[0], YLIM[1])
                    new_gen_pt = np.array([new_x, new_y])

                    st.session_state["anim_parent_idx"] = parent_idx_reset
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = distance_new
                    st.session_state["anim_all_pts"] = all_pts_reset
                    st.session_state["anim_config_complete_wait"] = False
                else:
                    # All configurations generated
                    st.session_state["anim_running"] = False
            else:
                # Prepare the next iteration for the same configuration
                n_k = k_points_plot.shape[0]
                n_l = l_points_plot.shape[0]
                total_original = n_k + n_l
                all_indices = list(range(total_original))
                if all_indices:
                    chosen_idx = int(np.random.choice(all_indices))
                else:
                    chosen_idx = 0

                youngest_point = None
                youngest_success_idx = None
                for idx, s in reversed(list(enumerate(successful_points))):
                    oi = s.get("original_parent_idx", None)
                    if oi is not None and int(oi) == chosen_idx:
                        youngest_point = s["point"]
                        youngest_success_idx = idx
                        break
                if youngest_point is not None and youngest_success_idx is not None:
                    parent_pt_new = youngest_point
                    parent_idx_new = total_original + youngest_success_idx
                else:
                    if chosen_idx < n_k:
                        parent_pt_new = k_points_plot[chosen_idx]
                    else:
                        parent_pt_new = l_points_plot[chosen_idx - n_k]
                    parent_idx_new = chosen_idx

                distance_new = maxdist
                max_attempts = 20
                for _ in range(max_attempts):
                    angle_local = float(np.random.uniform(0, 2 * np.pi))
                    new_x = parent_pt_new[0] + distance_new * np.cos(angle_local)
                    new_y = parent_pt_new[1] + distance_new * np.sin(angle_local)
                    if XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]:
                        break
                else:
                    new_x = np.clip(new_x, XLIM[0], XLIM[1])
                    new_y = np.clip(new_y, YLIM[0], YLIM[1])
                new_gen_pt = np.array([new_x, new_y])

                st.session_state["anim_parent_idx"] = parent_idx_new
                st.session_state["anim_angle"] = angle_local
                st.session_state["anim_generated_point"] = new_gen_pt
                st.session_state["anim_distance"] = distance_new
        else:
            # === Case 2: keep searching (halve radius etc.) ===
            search_steps += 1
            st.session_state["anim_search_steps"] = search_steps

            if search_steps >= max_search_steps:
                # If search did not converge, snap back to parent
                if gen_pt is not None and all_pts.size > 0:
                    n_k = k_points_plot.shape[0]
                    n_l = l_points_plot.shape[0]
                    total_original = n_k + n_l
                    if parent_idx < total_original:
                        parent_pt_cur = all_pts[parent_idx]
                    else:
                        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                        sidx = int(parent_idx - total_original)
                        if 0 <= sidx < len(succ_list):
                            parent_pt_cur = succ_list[sidx]["point"]
                        else:
                            parent_pt_cur = np.array([0.0, 0.0])

                    st.session_state["anim_generated_point"] = parent_pt_cur.copy()
                    st.session_state["anim_distance"] = 0.0
                    st.session_state["anim_in_search"] = True
            else:
                # Standard exponential search step: halve distance, tweak angle
                if gen_pt is not None and all_pts.size > 0:
                    n_k = k_points_plot.shape[0]
                    n_l = l_points_plot.shape[0]
                    total_original = n_k + n_l
                    if parent_idx < total_original:
                        parent_pt_cur = all_pts[parent_idx]
                    else:
                        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                        sidx = int(parent_idx - total_original)
                        if 0 <= sidx < len(succ_list):
                            parent_pt_cur = succ_list[sidx]["point"]
                        else:
                            parent_pt_cur = np.array([0.0, 0.0])

                    new_distance = distance / 2.0
                    min_distance = 1e-5
                    if new_distance < min_distance:
                        # If we get too small, reset angle randomly
                        new_distance = min_distance * 2.0
                        angle_local = float(np.random.uniform(0, 2 * np.pi))
                    else:
                        angle_local = angle
                    angle_local += float(np.random.uniform(-0.25, 0.25))
                    angle_local = angle_local % (2 * np.pi)
                    new_x = parent_pt_cur[0] + new_distance * np.cos(angle_local)
                    new_y = parent_pt_cur[1] + new_distance * np.sin(angle_local)
                    new_gen_pt = np.array([new_x, new_y])

                    # Keep candidate inside plotting window if possible
                    if not (XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]):
                        angle_local = (angle_local + np.pi) % (2 * np.pi)
                        new_x = parent_pt_cur[0] + new_distance * np.cos(angle_local)
                        new_y = parent_pt_cur[1] + new_distance * np.sin(angle_local)
                        new_gen_pt = np.array([new_x, new_y])

                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = new_distance
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_in_search"] = True

# ============= Animate button handler ============
if animate_btn or animate_5_btn:
    # Reset search diagnostics for a fresh animation run
    st.session_state["anim_delta"] = None

    if strategy == "exponential":
        num_configs_to_generate = 5 if animate_5_btn else 1  # kept only for possible future use

        all_pts = np.vstack([k_points_plot, l_points_plot])
        all_ts = np.concatenate([k_vals_plot, l_vals_plot])
        n_total = all_pts.shape[0]
        parent_idx = int(np.random.randint(0, n_total))  # type: ignore[arg-type]
        parent_pt = all_pts[parent_idx]
        distance = maxdist
        max_attempts = 20
        for _ in range(max_attempts):
            alfa = float(np.random.uniform(0, 2 * np.pi))
            gen_x = parent_pt[0] + distance * np.cos(alfa)
            gen_y = parent_pt[1] + distance * np.sin(alfa)
            if XLIM[0] <= gen_x <= XLIM[1] and YLIM[0] <= gen_y <= YLIM[1]:
                break
        else:
            gen_x = np.clip(gen_x, XLIM[0], XLIM[1])
            gen_y = np.clip(gen_y, YLIM[0], YLIM[1])
        generated_point = np.array([gen_x, gen_y])

        st.session_state["show_anim_circle"] = True
        st.session_state["anim_running"] = True
        st.session_state["anim_circle_idx"] = parent_idx
        st.session_state["anim_distance"] = distance
        st.session_state["anim_generated_point"] = generated_point
        st.session_state["anim_parent_idx"] = parent_idx
        st.session_state["anim_all_pts"] = all_pts
        st.session_state["anim_all_ts"] = all_ts
        st.session_state["anim_angle"] = alfa
        st.session_state["anim_iteration"] = 0
        st.session_state["anim_max_iterations"] = int(num_iterations)
        st.session_state["anim_iterations_per_run"] = int(num_iterations)
        st.session_state["anim_completed_iterations"] = 0
        st.session_state["anim_last_update"] = time.time()
        st.session_state["anim_successful_points"] = []
        st.session_state["anim_in_search"] = True
        # Number of configurations comes directly from the radio button
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_current_config"] = 1
        st.session_state["anim_all_configs"] = []
        st.session_state["anim_search_steps"] = 0
        st.session_state["anim_binary_mode"] = False
        st.session_state["anim_binary_step"] = 0
        st.session_state["diag_rows"] = []
        st.session_state["binary_iteration_summary"] = []
        st.session_state["anim_had_full_match"] = False

    elif strategy == "binary":
        num_configs_to_generate = 5 if animate_5_btn else 1  # kept only for possible future use

        all_pts = np.vstack([k_points_plot, l_points_plot])
        all_ts = np.concatenate([k_vals_plot, l_vals_plot])
        n_total = all_pts.shape[0]
        parent_idx = int(np.random.randint(0, n_total))  # type: ignore[arg-type]
        parent_pt = all_pts[parent_idx]

        distance = 0.5 * maxdist
        max_attempts = 20
        found_inside = False
        for _ in range(max_attempts):
            alfa = float(np.random.uniform(0, 2 * np.pi))
            gen_x = parent_pt[0] + distance * np.cos(alfa)
            gen_y = parent_pt[1] + distance * np.sin(alfa)
            # FIX hier: gebruik gen_y i.p.v. new_y
            if XLIM[0] <= gen_x <= XLIM[1] and YLIM[0] <= gen_y <= YLIM[1]:
                found_inside = True
                break

        if not found_inside:
            alfa = float(np.random.uniform(0, 2 * np.pi))
            gen_x = parent_pt[0] + distance * np.cos(alfa)
            gen_y = parent_pt[1] + distance * np.sin(alfa)

        generated_point = np.array([gen_x, gen_y])
        ok_point = generated_point.copy()

        st.session_state["show_anim_circle"] = True
        st.session_state["anim_running"] = True
        st.session_state["anim_circle_idx"] = parent_idx
        st.session_state["anim_distance"] = distance
        st.session_state["anim_generated_point"] = generated_point
        st.session_state["anim_parent_idx"] = parent_idx
        st.session_state["anim_all_pts"] = all_pts
        st.session_state["anim_all_ts"] = all_ts
        st.session_state["anim_angle"] = alfa
        st.session_state["anim_iteration"] = 0
        st.session_state["anim_max_iterations"] = int(num_iterations)
        st.session_state["anim_iterations_per_run"] = int(num_iterations)
        st.session_state["anim_completed_iterations"] = 0
        st.session_state["anim_last_update"] = time.time()
        st.session_state["anim_successful_points"] = []
        st.session_state["anim_in_search"] = True
        # Number of configurations comes directly from the radio button
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_current_config"] = 1
        st.session_state["anim_all_configs"] = []
        st.session_state["anim_search_steps"] = 0

        st.session_state["anim_ok_point"] = ok_point
        st.session_state["anim_binary_mode"] = True
        st.session_state["anim_binary_step"] = 0
        st.session_state["anim_delta"] = None
        st.session_state["diag_rows"] = []
        st.session_state["binary_iteration_summary"] = []
        st.session_state["anim_had_full_match"] = False

# ============= Generate button handler (non-animated exponential) ============
if generate_btn:
    # Reset state for a fresh non-animated generation run
    st.session_state["anim_all_configs"] = []
    st.session_state["anim_successful_points"] = []
    st.session_state["anim_completed_iterations"] = 0
    st.session_state["anim_current_config"] = 1
    st.session_state["anim_search_steps"] = 0
    st.session_state["anim_binary_mode"] = False
    st.session_state["anim_binary_step"] = 0
    st.session_state["anim_delta"] = None
    st.session_state["diag_rows"] = []
    st.session_state["binary_iteration_summary"] = []

    if strategy == "exponential":
        # Prepare initial search state for non-animated exponential generation.
        # The number of iterations and configurations comes from the radio buttons.
        all_pts = np.vstack([k_points_plot, l_points_plot])
        all_ts = np.concatenate([k_vals_plot, l_vals_plot])
        n_total = all_pts.shape[0]
        parent_idx = int(np.random.randint(0, n_total))  # type: ignore[arg-type]
        parent_pt = all_pts[parent_idx]
        distance = maxdist
        max_attempts = 20
        for _ in range(max_attempts):
            alfa = float(np.random.uniform(0, 2 * np.pi))
            gen_x = parent_pt[0] + distance * np.cos(alfa)
            gen_y = parent_pt[1] + distance * np.sin(alfa)
            if XLIM[0] <= gen_x <= XLIM[1] and YLIM[0] <= gen_y <= YLIM[1]:
                break
        else:
            gen_x = np.clip(gen_x, XLIM[0], XLIM[1])
            gen_y = np.clip(gen_y, YLIM[0], YLIM[1])
        generated_point = np.array([gen_x, gen_y])

        st.session_state["show_anim_circle"] = False  # no circle for generate
        st.session_state["anim_running"] = True
        st.session_state["anim_circle_idx"] = parent_idx
        st.session_state["anim_distance"] = distance
        st.session_state["anim_generated_point"] = generated_point
        st.session_state["anim_parent_idx"] = parent_idx
        st.session_state["anim_all_pts"] = all_pts
        st.session_state["anim_all_ts"] = all_ts
        st.session_state["anim_angle"] = alfa
        st.session_state["anim_iteration"] = 0
        st.session_state["anim_max_iterations"] = int(num_iterations)
        st.session_state["anim_iterations_per_run"] = int(num_iterations)
        st.session_state["anim_last_update"] = time.time()
        st.session_state["anim_in_search"] = True

        # Number of configurations comes from the radio button "Number of configurations"
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_config_complete_wait"] = False

        # Run non-animated exponential generator in one go
        generate_exp()
    else:
        st.warning("Generation for binary strategy is not implemented yet.")

# ============= Drawing (without gridlines) ============
def setup_square_axes(ax: matplotlib.axes.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    """Configure axes to be square, with simple ticks and labels d₁, d₂."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)  # type: ignore
        sp.set_color("#222")
    ax.tick_params(axis="both", labelsize=9, width=0.8, color="#222")  # type: ignore
    ax.set_xlabel("d₁", fontsize=11, labelpad=8)  # type: ignore
    ax.set_ylabel("d₂", fontsize=11, labelpad=8)  # type: ignore

def render_square_matplotlib_figure(
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
    setup_square_axes(ax, xlim, ylim)
    draw_fn(ax)
    fig.tight_layout(pad=0.9)
    return fig

BLUE = "C0"
ORANGE = "C1"
LABEL_FS = 9

def annotate_points(
    ax: matplotlib.axes.Axes,
    pts: np.ndarray,
    ts: np.ndarray,
    label_prefix: str,
    color: str,
) -> None:
    """Draw points plus LaTeX labels k_t or l_t with small offsets."""
    offsets = [(3, 3), (3, -8), (-8, 3)]
    for i, ((x, y), tval) in enumerate(zip(pts, ts)):
        ax.scatter([x], [y], s=40, zorder=3, color=color)  # type: ignore
        off = offsets[i % len(offsets)]
        try:
            tnum = float(tval)  # type: ignore[arg-type]
        except Exception:
            tnum = float(np.array(tval, dtype=float))
        lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
        ax.annotate(  # type: ignore
            rf"${label_prefix}_{lbl}$",
            xy=(x, y),
            xytext=off,
            textcoords="offset points",
            fontsize=LABEL_FS,
            color=color,
            ha="left" if off[0] >= 0 else "right",
            va="bottom" if off[1] >= 0 else "top",
        )

def draw_original(ax: matplotlib.axes.Axes) -> None:
    """Draw the original k and l curves in the left panel."""
    ax.plot(k_points_plot[:, 0], k_points_plot[:, 1], linewidth=2.0, color=BLUE)  # type: ignore
    annotate_points(ax, k_points_plot, k_vals_plot, "k", BLUE)
    ax.plot(l_points_plot[:, 0], l_points_plot[:, 1], linewidth=2.0, color=ORANGE)  # type: ignore
    annotate_points(ax, l_points_plot, l_vals_plot, "l", ORANGE)

def draw_generated_empty(ax: matplotlib.axes.Axes) -> None:
    """
    Draw the generated configuration in the right panel.

    This includes:
    - Original k and l (with transparent segments where we have generated points),
    - The latest generated copies of each parent,
    - Optional search circle and current candidate.
    """
    n_k = k_points_plot.shape[0]
    n_l_total = l_points_plot.shape[0]
    total_original = n_k + n_l_total

    current_config = st.session_state.get("anim_current_config", 1)
    completed_iters = st.session_state.get("anim_completed_iterations", 0)
    search_steps = st.session_state.get("anim_search_steps", 0)
    anim_running = st.session_state.get("anim_running", False)

    binary_mode = st.session_state.get("anim_binary_mode", False)
    binary_step = st.session_state.get("anim_binary_step", 0)
    current_strategy = st.session_state.get("cfg_strategy", "exponential")

    delta_val = st.session_state.get("anim_delta", None)

    # Status text: shows configuration, iteration and step
    if not anim_running and completed_iters > 0:
        if binary_mode:
            step_display = binary_step
        else:
            step_display = st.session_state.get("anim_last_step", 0)
        status_text = f"Config {current_config} | Iteration {completed_iters} | Step {step_display}"
    else:
        if binary_mode:
            step_display = binary_step
        else:
            step_display = search_steps
        status_text = f"Config {current_config} | Iteration {completed_iters + 1} | Step {step_display}"
        st.session_state["anim_last_step"] = step_display

    if current_strategy == "binary" and delta_val is not None and maxdist != 0:
        status_text += f" | Δ={delta_val / maxdist:.4f}·maxdist"

    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=9,  # type: ignore
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    has_animation = st.session_state.get("show_anim_circle", False)
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    gen_pt = st.session_state.get("anim_generated_point", None)
    parent_idx = st.session_state.get("anim_parent_idx", 0)
    in_search = st.session_state.get("anim_in_search", False)

    offsets = [(3, 3), (3, -8), (-8, 3)]

    def make_label(prefix: str, tval: float, gen_marker: str = "") -> str:
        """Helper to build a LaTeX label prefix_gen_marker_t."""
        try:
            tnum = float(tval)
        except Exception:
            tnum = float(np.array(tval, dtype=float))
        lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
        if gen_marker:
            return rf"${prefix}{gen_marker}_{lbl}$"
        return rf"${prefix}_{lbl}$"

    def _get_original_index(sp: SuccessfulPoint) -> int | None:
        """Return original parent index if present, otherwise None."""
        try:
            oi = int(sp["original_parent_idx"])
            return oi
        except Exception:
            return None

    # Determine which original segments should become transparent
    transparent_segments_k: set[tuple[int, int]] = set()
    transparent_segments_l: set[tuple[int, int]] = set()

    for succ_pt_data in successful_points:
        succ_parent_idx = succ_pt_data.get("parent_idx", -1)
        if succ_parent_idx < n_k:
            if succ_parent_idx > 0:
                transparent_segments_k.add((succ_parent_idx - 1, succ_parent_idx))
            if succ_parent_idx < n_k - 1:
                transparent_segments_k.add((succ_parent_idx, succ_parent_idx + 1))
        else:
            local_idx = succ_parent_idx - n_k
            if local_idx > 0:
                transparent_segments_l.add((local_idx - 1, local_idx))
            if local_idx < l_points_plot.shape[0] - 1:
                transparent_segments_l.add((local_idx, local_idx + 1))

    # Base k segments
    for i in range(len(k_points_plot) - 1):
        alpha = 0.2 if (i, i+1) in transparent_segments_k else 1.0
        ax.plot(
            [k_points_plot[i, 0], k_points_plot[i+1, 0]],
            [k_points_plot[i, 1], k_points_plot[i+1, 1]],
            linewidth=2.0, color=BLUE, alpha=alpha, zorder=1
        )

    # Base l segments
    for i in range(len(l_points_plot) - 1):
        alpha = 0.2 if (i, i+1) in transparent_segments_l else 1.0
        ax.plot(
            [l_points_plot[i, 0], l_points_plot[i+1, 0]],
            [l_points_plot[i, 1], l_points_plot[i+1, 1]],
            linewidth=2.0, color=ORANGE, alpha=alpha, zorder=1
        )

    # Track which original indices already have a generated replacement
    latest_indices: set[int] = set()
    for sp in successful_points:
        oi = _get_original_index(sp)
        if oi is not None:
            latest_indices.add(oi)

    # Draw original k points where there is no generated replacement yet
    for i, ((x, y), tval) in enumerate(zip(k_points_plot, k_vals_plot)):
        if i not in latest_indices:
            ax.scatter([x], [y], s=40, zorder=3, color=BLUE, alpha=1.0)  # type: ignore
            off = offsets[i % len(offsets)]
            label = make_label("k", float(tval))
            ax.annotate(  # type: ignore
                label,
                xy=(x, y),
                xytext=off,
                textcoords="offset points",
                fontsize=LABEL_FS,
                color=BLUE,
                ha="left" if off[0] >= 0 else "right",
                va="bottom" if off[1] >= 0 else "top",
            )

    # Draw original l points where there is no generated replacement yet
    for i, ((x, y), tval) in enumerate(zip(l_points_plot, l_vals_plot)):
        orig_idx = n_k + i
        if orig_idx not in latest_indices:
            ax.scatter([x], [y], s=40, zorder=3, color=ORANGE, alpha=1.0)  # type: ignore
            off = offsets[i % len(offsets)]
            label = make_label("l", float(tval))
            ax.annotate(  # type: ignore
                label,
                xy=(x, y),
                xytext=off,
                textcoords="offset points",
                fontsize=LABEL_FS,
                color=ORANGE,
                ha="left" if off[0] >= 0 else "right",
                va="bottom" if off[1] >= 0 else "top",
            )

    # If we have any successful points (or animation is done), build the updated paths
    if len(successful_points) > 0 or not anim_running:
        latest_by_original: dict[int, np.ndarray] = {}
        for sp in successful_points:
            oi = _get_original_index(sp)
            if oi is not None:
                latest_by_original[oi] = sp["point"]

        # Updated k path
        k_path_pts: list[np.ndarray] = []
        for i in range(n_k):
            pt_k = latest_by_original[i] if i in latest_by_original else k_points_plot[i]
            k_path_pts.append(pt_k)
        for i in range(len(k_path_pts) - 1):
            p0 = k_path_pts[i]
            p1 = k_path_pts[i + 1]
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                linewidth=2.2, color=BLUE, alpha=1.0, zorder=4
            )

        # Updated l path
        n_l = l_points_plot.shape[0]
        l_path_pts: list[np.ndarray] = []
        for j in range(n_l):
            orig_idx = n_k + j
            pt_l = latest_by_original[orig_idx] if orig_idx in latest_by_original else l_points_plot[j]
            l_path_pts.append(pt_l)
        for j in range(len(l_path_pts) - 1):
            q0 = l_path_pts[j]
            q1 = l_path_pts[j + 1]
            ax.plot(
                [q0[0], q1[0]], [q0[1], q1[1]],
                linewidth=2.2, color=ORANGE, alpha=1.0, zorder=4
            )

    # Draw latest generated points on top with primes or * marker
    if len(successful_points) > 0:
        latest_success: dict[int, SuccessfulPoint] = {}
        for sp in successful_points:
            oi = _get_original_index(sp)
            if oi is not None:
                latest_success[oi] = sp

        for original_parent_idx in sorted(latest_success.keys()):
            succ_pt_data = latest_success[original_parent_idx]
            succ_pt = succ_pt_data["point"]

            # Count how many times this parent has generated a point
            generation_count = 0
            for sp in successful_points:
                if _get_original_index(sp) == original_parent_idx:
                    generation_count += 1

            if generation_count == 1:
                gen_marker = "'"
            elif generation_count == 2:
                gen_marker = "''"
            else:
                gen_marker = "*"

            if original_parent_idx < n_k:
                prefix = "k"
                color = BLUE
                tval = k_vals_plot[original_parent_idx]
            else:
                prefix = "l"
                color = ORANGE
                local_idx = original_parent_idx - n_k
                tval = l_vals_plot[local_idx]

            ax.scatter([succ_pt[0]], [succ_pt[1]], s=60, zorder=6, color=color)  # type: ignore
            off = offsets[original_parent_idx % len(offsets)]
            try:
                tval_f = float(tval)  # type: ignore[arg-type]
            except Exception:
                tval_f = float(np.array(tval, dtype=float))
            label = make_label(prefix, tval_f, gen_marker)
            ax.annotate(  # type: ignore
                label,
                xy=(succ_pt[0], succ_pt[1]),
                xytext=off,
                textcoords="offset points",
                fontsize=LABEL_FS,
                color=color,
                ha="left" if off[0] >= 0 else "right",
                va="bottom" if off[1] >= 0 else "top",
            )

    # Draw the current candidate and its search circle if animation is on
    if has_animation and in_search and gen_pt is not None and st.session_state.get("anim_running", False):
        all_pts = st.session_state.get("anim_all_pts", np.array([]))
        distance = st.session_state.get("anim_distance", 0.0)

        if all_pts.size > 0:
            n_k = k_points_plot.shape[0]
            n_l = l_points_plot.shape[0]
            total_original = n_k + n_l
            if parent_idx < total_original:
                parent_pt = all_pts[parent_idx]
            else:
                succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                sidx = int(parent_idx - total_original)
                if 0 <= sidx < len(succ_list):
                    parent_pt = succ_list[sidx]["point"]
                else:
                    parent_pt = np.array([0.0, 0.0])
        else:
            parent_pt = np.array([0.0, 0.0])

        ax.scatter([gen_pt[0]], [gen_pt[1]], s=60, zorder=6, color='red')  # type: ignore
        circle = matplotlib.patches.Circle(
            (parent_pt[0], parent_pt[1]),
            radius=distance,
            edgecolor='red',
            facecolor='none',
            linewidth=2.0,
            zorder=5
        )
        ax.add_patch(circle)  # type: ignore

# ============= Helper: choose which config to display on demand ============
def _set_display_config(config_num: int) -> None:
    """
    Set the right-hand plot to show a specific generated configuration.

    This uses st.session_state['anim_all_configs'], finds the matching config_num,
    and updates anim_* state so that the existing drawing + LaTeX code show
    that configuration (without any extra UI changes).
    """
    all_configs: list = st.session_state.get("anim_all_configs", [])
    for cfg in all_configs:
        if cfg.get("config_num") == config_num:
            points: list[SuccessfulPoint] = cfg.get("points", [])
            st.session_state["anim_current_config"] = config_num
            st.session_state["anim_successful_points"] = points
            st.session_state["anim_running"] = False
            st.session_state["show_anim_circle"] = False
            st.session_state["anim_in_search"] = False
            st.session_state["anim_search_steps"] = 0

            # Base points for parent indices
            st.session_state["anim_all_pts"] = np.vstack([k_points_plot, l_points_plot])

            # Set "current" generated point to the last one of this config (for LaTeX etc.)
            if points:
                last_sp = points[-1]
                st.session_state["anim_generated_point"] = np.array(last_sp["point"], dtype=float)
                st.session_state["anim_parent_idx"] = int(last_sp["parent_idx"])
                iters = [int(sp.get("iteration", 0)) for sp in points]
                st.session_state["anim_completed_iterations"] = max(iters) + 1 if iters else 0
            else:
                st.session_state["anim_generated_point"] = None
                st.session_state["anim_parent_idx"] = 0
                st.session_state["anim_completed_iterations"] = 0

            # <<< belangrijk: direct order match herberekenen voor deze config >>>
            update_order_match_flags()
            break

# ============= Layout (two columns) ============
col1, col2 = st.columns(2, gap="small")

with col1:
    st.markdown("<div class='figure-title'>Original configuration</div>", unsafe_allow_html=True)
    st.latex(make_d1_order_latex())
    st.latex(make_d2_order_latex())
    fig_left = render_square_matplotlib_figure(draw_original, XLIM, YLIM)
    st.pyplot(fig_left, clear_figure=True)

    # Show textual comparison of orderings when animation / generation is active
    if "anim_generated_point" in st.session_state:
        left_d1 = make_d1_order_latex()
        right_d1 = make_d1_order_latex_generated()
        left_order = _extract_order_string(left_d1)
        right_order = _extract_order_string(right_d1)
        same_d1 = left_order == right_order
        st.caption(f"Left: {left_order}")
        st.caption(f"Right: {right_order}")
        st.markdown(f"**d₁ order match: {same_d1}**")

        left_d2 = make_d2_order_latex()
        right_d2 = make_d2_order_latex_generated()
        left_order_d2 = _extract_order_string(left_d2)
        right_order_d2 = _extract_order_string(right_d2)
        same_d2 = left_order_d2 == right_order_d2
        st.caption(f"Left d₂: {left_order_d2}")
        st.caption(f"Right d₂: {right_order_d2}")
        st.markdown(f"**d₂ order match: {same_d2}**")

    # Download original plot as PNG
    _buf_left: IO[bytes] = io.BytesIO()
    fig_left.savefig(_buf_left, format="png", dpi=160, bbox_inches="tight")  # type: ignore
    st.download_button(
        label="Save as PNG",
        data=_buf_left.getvalue(),
        file_name="original.png",
        mime="image/png",
        key="dl_left_png",
    )

with col2:
    st.markdown("<div class='figure-title'>Generated configuration</div>", unsafe_allow_html=True)
    st.latex(make_d1_order_latex_generated())
    st.latex(make_d2_order_latex_generated())
    fig_right = render_square_matplotlib_figure(draw_generated_empty, XLIM, YLIM)
    st.pyplot(fig_right, clear_figure=True)

    if "anim_generated_point" in st.session_state:
        left_d1 = make_d1_order_latex()
        right_d1 = make_d1_order_latex_generated()
        left_order = _extract_order_string(left_d1)
        right_order = _extract_order_string(right_d1)
        same_d1 = left_order == right_order
        st.caption(f"Left: {left_order}")
        st.caption(f"Right: {right_order}")
        st.markdown(f"**d₁ order match: {same_d1}**")

        left_d2 = make_d2_order_latex()
        right_d2 = make_d2_order_latex_generated()
        left_order_d2 = _extract_order_string(left_d2)
        right_order_d2 = _extract_order_string(right_d2)
        same_d2 = left_order_d2 == right_order_d2
        st.caption(f"Left d₂: {left_order_d2}")
        st.caption(f"Right d₂: {right_order_d2}")
        st.markdown(f"**d₂ order match: {same_d2}**")

    # Download generated plot as PNG + navigation buttons on ONE row
    _buf_right: IO[bytes] = io.BytesIO()
    fig_right.savefig(_buf_right, format="png", dpi=160, bbox_inches="tight")  # type: ignore

    # Determine navigation state
    all_configs_list: list = st.session_state.get("anim_all_configs", [])
    anim_running_flag = bool(st.session_state.get("anim_running", False))

    if all_configs_list:
        config_nums = sorted(cfg.get("config_num", 1) for cfg in all_configs_list)
        min_cfg = config_nums[0]
        max_cfg = config_nums[-1]
        current_cfg = int(st.session_state.get("anim_current_config", max_cfg))

        prev_disabled = anim_running_flag or (current_cfg <= min_cfg)
        next_disabled = anim_running_flag or (current_cfg >= max_cfg)
    else:
        config_nums = []
        min_cfg = max_cfg = current_cfg = 1
        prev_disabled = True
        next_disabled = True

    col_save, col_prev, col_next = st.columns([1, 1, 1], gap="small")
    with col_save:
        st.download_button(
            label="Save as PNG",
            data=_buf_right.getvalue(),
            file_name="generated.png",
            mime="image/png",
            key="dl_right_png",
        )
    with col_prev:
        prev_clicked = st.button(
            "Previous config",
            key="btn_prev_config",
            disabled=prev_disabled,
        )
    with col_next:
        next_clicked = st.button(
            "Next config",
            key="btn_next_config",
            disabled=next_disabled,
        )

    # Handle navigation clicks
    if all_configs_list and not anim_running_flag:
        if prev_clicked and not prev_disabled:
            new_cfg = max(min_cfg, current_cfg - 1)
            _set_display_config(new_cfg)
            st.rerun()

        if next_clicked and not next_disabled:
            new_cfg = min(max_cfg, current_cfg + 1)
            _set_display_config(new_cfg)
            st.rerun()

# ============= Animation progress (both strategies) ============
if st.session_state.get("anim_running", False):
    left_d1 = make_d1_order_latex()
    left_d2 = make_d2_order_latex()
    right_d1 = make_d1_order_latex_generated()
    right_d2 = make_d2_order_latex_generated()

    same_d1 = _extract_order_string(left_d1) == _extract_order_string(right_d1)
    same_d2 = _extract_order_string(left_d2) == _extract_order_string(right_d2)

    completed_iterations = int(st.session_state.get("anim_completed_iterations", 0))

    # Use radio-button value as default for max_iterations
    default_iterations = int(st.session_state.get("cfg_iterations", 3))
    max_iterations = int(st.session_state.get("anim_max_iterations", default_iterations))
    # Use radio-button value as default for number of configurations
    default_num_configs = int(st.session_state.get("cfg_num_configs", 1))

    search_steps = int(st.session_state.get("anim_search_steps", 0))
    max_search_steps = 7

    distance = float(st.session_state.get("anim_distance", maxdist))
    angle = float(st.session_state.get("anim_angle", 0.0))
    gen_pt = st.session_state.get("anim_generated_point", None)
    parent_idx = int(st.session_state.get("anim_parent_idx", 0))
    all_pts = st.session_state.get("anim_all_pts", np.array([]))
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    in_search = bool(st.session_state.get("anim_in_search", True))

    ok_point = st.session_state.get("anim_ok_point", gen_pt)
    binary_mode = bool(st.session_state.get("anim_binary_mode", False))
    binary_step = int(st.session_state.get("anim_binary_step", 0))

    current_strategy = st.session_state.get("cfg_strategy", strategy)


    # Wait time in seconds for all animation sleeps
    wait_ms = int(st.session_state.get("cfg_wait_ms", 2000))
    wait_s = wait_ms / 1000.0

    if st.session_state.get("anim_config_complete_wait", False):
        st.session_state["anim_config_complete_wait"] = False
        time.sleep(wait_s)
        st.rerun()

    def _get_parent_point(all_pts: np.ndarray, parent_idx: int) -> np.ndarray:
        """Return the effective parent point (original or generated) for a given parent_idx."""
        if all_pts.size == 0:
            return np.array([0.0, 0.0])
        n_k = k_points_plot.shape[0]
        n_l = l_points_plot.shape[0]
        total_original = n_k + n_l
        if parent_idx < total_original:
            return all_pts[parent_idx]
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - total_original)
        if 0 <= sidx < len(succ_list):
            return succ_list[sidx]["point"]
        return np.array([0.0, 0.0])

    # ===== Binary strategy =====
    if current_strategy == "binary":
        current_config = int(st.session_state.get("anim_current_config", 1))

        if not binary_mode:
            # First step: choose an initial candidate at D = 0.5 * maxdist
            parent_pt = _get_parent_point(all_pts, parent_idx)
            D = 0.5 * maxdist
            angle_local = float(np.random.uniform(0, 2 * np.pi))
            new_x = parent_pt[0] + D * np.cos(angle_local)
            new_y = parent_pt[1] + D * np.sin(angle_local)
            new_gen_pt = np.array([new_x, new_y])

            st.session_state["anim_distance"] = D
            st.session_state["anim_generated_point"] = new_gen_pt
            st.session_state["anim_ok_point"] = new_gen_pt.copy()
            st.session_state["anim_angle"] = angle_local
            st.session_state["anim_binary_mode"] = True
            st.session_state["anim_binary_step"] = 0
            st.session_state["anim_delta"] = None
            st.session_state["anim_search_steps"] = 0
            st.session_state["anim_had_full_match"] = False

            time.sleep(wait_s)
            st.rerun()
        else:
            # Subsequent binary steps: update D based on whether orders match
            n = binary_step + 1
            D = distance

            if n <= 7:
                both_match = same_d1 and same_d2
                D_before = D


                # At each step: if both_match → remember that this iteration saw a full match
                if both_match:
                    st.session_state["anim_had_full_match"] = True
                    if gen_pt is not None:
                        st.session_state["anim_ok_point"] = gen_pt.copy()

                delta = maxdist * (0.5 ** (n + 1))
                if both_match:
                    D = D + delta
                else:
                    D = D - delta

                if D < 0.0:
                    D = 0.0

                st.session_state["anim_delta"] = float(delta)

                diag_rows = st.session_state.get("diag_rows", [])
                diag_rows.append({
                    "n": int(n),
                    "order_match_d1": bool(same_d1),
                    "order_match_d2": bool(same_d2),
                    "D_before_update": float(D_before / maxdist) if maxdist != 0 else 0.0,
                    "delta": float(delta / maxdist) if maxdist != 0 else 0.0,
                })
                st.session_state["diag_rows"] = diag_rows

                parent_pt_cur = _get_parent_point(all_pts, parent_idx)
                new_x = parent_pt_cur[0] + D * np.cos(angle)
                new_y = parent_pt_cur[1] + D * np.sin(angle)
                new_gen_pt = np.array([new_x, new_y])

                st.session_state["anim_distance"] = D
                st.session_state["anim_generated_point"] = new_gen_pt
                st.session_state["anim_binary_step"] = n
                st.session_state["anim_search_steps"] = n

                time.sleep(wait_s)
                st.rerun()
            else:
                # End of the 7 binary steps: accept best candidate and move to next iteration/config
                time.sleep(wait_s)

                anim_ok_point = st.session_state.get("anim_ok_point", None)
                had_full_match = bool(st.session_state.get("anim_had_full_match", False))


                # New behavior:
                # - If there was at least one both_match: use the last ok_point
                # - If there was never a both_match: use the parent itself
                if had_full_match and anim_ok_point is not None:
                    final_ok = anim_ok_point.copy()
                else:
                    parent_pt_final = _get_parent_point(all_pts, parent_idx)
                    final_ok = parent_pt_final.copy()

                if final_ok is None:
                    st.session_state["anim_running"] = False
                    st.session_state["show_anim_circle"] = False
                    st.stop()

                # Zet definitieve punt in de sessie
                st.session_state["anim_generated_point"] = final_ok.copy()
                st.session_state["anim_distance"] = 0.0
                st.session_state["anim_binary_mode"] = False
                st.session_state["anim_binary_step"] = 0
                st.session_state["anim_delta"] = None
                st.session_state["anim_had_full_match"] = False  # reset voor volgende iteratie

                # Record successful point
                n_k = k_points_plot.shape[0]
                n_l = l_points_plot.shape[0]
                total_original = n_k + n_l
                if all_pts.size > 0 and parent_idx < total_original:
                    parent_point_val = all_pts[parent_idx]
                    original_parent_idx_val = parent_idx
                else:
                    succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                    sidx = int(parent_idx - total_original)
                    if 0 <= sidx < len(succ_list):
                        parent_point_val = succ_list[sidx]["point"]
                        original_parent_idx_val = succ_list[sidx]["original_parent_idx"]
                    else:
                        parent_point_val = np.array([0.0, 0.0])
                        original_parent_idx_val = 0

                sp: SuccessfulPoint = {
                    "point": np.array(final_ok, dtype=float),
                    "parent_idx": parent_idx,
                    "parent_point": parent_point_val,
                    "original_parent_idx": original_parent_idx_val,
                    "iteration": completed_iterations,
                }
                successful_points.append(sp)
                st.session_state["anim_successful_points"] = successful_points
                st.session_state["anim_completed_iterations"] = completed_iterations + 1
                st.session_state["anim_search_steps"] = 0
                st.session_state["anim_in_search"] = True

                # ---- Nieuw: check order match op het definitieve punt + log ----
                update_order_match_flags()

                final_left_d1 = make_d1_order_latex()
                final_right_d1 = make_d1_order_latex_generated()
                final_same_d1 = _extract_order_string(final_left_d1) == _extract_order_string(final_right_d1)

                final_left_d2 = make_d2_order_latex()
                final_right_d2 = make_d2_order_latex_generated()
                final_same_d2 = _extract_order_string(final_left_d2) == _extract_order_string(final_right_d2)

                iter_log = st.session_state.get("binary_iteration_summary", [])
                iter_log.append({
                    "config": current_config,
                    "iteration": completed_iterations + 1,
                    "match_d1": bool(final_same_d1),
                    "match_d2": bool(final_same_d2),
                })
                st.session_state["binary_iteration_summary"] = iter_log
                # ---- Einde diagnose ----

                # If we reached max_iterations, we may need to move to the next configuration
                if completed_iterations + 1 >= max_iterations:
                    current_config = int(st.session_state.get("anim_current_config", 1))
                    num_configs = int(st.session_state.get("anim_num_configs", default_num_configs))

                    all_configs: list = st.session_state.get("anim_all_configs", [])
                    all_configs.append({
                        "config_num": current_config,
                        "points": list(successful_points)
                    })
                    st.session_state["anim_all_configs"] = all_configs

                    for sp in successful_points:
                        sp["config_num"] = current_config  # type: ignore

                    if current_config < num_configs:
                        # Prepare a new configuration
                        st.session_state["anim_current_config"] = current_config + 1
                        st.session_state["anim_completed_iterations"] = 0
                        st.session_state["anim_search_steps"] = 0
                        st.session_state["anim_running"] = True
                        st.session_state["show_anim_circle"] = True

                        n_k_reset = k_points_plot.shape[0]
                        n_l_reset = l_points_plot.shape[0]
                        total_original_reset = n_k_reset + n_l_reset
                        all_pts_reset = np.vstack([k_points_plot, l_points_plot])
                        all_indices_reset = list(range(total_original_reset))
                        if all_indices_reset:
                            chosen_idx_reset = int(np.random.choice(all_indices_reset))
                        else:
                            chosen_idx_reset = 0

                        youngest_point_reset = None
                        youngest_success_idx_reset = None
                        for idx, s in reversed(list(enumerate(successful_points))):
                            oi = s.get("original_parent_idx", None)
                            if oi is not None and int(oi) == chosen_idx_reset:
                                youngest_point_reset = s["point"]
                                youngest_success_idx_reset = idx
                                break

                        if youngest_point_reset is not None and youngest_success_idx_reset is not None:
                            parent_pt_reset = youngest_point_reset
                            parent_idx_reset = total_original_reset + youngest_success_idx_reset
                        else:
                            parent_idx_reset = chosen_idx_reset
                            parent_pt_reset = all_pts_reset[parent_idx_reset]

                        distance_new = 0.5 * maxdist
                        max_attempts = 20
                        for _ in range(max_attempts):
                            angle_local = float(np.random.uniform(0, 2 * np.pi))
                            new_x = parent_pt_reset[0] + distance_new * np.cos(angle_local)
                            new_y = parent_pt_reset[1] + distance_new * np.sin(angle_local)
                            if XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]:
                                break
                        else:
                            new_x = np.clip(new_x, XLIM[0], XLIM[1])
                            new_y = np.clip(new_y, YLIM[0], YLIM[1])
                        new_gen_pt = np.array([new_x, new_y])

                        st.session_state["anim_parent_idx"] = parent_idx_reset
                        st.session_state["anim_angle"] = angle_local
                        st.session_state["anim_generated_point"] = new_gen_pt
                        st.session_state["anim_distance"] = distance_new
                        st.session_state["anim_all_pts"] = all_pts_reset
                        st.session_state["anim_config_complete_wait"] = True
                        st.session_state["anim_binary_mode"] = True
                        st.session_state["anim_binary_step"] = 0
                        st.session_state["anim_delta"] = None
                        st.session_state["anim_had_full_match"] = False
                    else:
                        # Done with all configurations
                        st.session_state["anim_running"] = False
                        st.session_state["show_anim_circle"] = False
                else:
                    # Prepare next iteration in the same configuration
                    n_k = k_points_plot.shape[0]
                    n_l = l_points_plot.shape[0]
                    total_original = n_k + n_l
                    all_indices = list(range(total_original))
                    if all_indices:
                        chosen_idx = int(np.random.choice(all_indices))
                    else:
                        chosen_idx = 0

                    youngest_point = None
                    youngest_success_idx = None
                    for idx, s in reversed(list(enumerate(successful_points))):
                        oi = s.get("original_parent_idx", None)
                        if oi is not None and int(oi) == chosen_idx:
                            youngest_point = s["point"]
                            youngest_success_idx = idx
                            break
                    if youngest_point is not None and youngest_success_idx is not None:
                        parent_pt_new = youngest_point
                        parent_idx_new = total_original + youngest_success_idx
                    else:
                        if chosen_idx < n_k:
                            parent_pt_new = k_points_plot[chosen_idx]
                        else:
                            parent_pt_new = l_points_plot[chosen_idx - n_k]
                        parent_idx_new = chosen_idx

                    distance_new = 0.5 * maxdist
                    max_attempts = 20
                    for _ in range(max_attempts):
                        angle_local = float(np.random.uniform(0, 2 * np.pi))
                        new_x = parent_pt_new[0] + distance_new * np.cos(angle_local)
                        new_y = parent_pt_new[1] + distance_new * np.sin(angle_local)
                        if XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]:
                            break
                    else:
                        new_x = np.clip(new_x, XLIM[0], XLIM[1])
                        new_y = np.clip(new_y, YLIM[0], YLIM[1])
                    new_gen_pt = np.array([new_x, new_y])

                    st.session_state["anim_parent_idx"] = parent_idx_new
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = distance_new
                    st.session_state["anim_ok_point"] = new_gen_pt.copy()
                    st.session_state["anim_binary_mode"] = True
                    st.session_state["anim_binary_step"] = 0
                    st.session_state["anim_delta"] = None
                    st.session_state["anim_had_full_match"] = False

                time.sleep(wait_s)
                st.rerun()

    # ===== Exponential strategy (animated version) =====
    else:
        # Same logic as generate_exp, but with sleeps and reruns
        if (same_d1 and same_d2 and gen_pt is not None) or (distance <= 0.0 and gen_pt is not None):
            n_k = k_points_plot.shape[0]
            n_l = l_points_plot.shape[0]
            total_original = n_k + n_l
            if all_pts.size > 0 and parent_idx < total_original:
                parent_point_val = all_pts[parent_idx]
                original_parent_idx_val = parent_idx
            else:
                succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                sidx = int(parent_idx - total_original)
                if 0 <= sidx < len(succ_list):
                    parent_point_val = succ_list[sidx]["point"]
                    original_parent_idx_val = succ_list[sidx]["original_parent_idx"]
                else:
                    parent_point_val = np.array([0.0, 0.0])
                    original_parent_idx_val = 0
            sp: SuccessfulPoint = {
                "point": np.array(gen_pt, dtype=float),
                "parent_idx": parent_idx,
                "parent_point": parent_point_val,
                "original_parent_idx": original_parent_idx_val,
                "iteration": completed_iterations,
            }
            successful_points.append(sp)
            st.session_state["anim_successful_points"] = successful_points
            st.session_state["anim_completed_iterations"] = completed_iterations + 1
            st.session_state["anim_search_steps"] = 0
            st.session_state["anim_in_search"] = True
            st.session_state["anim_delta"] = None

            # <<< hier opnieuw: match evalueren na plaatsing >>>
            update_order_match_flags()

            if completed_iterations + 1 >= max_iterations:
                current_config = int(st.session_state.get("anim_current_config", 1))
                num_configs = int(st.session_state.get("anim_num_configs", default_num_configs))

                all_configs: list = st.session_state.get("anim_all_configs", [])
                all_configs.append({
                    "config_num": current_config,
                    "points": list(successful_points)
                })
                st.session_state["anim_all_configs"] = all_configs

                for sp in successful_points:
                    sp["config_num"] = current_config  # type: ignore

                if current_config < num_configs:
                    st.session_state["anim_current_config"] = current_config + 1
                    st.session_state["anim_completed_iterations"] = 0
                    st.session_state["anim_search_steps"] = 0
                    st.session_state["anim_running"] = True
                    st.session_state["show_anim_circle"] = True

                    n_k_reset = k_points_plot.shape[0]
                    n_l_reset = l_points_plot.shape[0]
                    total_original_reset = n_k_reset + n_l_reset
                    all_pts_reset = np.vstack([k_points_plot, l_points_plot])
                    all_indices_reset = list(range(total_original_reset))
                    if all_indices_reset:
                        chosen_idx_reset = int(np.random.choice(all_indices_reset))
                    else:
                        chosen_idx_reset = 0

                    youngest_point_reset = None
                    youngest_success_idx_reset = None
                    for idx, s in reversed(list(enumerate(successful_points))):
                        oi = s.get("original_parent_idx", None)
                        if oi is not None and int(oi) == chosen_idx_reset:
                            youngest_point_reset = s["point"]
                            youngest_success_idx_reset = idx
                            break

                    if youngest_point_reset is not None and youngest_success_idx_reset is not None:
                        parent_pt_reset = youngest_point_reset
                        parent_idx_reset = total_original_reset + youngest_success_idx_reset
                    else:
                        parent_idx_reset = chosen_idx_reset
                        parent_pt_reset = all_pts_reset[parent_idx_reset]

                    distance_new = maxdist
                    max_attempts = 20
                    for _ in range(max_attempts):
                        angle_local = float(np.random.uniform(0, 2 * np.pi))
                        new_x = parent_pt_reset[0] + distance_new * np.cos(angle_local)
                        new_y = parent_pt_reset[1] + distance_new * np.sin(angle_local)
                        if XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]:
                            break
                    else:
                        new_x = np.clip(new_x, XLIM[0], XLIM[1])
                        new_y = np.clip(new_y, YLIM[0], YLIM[1])
                    new_gen_pt = np.array([new_x, new_y])

                    st.session_state["anim_parent_idx"] = parent_idx_reset
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = distance_new
                    st.session_state["anim_all_pts"] = all_pts_reset
                    st.session_state["anim_config_complete_wait"] = True
                else:
                    st.session_state["anim_running"] = False
                    st.session_state["show_anim_circle"] = False
        else:
            search_steps += 1
            st.session_state["anim_search_steps"] = search_steps

            if search_steps >= max_search_steps:
                if gen_pt is not None and all_pts.size > 0:
                    n_k = k_points_plot.shape[0]
                    n_l = l_points_plot.shape[0]
                    total_original = n_k + n_l
                    if parent_idx < total_original:
                        parent_pt_cur = all_pts[parent_idx]
                    else:
                        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                        sidx = int(parent_idx - total_original)
                        if 0 <= sidx < len(succ_list):
                            parent_pt_cur = succ_list[sidx]["point"]
                        else:
                            parent_pt_cur = np.array([0.0, 0.0])

                    st.session_state["anim_generated_point"] = parent_pt_cur.copy()
                    st.session_state["anim_distance"] = 0.0
                    st.session_state["anim_in_search"] = True
            else:
                if gen_pt is not None and all_pts.size > 0:
                    n_k = k_points_plot.shape[0]
                    n_l = l_points_plot.shape[0]
                    total_original = n_k + n_l
                    if parent_idx < total_original:
                        parent_pt_cur = all_pts[parent_idx]
                    else:
                        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                        sidx = int(parent_idx - total_original)
                        if 0 <= sidx < len(succ_list):
                            parent_pt_cur = succ_list[sidx]["point"]
                        else:
                            parent_pt_cur = np.array([0.0, 0.0])

                    new_distance = distance / 2.0
                    min_distance = 1e-5
                    if new_distance < min_distance:
                        new_distance = min_distance * 2.0
                        angle_local = float(np.random.uniform(0, 2 * np.pi))
                    else:
                        angle_local = angle
                    angle_local += float(np.random.uniform(-0.25, 0.25))
                    angle_local = angle_local % (2 * np.pi)
                    new_x = parent_pt_cur[0] + new_distance * np.cos(angle_local)
                    new_y = parent_pt_cur[1] + new_distance * np.sin(angle_local)
                    new_gen_pt = np.array([new_x, new_y])

                    if not (XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]):
                        angle_local = (angle_local + np.pi) % (2 * np.pi)
                        new_x = parent_pt_cur[0] + new_distance * np.cos(angle_local)
                        new_y = parent_pt_cur[1] + new_distance * np.sin(angle_local)
                        new_gen_pt = np.array([new_x, new_y])

                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = new_distance
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_in_search"] = True

        time.sleep(wait_s)
        st.rerun()


# ============= CSV Export Section ============
st.markdown("<hr />", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:1.5rem;'>Generated configuration (CSV)</h3>", unsafe_allow_html=True)

all_configs_list: list = st.session_state.get("anim_all_configs", [])
current_successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
current_config_num = int(st.session_state.get("anim_current_config", 1))

if all_configs_list or current_successful_points:
    # Collect all generated points, grouped per configuration
    all_points_by_config: dict[int, list[SuccessfulPoint]] = {}

    for config_data in all_configs_list:
        config_num = config_data["config_num"]
        points = config_data["points"]
        all_points_by_config[config_num] = points

    if current_successful_points:
        all_points_by_config[current_config_num] = current_successful_points

    # For each (config, original_index) keep the latest generated point
    latest_generated: dict[tuple[int, int], np.ndarray] = {}
    for config_num, points in all_points_by_config.items():
        for sp in points:
            orig_idx = sp.get("original_parent_idx", 0)
            latest_generated[(config_num, orig_idx)] = sp["point"]

    all_config_nums = sorted(all_points_by_config.keys())

    csv_rows: list[tuple[int, float, int, float, float]] = []

    n_k = k_points_plot.shape[0]
    n_l = l_points_plot.shape[0]

    # Build rows for each configuration, in the style (c, t, o, x, y)
    for config_num in all_config_nums:
        # Shift configuration id so each configuration has a unique c-value
        c_value = selected_c_int + config_num

        # k-points (o = 0)
        for i in range(n_k):
            t_val = float(k_vals_plot[i])
            if (config_num, i) in latest_generated:
                point = latest_generated[(config_num, i)]
            else:
                point = k_points_plot[i]
            csv_rows.append((c_value, t_val, 0, float(point[0]), float(point[1])))

        # l-points (o = 1)
        for j in range(n_l):
            t_val = float(l_vals_plot[j])
            orig_idx = n_k + j
            if (config_num, orig_idx) in latest_generated:
                point = latest_generated[(config_num, orig_idx)]
            else:
                point = l_points_plot[j]
            csv_rows.append((c_value, t_val, 1, float(point[0]), float(point[1])))

    csv_rows.sort(key=lambda row: (row[0], row[1], row[2]))

    csv_lines = ["c,t,o,x,y"]
    for c, t, o, x, y in csv_rows:
        csv_lines.append(f"{c},{t},{o},{x:.6f},{y:.6f}")

    csv_content = "\n".join(csv_lines)

    st.text_area(
        "Copy the generated configuration below:",
        value=csv_content,
        height=200,
        key="csv_export"
    )

    st.download_button(
        label="Download as CSV",
        data=csv_content,
        file_name=f"generated_config_c{selected_c_int}.csv",
        mime="text/csv",
        key="dl_csv"
    )
else:
    st.info("Run an animation or use 'Generate configurations' to generate configuration data.")


# ============= Diagnostic table for binary strategy ============
st.markdown("<hr />", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:1.5rem;'>Diagnostics binary strategy (per step)</h3>", unsafe_allow_html=True)

diag_rows = st.session_state.get("diag_rows", [])

if diag_rows:
    diag_df = pd.DataFrame(diag_rows, columns=[
        "n",
        "order_match_d1",
        "order_match_d2",
        "D_before_update",
        "delta",
    ])
    st.table(diag_df)
else:
    st.info("No binary approximation steps recorded yet.")


# ============= New diagnostic text box: result per iteration ============
st.markdown("<h3 style='margin-top:1.5rem;'>Order match per iteration (binary strategy)</h3>", unsafe_allow_html=True)

iter_log = st.session_state.get("binary_iteration_summary", [])

if iter_log:
    lines: list[str] = []
    for item in iter_log:
        cnum = item.get("config", 1)
        it = item.get("iteration", 0)
        m1 = item.get("match_d1", False)
        m2 = item.get("match_d2", False)
        lines.append(f"Config {cnum}, iteration {it}: d₁ match = {m1}, d₂ match = {m2}")
    summary_text = "\n".join(lines)
    st.text_area(
        "Overview of order match after final placement of the point",
        value=summary_text,
        height=160,
        key="binary_iter_overview"
    )
else:
    st.info("No final points placed with the binary strategy yet.")


