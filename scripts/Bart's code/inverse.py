# -*- coding: utf-8 -*-
# inverse.py
# Streamlit app - PDP Inverse Analysis
import sys
from pathlib import Path

# Ensure the script's own directory is on sys.path so local packages
# (pdp_utils, etc.) are importable even when run from a temp/remote dir.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from typing import Tuple, Callable, IO, Optional, Any
import io
import logging
import time
import traceback
from PIL import Image as PILImage

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot as plt

import plotly.graph_objects as go  # type: ignore[import-untyped]
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from pdp_utils.core import (
    COORD_DISPLAY_PRECISION,
    COORD_CSV_PRECISION,
    OBJECT_LABELS,
    SuccessfulPoint,
    IncrementalPDPChecker,
)
from pdp_utils.config import LANE_CONFIGURATIONS
from pdp_utils.data_loading import to_numeric_series, extract_points_from_df
from pdp_utils.order_comparison import (
    extract_order_string,
    check_pdp_match,
    check_pdp_match_detailed,
    check_pdp_match_frenet_detailed,
)
from pdp_utils.frenet_coordinates import FrenetFrame
from pdp_utils.lane_geometry import (
    safe_normalize as _safe_normalize_mod,
    calculate_vehicle_speeds as _calculate_vehicle_speeds_mod,
    determine_driving_direction as _determine_driving_direction_mod,
    vehicles_same_direction as _vehicles_same_direction_mod,
    offset_polyline as _offset_polyline_mod,
    lane_polylines_bounds as _lane_polylines_bounds_mod,
    extract_centerline_from_data as _extract_centerline_mod,
    build_lane_polylines as _build_lane_polylines_mod,
)
from pdp_utils.plotting import (
    add_lane_polylines_plotly as _add_lane_polylines_plotly,
    add_intersection_lanes_plotly as _add_intersection_lanes,
    draw_frenet_axes as _draw_frenet_axes,
    draw_intersection_lanes_matplotlib as _draw_intersection_lanes_matplotlib,
    annotate_points,
)
from pdp_utils.drawing import (
    BLUE,
    ORANGE,
    LABEL_FS,
    OBJECT_COLORS,
    OBJECT_COLORS_PLOTLY,
)

# Type annotations for imported items with incomplete type stubs
LANE_CONFIGURATIONS: dict[int, dict[str, Any]]
check_pdp_match_detailed: Callable[..., dict[str, Any]]
check_pdp_match_frenet_detailed: Callable[..., dict[str, Any]]

logger = logging.getLogger(__name__)

# ============= Named constants (avoid magic numbers) =============
MAX_DIRECTION_ATTEMPTS: int = 50       # Max attempts to find a valid direction vector
MAX_RESET_ATTEMPTS: int = 20           # Max attempts when resetting/placing a new point
MAX_BINARY_SEARCH_STEPS: int = 7       # Steps in binary search for boundary finding
MAX_GENERATION_ITERATIONS: int = 1000  # Max configs for ext30 generation
MAX_FILTER_CONFIGS: int = 100          # Max configs for filtered timestamp generation
MAX_FILTER_ITERATIONS: int = 2500      # Max iterations per config for filtered generation
EARLY_STOP_PATIENCE: int = 50          # Stop config early after this many stagnant iterations
EARLY_STOP_EPSILON: float = 1e-9       # Movement threshold below which an iteration counts as stagnant
GIF_FRAME_DURATION_MS: int = 200       # Milliseconds per GIF frame
GIF_LAST_FRAME_PAUSE_MS: int = 1500    # Milliseconds to pause on the last GIF frame
DEFAULT_BUFFER_X: float = 25.0         # Default x-axis buffer margin
DEFAULT_BUFFER_Y: float = 10.0         # Default y-axis buffer margin
DEFAULT_MAXDIST_FALLBACK: float = 10.0 # Fallback max distance when no pairwise data
CURVED_ROAD_CONFIGS: list[int] = [15, 17]  # S-curve config numbers for Frenet coords

# ============= Page configuration =============
st.set_page_config(
    page_title="pdp inverse",
    page_icon="ðŸ“",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============= Authentication =============
def check_password():
    """Returns `True` if the user had the correct password."""

    auth_qp_key = "pdp_auth"

    def has_query_auth_flag() -> bool:
        try:
            return str(st.query_params.get(auth_qp_key, "0")) == "1"
        except Exception as e:
            logger.warning(f"[AUTH] Could not read query param '{auth_qp_key}': {e}")
            return False

    def set_query_auth_flag(enabled: bool) -> None:
        try:
            if enabled:
                st.query_params[auth_qp_key] = "1"
            elif auth_qp_key in st.query_params:
                del st.query_params[auth_qp_key]
        except Exception as e:
            logger.warning(f"[AUTH] Could not set query param '{auth_qp_key}': {e}")

    # If already authenticated in this session, keep access on reruns.
    if bool(st.session_state.get("password_correct", False)):
        set_query_auth_flag(True)
        return True

    # If query flag is present, restore authenticated state for this session.
    if has_query_auth_flag():
        st.session_state["password_correct"] = True
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Use .get() to safely access the password key, avoiding KeyError if not present
        entered_password = str(st.session_state.get("password", "")).strip()
        if not entered_password:
            return
        if entered_password == "pdp2025":
            st.session_state["password_correct"] = True
            set_query_auth_flag(True)
            # Safely delete password from state if it exists
            if "password" in st.session_state:
                del st.session_state["password"]  # don't store password
        else:
            # Only mark explicit failed login attempts as False.
            st.session_state["password_correct"] = False
            set_query_auth_flag(False)

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

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
/* Force both plot columns to have identical width */
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    width: calc(50% - 0.5rem) !important;
    flex: 0 0 calc(50% - 0.5rem) !important;
}
/* Force matplotlib figures to have same size in both columns */
[data-testid="column"] [data-testid="stImage"],
[data-testid="column"] .stPlotlyChart,
[data-testid="column"] > div > div > img {
    max-width: 100% !important;
    width: 100% !important;
}
/* LaTeX formulas should not overflow and have fixed height */
.stLatex {
    overflow-x: auto !important;
    max-width: 100% !important;
    min-height: 2em !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============= Main title =============
st.markdown("<h1 class='headline'>pdp inverse</h1>", unsafe_allow_html=True)
st.markdown("<hr />", unsafe_allow_html=True)

# ---------- to_numeric_series imported from pdp_utils.data_loading ----------

# ============= Load CSV (recognizes 'header: c,t,o,x,y') ============
def load_points(csv_name: str = "voorbeeld.csv", o_val: int = 0, c_val: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file with columns (c, t, o, x, y). If the first line starts with
    'header:', it is skipped and hard-coded column names ['c','t','o','x','y'] are used.

    Filters:
      - c == c_val
      - o == o_val

    Returns:
      - pts: (N,2) numpy array [x,y] sorted by t
      - ts:  (N,) numpy array with t-values (sorted)
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

# ============= Settings (configuration and time window) ============
def _read_clean_df(csv_name: str) -> pd.DataFrame:
    """Read the CSV once into a clean DataFrame for sidebar settings."""
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

# ============= Data Source Selection ============
st.markdown("""
<div class='settings-card'>
  <h3>Reference Configuration Source</h3>
""", unsafe_allow_html=True)

data_source = st.radio(
    "Select data source",
    options=["Preset configurations", "Upload custom file", "Create random configuration"],
    index=0,
    horizontal=True,
    key="data_source",
    help="""Choose how to load the reference configuration:

• **Preset configurations**: Load from the built-in 'voorbeeld.csv' file containing 11 predefined configurations.

• **Upload custom file**: Upload your own CSV file with columns (c, t, o, x, y) where c=configuration ID, t=timestamp, o=object type (0=k, 1=l), x/y=coordinates.

• **Create random configuration**: Generate a random configuration with specified number of points and timestamps. You can then interactively edit the coordinates."""
)

# Initialize variables that will be set based on data source
_df_all = None
available_configs = []
selected_c_int = 0
_is_custom_upload = (data_source == "Upload custom file")

# When using custom upload, force external points off by clearing session state keys
if _is_custom_upload:
    if st.session_state.get("cfg_use_external_points", False):
        st.session_state["cfg_use_external_points"] = False
    st.session_state["use_external_points"] = False
    if "external_points" in st.session_state:
        del st.session_state["external_points"]
    # Default PDP variant to fundamental for custom uploads
    st.session_state["cfg_pdp_variants"] = ["fundamental"]

if data_source == "Preset configurations":
    _df_all = _read_clean_df("voorbeeld.csv")
    _mask_t = _df_all["t"].isin(_df_all["t"].unique())  # type: ignore
    _df_all = _df_all[_mask_t]
    
    available_configs = sorted(_df_all["c"].dropna().unique().astype(int).tolist())  # type: ignore
    if not available_configs:
        st.error("No configurations found (column 'c' is empty).")
        st.stop()

elif data_source == "Upload custom file":
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="uploaded_csv",
        help="Upload a CSV file with columns: c (configuration ID), t (timestamp), o (object: 0=k, 1=l), x (x-coordinate), y (y-coordinate). The file can have a header row or start with 'header: c,t,o,x,y'."
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            content = uploaded_file.read().decode("utf-8")
            lines = content.strip().split("\n")
            first_line = lines[0].strip().lower()
            
            names = ["c", "t", "o", "x", "y"]
            # Column name aliases for common alternative naming conventions
            _col_aliases: dict[str, str] = {
                "conid": "c", "configid": "c", "config": "c", "configuration": "c",
                "tstid": "t", "timestamp": "t", "time": "t",
                "poiid": "o", "objectid": "o", "object": "o", "obj": "o",
            }
            if first_line.startswith("header:"):
                # Skip header line
                from io import StringIO
                _df_all = pd.read_csv(StringIO("\n".join(lines[1:])), header=None, names=names)  # type: ignore[call-overload]
            else:
                from io import StringIO
                uploaded_file.seek(0)
                _df_all = pd.read_csv(StringIO(content))  # type: ignore[call-overload]
                # Auto-rename known aliases (case-insensitive)
                _rename_map = {col: _col_aliases[col.strip().lower()] for col in _df_all.columns if col.strip().lower() in _col_aliases}
                if _rename_map:
                    _df_all = _df_all.rename(columns=_rename_map)  # type: ignore[assignment]
                if not set(names).issubset(_df_all.columns):
                    _df_all = pd.read_csv(StringIO(content), header=None, names=names)  # type: ignore[call-overload]
            
            # Clean the dataframe
            for col in names:
                _df_all[col] = pd.to_numeric(_df_all[col], errors="coerce")  # type: ignore[assignment]
            _df_all = _df_all.dropna(subset=names)  # type: ignore[assignment]
            _df_all = _df_all.reset_index(drop=True)
            
            available_configs = sorted(_df_all["c"].dropna().unique().astype(int).tolist())
            if not available_configs:
                st.error("No configurations found in uploaded file (column 'c' is empty).")
                st.stop()
            
            st.success(f"Loaded {len(_df_all)} rows with {len(available_configs)} configuration(s).")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()

elif data_source == "Create random configuration":
    # Initialize bounds for random config if not set (defaults: 0-100)
    if "coord_min_x" not in st.session_state:
        st.session_state["coord_min_x"] = 0.0
    if "coord_max_x" not in st.session_state:
        st.session_state["coord_max_x"] = 100.0
    if "coord_min_y" not in st.session_state:
        st.session_state["coord_min_y"] = 0.0
    if "coord_max_y" not in st.session_state:
        st.session_state["coord_max_y"] = 100.0
    
    st.markdown("**Random Configuration Generator**")
    
    rand_col1, rand_col2 = st.columns([1, 1])
    with rand_col1:
        num_points = st.number_input(
            "Number of points",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            key="rand_num_points",
            help="Number of moving objects (points). Each point will have its own trajectory over time."
        )
    with rand_col2:
        # Halve the number of timestamps (default and max)
        orig_max_timestamps = 20
        halved_max_timestamps = orig_max_timestamps // 2
        halved_default = 3 // 2 if 3 > 1 else 1
        num_timestamps = st.number_input(
            "Number of timestamps (halved)",
            min_value=1,
            max_value=halved_max_timestamps,
            value=halved_max_timestamps,
            step=1,
            key="rand_num_timestamps",
            help=f"Number of timestamps (time moments). Each point will have a position at each timestamp. [Halved: {halved_max_timestamps}]"
        )
    
    # Generate or load random configuration
    if st.button("Generate Random Configuration", key="btn_gen_random", 
                 help="Click to generate a new random configuration within the Coordinate Bounds specified below."):
        # Generate random points - each object has num_timestamps positions
        np.random.seed(None)  # Use current time as seed for true randomness
        
        # Get coordinate bounds from session state
        # Use cfg_coord_* keys (from the number inputs) if available, otherwise use coord_* session state
        gen_min_x = float(st.session_state.get("cfg_coord_min_x", st.session_state.get("coord_min_x", 0.0)))
        gen_max_x = float(st.session_state.get("cfg_coord_max_x", st.session_state.get("coord_max_x", 100.0)))
        gen_min_y = float(st.session_state.get("cfg_coord_min_y", st.session_state.get("coord_min_y", 0.0)))
        gen_max_y = float(st.session_state.get("cfg_coord_max_y", st.session_state.get("coord_max_y", 100.0)))
        
        # Generate coordinates for all points within the specified bounds
        all_coords = {}
        for p in range(num_points):
            p_x = np.random.uniform(gen_min_x, gen_max_x, num_timestamps)
            p_y = np.random.uniform(gen_min_y, gen_max_y, num_timestamps)
            all_coords[p] = list(zip(p_x, p_y))
        
        # Store in session state
        st.session_state["random_all_coords"] = all_coords
        st.session_state["random_num_points"] = num_points
        st.session_state["random_config_generated"] = True
    
    # Check if we have a generated configuration
    if not st.session_state.get("random_config_generated", False):
        st.info("Click 'Generate Random Configuration' to create a new configuration, or edit existing coordinates below.")
        # Initialize with default values if not present - use bounds from session state
        if "random_all_coords" not in st.session_state:
            # Get bounds for generating sensible defaults
            _init_min_x = st.session_state.get("coord_min_x", 0.0)
            _init_max_x = st.session_state.get("coord_max_x", 100.0)
            _init_min_y = st.session_state.get("coord_min_y", 0.0)
            _init_max_y = st.session_state.get("coord_max_y", 100.0)
            # Create default points within bounds
            _mid_x = (_init_min_x + _init_max_x) / 2
            _mid_y = (_init_min_y + _init_max_y) / 2
            _step_x = (_init_max_x - _init_min_x) / 10
            _step_y = (_init_max_y - _init_min_y) / 10
            st.session_state["random_all_coords"] = {
                0: [(_mid_x, _mid_y), (_mid_x + _step_x, _mid_y + _step_y), (_mid_x + 2*_step_x, _mid_y + 2*_step_y)],
                1: [(_mid_x - _step_x, _mid_y - _step_y), (_mid_x, _mid_y), (_mid_x + _step_x, _mid_y + _step_y)]
            }
            st.session_state["random_num_points"] = 2
    
    # Display editable coordinates
    st.markdown("**Edit Coordinates** (modify values to adjust point positions)")
    
    # Get bounds for fallback values
    _fb_min_x = st.session_state.get("coord_min_x", 0.0)
    _fb_max_x = st.session_state.get("coord_max_x", 100.0)
    _fb_min_y = st.session_state.get("coord_min_y", 0.0)
    _fb_max_y = st.session_state.get("coord_max_y", 100.0)
    _fb_mid_x = (_fb_min_x + _fb_max_x) / 2
    _fb_mid_y = (_fb_min_y + _fb_max_y) / 2
    
    all_coords = st.session_state.get("random_all_coords", {0: [(_fb_mid_x, _fb_mid_y)], 1: [(_fb_mid_x, _fb_mid_y)]})
    stored_num_points = st.session_state.get("random_num_points", 2)
    
    # Create editable dataframes for each point
    edited_coords = {}
    # Display editors in columns to make them narrower
    cols = st.columns(stored_num_points)
    for p in range(stored_num_points):
        point_label = chr(ord('k') + p) if p < 26 else f"p{p}"  # k, l, m, n, ... or p0, p1, ...
        coords = all_coords.get(p, [(50.0, 50.0)])
        
        with cols[p]:
            st.markdown(f"**Point {point_label}**")
            p_df = pd.DataFrame({
                "t": list(range(len(coords))),
                "x": [round(c[0], 2) for c in coords],
                "y": [round(c[1], 2) for c in coords]
            })
            edited_p_df = st.data_editor(
                p_df,
                key=f"edit_point_{p}_coords",
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "t": st.column_config.NumberColumn("t", width="small"),
                    "x": st.column_config.NumberColumn("x", format="%.2f", width="small"),
                    "y": st.column_config.NumberColumn("y", format="%.2f", width="small"),
                }
            )
            
            # st.data_editor always returns a DataFrame, so we can use it directly
            edited_coords[p] = list(zip(edited_p_df["x"].tolist(), edited_p_df["y"].tolist()))
    
    # Update session state with edited values
    if edited_coords:
        st.session_state["random_all_coords"] = edited_coords
    
    # Build the dataframe from edited coordinates
    all_coords_final = st.session_state.get("random_all_coords", {})
    
    rows: list[dict[str, int | float]] = []
    for p, coords in all_coords_final.items():
        for t_idx, (x, y) in enumerate(coords):
            rows.append({"c": 0, "t": t_idx, "o": p, "x": x, "y": y})
    
    _df_all = pd.DataFrame(rows)
    available_configs = [0]
    
    # Save configuration button
    # Build CSV content once for both Save and Download buttons
    csv_content = "header: c,t,o,x,y\n"
    for _, row in _df_all.iterrows():
        csv_content += f"{int(row['c'])},{int(row['t'])},{int(row['o'])},{row['x']:.{COORD_CSV_PRECISION}f},{row['y']:.{COORD_CSV_PRECISION}f}\n"

    save_col1, save_col2 = st.columns([1, 2])
    with save_col1:
        save_filename = st.text_input(
            "Filename",
            value="custom_reference.csv",
            key="save_filename",
            help="Name of the file to save the configuration to."
        )
    with save_col2:
        st.markdown("<div style='margin-top:1.7rem;'>", unsafe_allow_html=True)
        if st.button("Save Configuration", key="btn_save_config",
                     help="Save the current configuration to a CSV file in the application directory."):
            try:
                save_path = Path(__file__).parent / save_filename
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(csv_content)
                
                st.success(f"Configuration saved to: {save_path}")
            except Exception as e:
                st.error(f"Error saving file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Also provide download button
    st.download_button(
        label="Download Configuration as CSV",
        data=csv_content,
        file_name="custom_reference.csv",
        mime="text/csv",
        key="dl_custom_config",
        help="Download the current configuration as a CSV file that can be loaded later via 'Upload custom file'."
    )

# Validate that we have data
if _df_all is None or len(_df_all) == 0:
    st.error("No data loaded. Please select a valid data source.")
    st.stop()

# ============= Coordinate Bounds (auto-calculated from data) =============
if data_source == "Create random configuration":
    # For random config, use session state values (already initialized to 0-100)
    _auto_min_x = st.session_state.get("coord_min_x", 0.0)
    _auto_max_x = st.session_state.get("coord_max_x", 100.0)
    _auto_min_y = st.session_state.get("coord_min_y", 0.0)
    _auto_max_y = st.session_state.get("coord_max_y", 100.0)
    _default_min_x, _default_max_x = 0.0, 100.0
    _default_min_y, _default_max_y = 0.0, 100.0
else:
    # Calculate bounds from ALL loaded data (across all configurations)
    # This ensures bounds cover the entire dataset regardless of which config is selected
    _data_min_x = float(_df_all["x"].min())
    _data_max_x = float(_df_all["x"].max())
    _data_min_y = float(_df_all["y"].min())
    _data_max_y = float(_df_all["y"].max())

    # Add 10% margin to auto-calculated bounds
    _data_range_x = _data_max_x - _data_min_x
    _data_range_y = _data_max_y - _data_min_y
    _auto_min_x = _data_min_x - 0.1 * _data_range_x if _data_range_x > 0 else _data_min_x - 10
    _auto_max_x = _data_max_x + 0.1 * _data_range_x if _data_range_x > 0 else _data_max_x + 10
    _auto_min_y = _data_min_y - 0.1 * _data_range_y if _data_range_y > 0 else _data_min_y - 10
    _auto_max_y = _data_max_y + 0.1 * _data_range_y if _data_range_y > 0 else _data_max_y + 10

    # Round to nice values
    _auto_min_x = float(np.floor(_auto_min_x / 10) * 10)
    _auto_max_x = float(np.ceil(_auto_max_x / 10) * 10)
    _auto_min_y = float(np.floor(_auto_min_y / 10) * 10)
    _auto_max_y = float(np.ceil(_auto_max_y / 10) * 10)
    
    _default_min_x, _default_max_x = _auto_min_x, _auto_max_x
    _default_min_y, _default_max_y = _auto_min_y, _auto_max_y

    # Initialize session state with auto-calculated values ONLY if:
    # 1. The bounds have never been set (coord_min_x not in session_state), OR
    # 2. The underlying data has changed (different data file/config)
    # Do NOT overwrite if user has used Auto Detect or manually adjusted bounds
    _current_data_hash = f"{_data_min_x:.2f}_{_data_max_x:.2f}_{_data_min_y:.2f}_{_data_max_y:.2f}"
    _existing_hash = st.session_state.get("_bounds_data_hash", "")
    
    # Only auto-initialize if no bounds exist OR if we're loading completely new data
    # (i.e., the existing hash doesn't start with "auto_detect_" and doesn't match current data)
    _bounds_never_set = "cfg_coord_min_x" not in st.session_state
    _data_changed = _existing_hash != _current_data_hash and not _existing_hash.startswith("auto_detect_")
    
    if _bounds_never_set or _data_changed:
        # Update BOTH coord_ keys AND cfg_coord_ widget keys
        st.session_state["coord_min_x"] = _auto_min_x
        st.session_state["coord_max_x"] = _auto_max_x
        st.session_state["coord_min_y"] = _auto_min_y
        st.session_state["coord_max_y"] = _auto_max_y
        st.session_state["cfg_coord_min_x"] = _auto_min_x
        st.session_state["cfg_coord_max_x"] = _auto_max_x
        st.session_state["cfg_coord_min_y"] = _auto_min_y
        st.session_state["cfg_coord_max_y"] = _auto_max_y
        st.session_state["_bounds_data_hash"] = _current_data_hash

st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
st.markdown("**Coordinate Bounds**")
if data_source == "Create random configuration":
    st.caption("Define the valid coordinate range for generated points. Click 'Generate Random Configuration' after adjusting bounds.")
else:
    st.caption("Define the valid coordinate range for generated points. Auto-calculated from loaded data with 10% margin. Visualizations will show an additional 10% margin for display.")

# Check if Auto Detect was triggered - if so, apply the pending bounds BEFORE widgets are created
# CRITICAL: We must update BOTH the coord_ keys AND the cfg_coord_ widget keys
# because Streamlit widgets use the key value, not the value parameter, after first render
if st.session_state.get("_pending_bounds_update", False):
    _pending = st.session_state.get("_pending_bounds", {})
    if _pending:
        _new_min_x = _pending.get("min_x", 0)
        _new_max_x = _pending.get("max_x", 100)
        _new_min_y = _pending.get("min_y", 0)
        _new_max_y = _pending.get("max_y", 100)
        # Update both the source-of-truth keys AND the widget keys
        st.session_state["coord_min_x"] = _new_min_x
        st.session_state["coord_max_x"] = _new_max_x
        st.session_state["coord_min_y"] = _new_min_y
        st.session_state["coord_max_y"] = _new_max_y
        # Update widget keys directly - this is what the widgets actually read
        st.session_state["cfg_coord_min_x"] = _new_min_x
        st.session_state["cfg_coord_max_x"] = _new_max_x
        st.session_state["cfg_coord_min_y"] = _new_min_y
        st.session_state["cfg_coord_max_y"] = _new_max_y
        # Clear pending state
        st.session_state["_pending_bounds_update"] = False
        st.session_state["_pending_bounds"] = {}

bounds_col1, bounds_col2, bounds_col3, bounds_col4 = st.columns([1, 1, 1, 1], gap="small")

# Initialize session state for coordinate bounds if not already set
# This avoids the Streamlit warning about setting both value and session state
if "cfg_coord_min_x" not in st.session_state:
    st.session_state["cfg_coord_min_x"] = st.session_state.get("coord_min_x", _auto_min_x)
if "cfg_coord_max_x" not in st.session_state:
    st.session_state["cfg_coord_max_x"] = st.session_state.get("coord_max_x", _auto_max_x)
if "cfg_coord_min_y" not in st.session_state:
    st.session_state["cfg_coord_min_y"] = st.session_state.get("coord_min_y", _auto_min_y)
if "cfg_coord_max_y" not in st.session_state:
    st.session_state["cfg_coord_max_y"] = st.session_state.get("coord_max_y", _auto_max_y)

with bounds_col1:
    coord_min_x = st.number_input(
        "Min X",
        step=10.0,
        key="cfg_coord_min_x",
        help="Minimum X coordinate. Generated points cannot have x < this value."
    )
with bounds_col2:
    coord_max_x = st.number_input(
        "Max X",
        step=10.0,
        key="cfg_coord_max_x",
        help="Maximum X coordinate. Generated points cannot have x > this value."
    )
with bounds_col3:
    coord_min_y: float = st.number_input(
        "Min Y",
        step=10.0,
        key="cfg_coord_min_y",
        help="Minimum Y coordinate. Generated points cannot have y < this value."
    )
with bounds_col4:
    coord_max_y: float = st.number_input(
        "Max Y",
        step=10.0,
        key="cfg_coord_max_y",
        help="Maximum Y coordinate. Generated points cannot have y > this value."
    )

# Store bounds in session state for access throughout the app
st.session_state["coord_min_x"] = coord_min_x
st.session_state["coord_max_x"] = coord_max_x
st.session_state["coord_min_y"] = coord_min_y
st.session_state["coord_max_y"] = coord_max_y

# Validate bounds
if coord_min_x >= coord_max_x:
    st.warning("Min X must be less than Max X")
if coord_min_y >= coord_max_y:
    st.warning("Min Y must be less than Max Y")

# ============= Lane Drawing Helper Functions =============
# These functions are needed by _auto_detect_bounds_logic for data_path mode
# They must be defined BEFORE _auto_detect_bounds_logic

def _remove_duplicate_points(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    if points.size == 0:
        return points
    filtered = [points[0]]
    for pt in points[1:]:
        if np.linalg.norm(pt - filtered[-1]) > tolerance:
            filtered.append(pt)  # type: ignore[arg-type]
    return np.array(filtered, dtype=float)  # type: ignore[call-overload]

def _extract_longest_object_path(config_df: pd.DataFrame):
    best_pts = None
    best_score = -np.inf
    for obj_id, obj_df in config_df.groupby("o"):  # type: ignore[attr-defined]
        obj_sorted = obj_df.sort_values("t")  # type: ignore[attr-defined]
        pts = obj_sorted[["x", "y"]].to_numpy(dtype=float)  # type: ignore[assignment]
        pts = _remove_duplicate_points(pts)  # type: ignore[arg-type]
        if pts.shape[0] < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        score = total_length + (10.0 if obj_id == 0 else 0.0)
        if score > best_score:
            best_score = score
            best_pts = pts
    return best_pts

def _calculate_vehicle_speeds(config_df: pd.DataFrame) -> dict[int, float]:
    """
    Calculate average speed for each vehicle in km/h.
    Assumes: x,y in meters, t in seconds or deciseconds.
    Returns dict {obj_id: speed_kmh}
    """
    speeds: dict[int, float] = {}
    for obj_id in config_df['o'].unique():
        obj_df: pd.DataFrame = config_df[config_df['o'] == obj_id].sort_values('t')  # type: ignore[assignment]
        if len(obj_df) < 2:
            speeds[obj_id] = 0.0
            continue
        
        # Calculate distances between consecutive timestamps (in meters)
        positions: np.ndarray = obj_df[['x', 'y']].to_numpy()  # type: ignore[assignment]
        times: np.ndarray = obj_df['t'].to_numpy()  # type: ignore[assignment]
        
        distances: np.ndarray = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        time_diffs: np.ndarray = np.diff(times)
        
        # Speed in m/s
        speeds_ms = distances / time_diffs
        avg_speed_ms = np.mean(speeds_ms) if len(speeds_ms) > 0 else 0.0
        
        # Convert to km/h (multiply by 3.6)
        avg_speed_kmh = avg_speed_ms * 3.6
        speeds[obj_id] = avg_speed_kmh
        
        logger.debug(f"[SPEED] Object {obj_id}: {avg_speed_kmh:.1f} km/h")
    
    return speeds

def _safe_normalize(direction: np.ndarray, fallback: np.ndarray = None) -> np.ndarray:
    """Normalize a direction vector, returning a fallback for zero-length or NaN vectors."""
    if fallback is None:
        fallback = np.array([1.0, 0.0])
    norm = np.linalg.norm(direction)
    if np.isfinite(norm) and norm > 1e-6:
        return direction / norm
    return fallback


def _determine_driving_direction(config_df: pd.DataFrame, obj_id: int = None) -> np.ndarray:
    """
    Determine the main driving direction based on movement from timestamp 0.
    If obj_id is provided, calculate direction for that specific object.
    Returns a unit vector representing the driving direction.
    """
    _default = np.array([1.0, 0.0])

    if obj_id is not None:
        obj_df = config_df[config_df['o'] == obj_id].sort_values('t')
        if len(obj_df) < 2:
            return _default
        
        # Use first and last position to get overall direction
        p0 = obj_df.iloc[0][['x', 'y']].to_numpy()
        p1 = obj_df.iloc[-1][['x', 'y']].to_numpy()
        return _safe_normalize(p1 - p0, _default)
    
    # Get positions at first two timestamps for all objects
    t_values = sorted(config_df['t'].unique())
    if len(t_values) < 2:
        return _default
    
    t0_df = config_df[config_df['t'] == t_values[0]]
    t1_df = config_df[config_df['t'] == t_values[1]]
    
    # Calculate center of mass for both timestamps
    p0 = np.array([t0_df['x'].mean(), t0_df['y'].mean()])
    p1 = np.array([t1_df['x'].mean(), t1_df['y'].mean()])
    
    return _safe_normalize(p1 - p0, _default)

def _vehicles_same_direction(config_df: pd.DataFrame, angle_threshold: float = 45.0) -> bool:
    """
    Determine if vehicles are traveling in roughly the same direction.
    Returns True if angle between vehicle directions is less than angle_threshold degrees.
    """
    object_ids = sorted(config_df['o'].unique())
    
    if len(object_ids) < 2:
        return True  # Single vehicle, consider as "same direction"
    
    # Calculate direction vectors for each vehicle
    directions: list[np.ndarray] = []
    for obj_id in object_ids:
        direction = _determine_driving_direction(config_df, obj_id)
        directions.append(direction)  # type: ignore[arg-type]
        logger.debug(f"[DIRECTION] Object {obj_id}: direction={direction}")
    
    # Compare all pairs of directions
    angle_deg: float = 0.0  # Initialize to avoid 'possibly unbound' error
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            # Calculate angle between directions using dot product
            dot_product = np.dot(directions[i], directions[j])
            # Clamp to [-1, 1] to avoid numerical issues with arccos
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = float(np.degrees(angle_rad))
            logger.debug(f"[DIRECTION] Angle between obj {object_ids[i]} and obj {object_ids[j]}: {angle_deg:.1f}°")
            
            # If any pair has large angle difference, they're not in same direction
            if angle_deg > angle_threshold:
                return False
    
    return True

def _extract_centerline_from_data(c_value: int) -> np.ndarray | None:
    if _df_all is None:
        return None
    config_df = _df_all[_df_all["c"] == c_value]
    if config_df.empty:
        return None

    lane_cfg: dict[str, Any] = LANE_CONFIGURATIONS.get(c_value, {})
    force_horizontal = bool(lane_cfg.get("force_horizontal", False))

    # Calculate vehicle speeds to identify slowest vehicle (should be on right)
    speeds = _calculate_vehicle_speeds(config_df)
    
    # Determine driving direction from timestamp 0
    driving_direction = _determine_driving_direction(config_df)
    
    # Find slowest vehicle (should be on the right lane)
    slowest_vehicle = None
    if speeds:
        slowest_vehicle = min(speeds.items(), key=lambda x: x[1])[0]
        
        # For curved roads, use the slowest vehicle's path directly as the RIGHT lane
        # Then offset to create centerline
        if c_value in [15]:
            slowest_df = config_df[config_df['o'] == slowest_vehicle].sort_values('t')
            right_lane_path = slowest_df[['x', 'y']].to_numpy(dtype=float)
            right_lane_path = _remove_duplicate_points(right_lane_path)
            if right_lane_path.shape[0] >= 2:
                # This is the rightmost lane, we'll offset it in _build_lane_polylines_from_data
                return right_lane_path
    
    # Calculate initial centerline as average between all vehicles at each timestamp
    center_samples: list[tuple[float, float, float]] = []
    for t_val, group in config_df.groupby("t"):  # type: ignore[attr-defined]
        center_samples.append((float(t_val), float(group["x"].mean()), float(group["y"].mean())))

    center_samples.sort(key=lambda item: item[0])

    if center_samples:
        centerline = np.array([[row[1], row[2]] for row in center_samples], dtype=float)
        centerline = _remove_duplicate_points(centerline)
    else:
        centerline = np.empty((0, 2), dtype=float)

    if centerline.shape[0] < 2:
        return _extract_longest_object_path(config_df)
    
    # Now adjust centerline so that the slowest vehicle is on the RIGHT lane
    # "Right" is relative to the driving direction
    if slowest_vehicle is not None and len(config_df['o'].unique()) > 1:
        # Get slowest vehicle's average position
        slowest_df = config_df[config_df['o'] == slowest_vehicle]
        slowest_positions = slowest_df[['x', 'y']].to_numpy(dtype=float)
        slowest_avg = np.mean(slowest_positions, axis=0)
        
        # Get centerline midpoint
        centerline_mid = np.mean(centerline, axis=0)
        
        # Vector from centerline to slowest vehicle
        to_slowest = slowest_avg - centerline_mid
        
        # Calculate perpendicular to driving direction (right side when facing forward)
        # Right = rotate driving direction 90 degrees clockwise
        # In 2D: (x, y) rotated 90° clockwise = (y, -x)
        right_direction = np.array([driving_direction[1], -driving_direction[0]])
        
        # Check if slowest vehicle is on the left or right of centerline
        # Positive dot product means slowest is on the right side
        side = np.dot(to_slowest, right_direction)
        
        if side < 0:
            # Slowest vehicle is on the LEFT, we need to shift centerline LEFT
            # so that slowest ends up on the RIGHT
            # Calculate how much to shift: distance from centerline to slowest vehicle
            shift_distance = np.linalg.norm(to_slowest)
            # Shift centerline in the direction opposite to slowest vehicle
            shift_vector = -to_slowest / np.linalg.norm(to_slowest) * shift_distance
            centerline = centerline + shift_vector

    # Check if horizontal lanes should be forced (for certain configs like 7)
    lane_cfg = LANE_CONFIGURATIONS.get(c_value, {})
    force_horizontal = lane_cfg.get("force_horizontal", False)
    
    if force_horizontal and centerline.shape[0] >= 2:
        # Force completely horizontal lanes.
        # If provided, use explicit configured centerline y-position.
        forced_centerline_y = lane_cfg.get("centerline_y")
        if forced_centerline_y is not None:
            avg_y = float(forced_centerline_y)
        else:
            avg_y = float(np.mean(centerline[:, 1]))
        # Create a horizontal line from min to max x-coordinate at constant y
        x_min = np.min(centerline[:, 0])
        x_max = np.max(centerline[:, 0])
        centerline = np.array([[x_min, avg_y], [x_max, avg_y]])
        logger.debug(f"[CENTERLINE] Config {c_value}: Forced horizontal lanes at y={avg_y:.2f}")
        return centerline

    # Check if the path is roughly straight (skip for curved configs)
    # IMPORTANT: Always preserve the original slope angle from start to end point!
    if c_value not in [15] and centerline.shape[0] >= 3:
        p_start = centerline[0]
        p_end = centerline[-1]
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            # Calculate distance of all points to the line segment
            unit_vec = vec / norm
            vecs = centerline - p_start
            # 2D cross product: x1*y2 - x2*y1 gives signed distance * norm
            cross_products = vecs[:, 0] * unit_vec[1] - vecs[:, 1] * unit_vec[0]
            max_deviation = np.max(np.abs(cross_products))
            
            # If deviation is small (e.g. < 5.0m), simplify to straight line
            # Keep original start/end points to preserve the slope angle!
            if max_deviation < 5.0:
                centerline = np.array([p_start, p_end])
    elif centerline.shape[0] == 2:
        pass # Already straight

    return centerline

def _offset_polyline(points: np.ndarray, offset: float) -> np.ndarray:
    if points.shape[0] < 2:
        return points.copy()

    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]

    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1.0
    normalized = tangents / norms[:, np.newaxis]
    normals = np.column_stack((-normalized[:, 1], normalized[:, 0]))

    return points + offset * normals

def _build_lane_polylines_from_data(c_value: int, lane_width: float, lane_count: int, xlim: Optional[Tuple[float, float]] = None, config_offset: float = 0.0) -> dict[str, Any] | None:
    """
    Build lane polylines for a configuration. Handles two cases:
    1. Vehicles traveling in same direction: creates parallel lanes
    2. Vehicles traveling in different directions: creates separate road segments with merge
    
    Dynamically determines lane count based on max speed (>100 km/h = 3 lanes, else 2 lanes).
    Positions slower vehicle on the rightmost lane.
    """
    if lane_count < 1:
        return None
    
    # Check if we have data and if vehicles are traveling in same direction
    if _df_all is None:
        return None
    
    config_df = _df_all[_df_all["c"] == c_value]
    if config_df.empty:
        return None

    lane_cfg: dict[str, Any] = LANE_CONFIGURATIONS.get(c_value, {})
    force_horizontal = bool(lane_cfg.get("force_horizontal", False))
    
    # Calculate speeds to determine lane count and positioning
    speeds = _calculate_vehicle_speeds(config_df)
    max_speed = max(speeds.values()) if speeds else 0.0
    
    # Check vehicle y-positions to determine if they span multiple lanes
    object_ids = sorted(config_df['o'].unique())
    vehicle_y_positions: dict[int, float] = {}
    for obj_id in object_ids:
        obj_df: pd.DataFrame = config_df[config_df['o'] == obj_id]  # type: ignore[assignment]
        vehicle_y_positions[obj_id] = float(obj_df['y'].mean())  # type: ignore[arg-type]
    
    # Calculate y-span across all vehicles
    if len(vehicle_y_positions) > 0:
        y_values = list(vehicle_y_positions.values())
        y_span = max(y_values) - min(y_values)
    else:
        y_span = 0.0
    
    # Determine lane count based on:
    # 1. Speed: >100 km/h suggests highway (3 lanes)
    # 2. Y-span: if vehicles are separated by more than 1.5 * lane_width, need 3 lanes
    # 3. Default: 2 lanes
    needs_3_lanes: bool = (max_speed > 100.0) or (y_span > 1.5 * lane_width)
    
    if needs_3_lanes:
        lane_count = 3
        logger.debug(f"[LANE BUILD] Config {c_value}: max_speed={max_speed:.1f} km/h, y_span={y_span:.2f}m -> 3 lanes")
    else:
        lane_count = 2
        logger.debug(f"[LANE BUILD] Config {c_value}: max_speed={max_speed:.1f} km/h, y_span={y_span:.2f}m -> 2 lanes")
    
    same_direction = _vehicles_same_direction(config_df)
    logger.debug(f"[LANE BUILD] Config {c_value}: same_direction={same_direction}")
    
    if same_direction:
        # Case 1: Same direction - use traditional parallel lane approach
        centerline: np.ndarray | None = _extract_centerline_from_data(c_value)
        if centerline is None or centerline.shape[0] < 2:
            return None

        # Extend centerline to xlim if provided
        if xlim is not None:
            p1 = centerline[0]
            p2 = centerline[-1]
            
            # For curved lines (>2 points), use LOCAL tangent at endpoints for extension
            if centerline.shape[0] > 2:
                # Tangent at start: direction from point 0 to point 1
                start_tangent = centerline[1] - centerline[0]
                start_norm = np.linalg.norm(start_tangent)
                if start_norm > 1e-6:
                    start_unit = start_tangent / start_norm
                else:
                    start_unit = np.array([1.0, 0.0])
                
                # Tangent at end: direction from second-to-last to last point
                end_tangent = centerline[-1] - centerline[-2]
                end_norm = np.linalg.norm(end_tangent)
                if end_norm > 1e-6:
                    end_unit = end_tangent / end_norm
                else:
                    end_unit = np.array([1.0, 0.0])
                
                # Extend start if needed (using start tangent, going backwards)
                if abs(start_unit[0]) > 1e-6 and xlim[0] < p1[0] - 0.1:
                    t_start = (xlim[0] - p1[0]) / start_unit[0]
                    new_start = p1 + t_start * start_unit
                    centerline = np.vstack([[new_start], centerline])
                
                # Extend end if needed (using end tangent, going forwards)
                if abs(end_unit[0]) > 1e-6 and xlim[1] > p2[0] + 0.1:
                    t_end = (xlim[1] - p2[0]) / end_unit[0]
                    new_end = p2 + t_end * end_unit
                    centerline = np.vstack([centerline, [new_end]])
            else:
                # For straight lines (2 points), use overall direction
                direction = p2 - p1
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    unit_dir = direction / norm
                    if abs(unit_dir[0]) > 1e-6:
                        t_start = (xlim[0] - p1[0]) / unit_dir[0]
                        t_end = (xlim[1] - p1[0]) / unit_dir[0]
                        if t_start > t_end:
                            t_start, t_end = t_end, t_start
                        new_start = p1 + t_start * unit_dir
                        new_end = p1 + t_end * unit_dir
                        centerline = np.array([new_start, new_end])

        # Adjust offset to position vehicles realistically in lanes
        # STRATEGY: The road centerline should be positioned such that vehicles
        # are centered in their assigned lanes. The "centerline" variable represents
        # the reference line for creating parallel lane boundaries.
        #
        # For a road with N lanes:
        # - Boundary offsets: [-half_width, -half_width + lane_width, ..., +half_width]
        # - Lane k center (0-indexed, rightmost): -half_width + (k + 0.5) * lane_width
        #
        # We position the reference centerline so that after applying boundary offsets,
        # vehicles end up centered in their assigned lanes.
        
        half_width = (lane_width * lane_count) / 2.0
        forced_centerline_y = lane_cfg.get("centerline_y")
        lock_centerline = bool(force_horizontal and forced_centerline_y is not None)

        if lock_centerline:
            offset = 0.0
            logger.debug(f"[LANE BUILD] Config {c_value}: centerline locked at y={float(forced_centerline_y):.2f} (auto-offset disabled)")
        elif len(speeds) == 1:
            # Single vehicle: place it in the center lane
            single_obj = list(speeds.keys())[0]
            single_df = config_df[config_df['o'] == single_obj]
            avg_y_vehicle = float(single_df['y'].mean())
            
            # For a 2-lane road, center lane is lane 0 with offset around 0
            # We want the reference line to be positioned such that the vehicle
            # is centered in a lane. Simply position ref line at vehicle position.
            target_ref_y = avg_y_vehicle
            
            current_ref_y = float(np.mean(centerline[:, 1]))
            offset = target_ref_y - current_ref_y
            
            logger.debug(f"[LANE BUILD] Single vehicle obj {single_obj}: vehicle_y={avg_y_vehicle:.2f}, target_ref={target_ref_y:.2f}, offset={offset:.2f}m")
            
        elif len(speeds) > 1:
            # Multiple vehicles: position reference line at average of all vehicles
            avg_vehicle_y = sum(vehicle_y_positions.values()) / len(vehicle_y_positions)
            target_ref_y = avg_vehicle_y
            
            current_ref_y = float(np.mean(centerline[:, 1]))
            offset = target_ref_y - current_ref_y
            logger.debug(f"[LANE BUILD] Multi-vehicle: avg_vehicle_y={avg_vehicle_y:.2f}, target_ref={target_ref_y:.2f}, offset={offset:.2f}m")
        else:
            # No vehicles
            offset = 0.0

        # Apply vertical offset to centerline
        # NOTE: We ignore config_offset parameter as our calculated offset already
        # positions vehicles correctly in their lanes. The config_offset was a legacy
        # manual adjustment that is no longer needed.
        centerline[:, 1] += offset
        logger.debug(f"[LANE BUILD] After offset (calculated={offset:.2f}m, config_offset={config_offset:.2f}m ignored), centerline y-range: [{np.min(centerline[:, 1]):.2f}, {np.max(centerline[:, 1]):.2f}]")

        # half_width already calculated above, reuse it
        boundary_offsets = [-half_width + i * lane_width for i in range(lane_count + 1)]
        boundaries = [_offset_polyline(centerline, off) for off in boundary_offsets]
        logger.debug(f"[LANE BUILD] Created {len(boundaries)} boundaries with offsets: {boundary_offsets}")
        if boundaries:
            for i, boundary in enumerate(boundaries):
                y_range: list[float] = [float(np.min(boundary[:, 1])), float(np.max(boundary[:, 1]))]
                logger.debug(f"[LANE BUILD]   Boundary {i} y-range: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
        if centerline.shape[0] > 0:
            centerline_y_dbg = float(np.mean(centerline[:, 1]))
            logger.debug(f"[LANE BUILD] Lane y-positions -> lower edge: {centerline_y_dbg - half_width:.2f}, dashed divider: {centerline_y_dbg:.2f}, upper edge: {centerline_y_dbg + half_width:.2f}")

        interior_count = max(0, lane_count - 1)
        center_offsets = [-half_width + (i + 1) * lane_width for i in range(interior_count)]
        center_lines = [_offset_polyline(centerline, off) for off in center_offsets]

        return {"boundaries": boundaries, "center_lines": center_lines, "centerline": centerline}
    
    else:
        # Case 2: Different directions - create separate road segments for each vehicle
        # and merge them realistically
        object_ids = sorted(config_df['o'].unique())
        
        # Create road segments for each vehicle
        all_boundaries = []
        
        for obj_id in object_ids:
            obj_df: pd.DataFrame = config_df[config_df['o'] == obj_id].sort_values('t')  # type: ignore[assignment]
            if len(obj_df) < 2:
                continue
            
            # Get vehicle path
            vehicle_path: np.ndarray = obj_df[['x', 'y']].to_numpy(dtype=float)
            vehicle_path = _remove_duplicate_points(vehicle_path)
            
            if vehicle_path.shape[0] < 2:
                continue
            
            # Position vehicle in the rightmost lane of its road
            # Vehicle is currently on the path, we need to shift the road so vehicle is in rightmost lane center
            half_width = (lane_width * lane_count) / 2.0
            
            # Rightmost lane center is at -half_width + lane_width/2 relative to road centerline
            rightmost_lane_center = -half_width + lane_width / 2.0
            
            # The vehicle path is at y-position of vehicle
            # We want road centerline offset so: vehicle_y - centerline_y = rightmost_lane_center
            # Therefore: centerline should be at vehicle_y - rightmost_lane_center
            # Since vehicle_path is the vehicle position, offset = -rightmost_lane_center
            road_offset = -rightmost_lane_center
            
            # Shift vehicle path to become road centerline
            road_centerline = vehicle_path.copy()
            road_centerline[:, 1] += road_offset
            
            # Create lane boundaries relative to this centerline
            boundary_offsets = [-half_width + i * lane_width for i in range(lane_count + 1)]
            obj_boundaries = [_offset_polyline(road_centerline, off) for off in boundary_offsets]
            all_boundaries.extend(obj_boundaries)
            
            logger.debug(f"[LANE BUILD] Case 2: obj {obj_id} - road offset={road_offset:.2f}m to center vehicle in rightmost lane")
        
        # For multi-path scenarios (different directions), don't draw center lines
        # as they would overlap/conflict where paths cross or merge
        # Only draw the road boundaries for each vehicle path
        
        # Use first vehicle path as "centerline" reference for bounds calculation
        first_obj = object_ids[0]
        centerline_df: pd.DataFrame = config_df[config_df['o'] == first_obj].sort_values('t')  # type: ignore[assignment]
        centerline: np.ndarray = centerline_df[['x', 'y']].to_numpy(dtype=float)  # type: ignore[assignment]
        centerline = _remove_duplicate_points(centerline)
        
        return {
            "boundaries": all_boundaries,
            "center_lines": [],  # No center lines for multi-path scenarios
            "centerline": centerline,
            "multi_path": True  # Flag to indicate this is a multi-path scenario
        }

def _lane_polylines_bounds(lane_polylines: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not lane_polylines:
        return None

    arrays: list[np.ndarray] = []
    boundaries = lane_polylines.get("boundaries")  # type: ignore[union-attr]
    if boundaries:
        arrays.extend(boundaries)  # type: ignore[arg-type]

    centerline = lane_polylines.get("centerline")  # type: ignore[union-attr]
    if centerline is not None and centerline.size:
        arrays.append(centerline)  # type: ignore[arg-type]

    if not arrays:
        return None

    stacked = np.vstack(arrays)
    return (
        float(np.min(stacked[:, 0])),
        float(np.max(stacked[:, 0])),
        float(np.min(stacked[:, 1])),
        float(np.max(stacked[:, 1])),
    )

# ============= Auto Detect Coordinate Bounds =============
# This logic recalculates the coordinate bounds based on the currently selected
# configuration (c) and timestamp window. Useful when switching between configurations
# or changing the number of timestamps, as parent points may fall outside the current bounds.

def _auto_detect_bounds_logic() -> bool:
    """
    Calculate and update coordinate bounds based on current configuration and timestamp window.
    Returns True if bounds were successfully updated, False otherwise.
    """
    if _df_all is None:
        return False
    
    _detect_c = int(st.session_state.get("cfg_c", available_configs[0]))
    _detect_k = int(st.session_state.get("cfg_k", 3))  # Number of timestamps
    _detect_start_t = st.session_state.get("cfg_start_t", None)
    
    config_df: pd.DataFrame = _df_all[_df_all["c"] == _detect_c]  # type: ignore[assignment]
    if config_df.empty:
        return False

    object_ids: list[int] = sorted(config_df["o"].unique().tolist())  # type: ignore[assignment]
    time_values_by_object: dict[int, list[float]] = {
        o_id: sorted(config_df[config_df["o"] == o_id]["t"].unique().tolist())
        for o_id in object_ids
    }

    _detect_t_k = time_values_by_object.get(0, [])
    comparison_obj: int | None = next((o_id for o_id in object_ids if o_id != 0), None)
    if comparison_obj is not None:
        _detect_t_l = time_values_by_object.get(comparison_obj, [])
        _detect_t_common = [t for t in _detect_t_k if t in _detect_t_l]
    else:
        _detect_t_l = []
        _detect_t_common = list(_detect_t_k)

    if not _detect_t_common:
        # Fallback to union of timestamps across all objects
        union_times = sorted({t for times in time_values_by_object.values() for t in times})
        _detect_t_common = union_times
    if not _detect_t_common:
        return False
    
    # Determine the timestamp window
    if _detect_start_t is not None and _detect_start_t in _detect_t_common:
        _detect_start_idx = _detect_t_common.index(_detect_start_t)
    else:
        _detect_start_idx = 0
    _detect_end_idx = min(_detect_start_idx + _detect_k, len(_detect_t_common))
    if _detect_end_idx <= _detect_start_idx:
        _detect_end_idx = min(len(_detect_t_common), _detect_start_idx + 1)
    _detect_ts_window = _detect_t_common[_detect_start_idx:_detect_end_idx]
    if not _detect_ts_window:
        _detect_ts_window = _detect_t_common[:max(1, _detect_k)]
    
    # Filter dataframe to selected configuration and timestamp window
    _df_filtered: pd.DataFrame = config_df[config_df["t"].isin(_detect_ts_window)]  # type: ignore[assignment]
    if _df_filtered.empty:  # type: ignore[union-attr]
        _df_filtered = config_df  # type: ignore[assignment]
    
    if len(_df_filtered) > 0:  # type: ignore[arg-type]
        # Calculate bounds from filtered data
        _new_min_x = float(_df_filtered["x"].min())  # type: ignore[arg-type]
        _new_max_x = float(_df_filtered["x"].max())  # type: ignore[arg-type]
        _new_min_y = float(_df_filtered["y"].min())  # type: ignore[arg-type]
        _new_max_y = float(_df_filtered["y"].max())  # type: ignore[arg-type]

        # Include lane bounds if available
        lane_cfg = LANE_CONFIGURATIONS.get(_detect_c)
        if lane_cfg:
            lane_bounds = lane_cfg.get("bounds")
            if lane_bounds:
                x_bounds: Any = lane_bounds.get("x")
                y_bounds: Any = lane_bounds.get("y")
            else:
                x_bounds: Any = None
                y_bounds: Any = None
            if x_bounds:
                _new_min_x = min(_new_min_x, float(x_bounds[0]))  # type: ignore[index]
                _new_max_x = max(_new_max_x, float(x_bounds[1]))  # type: ignore[index]
            if y_bounds:
                _new_min_y = min(_new_min_y, float(y_bounds[0]))  # type: ignore[index]
                _new_max_y = max(_new_max_y, float(y_bounds[1]))  # type: ignore[index]

            if lane_cfg.get("mode", "data_path") == "data_path":
                lane_width = float(lane_cfg.get("lane_width", 3.0))  # type: ignore[arg-type]
                lane_count = int(lane_cfg.get("lanes", 3))  # type: ignore[arg-type]
                lane_polylines = _build_lane_polylines_from_data(_detect_c, lane_width, lane_count)
                poly_bounds = _lane_polylines_bounds(lane_polylines) if lane_polylines else None
                if poly_bounds:
                    _new_min_x = min(_new_min_x, poly_bounds[0])
                    _new_max_x = max(_new_max_x, poly_bounds[1])
                    _new_min_y = min(_new_min_y, poly_bounds[2])
                    _new_max_y = max(_new_max_y, poly_bounds[3])
        
        # Add 10% margin (or minimum margin for very small ranges)
        _new_range_x = _new_max_x - _new_min_x
        _new_range_y = _new_max_y - _new_min_y
        
        # Use 10% margin, but ensure at least some minimum margin for very tight data
        _margin_x = max(0.1 * _new_range_x, 0.5) if _new_range_x > 0 else 1.0
        _margin_y = max(0.1 * _new_range_y, 0.5) if _new_range_y > 0 else 1.0
        
        _new_min_x = _new_min_x - _margin_x
        _new_max_x = _new_max_x + _margin_x
        _new_min_y = _new_min_y - _margin_y
        _new_max_y = _new_max_y + _margin_y
        
        # Smart rounding: choose rounding unit based on data range
        def smart_round_min(val: float, data_range: float) -> float:
            if data_range < 10:
                return float(np.floor(val))
            elif data_range < 100:
                return float(np.floor(val / 5) * 5)
            else:
                return float(np.floor(val / 10) * 10)
        
        def smart_round_max(val: float, data_range: float) -> float:
            if data_range < 10:
                return float(np.ceil(val))
            elif data_range < 100:
                return float(np.ceil(val / 5) * 5)
            else:
                return float(np.ceil(val / 10) * 10)
        
        _new_range_x = _new_max_x - _new_min_x
        _new_range_y = _new_max_y - _new_min_y

        _new_min_x = smart_round_min(_new_min_x, _new_range_x)
        _new_max_x = smart_round_max(_new_max_x, _new_range_x)
        _new_min_y = smart_round_min(_new_min_y, _new_range_y)
        _new_max_y = smart_round_max(_new_max_y, _new_range_y)
        
        # Store pending bounds
        st.session_state["_pending_bounds_update"] = True
        st.session_state["_pending_bounds"] = {
            "min_x": _new_min_x,
            "max_x": _new_max_x,
            "min_y": _new_min_y,
            "max_y": _new_max_y
        }
        
        # Set a special hash to prevent auto-recalculation from overwriting
        st.session_state["_bounds_data_hash"] = f"auto_detect_{_detect_c}_{_new_min_x:.2f}_{_new_max_x:.2f}_{_new_min_y:.2f}_{_new_max_y:.2f}"
        return True
    return False

if data_source != "Create random configuration":
    # Only show Auto Detect button when using preset or uploaded data
    st.markdown('<div class="auto-detect-bounds-wrapper" style="margin-top:0.5rem;">', unsafe_allow_html=True)
    if st.button("Auto Detect Coordinate Bounds", key="btn_auto_detect_bounds", 
                 help="Recalculate axis bounds based on currently selected configuration (c) and timestamp window. Use this when parent points fall outside the visible area after changing settings."):
        # Use the shared auto-detect logic
        _detect_c = int(st.session_state.get("cfg_c", available_configs[0]))
        if _auto_detect_bounds_logic():
            _bounds = st.session_state.get("_pending_bounds", {})
            st.success(f"Bounds updated for config {_detect_c}: X=[{_bounds.get('min_x', 0):.0f}, {_bounds.get('max_x', 100):.0f}], Y=[{_bounds.get('min_y', 0):.0f}, {_bounds.get('max_y', 100):.0f}]")
            st.rerun()
        else:
            st.warning("No data found for the selected configuration and timestamps.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============= Settings Card (UI) ============
st.markdown("""
<div class='settings-card'>
  <h3>Settings</h3>
""", unsafe_allow_html=True)

# Pre-apply cfg_c for C68 presets BEFORE the widget renders
if st.session_state.get("_c68r_preset_pending", False) or st.session_state.get("_c68f_preset_pending", False):
    st.session_state["cfg_c"] = 68

sc1, sc2, sc3 = st.columns([1,1,2], gap="small")
with sc1:
    # Configuration selector for c (only show if multiple configs available)
    if len(available_configs) > 1:
        selected_c = st.selectbox(
            "Configuration (c)",
            options=available_configs,
            index=available_configs.index(7) if (not _is_custom_upload and 7 in available_configs) else (available_configs.index(0) if 0 in available_configs else 0),
            key="cfg_c",
            help="Select which configuration to use as the reference. Each configuration has its own set of k and l trajectories."
        )
    else:
        selected_c = available_configs[0]
        st.markdown(f"**Configuration:** {selected_c}")
selected_c_int: int = int(selected_c) if selected_c is not None else int(available_configs[0])

# Gather data for the selected configuration
config_df = _df_all[_df_all["c"] == selected_c_int]
all_object_ids: list[int] = sorted(config_df["o"].unique().tolist())  # type: ignore[attr-defined]

# Time values per object (supports single-object configurations)
time_values_by_object: dict[int, list[float]] = {
    int(o_id): sorted(config_df[config_df["o"] == o_id]["t"].unique().tolist())  # type: ignore[attr-defined]
    for o_id in all_object_ids
}

_t_k = time_values_by_object.get(0, [])
comparison_obj: int | None = next((o_id for o_id in all_object_ids if o_id != 0), None)
if comparison_obj is not None:
    _t_l = time_values_by_object.get(comparison_obj, [])
    _t_common = [t for t in _t_k if t in _t_l]
    if not _t_common:
        st.error(f"No overlapping t-values for c={selected_c} between o=0 and o={comparison_obj}.")
        st.stop()
else:
    _t_l: list[float] = []
    _t_common = list(_t_k)

if not _t_common:
    # fallback to union of all timestamps if intersection is empty (e.g., single-object data)
    _t_common = sorted({t for times in time_values_by_object.values() for t in times})
if not _t_common:
    st.error(f"No timestamps found for configuration c={selected_c}.")
    st.stop()

    n_timepoints = len(_t_common)
    halved_timepoints = max(1, n_timepoints // 2)
    default_window = n_timepoints if _is_custom_upload else min(halved_timepoints, n_timepoints)

# Apply pending cfg_k override (ext30/ext30_fe set full timestamps)
if "_pending_cfg_k" in st.session_state:
    st.session_state["cfg_k"] = st.session_state.pop("_pending_cfg_k")

# Apply reinsertion preset if pending (must happen BEFORE widgets render)
if st.session_state.pop("_reinsertion_preset_pending", False):
    st.session_state["cfg_start_t"] = 131
    st.session_state["cfg_k"] = 8

# Apply ext30-half preset: just set timestamp step to 2 (keep all other settings)
if st.session_state.pop("_ext30_half_preset_pending", False):
    st.session_state["_cfg_timestamp_step"] = 2

# Apply Config 68 Realistic preset if pending
if st.session_state.pop("_c68r_preset_pending", False):
    st.session_state["cfg_start_t"] = 82
    # timestamps 82..160 → 79 entries; with step 2 we get 40 timestamps
    st.session_state["cfg_k"] = min(79, n_timepoints)
    st.session_state["_cfg_timestamp_step"] = 2
    st.session_state["cfg_pdp_variants"] = ["realistic"]
    st.session_state["cfg_buffer_x"] = 1.5
    st.session_state["cfg_buffer_y"] = 0.0
    st.session_state["cfg_rough_x"] = 0.0
    st.session_state["cfg_rough_y"] = 0.40
    st.session_state["cfg_point_selection_mode"] = "Multiple random points"
    st.session_state["cfg_movement_direction"] = "Same direction"
    st.session_state["cfg_use_external_points"] = True
    st.session_state["use_external_points"] = True
    # Compute external points at lane centers for config 68
    _c68_cfg = LANE_CONFIGURATIONS.get(68, {})
    _c68_lw = float(_c68_cfg.get("lane_width", 3.0))
    _c68_nlanes = int(_c68_cfg.get("lanes", 2))
    _c68_cl_y = float(_c68_cfg.get("centerline_y", -5.0))
    _c68_hw = (_c68_lw * _c68_nlanes) / 2.0
    st.session_state["external_points"] = [
        (500.0, round(_c68_cl_y - _c68_hw + (i + 0.5) * _c68_lw, 3))
        for i in range(_c68_nlanes)
    ]

# Apply Config 68 Fundamental preset if pending
if st.session_state.pop("_c68f_preset_pending", False):
    # Ensure n_timepoints is defined
    try:
        n_timepoints
    except NameError:
        n_timepoints = len(_t_common) if '_t_common' in locals() else 137
    st.session_state["cfg_start_t"] = 82
    st.session_state["cfg_k"] = min(79, n_timepoints)
    st.session_state["_cfg_timestamp_step"] = 2
    st.session_state["cfg_pdp_variants"] = ["fundamental"]
    st.session_state["cfg_buffer_x"] = 0.0
    st.session_state["cfg_buffer_y"] = 0.0
    st.session_state["cfg_rough_x"] = 0.0
    st.session_state["cfg_rough_y"] = 0.0
    st.session_state["cfg_point_selection_mode"] = "Single point"
    st.session_state["cfg_movement_direction"] = "Same direction"
    st.session_state["cfg_use_external_points"] = True
    st.session_state["use_external_points"] = True
    _c68_cfg = LANE_CONFIGURATIONS.get(68, {})
    _c68_lw = float(_c68_cfg.get("lane_width", 3.0))
    _c68_nlanes = int(_c68_cfg.get("lanes", 2))
    _c68_cl_y = float(_c68_cfg.get("centerline_y", -5.0))
    _c68_hw = (_c68_lw * _c68_nlanes) / 2.0
    st.session_state["external_points"] = [
        (500.0, round(_c68_cl_y - _c68_hw + (i + 0.5) * _c68_lw, 3))
        for i in range(_c68_nlanes)
    ]

with sc2:
    # Number of timestamps in the sliding time window (dropdown instead of slider)
    # Ensure n_timepoints and default_window are always defined
    try:
        n_timepoints
    except NameError:
        n_timepoints = len(_t_common) if '_t_common' in locals() else 137
    try:
        default_window
    except NameError:
        halved_timepoints = max(1, n_timepoints // 2)
        default_window = n_timepoints
    if n_timepoints > 1:
        # Allow selecting up to all timestamps for any data source
        max_selectable = n_timepoints
        timestamp_options = list(range(2, max_selectable + 1))
        default_idx = timestamp_options.index(default_window) if default_window in timestamp_options else len(timestamp_options) - 1
        num_timestamps = st.selectbox(
            f"Number of timestamps (max: {max_selectable})",
            options=timestamp_options,
            index=default_idx,
            key="cfg_k",
            help=f"Select the number of timestamps to include in the analysis window. [Max: {max_selectable}]"
        )
    else:
        st.markdown("**Number of timestamps**")
        st.code(str(n_timepoints))
        num_timestamps = n_timepoints


with sc3:
    # Starting t value of the window (dropdown instead of slider)
    valid_start_count = max(1, n_timepoints - num_timestamps + 1)
    valid_starts = _t_common[:valid_start_count]
    # Utility for halved timestamp titles
    def _halved_ts_title(base, count):
        return f"{base} ({count} timestamps, halved)"
    if len(valid_starts) > 1:
        # Default to start timestamp 38 if available, otherwise use first
        default_start_idx = valid_starts.index(38) if 38 in valid_starts else 0
        start_t = st.selectbox(
            "Starting time (t)",
            options=valid_starts,
            index=default_start_idx,
            key="cfg_start_t",
            help="Select the starting timestamp for the analysis window."
        )
    else:
        st.markdown("**Starting time (t)**")
        st.code(str(valid_starts[0]))
        start_t = valid_starts[0]

# Fallback: ensure start_t is always defined for later use
try:
    start_t
except NameError:
    if 'valid_starts' in locals() and len(valid_starts) > 0:
        start_t = valid_starts[0]
    elif '_t_common' in locals() and len(_t_common) > 0:
        start_t = _t_common[0]
    else:
        start_t = 0

# Strategy, iterations, configurations - on one row with 4 columns for compactness
sc4, sc5, sc6, sc7 = st.columns([1.2, 1, 1, 1], gap="small")
with sc4:
    # Choice of search strategy for generating new configurations
    strategy = st.radio(
        "Strategy",
        options=["exponential", "linear", "binary"],
        index=0,
        key="cfg_strategy",
        horizontal=True,
        help="Choose the search strategy for configuration generation.\n- exponential: halve distance until match\n- linear: decrease by 10% of maxdist per step\n- binary: 7-step binary search"
    )
with sc5:
    # Number of iterations per configuration (used by both animate and generate)
    num_iterations = st.number_input(
        "Iterations",
        min_value=1,
        max_value=1200,
        value=15,
        step=1,
        key="cfg_iterations",
        help="Number of points to generate per configuration. Each iteration replaces one original point with a new generated point that preserves the PDP pattern."
    )
with sc6:
    # How many configurations to generate (used by both animate and generate)
    num_configs = st.number_input(
        "Configurations",
        min_value=1,
        max_value=1000,
        value=10,
        step=1,
        key="cfg_num_configs",
        help="How many independent configurations to create when clicking 'Generate configurations'. Each configuration is a complete new set of generated points (all k and l points) that preserves the original PDP pattern. Use this for batch generation without animation."
    )
with sc7:
    # Match threshold selection - two modes: percentage or absolute
    threshold_mode = st.radio(
        "Match threshold",
        options=["Percentage", "Max mismatches"],
        index=0,
        horizontal=True,
        key="cfg_threshold_mode",
        help="Percentage: require X% match. Max mismatches: allow up to N differing cells."
    )
    
    if threshold_mode == "Percentage":
        threshold_pct = st.slider(
            "Match %",
            min_value=25,
            max_value=100,
            value=100,
            step=5,
            key="cfg_threshold_pct",
            help="Minimum percentage of cells that must match (100% = strict, lower = permissive)"
        )
    else:
        threshold_abs = st.number_input(
            "Max mismatches",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key="cfg_threshold_abs",
            help="Maximum number of differing cells allowed between original and generated heatmaps (0 = strict)"
        )

# Advanced Settings in collapsible expander
with st.expander("Advanced Point Selection", expanded=False):
    st.markdown("**Point Selection (per iteration)**")
    
    ps_col1, ps_col2 = st.columns([1, 1], gap="small")
    with ps_col1:
        point_selection_mode = st.selectbox(
            "Selection mode",
            options=["Single point", "Multiple random points", "Consecutive time stamps"],
            index=0,
            key="cfg_point_selection_mode",
            help="""How to select points to move in each iteration:
            
• **Single point**: Move 1 random point per iteration (default, current behavior)

• **Multiple random points**: Move N randomly selected points together

• **Consecutive time stamps**: Move consecutive timestamps of a single object. Select which object (k or l) and the starting timestamp, then T consecutive timestamps are moved together."""
        )
    
    with ps_col2:
        movement_direction = st.selectbox(
            "Movement direction",
            options=["Same direction", "Random directions"],
            index=0,
            key="cfg_movement_direction",
            help="""How selected points move together:
            
• **Same direction**: All points move with the same angle and distance (coherent movement)

• **Random directions**: Each point gets its own random angle and distance (independent movement)"""
        )
    
    # Damping factor settings
    st.markdown("**Damping Factor (per iteration)**")
    damp_col1, damp_col2, damp_col3 = st.columns([1, 1, 1], gap="small")
    with damp_col1:
        use_damping = st.checkbox(
            "Enable damping",
            value=False,
            key="cfg_use_damping",
            help="When enabled, the final distance from parent to child point is multiplied by a random damping factor, making the child point closer to the parent."
        )
    with damp_col2:
        damping_min = st.number_input(
            "Min damping",
            min_value=0.00,
            max_value=1.00,
            value=0.00,
            step=0.05,
            format="%.2f",
            key="cfg_damping_min",
            disabled=not use_damping,
            help="Minimum value for the random damping factor (0.00 = point at parent, 1.00 = no damping)"
        )
    with damp_col3:
        damping_max = st.number_input(
            "Max damping",
            min_value=0.00,
            max_value=1.00,
            value=1.00,
            step=0.05,
            format="%.2f",
            key="cfg_damping_max",
            disabled=not use_damping,
            help="Maximum value for the random damping factor (0.00 = point at parent, 1.00 = no damping)"
        )
    
    # Show additional inputs based on selection mode
    if point_selection_mode == "Multiple random points":
        num_random_points = st.number_input(
            "Number of points to move together",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            key="cfg_num_random_points",
            help="How many random points to select and move together in each iteration."
        )
    elif point_selection_mode == "Consecutive time stamps":
        # Get available objects from data (use config_df which is already available)
        available_objects: list[int] = sorted(config_df["o"].unique().tolist())  # type: ignore[attr-defined]
        if not available_objects:
            available_objects = [0, 1]  # Default: assume k and l
        object_labels = [OBJECT_LABELS[i] if i < len(OBJECT_LABELS) else f"obj_{i}" for i in available_objects]
        
        # Get max timestamps for selected object
        def get_max_timestamps_for_object(obj_id: int) -> int:
            obj_data = config_df[config_df["o"] == obj_id]  # type: ignore[attr-defined]
            if len(obj_data) == 0:  # type: ignore[arg-type]
                return 3  # Default assumption
            return int(obj_data["t"].nunique())  # type: ignore[attr-defined]
        
        gp_col1, gp_col2, gp_col3 = st.columns([1, 1, 1], gap="small")
        with gp_col1:
            selected_object_label = st.selectbox(
                "Object",
                options=object_labels,
                index=0,
                key="cfg_consecutive_object",
                help="Select which object's points to move (k, l, etc.)"
            )
            # Convert label back to object id
            selected_object_idx = object_labels.index(selected_object_label) if selected_object_label in object_labels else 0
            selected_object_id = available_objects[selected_object_idx] if selected_object_idx < len(available_objects) else 0
            st.session_state["cfg_consecutive_object_id"] = selected_object_id
        
        with gp_col2:
            max_ts = get_max_timestamps_for_object(selected_object_id)
            group_num_timestamps = st.number_input(
                "Consecutive timestamps (t)",
                min_value=1,
                max_value=max(1, max_ts),
                value=min(2, max_ts),
                step=1,
                key="cfg_group_num_timestamps",
                help="Number of consecutive timestamps to move together."
            )
        
        with gp_col3:
            # First timestamp selection (0-indexed, max depends on num_timestamps)
            max_first_ts = max(0, max_ts - int(group_num_timestamps))
            first_timestamp = st.number_input(
                "First timestamp",
                min_value=0,
                max_value=max_first_ts,
                value=0,
                step=1,
                key="cfg_consecutive_first_timestamp",
                help=f"Starting timestamp index (0 to {max_first_ts}). The next {group_num_timestamps} consecutive timestamps will be selected."
            )

# PDP Variant Selection (Multiple variants) - in expander for compactness
with st.expander("PDP Variant Configuration", expanded=False):
    # Multi-select for PDP variants
    pdp_variants_selected = st.multiselect(
        "PDP Variants to calculate",
        options=["fundamental", "buffer", "rough", "bufferrough", "realistic", "frenet"],
        default=["bufferrough"],
        key="cfg_pdp_variants",
        help="""Select PDP variants for configuration generation:

• **fundamental**: Basic PDP with N×N inequality matrix. Two configurations match if ALL pairwise orderings are identical.

• **buffer**: Expands each point to 5 variants (±buffer in x and y directions). Creates 5N×5N matrix. More restrictive - requires all buffer variants to match.

• **rough**: Adds equality tolerance. Points within roughness distance are considered EQUAL (matrix value 1). More permissive - allows small variations.

• **bufferrough**: Combines buffer expansion AND roughness tolerance. 5N×5N matrix with fuzzy equality.

• **realistic**: Designed for traffic scenarios. Uses buffer ONLY on d1 (x-axis, driving direction) and roughness ONLY on d2 (y-axis, lateral position).

• **frenet** (NEW): Uses road-relative coordinates for curved roads. Instead of global x/y, uses:
  - **s**: distance along road centerline (longitudinal)
  - **n**: perpendicular distance from centerline (lateral)
  This ensures PDP orderings are computed relative to the driving direction, essential for curved roads where x/y axes don't align with traffic flow."""
    )
    
    # Show parameter inputs if any variant needs them
    # "realistic" uses buffer_x AND rough_y (but not buffer_y or rough_x)
    # "frenet" uses buffer_s (along road) and rough_n (lateral) with centerline extraction
    needs_buffer = any(v in ["buffer", "bufferrough", "realistic", "frenet"] for v in pdp_variants_selected)
    needs_rough = any(v in ["rough", "bufferrough", "realistic", "frenet"] for v in pdp_variants_selected)
    needs_realistic = "realistic" in pdp_variants_selected
    needs_frenet = "frenet" in pdp_variants_selected
    
    # Show info about Frenet coordinate system
    if needs_frenet:
        st.info("🛣️ **Frenet mode**: Using road-relative coordinates (s, n) instead of (x, y). "
                "The centerline is automatically extracted from vehicle trajectories. "
                "Buffer/roughness parameters apply to road-relative dimensions.")
    
    if needs_buffer or needs_rough:
        st.markdown("**Parameters for selected variants:**")
        param_col1, param_col2 = st.columns([1, 1], gap="small")
        
        with param_col1:
            if needs_buffer:
                # Show buffer_x for all buffer variants including realistic
                buffer_x_default = 1.5 if needs_realistic else 25.0
                buffer_x = st.number_input(
                    "Buffer X (d1)",
                    min_value=0.0,
                    max_value=100.0,
                    value=buffer_x_default,
                    step=0.5,
                    key="cfg_buffer_x",
                    help="Buffer distance in x-direction (d1, driving direction). Used by: buffer, bufferrough, realistic. For 'realistic' this allows variation in longitudinal position along the road. Recommended: 1–2 m for realistic alternatives."
                )
                # Only show buffer_y if NOT exclusively using realistic
                needs_buffer_y = any(v in ["buffer", "bufferrough"] for v in pdp_variants_selected)
                if needs_buffer_y:
                    buffer_y = st.number_input(
                        "Buffer Y (d2)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        key="cfg_buffer_y",
                        help="Buffer distance in y-direction (d2, lateral position). Used by: buffer, bufferrough. NOT used by 'realistic' (which uses roughness on y instead)."
                    )
                else:
                    buffer_y = 0.0
                    if needs_realistic:
                        st.info("ℹï¸ 'realistic' uses roughness on y instead of buffer")
            else:
                buffer_x = 0.0
                buffer_y = 0.0
        
        with param_col2:
            if needs_rough:
                # Only show rough_x if NOT exclusively using realistic
                needs_rough_x = any(v in ["rough", "bufferrough"] for v in pdp_variants_selected)
                if needs_rough_x:
                    rough_x = st.number_input(
                        "Roughness X (d1)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1,
                        key="cfg_rough_x",
                        help="Equality tolerance in x-direction (d1). Used by: rough, bufferrough. NOT used by 'realistic' (which uses buffer on x instead)."
                    )
                else:
                    rough_x = 0.0
                    if needs_realistic:
                        st.info("ℹï¸ 'realistic' uses buffer on x instead of roughness")
                
                # Show rough_y for all rough variants including realistic
                rough_y_default = 0.4 if needs_realistic else 0.8
                rough_y = st.number_input(
                    "Roughness Y (d2)",
                    min_value=0.0,
                    max_value=100.0,
                    value=rough_y_default,
                    step=0.05,
                    key="cfg_rough_y",
                    help="Equality tolerance in y-direction (d2, lateral position). Used by: rough, bufferrough, realistic. For 'realistic' this defines the lane tolerance - positions within the same lane are considered equivalent. Recommended: 0.3–0.5 m for realistic alternatives."
                )
            else:
                rough_x = 0.0
                rough_y = 0.0
    else:
        buffer_x = 0.0
        buffer_y = 0.0
        rough_x = 0.0
        rough_y = 0.0

# External (fixed) reference points - in expander for compactness
with st.expander("External Reference Points", expanded=False):
    # For custom uploads, force external points off by default
    _ext_default = False if _is_custom_upload else st.session_state.get("use_external_points", True)
    use_external_points = st.checkbox(
        "Use external reference points",
        value=_ext_default,
        key="cfg_use_external_points",
        help="Enable fixed external reference points that constrain absolute positions. "
             "These points (e.g., corners of a tennis court, field boundaries, landmarks) "
             "are included in the PDP inequality matrix but do NOT move during configuration generation. "
             "This anchors the generated configurations to real-world positions."
    )
    
    # Store in session state for access later
    st.session_state["use_external_points"] = use_external_points
    
    if use_external_points:
        st.markdown("Define fixed reference points (these remain stationary during generation):")
        st.caption("Each row is a fixed point with coordinates (x, y). These points apply to all timestamps and constrain the absolute positions of generated configurations.")
        
        # Initialize external points if not present: place one point per lane center at x=0
        if "external_points" not in st.session_state:
            _ext_lane_cfg = LANE_CONFIGURATIONS.get(selected_c_int, {})
            _ext_lw = float(_ext_lane_cfg.get("lane_width", 3.0))
            # Determine lane count dynamically (same logic as _build_lane_polylines_from_data)
            _ext_ys = [float(config_df[config_df['o'] == o]['y'].mean()) for o in all_object_ids]
            _ext_y_span = (max(_ext_ys) - min(_ext_ys)) if len(_ext_ys) > 1 else 0.0
            _ext_nlanes = 3 if _ext_y_span > 1.5 * _ext_lw else 2
            # Road centerline y = average of all vehicle y-positions
            _ext_center_y = float(config_df['y'].mean())
            _ext_hw = (_ext_lw * _ext_nlanes) / 2.0
            # Lane center i (0=rightmost): center_y - hw + (i + 0.5) * lw
            st.session_state["external_points"] = [
                (500.0, round(_ext_center_y - _ext_hw + (i + 0.5) * _ext_lw, 3))
                for i in range(_ext_nlanes)
            ]
        
        external_pts = st.session_state["external_points"]
        
        # Create dataframe for editing (only x and y)
        ext_df = pd.DataFrame({
            "x": [p[0] for p in external_pts],
            "y": [p[1] for p in external_pts]
        })
        
        edited_ext_df = st.data_editor(
            ext_df,
            key="edit_external_points",
            num_rows="dynamic",
            width="content",
            column_config={
                "x": st.column_config.NumberColumn("x", format=f"%.{COORD_DISPLAY_PRECISION}f", width="small"),
                "y": st.column_config.NumberColumn("y", format=f"%.{COORD_DISPLAY_PRECISION}f", width="small"),
            },
        )
        
        # Update session state with edited values (store as x, y tuples)
        if edited_ext_df is not None and len(edited_ext_df) > 0:
            st.session_state["external_points"] = [
                (float(row["x"]), float(row["y"])) 
                for _, row in edited_ext_df.iterrows()
            ]
        else:
            st.session_state["external_points"] = []

# Animation settings - compact layout
st.markdown("**Animation Mode**")
anim_col1, anim_col2 = st.columns([3, 1], gap="small")
with anim_col1:
    anim_mode = st.radio(
        "Animation mode",
        options=["Auto-advance", "Manual step-by-step", "Manual iteration-by-iteration", "Manual config-by-config"],
        index=0,
        horizontal=True,
        key="cfg_anim_mode",
        label_visibility="collapsed",
        help=("Choose how the animation advances:\n"
              "• **Auto-advance**: Automatically moves to the next step after a set time interval.\n"
              "• **Manual step-by-step**: Click to advance each search step manually.\n"
              "• **Manual iteration-by-iteration**: Click to complete one full iteration (all search steps until point is placed).\n"
              "• **Manual config-by-config**: Click to complete one full configuration (all iterations).")
    )
with anim_col2:
    if anim_mode == "Auto-advance":
        wait_interval_ms = st.selectbox(
            "Wait (ms)",
            options=[100, 200, 500, 1000, 2000, 5000],
            index=4,  # 2000 ms als default
            key="cfg_wait_ms",
            help="Time in milliseconds between each animation step. Lower values = faster animation."
        )
    else:
        wait_interval_ms = None  # Manual mode - no auto interval

# Custom CSS for Reset button styling: white text on black background
# This provides a visually distinct button that stands out as a "stop/reset" action
# Uses a class-based approach with a wrapper div for reliable targeting
# Also includes CSS for red Auto Detect Coordinate Bounds button
st.markdown("""
<style>
    /* Style Reset buttons using wrapper div class */
    .reset-button-wrapper button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #000000 !important;
    }
    .reset-button-wrapper button:hover:not(:disabled) {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    .reset-button-wrapper button:disabled {
        background-color: #666666 !important;
        color: #cccccc !important;
        border: 1px solid #666666 !important;
        opacity: 0.6 !important;
    }
    /* Target the button's inner paragraph element for text color */
    .reset-button-wrapper button p {
        color: #ffffff !important;
    }
    .reset-button-wrapper button:disabled p {
        color: #cccccc !important;
    }
    /* Style Auto Detect Coordinate Bounds button: red background with white text */
    /* This button recalculates axis bounds based on currently selected configuration and timestamps */
    .auto-detect-bounds-wrapper button {
        background-color: #dc3545 !important;
        color: #ffffff !important;
        border: 1px solid #dc3545 !important;
    }
    .auto-detect-bounds-wrapper button:hover:not(:disabled) {
        background-color: #c82333 !important;
        color: #ffffff !important;
        border: 1px solid #c82333 !important;
    }
    .auto-detect-bounds-wrapper button p {
        color: #ffffff !important;
    }
    /* Style Generate 5 button: solid red background */
    .element-container:has(.generate-5000-marker) + div[data-testid="stButton"] > button,
    .element-container:has(.generate-5000-marker) + div[data-testid="stButton"] button,
    .element-container:has(.generate-5000-marker) ~ div[data-testid="stButton"] > button[kind="primary"],
    .element-container:has(.generate-5000-marker) ~ div[data-testid="stButton"] button[kind="primary"],
    .generate-5000-marker ~ div[data-testid="stButton"] > button[kind="primary"],
    .generate-5000-marker ~ div[data-testid="stButton"] button[kind="primary"] {
        background-color: #dc2626 !important;
        color: #ffffff !important;
        border: 1px solid #dc2626 !important;
    }
    .element-container:has(.generate-5000-marker) + div[data-testid="stButton"] > button:hover:not(:disabled),
    .element-container:has(.generate-5000-marker) + div[data-testid="stButton"] button:hover:not(:disabled),
    .element-container:has(.generate-5000-marker) ~ div[data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled),
    .element-container:has(.generate-5000-marker) ~ div[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled),
    .generate-5000-marker ~ div[data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled),
    .generate-5000-marker ~ div[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled) {
        background-color: #b91c1c !important;
        color: #ffffff !important;
        border: 1px solid #b91c1c !important;
    }
    .element-container:has(.generate-5000-marker) + div[data-testid="stButton"] > button p,
    .element-container:has(.generate-5000-marker) + div[data-testid="stButton"] button p,
    .element-container:has(.generate-5000-marker) ~ div[data-testid="stButton"] > button[kind="primary"] p,
    .element-container:has(.generate-5000-marker) ~ div[data-testid="stButton"] button[kind="primary"] p,
    .generate-5000-marker ~ div[data-testid="stButton"] > button[kind="primary"] p,
    .generate-5000-marker ~ div[data-testid="stButton"] button[kind="primary"] p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Action buttons
st.markdown("<div style='display:flex;gap:1.2rem;margin-top:0.7rem;'>", unsafe_allow_html=True)

# Determine if we're in any manual mode
is_manual_step_mode = (anim_mode == "Manual step-by-step")
is_manual_iteration_mode = (anim_mode == "Manual iteration-by-iteration")
is_manual_config_mode = (anim_mode == "Manual config-by-config")
is_any_manual_mode = is_manual_step_mode or is_manual_iteration_mode or is_manual_config_mode

anim_is_running = st.session_state.get("anim_running", False)
# Check if there are generated configurations or successful points that can be cleared
has_generated_configs = len(st.session_state.get("anim_all_configs", [])) > 0
has_successful_points = len(st.session_state.get("anim_successful_points", [])) > 0
has_generated_point = st.session_state.get("anim_generated_point") is not None
# Reset button should be enabled if animation is running OR if there are any generated points/configs to clear
reset_btn_should_be_enabled = anim_is_running or has_generated_configs or has_successful_points or has_generated_point

if is_any_manual_mode:
    # Manual mode: show 4 buttons (Generate, Previous, Next, Reset)
    # Button labels change based on the manual mode type
    
    # Check if there's a redo stack (for dynamic labels)
    has_redo_for_labels = len(st.session_state.get("anim_state_redo", [])) > 0
    
    # Determine button labels based on mode
    # Button always shows "Complete iteration/config" - the action either completes new work or redoes previous
    if is_manual_step_mode:
        prev_label = "Previous step"
        next_label = "Next step"
        prev_help = "Click to go back to the previous animation step."
        next_help = "Click to redo the next step." if has_redo_for_labels else "Click to advance the animation by one step."
        generate_help = "Start generating configurations step-by-step. Click 'Next step' to advance each step manually."
    elif is_manual_iteration_mode:
        prev_label = "Previous"
        next_label = "▶ Complete iteration"
        prev_help = "Click to go back to the previous iteration state."
        next_help = "Click to restore the next iteration." if has_redo_for_labels else "Click to complete the current iteration (finish all search steps and place the point)."
        generate_help = "Start generating configurations. Click 'Complete iteration' to finish each iteration."
    else:  # is_manual_config_mode
        prev_label = "Previous"
        next_label = "▶ Complete config"
        prev_help = "Click to go back to the previous configuration state."
        next_help = "Click to restore the next configuration." if has_redo_for_labels else "Click to complete the current configuration (finish all remaining iterations)."
        generate_help = "Start generating configurations. Click 'Complete config' to finish each configuration."
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 0.6], gap="small")
    # No "Generate without animation" button in manual mode
    generate_btn = False  # Set to False so the generate_btn handler doesn't trigger
    
    # Check if animation is already running (has history or is actively running)
    # Generate button should be disabled when navigating through previous states
    anim_history = st.session_state.get("anim_state_history", [])
    anim_redo_stack = st.session_state.get("anim_state_redo", [])
    has_history = len(anim_history) > 0
    has_redo = len(anim_redo_stack) > 0
    # Disable Generate when animation is running or when there's history to navigate
    generate_disabled = anim_is_running or has_history
    
    with col_btn1:
        animate_btn = st.button(
            "Generate", 
            key="btn_animate",
            disabled=generate_disabled,
            help=generate_help if not generate_disabled else "Reset first to start a new generation."
        )
    with col_btn2:
        # Show "Previous" button - enabled when there is history to go back to
        # In manual iteration/config mode, allow going back even when waiting for user input
        # Enable Previous button whenever there is history - don't require animation to be running
        # This allows users to go back after an iteration completes in manual mode
        prev_enabled = has_history
        prev_step_clicked = st.button(
            prev_label, 
            key="btn_prev_step", 
            disabled=not prev_enabled,
            help=prev_help
        )
        if prev_step_clicked and has_history:
            # Save current state to redo stack before going back
            import copy
            current_state_for_redo: dict[str, Any] = {}
            anim_state_keys = [
                "anim_generated_point", "anim_parent_idx", "anim_successful_points",
                "anim_distance", "anim_angle", "anim_search_steps", "anim_completed_iterations",
                "anim_current_config", "anim_in_search", "anim_binary_mode", "anim_binary_step",
                "anim_ok_point", "anim_delta", "anim_had_full_match", "anim_linear_mode",
                "anim_linear_step", "anim_linear_current_distance", "anim_linear_maxdist",
                "anim_linear_step_size", "anim_all_pts", "anim_all_ts", "diag_rows",
                "binary_iteration_summary", "anim_circle_idx", "show_anim_circle",
                "anim_selected_indices", "anim_generated_points", "anim_movement_vectors",
                "anim_pdp_variants_list", "anim_current_variant_idx", "anim_current_variant",
                "anim_running"
            ]
            for key in anim_state_keys:
                if key in st.session_state:
                    value: Any = st.session_state[key]
                    if isinstance(value, np.ndarray):
                        current_state_for_redo[key] = value.copy()
                    elif isinstance(value, (list, dict)):
                        current_state_for_redo[key] = copy.deepcopy(value)  # type: ignore[arg-type]
                    else:
                        current_state_for_redo[key] = value
            anim_redo_stack.append(current_state_for_redo)
            st.session_state["anim_state_redo"] = anim_redo_stack
            
            # Pop the last state from history and restore it
            previous_state = anim_history.pop()
            st.session_state["anim_state_history"] = anim_history
            # Restore all animation state variables from the previous state
            for key, value in previous_state.items():
                st.session_state[key] = value
            st.rerun()
    with col_btn3:
        # Show "Next/Complete" button
        # In iteration/config mode: enabled when animation is running (more iterations to complete)
        # OR when there's a redo stack (can go forward through previously completed iterations)
        # The button completes the current iteration/config, so it should be enabled while there's work to do
        can_go_forward = has_redo or anim_is_running
        next_step_clicked = st.button(
            next_label, 
            key="btn_next_step", 
            type="primary", 
            disabled=not can_go_forward,
            help=next_help
        )
        if next_step_clicked:
            if has_redo:
                # Redo: restore the next state from redo stack
                # First save current state to history
                import copy
                current_state_for_history: dict[str, Any] = {}
                anim_state_keys = [
                    "anim_generated_point", "anim_parent_idx", "anim_successful_points",
                    "anim_distance", "anim_angle", "anim_search_steps", "anim_completed_iterations",
                    "anim_current_config", "anim_in_search", "anim_binary_mode", "anim_binary_step",
                    "anim_ok_point", "anim_delta", "anim_had_full_match", "anim_linear_mode",
                    "anim_linear_step", "anim_linear_current_distance", "anim_linear_maxdist",
                    "anim_linear_step_size", "anim_all_pts", "anim_all_ts", "diag_rows",
                    "binary_iteration_summary", "anim_circle_idx", "show_anim_circle",
                    "anim_selected_indices", "anim_generated_points", "anim_movement_vectors",
                    "anim_pdp_variants_list", "anim_current_variant_idx", "anim_current_variant",
                    "anim_running"
                ]
                for key in anim_state_keys:
                    if key in st.session_state:
                        value: Any = st.session_state[key]
                        if isinstance(value, np.ndarray):
                            current_state_for_history[key] = value.copy()
                        elif isinstance(value, (list, dict)):
                            current_state_for_history[key] = copy.deepcopy(value)  # type: ignore[arg-type]
                        else:
                            current_state_for_history[key] = value
                anim_history.append(current_state_for_history)
                st.session_state["anim_state_history"] = anim_history
                
                # Pop from redo stack and restore
                next_state = anim_redo_stack.pop()
                st.session_state["anim_state_redo"] = anim_redo_stack
                for key, value in next_state.items():
                    st.session_state[key] = value
                st.rerun()
            elif anim_is_running:
                # Clear redo stack when making new progress (branching off)
                st.session_state["anim_state_redo"] = []
                # Set flags to indicate what type of manual advance was requested
                # The animation progress code will check these flags
                if is_manual_step_mode:
                    st.session_state["anim_manual_step_requested"] = True
                elif is_manual_iteration_mode:
                    st.session_state["anim_manual_iteration_requested"] = True
                else:  # is_manual_config_mode
                    st.session_state["anim_manual_config_requested"] = True
    with col_btn4:
        # Reset button - halts animation and resets all graphs to initial state
        # Enabled when animation is running OR when there are generated configurations to clear
        # Styled with white text on black background via custom CSS wrapper class
        st.markdown('<div class="reset-button-wrapper">', unsafe_allow_html=True)
        reset_btn_manual = st.button(
            "Reset",
            key="btn_reset_manual",
            disabled=not reset_btn_should_be_enabled,
            help="Halt the animation and reset all graphs to their initial values. Clears all generated points and search state."
        )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Auto mode: show 3 buttons (Generate without animation, Generate with animation, Reset)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 0.5], gap="small")
    with col_btn1:
        generate_btn = st.button(
            "Generate without animation", 
            key="btn_generate",
            help="Instantly generate all configurations without showing the step-by-step process. Uses the 'Number of configurations' setting above. Results appear when complete - this is the fastest option."
        )
    with col_btn2:
        animate_btn = st.button(
            "Generate with animation", 
            key="btn_animate",
            help="Generate configurations while showing each step visually. Uses the 'Number of configurations' setting above. You will see each point being placed one-by-one. Animation advances automatically based on the wait interval."
        )
    with col_btn3:
        # Reset button - halts animation and resets all graphs to initial state
        # Enabled when animation is running OR when there are generated configurations to clear
        # Styled with white text on black background via custom CSS wrapper class
        st.markdown('<div class="reset-button-wrapper">', unsafe_allow_html=True)
        reset_btn_auto = st.button(
            "Reset",
            key="btn_reset_auto",
            disabled=not reset_btn_should_be_enabled,
            help="Halt the animation and reset all graphs to their initial values. Clears all generated points and search state."
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ============= Advanced: Batch generation & deviation analysis ============
st.markdown("<hr style='margin:1.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
st.markdown("**Advanced Generation & Analysis**")

with st.expander("ℹ️ Early Stopping — waarom sommige configuraties sneller klaar zijn", expanded=False):
    st.markdown(f"""
**Wat is early stopping?**
Elke configuratie wordt gegenereerd door iteratief punten te verplaatsen terwijl de PDP-ordening behouden blijft.
Soms raakt een configuratie *gestagneerd*: de zoekruimte is uitgeput en elk punt wordt teruggeplaatst op zijn
vorige positie (nulbeweging). Verdere iteraties produceren dan geen extra afwijking meer.

**Wanneer wordt het toegepast?**
Als **{EARLY_STOP_PATIENCE} opeenvolgende iteraties** geen punt meer dan {EARLY_STOP_EPSILON:.0e} m verplaatsen,
stopt de generatie voor die configuratie vroegtijdig. De maximaal {MAX_FILTER_ITERATIONS} iteraties worden dan niet
volledig doorlopen.

**Waarom?**
- **Snelheid**: gestagneerde iteraties kosten rekentijd zonder de uitkomst te veranderen.
  Bij veel configuraties kan dit 30–70 % van de totale rekentijd besparen.
- **Kwaliteit**: het eindresultaat is identiek — early stopping slaat alleen iteraties over
  die toch geen effect meer zouden hebben.
- Na afloop toont de timing-balk hoeveel iteraties er daadwerkelijk gebruikt zijn en hoeveel procent bespaard is.
""")

advanced_col1, advanced_col2 = st.columns([3, 1], gap="small")
with advanced_col1:
    st.caption("""Generate 1000 configurations automatically and analyze the 100 most deviating from the original pattern.
    
**Metrics explained:**
- **Perpendicular Variance**: Variance of the perpendicular distances (in m²) from generated points to the original trajectory path
- **Max Angle Deviation**: Maximum change in trajectory angle (in degrees) between consecutive timestamps
- **Max Distance Deviation**: Maximum change in distance (in meters) between consecutive timestamps

The analysis identifies configurations with the largest spatial variations while preserving the PDP inequality pattern.""")
with advanced_col2:
    generate_30_btn = st.button(
        "Generate 1000 & Show Top 100",
        key="btn_generate_30",
        help="Automatically generates 1000 configurations using your current settings (iterations, PDP variant, buffer, roughness, threshold). Displays detailed analysis of the 100 configurations that deviate most from the original, including visualizations, statistics, and downloadable data."
    )
    st.markdown('<div class="generate-5000-marker"></div>', unsafe_allow_html=True)
    generate_5000_btn = st.button(
        "Generate 1 and show",
        key="btn_generate_5000",
        type="primary",
        help="Automatically generates 5 configurations using your current settings (PDP variant, buffer, roughness, threshold). Uses 50 iterations only for this button and shows the most deviating configurations with full analysis."
    )
    generate_50_btn = st.button(
        "Generate 200 & Show Top 25 (reinsertion)",
        key="btn_generate_50",
        help="Generates 200 configurations with 200 iterations each, using 8 timestamps from t=131 to t=138 (reinsertion zone). Shows the 25 most deviating configurations."
    )
    generate_ext30_btn = st.button(
        "Generate 100 & Show Top 3 (with GIF)",
        key="btn_generate_ext30",
        help="Generates 100 configurations with 2500 iterations each. Shows the 3 most deviating configurations and lets you download an animated GIF of each."
    )
    generate_ext30_fe_btn = st.button(
        "Generate 100 & Top 3 GIF (fixed endpoints)",
        key="btn_generate_ext30_fe",
        help="Zelfde als 'Generate 100 & Show Top 3 (with GIF)' maar de eerste en laatste timestamp per object worden NIET verplaatst. Ze fungeren als vaste ankerpunten."
    )
    generate_ext30_half_btn = st.button(
        "100 configs × Top 3 (½ ts, sidebar-afhankelijk)",
        key="btn_generate_ext30_half",
        help="100 configs × 2500 iter | Gebruikt sidebar PDP-variant & step=2. Afhankelijk van sidebar-instellingen."
    )
    generate_half_ts_btn = st.button(
        "100 configs × Top 10 (½ ts, eigen dataset)",
        key="btn_generate_half_ts",
        help="100 configs × 2500 iter | PDP fundamental | Bouwt zelf een gefilterde dataset (elke 2e timestamp + laatste). Onafhankelijk van sidebar."
    )
    generate_quarter_ts_btn = st.button(
        "Generate 100 & Top 10 GIF (¼ ts, filtered)",
        key="btn_generate_quarter_ts",
        help="100 configs × 2500 iter | exponential | PDP fundamental | Houdt elke 4e timestamp (0,4,8,...,136) + laatste (136). Tekent blauwe en oranje curves apart per object."
    )
    generate_eighth_ts_btn = st.button(
        "Generate 100 & Top 10 GIF (⅛ ts, filtered)",
        key="btn_generate_eighth_ts",
        help="100 configs × 2500 iter | exponential | PDP fundamental | Houdt elke 8e timestamp (0,8,16,...,136) + laatste (136). ±18 timestamps per object."
    )
    generate_sixteenth_ts_btn = st.button(
        "Generate 100 & Top 10 GIF (1/16 ts, filtered)",
        key="btn_generate_sixteenth_ts",
        help="100 configs × 2500 iter | exponential | PDP fundamental | Houdt elke 16e timestamp (0,16,32,...,128) + laatste (136). ±9 timestamps per object."
    )
    generate_four_ts_btn = st.button(
        "Generate 100 & Top 10 GIF (4 ts)",
        key="btn_generate_four_ts",
        help="100 configs × 2500 iter | exponential | PDP fundamental | Slechts 4 timestamps: 0, 46, 92, 136."
    )
    generate_two_ts_btn = st.button(
        "Generate 100 & Top 10 GIF (2 ts)",
        key="btn_generate_two_ts",
        help="100 configs × 2500 iter | exponential | PDP fundamental | Slechts 2 timestamps: 0 en 136."
    )
    generate_c68_realistic_btn = st.button(
        "C68 Realistic (multi-pt, buffer+rough)",
        key="btn_generate_c68_realistic",
        help="Config 68 | t=82..160 step 2 | Multiple points same direction | realistic (d1 buffer 5m, d2 rough 0.30m) | External pts at lane centers | 100 configs × 2500 iterations | Top 10"
    )
    generate_c68_fundamental_btn = st.button(
        "C68 Fundamental (single-pt, ext pts)",
        key="btn_generate_c68_fundamental",
        help="Config 68 | t=82..160 step 2 | Single point | fundamental | External pts at lane centers | 100 configs × 2500 iterations | Top 10"
    )
    generate_ext30_rough_btn = st.button(
        "Generate 100 & Top 3 GIF (rough d1=d2=0.30m)",
        key="btn_generate_ext30_rough",
        help="Zelfde als 'Generate 100 & Show Top 3 (with GIF)' maar forceert rough d1=d2=0.30m ongeacht sidebar-instellingen."
    )
    generate_br_consec_btn = st.button(
        "100 configs × Top 3 GIF (bufferrough, consec ts)",
        key="btn_generate_br_consec",
        help="100 configs × 2500 iter | early stopping | bufferrough (buffer x/y=1m, rough x/y=0.30m) | Consecutive timestamps (2-3) same direction | ALL timestamps"
    )
    generate_br_consec_rd_btn = st.button(
        "100 configs × Top 3 GIF (bufferrough, consec ts, random dirs)",
        key="btn_generate_br_consec_rd",
        help="100 configs × 2500 iter | early stopping | bufferrough (buffer x/y=1m, rough x/y=0.30m) | Consecutive timestamps (2-3) RANDOM directions | ALL timestamps"
    )
    generate_recursive_6event_btn = st.button(
        "🔄 Recursive 6-Event Generation",
        key="btn_generate_recursive_6event",
        help="Detects 6 overtake events (start, lane-change start/end, return start/end, last). "
             "Generates recursively in batches of 100 configs — each batch starts from the best "
             "result of the previous batch. Only the 6 event timestamps are used for generation and PDP comparison."
    )
    generate_6ev_single_btn = st.button(
        "🎯 6-Event Single Iteration",
        key="btn_generate_6ev_single",
        type="primary",
        help="Selects 6 event timestamps (0, 34, 67, 183, 213, 249). "
             "Generates 1 config × 1 iteration using exponential strategy + fundamental PDP variant. "
             "Moves a single random point per iteration. Y-axis fixed at [-10, +10]."
    )

# Handle Reset button click for both modes
# This resets all animation state variables to their initial values
# We need to check which button variable exists based on the current mode
if is_any_manual_mode:
    reset_btn_clicked = reset_btn_manual
else:
    reset_btn_clicked = reset_btn_auto

# Reset is triggered when clicked and there's something to reset (animation running OR configs exist)
if reset_btn_clicked and reset_btn_should_be_enabled:
    # Clear all animation-related session state variables
    # This performs a complete reset:
    # 1. Halts any running animation
    # 2. Clears all generated daughter points, restoring graphs to show only original parent points
    # 3. Clears the Generated configuration (CSV) section
    # 4. Clears all search state and diagnostics
    animation_keys_to_clear = [
        # Animation control flags
        "anim_running",
        "anim_manual_step_requested",
        "anim_manual_iteration_requested",
        "anim_manual_config_requested",
        "anim_manual_mode",
        "anim_manual_step_mode",
        "anim_manual_iteration_mode",
        "anim_manual_config_mode",
        "_iteration_in_progress",
        "_config_in_progress",
        "_iteration_just_completed",
        "_config_just_completed",
        "anim_in_search",
        # Point generation state - clearing these removes all daughter points
        "anim_generated_point",
        "anim_generated_points",
        "anim_successful_points",  # Current configuration's successful points
        "anim_all_configs",        # All completed configurations (used for CSV export)
        # Search parameters
        "anim_distance",
        "anim_angle",
        "anim_parent_idx",
        "anim_all_pts",
        "anim_all_ts",
        # Iteration tracking
        "anim_iteration",
        "anim_completed_iterations",
        "anim_search_steps",
        "anim_current_config",
        "anim_num_configs",
        "anim_max_iterations",
        "anim_iterations_per_run",
        "anim_last_update",
        "anim_last_step",
        # Binary search state
        "anim_binary_mode",
        "anim_binary_step",
        "anim_ok_point",
        "anim_delta",
        "anim_delta_vector",  # Binary search delta vector
        "anim_had_full_match",
        # Linear search state
        "anim_linear_mode",
        "anim_linear_step",
        "anim_linear_current_distance",
        "anim_linear_maxdist",
        "anim_linear_step_size",
        # Circle visualization
        "anim_circle_idx",
        "show_anim_circle",
        # Multi-point animation support
        "anim_selected_indices",
        "anim_movement_vectors",
        # Multi-variant support
        "anim_pdp_variants_list",
        "anim_current_variant_idx",
        "anim_current_variant",
        # Diagnostics
        "diag_rows",
        "binary_iteration_summary",
        # History (for Previous step functionality)
        "anim_state_history",
    ]
    for key in animation_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Trigger a rerun to update the UI with reset state
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ============= Auto-detect bounds when configuration settings change =============
# This must happen AFTER all widgets are rendered, so we have access to the NEW values
# Track previous values to detect changes - includes ALL settings that affect the animation
_prev_cfg_k = st.session_state.get("_prev_cfg_k", None)
_prev_cfg_start_t = st.session_state.get("_prev_cfg_start_t", None)
_prev_cfg_c = st.session_state.get("_prev_cfg_c", None)
_prev_cfg_num_timestamps = st.session_state.get("_prev_cfg_num_timestamps", None)
_prev_cfg_strategy = st.session_state.get("_prev_cfg_strategy", None)
_prev_cfg_iterations = st.session_state.get("_prev_cfg_iterations", None)
_prev_cfg_num_configs = st.session_state.get("_prev_cfg_num_configs", None)
_prev_cfg_pdp_variants = st.session_state.get("_prev_cfg_pdp_variants", None)
_prev_cfg_threshold_mode = st.session_state.get("_prev_cfg_threshold_mode", None)
_prev_cfg_threshold_pct = st.session_state.get("_prev_cfg_threshold_pct", None)
_prev_cfg_threshold_abs = st.session_state.get("_prev_cfg_threshold_abs", None)
_prev_cfg_point_selection = st.session_state.get("_prev_cfg_point_selection", None)
_prev_cfg_movement_direction = st.session_state.get("_prev_cfg_movement_direction", None)

_curr_cfg_k = st.session_state.get("cfg_k", None)
_curr_cfg_start_t = st.session_state.get("cfg_start_t", None)
_curr_cfg_c = st.session_state.get("cfg_c", None)
_curr_cfg_num_timestamps = st.session_state.get("cfg_num_timestamps", None)
_curr_cfg_strategy = st.session_state.get("cfg_strategy", None)
_curr_cfg_iterations = st.session_state.get("cfg_iterations", None)
_curr_cfg_num_configs = st.session_state.get("cfg_num_configs", None)
_curr_cfg_pdp_variants = st.session_state.get("cfg_pdp_variants", None)
_curr_cfg_threshold_mode = st.session_state.get("cfg_threshold_mode", None)
_curr_cfg_threshold_pct = st.session_state.get("cfg_threshold_pct", None)
_curr_cfg_threshold_abs = st.session_state.get("cfg_threshold_abs", None)
_curr_cfg_point_selection = st.session_state.get("cfg_point_selection", None)
_curr_cfg_movement_direction = st.session_state.get("cfg_movement_direction", None)

# Check if any configuration changed (but only if previous values were set, i.e., not on first run)
# Also check if animation is running or has history - only then do we need to reset
_has_animation_state = (
    st.session_state.get("anim_running", False) or 
    len(st.session_state.get("anim_state_history", [])) > 0 or
    len(st.session_state.get("anim_all_configs", [])) > 0 or
    st.session_state.get("anim_generated_point") is not None
)

_config_changed = (
    _prev_cfg_k is not None and 
    _has_animation_state and
    (
        _prev_cfg_k != _curr_cfg_k or 
        _prev_cfg_start_t != _curr_cfg_start_t or 
        _prev_cfg_c != _curr_cfg_c or
        _prev_cfg_num_timestamps != _curr_cfg_num_timestamps or
        _prev_cfg_strategy != _curr_cfg_strategy or
        _prev_cfg_iterations != _curr_cfg_iterations or
        _prev_cfg_num_configs != _curr_cfg_num_configs or
        _prev_cfg_pdp_variants != _curr_cfg_pdp_variants or
        _prev_cfg_threshold_mode != _curr_cfg_threshold_mode or
        _prev_cfg_threshold_pct != _curr_cfg_threshold_pct or
        _prev_cfg_threshold_abs != _curr_cfg_threshold_abs or
        _prev_cfg_point_selection != _curr_cfg_point_selection or
        _prev_cfg_movement_direction != _curr_cfg_movement_direction
    )
)

# Store current values for next comparison
st.session_state["_prev_cfg_k"] = _curr_cfg_k
st.session_state["_prev_cfg_start_t"] = _curr_cfg_start_t
st.session_state["_prev_cfg_c"] = _curr_cfg_c
st.session_state["_prev_cfg_num_timestamps"] = _curr_cfg_num_timestamps
st.session_state["_prev_cfg_strategy"] = _curr_cfg_strategy
st.session_state["_prev_cfg_iterations"] = _curr_cfg_iterations
st.session_state["_prev_cfg_num_configs"] = _curr_cfg_num_configs
st.session_state["_prev_cfg_pdp_variants"] = _curr_cfg_pdp_variants
st.session_state["_prev_cfg_threshold_mode"] = _curr_cfg_threshold_mode
st.session_state["_prev_cfg_threshold_pct"] = _curr_cfg_threshold_pct
st.session_state["_prev_cfg_threshold_abs"] = _curr_cfg_threshold_abs
st.session_state["_prev_cfg_point_selection"] = _curr_cfg_point_selection
st.session_state["_prev_cfg_movement_direction"] = _curr_cfg_movement_direction

# Reset animation state when any settings change - same as Reset button
if _config_changed:
    # Clear all animation-related session state keys (same list as Reset button)
    animation_keys_to_clear = [
        # Animation control flags
        "anim_running",
        "anim_manual_step_requested",
        "anim_manual_iteration_requested",
        "anim_manual_config_requested",
        "anim_manual_mode",
        "anim_manual_step_mode",
        "anim_manual_iteration_mode",
        "anim_manual_config_mode",
        "_iteration_in_progress",
        "_config_in_progress",
        "_iteration_just_completed",
        "_config_just_completed",
        "anim_in_search",
        # Point generation state - clearing these removes all daughter points
        "anim_generated_point",
        "anim_generated_points",
        "anim_successful_points",  # Current configuration's successful points
        "anim_all_configs",        # All completed configurations (used for CSV export)
        # Search parameters
        "anim_distance",
        "anim_angle",
        "anim_parent_idx",
        "anim_all_pts",
        "anim_all_ts",
        # Iteration tracking
        "anim_iteration",
        "anim_completed_iterations",
        "anim_search_steps",
        "anim_current_config",
        "anim_num_configs",
        "anim_max_iterations",
        "anim_iterations_per_run",
        "anim_last_update",
        "anim_last_step",
        # Binary search state
        "anim_binary_mode",
        "anim_binary_step",
        "anim_ok_point",
        "anim_delta",
        "anim_delta_vector",
        "anim_had_full_match",
        # Linear search state
        "anim_linear_mode",
        "anim_linear_step",
        "anim_linear_current_distance",
        "anim_linear_maxdist",
        "anim_linear_step_size",
        # Circle visualization
        "anim_circle_idx",
        "show_anim_circle",
        # Multi-point animation support
        "anim_selected_indices",
        "anim_movement_vectors",
        # Multi-variant support
        "anim_pdp_variants_list",
        "anim_current_variant_idx",
        "anim_current_variant",
        # Diagnostics
        "diag_rows",
        "binary_iteration_summary",
        # History (for Previous step functionality)
        "anim_state_history",
        "anim_state_redo",
    ]
    for key in animation_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Trigger auto-detect bounds if config changed
if _config_changed and data_source != "Create random configuration":
    if _auto_detect_bounds_logic():
        st.rerun()

# ============= Data window (select subset of k and l) ============
# Extract points from _df_all (works for all data sources: preset, uploaded, random)
# extract_points_from_df imported from pdp_utils.data_loading

# Extract points for all objects into a unified structure
all_objects_points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
for o_id in all_object_ids:
    pts, ts = extract_points_from_df(_df_all, o_val=o_id, c_val=selected_c_int)
    all_objects_points[o_id] = (pts, ts)

# Determine which time indices are included in the chosen window
try:
    start_idx = _t_common.index(start_t)  # type: ignore[arg-type]
except ValueError:
    start_idx = 0
end_idx = start_idx + int(num_timestamps)
_ts_step = int(st.session_state.get("_cfg_timestamp_step", 1))
selected_ts_window = list(_t_common[start_idx:end_idx:_ts_step])
# When using step > 1 (½ timestamps), always include the LAST timestamp
# of the full window (e.g. 136) even if it would be skipped by the step.
if _ts_step > 1:
    _full_window_last = _t_common[min(end_idx - 1, len(_t_common) - 1)]
    if _full_window_last not in selected_ts_window:
        selected_ts_window.append(_full_window_last)
selected_ts_set = set(selected_ts_window)

# Filter all objects to the time window - unified structure
all_points_plot: dict[int, np.ndarray] = {}
all_vals_plot: dict[int, np.ndarray] = {}
for o_id in all_object_ids:
    pts, ts = all_objects_points[o_id]
    mask = np.isin(ts, list(selected_ts_set))
    all_points_plot[o_id] = pts[mask]
    all_vals_plot[o_id] = ts[mask]

# ============= External (fixed) reference points =============
# These points are included in PDP comparison but do NOT move during generation
# External points always apply to ALL timestamps in the window
external_points_list: list[tuple[float, float]] = []  # (x, y) only - no timestamp needed
if st.session_state.get("use_external_points", False):
    raw_external = st.session_state.get("external_points", [])
    # Handle both old format (x, y, t) and new format (x, y)
    for pt in raw_external:
        if len(pt) >= 2:
            external_points_list.append((float(pt[0]), float(pt[1])))

# Build external points arrays for the selected time window
# Each external point is expanded to ALL timestamps in the window
external_pts_for_window: list[np.ndarray] = []
external_ts_for_window: list[float] = []
for ext_x, ext_y in external_points_list:
    # External points always apply to ALL timestamps
    for t_val in selected_ts_window:
        external_pts_for_window.append(np.array([ext_x, ext_y]))
        external_ts_for_window.append(float(t_val))

n_external_points = len(external_pts_for_window)

# Create flattened arrays of ALL points for PDP algorithm
# This combines all objects into single arrays with tracking info
def build_flattened_points() -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[bool]]:
    """
    Build flattened arrays of all points across all objects.
    Returns:
        all_pts: (N_total, 2) array of all points
        all_ts: (N_total,) array of timestamps
        all_obj_ids: list of object IDs for each point (-1 for external points)
        all_local_indices: list of local index within each object
        all_is_fixed: list of booleans (True for external points that don't move)
    """
    pts_list: list[np.ndarray] = []
    ts_list: list[float] = []
    obj_ids: list[int] = []
    local_indices: list[int] = []
    is_fixed: list[bool] = []
    
    # First add all movable points (from objects)
    for o_id in sorted(all_points_plot.keys()):
        pts = all_points_plot[o_id]
        ts = all_vals_plot[o_id]
        for local_idx in range(pts.shape[0]):
            pts_list.append(pts[local_idx])
            ts_list.append(ts[local_idx])
            obj_ids.append(o_id)
            local_indices.append(local_idx)
            is_fixed.append(False)
    
    # Then add external (fixed) points
    for ext_idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):  # type: ignore[misc]
        pts_list.append(ext_pt)
        ts_list.append(ext_t)
        obj_ids.append(-1)  # -1 indicates external point
        local_indices.append(ext_idx)
        is_fixed.append(True)
    
    if pts_list:
        return np.array(pts_list), np.array(ts_list), obj_ids, local_indices, is_fixed
    else:
        return np.array([]).reshape(0, 2), np.array([]), [], [], []

# Flattened representation for PDP
all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat, all_is_fixed_flat = build_flattened_points()
n_total_points = all_pts_flat.shape[0]
n_movable_points = n_total_points - n_external_points  # Points that can be moved during generation

# Helper functions for converting between flat index and object info
def get_object_info_for_flat_idx(flat_idx: int) -> tuple[int, int, str]:
    """
    Get object ID, local index, and label for a flat index.
    Returns: (object_id, local_idx_in_object, label_character)
    For external points, object_id is -1 and label is "ext"
    """
    if 0 <= flat_idx < n_total_points:
        o_id = all_obj_ids_flat[flat_idx]
        local_idx = all_local_idx_flat[flat_idx]
        if o_id == -1:
            # External point
            return -1, local_idx, "ext"
        # Find which position this object is in (for label lookup)
        sorted_obj_ids = sorted(all_points_plot.keys())
        obj_position = sorted_obj_ids.index(o_id) if o_id in sorted_obj_ids else 0
        label = OBJECT_LABELS[obj_position % len(OBJECT_LABELS)]
        return o_id, local_idx, label
    return 0, 0, "k"

def is_fixed_point(flat_idx: int) -> bool:
    """Check if a flat index refers to a fixed (external) point."""
    if 0 <= flat_idx < n_total_points:
        return all_is_fixed_flat[flat_idx]
    return False

def get_movable_indices() -> list[int]:
    """Get list of flat indices for movable (non-fixed) points only.
    Also excludes any indices in session-state '_frozen_endpoints' set."""
    frozen: set[int] = st.session_state.get("_frozen_endpoints", set())
    return [i for i in range(n_total_points) if not is_fixed_point(i) and i not in frozen]

def _build_d2_change_weights(movable_indices: list[int]) -> dict[int, float]:
    """Build per-index weights based on local d2 (y) change magnitude."""
    weights: dict[int, float] = {}
    eps = 1e-6
    for flat_idx in movable_indices:
        o_id = all_obj_ids_flat[flat_idx]
        local_idx = all_local_idx_flat[flat_idx]
        pts = all_points_plot.get(o_id)
        if pts is None or pts.ndim < 2 or pts.shape[0] < 2:
            # Need at least 2 points to compute any d2 difference.
            weights[flat_idx] = 1.0
            continue

        y_vals = pts[:, 1]
        n_pts = y_vals.shape[0]
        # Clamp local_idx to valid range so every access is in-bounds.
        li = max(0, min(local_idx, n_pts - 1))
        if li == 0:
            # First point → forward difference (index 1 guaranteed by n_pts >= 2)
            strength = abs(float(y_vals[1] - y_vals[0]))
        elif li >= n_pts - 1:
            # Last point → backward difference
            strength = abs(float(y_vals[n_pts - 1] - y_vals[n_pts - 2]))
        else:
            # Interior point → average of backward and forward differences
            prev_jump = abs(float(y_vals[li] - y_vals[li - 1]))
            next_jump = abs(float(y_vals[li + 1] - y_vals[li]))
            strength = 0.5 * (prev_jump + next_jump)
        weights[flat_idx] = strength + eps
    return weights

def detect_overtake_events(
    objects_points: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[int, list[float]] | None:
    """Detect 6 key events in an overtake manoeuvre.

    Identifies the overtaking vehicle (largest lateral / d2 range) and finds
    six characteristic timestamps using **raw consecutive d2 differences**
    (no smoothing, so no temporal bleed):

    1. First timestamp
    2. Start of first lane change (first t where d2 visibly moves)
    3. Arrival at second lane (d2 stabilises at new position)
    4. Departure from second lane (d2 begins returning)
    5. Arrival back in original lane (d2 stabilises)
    6. Last timestamp

    Parameters
    ----------
    objects_points : dict mapping object-id → (pts (N,2), ts (N,))
    change_threshold_frac : fraction of the max |Δd2| step used as threshold
        to distinguish "changing" from "stable" timestamps.

    Returns
    -------
    (overtaking_oid, event_timestamps) with 6 timestamps, or *None* when
    detection is not feasible (e.g. no clear lane change).
    """
    # --- 1. Identify the overtaking vehicle (largest d2 range) ---
    best_oid: int | None = None
    best_range = 0.0
    for oid, (pts, _ts) in objects_points.items():
        d2_range = float(np.ptp(pts[:, 1]))
        if d2_range > best_range:
            best_range = d2_range
            best_oid = oid
    if best_oid is None or best_range < 0.5:
        return None

    pts, ts = objects_points[best_oid]
    d2 = pts[:, 1].astype(float)
    n = len(d2)
    if n < 10:
        return None

    # --- 2. Raw absolute d2 step sizes (no smoothing!) ---
    # delta[i] = |d2[i+1] - d2[i]|   (length n-1)
    delta = np.abs(np.diff(d2))
    max_delta = float(np.max(delta))
    if max_delta < 1e-9:
        return None

    # Adaptive threshold based on the noise floor of the stable regions.
    # The lower quartile of delta values represents the "noise" during
    # stable portions of the trajectory.  A multiplier well above the noise
    # guarantees we don't confuse jitter for a lane change, while still
    # catching the very first (smaller) step of a lane change.
    sorted_delta = np.sort(delta)
    q25 = float(sorted_delta[max(0, len(sorted_delta) // 4)])
    if q25 < 1e-12:
        # Almost-zero noise → fall back to a small fraction of max
        threshold = max_delta * 0.03
    else:
        threshold = q25 * 10.0
        # Cap so we never exceed half the max step
        threshold = min(threshold, max_delta * 0.50)

    # --- 3. Mark each timestamp as "changing" or "stable" ---
    # Timestamp i is "changing" when the step *arriving* at i is large,
    # i.e. |d2[i] - d2[i-1]| > threshold.  This means the event index
    # directly corresponds to the first data point that has visibly moved.
    is_changing = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if delta[i - 1] > threshold:
            is_changing[i] = True

    # --- 3b. Bridge small stable gaps (≤ 2 timestamps) between changing
    #     regions so that a single step dipping below threshold mid-lane-
    #     change doesn't fragment the detection into extra regions. ---
    _gap_limit = 2
    _i = 0
    while _i < n:
        if is_changing[_i]:
            _j = _i
            while _j < n and is_changing[_j]:
                _j += 1
            # _j = first stable index after the run
            _k = _j
            while _k < n and not is_changing[_k] and (_k - _j) <= _gap_limit:
                _k += 1
            if _k < n and is_changing[_k] and (_k - _j) <= _gap_limit:
                for _m in range(_j, _k):
                    is_changing[_m] = True
                _i = _k
            else:
                _i = _j
        else:
            _i += 1

    # Find transition indices (start / end of each changing region)
    transitions = np.diff(is_changing.astype(np.int8))
    starts = list(np.where(transitions == 1)[0] + 1)   # first "changing"
    ends   = list(np.where(transitions == -1)[0] + 1)   # first "stable" after

    if is_changing[0]:
        starts.insert(0, 0)
    if is_changing[-1]:
        ends.append(n - 1)

    if len(starts) < 2 or len(ends) < 2:
        return None

    # --- 4. Build event indices ---
    # For E2/E4 (start of lane change): starts[] = first index where d2
    #   has visibly moved → correct as-is.
    # For E3/E5 (arrival / stabilisation): ends[] gives the first STABLE
    #   index after the changing region, but the arrival point is one
    #   earlier — the last index where d2 was still changing, i.e. the
    #   timestamp where d2 reached its new value.  Hence ends[] - 1.
    idx_event = [
        0,               # event 1: first timestamp
        starts[0],       # event 2: first t where d2 visibly moves
        ends[0] - 1,     # event 3: last t of first lane-change (arrival)
        starts[1],       # event 4: first t where d2 starts returning
        ends[1] - 1,     # event 5: last t of return lane-change (arrival back)
        n - 1,           # event 6: last timestamp
    ]
    # Ensure strict ordering and clamp
    for i in range(1, len(idx_event)):
        if idx_event[i] <= idx_event[i - 1]:
            idx_event[i] = min(idx_event[i - 1] + 1, n - 1)

    event_timestamps = [float(ts[i]) for i in idx_event]
    return best_oid, event_timestamps


# ============= 6-Event Overtake Detection (for visualisation) ============
# Run once and store so draw_original can mark the events on the plot.
_6ev_vis_result = detect_overtake_events(all_objects_points)
if _6ev_vis_result is not None:
    _6ev_vis_oid, _6ev_vis_timestamps = _6ev_vis_result
    _6ev_vis_map: dict[float, int] = {t: i + 1 for i, t in enumerate(_6ev_vis_timestamps)}
else:
    _6ev_vis_oid = None
    _6ev_vis_map = {}


def _weighted_pick(indices: list[int], count: int, weights_map: dict[int, float]) -> list[int]:
    """Weighted sampling without replacement over candidate flat indices."""
    if not indices or count <= 0:
        return []
    count = min(count, len(indices))
    weights = np.array([max(0.0, float(weights_map.get(idx, 1.0))) for idx in indices], dtype=float)
    if weights.sum() <= 0.0:
        selected_uniform = np.random.choice(indices, size=count, replace=False)
        return [int(i) for i in selected_uniform]
    probs = weights / weights.sum()
    selected = np.random.choice(indices, size=count, replace=False, p=probs)
    return [int(i) for i in selected]

def select_points_for_iteration() -> list[int]:
    """
    Select points to move in this iteration based on the point selection mode.
    Returns a list of flat indices of points to move together.
    """
    movable_indices = get_movable_indices()
    if not movable_indices:
        return []

    prefer_high_d2_change = bool(st.session_state.get("_prefer_high_d2_change_sampling", False))
    d2_weights = _build_d2_change_weights(movable_indices) if prefer_high_d2_change else {}
    
    # Check for temporary overrides first (used by preset buttons), then fall back to widget keys
    point_selection_mode = st.session_state.get("_override_point_selection_mode", st.session_state.get("cfg_point_selection_mode", "Single point"))
    
    if point_selection_mode == "Single point":
        # Current default behavior: select one random point
        if prefer_high_d2_change:
            return _weighted_pick(movable_indices, 1, d2_weights)
        return [int(np.random.choice(movable_indices))]
    
    elif point_selection_mode == "Multiple random points":
        # Select N random points
        num_points = int(st.session_state.get("cfg_num_random_points", 2))
        num_points = min(num_points, len(movable_indices))  # Can't select more than available
        if prefer_high_d2_change:
            return _weighted_pick(movable_indices, num_points, d2_weights)
        selected = list(np.random.choice(movable_indices, size=num_points, replace=False))
        return [int(idx) for idx in selected]
    
    elif point_selection_mode == "Consecutive time stamps":
        # Select consecutive timestamps from a single user-chosen object
        selected_object_id = int(st.session_state.get("_override_consecutive_object_id", st.session_state.get("cfg_consecutive_object_id", 0)))
        num_timestamps = int(st.session_state.get("_override_group_num_timestamps", st.session_state.get("cfg_group_num_timestamps", 2)))
        first_timestamp_idx = int(st.session_state.get("_override_consecutive_first_timestamp", st.session_state.get("cfg_consecutive_first_timestamp", 0)))
        
        # Get indices for the selected object, sorted by timestamp
        indices_for_object: list[tuple[int, float]] = []  # (flat_idx, timestamp)
        for flat_idx in movable_indices:
            o_id = all_obj_ids_flat[flat_idx]
            if o_id == selected_object_id:
                t = all_ts_flat[flat_idx]
                indices_for_object.append((flat_idx, t))
        
        # Sort by timestamp
        indices_for_object.sort(key=lambda x: x[1])
        
        if not indices_for_object:
            # Fall back to single point if no points for this object
            return [int(np.random.choice(movable_indices))]
        
        # Clamp first_timestamp_idx to valid range
        max_start = max(0, len(indices_for_object) - num_timestamps)
        if prefer_high_d2_change and max_start > 0:
            start_candidates = list(range(max_start + 1))
            start_scores: list[float] = []
            for s in start_candidates:
                end_s = min(len(indices_for_object), s + num_timestamps)
                window_indices = [indices_for_object[j][0] for j in range(s, end_s)]
                if not window_indices:
                    start_scores.append(1.0)
                else:
                    start_scores.append(float(np.mean([d2_weights.get(idx, 1.0) for idx in window_indices])))
            score_arr = np.array(start_scores, dtype=float)
            if score_arr.sum() > 0:
                probs = score_arr / score_arr.sum()
                first_timestamp_idx = int(np.random.choice(start_candidates, p=probs))
            else:
                first_timestamp_idx = min(first_timestamp_idx, max_start)
        else:
            first_timestamp_idx = min(first_timestamp_idx, max_start)
        
        # Select consecutive points starting from first_timestamp_idx
        selected_indices = []
        for i in range(num_timestamps):
            if first_timestamp_idx + i < len(indices_for_object):
                selected_indices.append(indices_for_object[first_timestamp_idx + i][0])
        
        return [int(idx) for idx in selected_indices] if selected_indices else [int(np.random.choice(movable_indices))]
    
    # Default fallback
    return [int(np.random.choice(movable_indices))]

def generate_movement_vectors(selected_indices: list[int], base_distance: float) -> dict[int, tuple[float, float]]:
    """
    Generate movement vectors for selected points based on movement direction mode.
    Returns a dict mapping flat_idx -> (delta_x, delta_y)
    
    For multi-point mode, ensures that the chosen direction keeps ALL points within bounds.
    If after max_attempts no valid direction is found, uses the best direction found.
    """
    if not selected_indices:
        return {}
    
    # Check for temporary override first (used by preset buttons), then fall back to widget key
    movement_direction = st.session_state.get("_override_movement_direction", st.session_state.get("cfg_movement_direction", "Same direction"))
    
    # Get visualization bounds (XLIM, YLIM) to keep points within the graph
    # These are more restrictive than coordinate bounds and ensure points stay visible
    try:
        coord_min_x, coord_max_x = XLIM
        coord_min_y, coord_max_y = YLIM
    except NameError:
        # Fallback to session state bounds if XLIM/YLIM not yet computed
        coord_min_x = float(st.session_state.get("coord_min_x", -50.0))
        coord_max_x = float(st.session_state.get("coord_max_x", 150.0))
        coord_min_y = float(st.session_state.get("coord_min_y", -50.0))
        coord_max_y = float(st.session_state.get("coord_max_y", 150.0))
    
    # Check if buffer variants are active and add extra margin for bounds checking
    buffer_margin_x = 0.0
    buffer_margin_y = 0.0
    try:
        pdp_variants = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        if any(v in ["buffer", "bufferrough", "realistic"] for v in pdp_variants):
            buffer_margin_x = float(st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X))
            buffer_margin_y = float(st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y))
            # For realistic variant, only x buffer is used
            if "realistic" in pdp_variants and "buffer" not in pdp_variants and "bufferrough" not in pdp_variants:
                buffer_margin_y = 0.0
    except (TypeError, ValueError) as e:
        logger.warning(f"[WARNING] Failed to read buffer margins from session state: {e}")
    
    # Adjust bounds to account for buffer transformation
    check_min_x = coord_min_x + buffer_margin_x
    check_max_x = coord_max_x - buffer_margin_x
    check_min_y = coord_min_y + buffer_margin_y
    check_max_y = coord_max_y - buffer_margin_y
    
    def point_in_bounds(x: float, y: float) -> bool:
        """Check if a point is within visualization bounds (accounting for buffer margin)."""
        return check_min_x <= x <= check_max_x and check_min_y <= y <= check_max_y
    
    def get_parent_point(idx: int) -> np.ndarray:
        """Get the current parent point position for a given index."""
        # Check successful_points first for updated parent position
        successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        for s in reversed(successful_points):
            if int(s.get("original_parent_idx", -1)) == idx:
                return np.array(s["point"])
        # Fall back to original position
        if 0 <= idx < len(all_pts_flat):
            return all_pts_flat[idx]
        return np.array([0.0, 0.0])
    
    if movement_direction == "Same direction":
        # All points move with the same angle - find a direction that keeps ALL points in bounds
        max_attempts = MAX_DIRECTION_ATTEMPTS
        best_angle = None
        best_in_bounds_count = 0
        
        for _ in range(max_attempts):
            angle = float(np.random.uniform(0, 2 * np.pi))
            delta_x = base_distance * np.cos(angle)
            delta_y = base_distance * np.sin(angle)
            
            # Check if all points would be in bounds with this direction
            in_bounds_count = 0
            all_in_bounds = True
            for idx in selected_indices:
                parent_pt = get_parent_point(idx)
                new_x = parent_pt[0] + delta_x
                new_y = parent_pt[1] + delta_y
                if point_in_bounds(new_x, new_y):
                    in_bounds_count += 1
                else:
                    all_in_bounds = False
            
            # Track best attempt
            if in_bounds_count > best_in_bounds_count:
                best_in_bounds_count = in_bounds_count
                best_angle = angle
            
            if all_in_bounds:
                # Found a valid direction
                return {int(idx): (delta_x, delta_y) for idx in selected_indices}
        
        # Use best direction found (may not keep all points in bounds, but maximizes in-bounds count)
        if best_angle is not None:
            delta_x = base_distance * np.cos(best_angle)
            delta_y = base_distance * np.sin(best_angle)
            return {int(idx): (delta_x, delta_y) for idx in selected_indices}
        
        # Fallback: use first random angle
        angle = float(np.random.uniform(0, 2 * np.pi))
        delta_x = base_distance * np.cos(angle)
        delta_y = base_distance * np.sin(angle)
        return {int(idx): (delta_x, delta_y) for idx in selected_indices}
    
    else:  # Random directions
        # Each point gets its own random angle - ensure each point stays in bounds
        vectors: dict[int, tuple[float, float]] = {}
        max_attempts = MAX_DIRECTION_ATTEMPTS
        
        for idx in selected_indices:
            parent_pt = get_parent_point(idx)
            best_angle = None
            
            for _ in range(max_attempts):
                angle = float(np.random.uniform(0, 2 * np.pi))
                delta_x = base_distance * np.cos(angle)
                delta_y = base_distance * np.sin(angle)
                
                new_x = parent_pt[0] + delta_x
                new_y = parent_pt[1] + delta_y
                
                if point_in_bounds(new_x, new_y):
                    vectors[int(idx)] = (delta_x, delta_y)
                    break
                elif best_angle is None:
                    best_angle = angle
            else:
                # No valid angle found after max_attempts, use first angle tried
                if best_angle is not None:
                    delta_x = base_distance * np.cos(best_angle)
                    delta_y = base_distance * np.sin(best_angle)
                    vectors[int(idx)] = (delta_x, delta_y)
                else:
                    # Complete fallback
                    angle = float(np.random.uniform(0, 2 * np.pi))
                    delta_x = base_distance * np.cos(angle)
                    delta_y = base_distance * np.sin(angle)
                    vectors[int(idx)] = (delta_x, delta_y)
        
        return vectors

def scale_movement_vectors(vectors: dict[int, tuple[float, float]], scale: float) -> dict[int, tuple[float, float]]:
    """Scale all movement vectors by a factor (e.g., 0.5 to halve distances)."""
    return {int(idx): (dx * scale, dy * scale) for idx, (dx, dy) in vectors.items()}

def apply_damping_factor(parent_pt: np.ndarray, child_pt: np.ndarray) -> np.ndarray:
    """
    Apply a random damping factor to reduce the distance between parent and child point.
    
    The damping factor is randomized for each call (per iteration).
    When damping is disabled, returns the original child_pt unchanged.
    
    Args:
        parent_pt: Coordinates of the parent/mother point
        child_pt: Coordinates of the child/daughter point (before damping)
    
    Returns:
        New child point coordinates, with distance reduced by the damping factor
    """
    use_damping = st.session_state.get("cfg_use_damping", False)
    if not use_damping:
        return child_pt
    
    damping_min = float(st.session_state.get("cfg_damping_min", 0.0))
    damping_max = float(st.session_state.get("cfg_damping_max", 1.0))
    
    # Ensure min <= max
    if damping_min > damping_max:
        damping_min, damping_max = damping_max, damping_min
    
    # Generate random damping factor for this iteration
    damping_factor = float(np.random.uniform(damping_min, damping_max))
    
    # Calculate the vector from parent to child
    direction = child_pt - parent_pt
    
    # Apply damping: new_child = parent + direction * damping_factor
    damped_child = parent_pt + direction * damping_factor
    
    return damped_child

def apply_movement_vectors(base_points: np.ndarray, vectors: dict[int, tuple[float, float]]) -> dict[int, np.ndarray]:
    """
    Apply movement vectors to base points and return new positions.
    Returns dict mapping flat_idx -> new_position (clipped to visualization bounds)
    
    When buffer variants are active, clips positions with extra margin so that
    buffer-transformed points (x Â± buffer_x, y Â± buffer_y) stay within bounds.
    """
    # Use visualization bounds (XLIM, YLIM) to keep points within the graph
    # These are computed from the actual data and ensure points stay visible
    try:
        x_min, x_max = XLIM
        y_min, y_max = YLIM
    except NameError:
        # Fallback to coordinate bounds if XLIM/YLIM not yet computed
        x_min, x_max = COORD_MIN_X, COORD_MAX_X
        y_min, y_max = COORD_MIN_Y, COORD_MAX_Y
    
    # Check if buffer variants are active and add extra margin for clipping
    buffer_margin_x = 0.0
    buffer_margin_y = 0.0
    try:
        pdp_variants = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        if any(v in ["buffer", "bufferrough", "realistic"] for v in pdp_variants):
            buffer_margin_x = float(st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X))
            buffer_margin_y = float(st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y))
            # For realistic variant, only x buffer is used
            if "realistic" in pdp_variants and "buffer" not in pdp_variants and "bufferrough" not in pdp_variants:
                buffer_margin_y = 0.0
    except (TypeError, ValueError) as e:
        logger.warning(f"[WARNING] Failed to read buffer margins from session state: {e}")
    
    # Adjust clipping bounds to account for buffer transformation
    clip_x_min = x_min + buffer_margin_x
    clip_x_max = x_max - buffer_margin_x
    clip_y_min = y_min + buffer_margin_y
    clip_y_max = y_max - buffer_margin_y
    
    new_positions = {}
    for idx, (dx, dy) in vectors.items():
        if 0 <= idx < len(base_points):
            new_x = base_points[idx, 0] + dx
            new_y = base_points[idx, 1] + dy
            # Clip to visualization bounds (with buffer margin) to keep points within the graph
            new_x = np.clip(new_x, clip_x_min, clip_x_max)
            new_y = np.clip(new_y, clip_y_min, clip_y_max)
            new_positions[idx] = np.array([new_x, new_y])
    return new_positions

def get_timestamp_for_flat_idx(flat_idx: int) -> float:
    """Get the timestamp for a flat index."""
    if 0 <= flat_idx < n_total_points:
        return float(all_ts_flat[flat_idx])
    return 0.0

def get_point_for_flat_idx(flat_idx: int) -> np.ndarray:
    """Get the point coordinates for a flat index."""
    if 0 <= flat_idx < n_total_points:
        return all_pts_flat[flat_idx]
    return np.array([0.0, 0.0])

# For backward compatibility (used in some places still)
k_points_plot = all_points_plot.get(0, np.array([]).reshape(0, 2))
k_vals_plot = all_vals_plot.get(0, np.array([]))
l_points_plot = all_points_plot.get(1, np.array([]).reshape(0, 2))
l_vals_plot = all_vals_plot.get(1, np.array([]))

# ============= maxdist & axis limits ============
def max_consecutive_dist(pts: np.ndarray) -> float:
    """Return the maximum distance between consecutive points in pts."""
    n = pts.shape[0]
    if n < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    return float(np.max(dists))

# Calculate maxdist from all objects
all_max_dists = [max_consecutive_dist(pts) for pts in all_points_plot.values()]
_maxdist_consecutive: float = max(all_max_dists) if all_max_dists else 0.0

if _maxdist_consecutive > 0:
    maxdist = _maxdist_consecutive
else:
    # For single timestamp: use distance between all pairs of points, or 10% of coordinate range as fallback
    # Gather all point arrays from all objects
    all_point_arrays = [pts for pts in all_points_plot.values() if pts.shape[0] > 0]
    
    if len(all_point_arrays) > 1:
        # Calculate max pairwise distance — vectorized O(n²) via scipy
        all_pts = np.vstack(all_point_arrays)
        pw = pdist(all_pts)  # condensed distance matrix, Euclidean by default
        maxdist = float(pw.max()) if pw.size > 0 else DEFAULT_MAXDIST_FALLBACK
    else:
        maxdist = DEFAULT_MAXDIST_FALLBACK  # Default fallback

def square_limits_with_margin(
    pts: np.ndarray, margin: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute square axis limits around pts with a given margin.
    Ensures a square window and at least 'margin' distance from points to borders.
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
    effective_margin = max(margin, data_range * 0.1, 5.0)  # At least 5 units margin
    
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

# ============= Coordinate Bounds from User Input =============
# Get user-defined coordinate bounds (these define valid coordinate range)
# These are used as fallback, but we'll compute actual visualization bounds from the data
COORD_MIN_X = float(st.session_state.get("coord_min_x", -50.0))
COORD_MAX_X = float(st.session_state.get("coord_max_x", 150.0))
COORD_MIN_Y = float(st.session_state.get("coord_min_y", -50.0))
COORD_MAX_Y = float(st.session_state.get("coord_max_y", 150.0))

# ============= Compute XLIM/YLIM from ACTUAL plotted data =============
# This ensures the visualization always shows the actual data points, regardless of bounds settings
def _compute_viz_bounds_from_data() -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute visualization bounds from the actual plotted data points.
    
    Adds a reasonable margin around the data points.
    """
    # Gather all points from all objects
    all_pts_arrays = [pts for pts in all_points_plot.values() if pts.shape[0] > 0]
    
    # Also include external reference points if any
    if external_pts_for_window:
        ext_arr = np.array(external_pts_for_window)
        if ext_arr.shape[0] > 0:
            all_pts_arrays.append(ext_arr)
    
    if not all_pts_arrays:
        # No data - use coordinate bounds as fallback
        return ((COORD_MIN_X, COORD_MAX_X), (COORD_MIN_Y, COORD_MAX_Y))
    
    all_pts = np.vstack(all_pts_arrays)
    data_min_x = float(np.min(all_pts[:, 0]))
    data_max_x = float(np.max(all_pts[:, 0]))
    data_min_y = float(np.min(all_pts[:, 1]))
    data_max_y = float(np.max(all_pts[:, 1]))
    
    # Add 10% margin (original behavior - no buffer margin at module load)
    data_range_x = data_max_x - data_min_x
    data_range_y = data_max_y - data_min_y
    
    # Ensure minimum range of 1.0 to avoid division by zero
    data_range_x = max(data_range_x, 1.0)
    data_range_y = max(data_range_y, 1.0)
    
    margin_x = max(data_range_x * 0.10, 1.0)  # At least 1 unit margin
    margin_y = max(data_range_y * 0.10, 1.0)
    
    viz_min_x = data_min_x - margin_x
    viz_max_x = data_max_x + margin_x
    viz_min_y = data_min_y - margin_y
    viz_max_y = data_max_y + margin_y
    
    # Make it square
    total_width = viz_max_x - viz_min_x
    total_height = viz_max_y - viz_min_y
    viz_side = max(total_width, total_height)
    
    coord_cx = 0.5 * (viz_min_x + viz_max_x)
    coord_cy = 0.5 * (viz_min_y + viz_max_y)
    
    xlim = (coord_cx - viz_side / 2.0, coord_cx + viz_side / 2.0)
    ylim = (coord_cy - viz_side / 2.0, coord_cy + viz_side / 2.0)
    
    return (xlim, ylim)

# Compute actual visualization bounds from the data
XLIM, YLIM = _compute_viz_bounds_from_data()
logger.debug(f"[BOUNDS COMPUTED] XLIM={XLIM}, YLIM={YLIM}")
logger.debug(f"[BOUNDS COMPUTED] all_points_plot.keys()={list(all_points_plot.keys())}")
for _dbg_oid in all_points_plot:
    logger.debug(f"[BOUNDS COMPUTED] o_id={_dbg_oid}, first_pt={all_points_plot[_dbg_oid][0] if all_points_plot[_dbg_oid].shape[0] > 0 else 'EMPTY'}")

# ============= d1/d2 order strings (LaTeX) ============
def _format_t_subscript(tval: float) -> str:
    """Format t-value as an integer subscript if possible, otherwise as a float."""
    try:
        tnum = float(tval)
    except (TypeError, ValueError):
        tnum = float(np.array(tval, dtype=float))
    return str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"

def _get_frenet_coordinates_for_ordering() -> Optional[dict[int, np.ndarray]]:
    """
    Get Frenet coordinates (s, n) for all points if on a curved road config.
    Returns dict mapping object_id -> (N, 2) array of [s, n] coordinates.
    Returns None if not a curved road config or if Frenet transform fails.
    """
    # Use the module-level selected_c_int which holds the current config number
    current_config = selected_c_int
    
    # Only apply Frenet for curved road configs
    # Use module-level CURVED_ROAD_CONFIGS
    if current_config not in CURVED_ROAD_CONFIGS:
        return None
    
    # Get the centerline
    try:
        centerline = _extract_centerline_from_data(current_config)
        if centerline is None or centerline.shape[0] < 2:
            logger.warning(f"[FRENET] Centerline invalid for config {current_config}")
            return None
        
        # Create Frenet frame
        frenet_frame = FrenetFrame(centerline)
        
        # Convert all points to Frenet coordinates
        frenet_coords: dict[int, np.ndarray] = {}
        for o_id in all_points_plot.keys():
            pts = all_points_plot[o_id]
            if pts.shape[0] > 0:
                frenet_pts = frenet_frame.to_frenet(pts)
                frenet_coords[o_id] = frenet_pts
        
        logger.debug(f"[FRENET] Successfully computed Frenet coords for {len(frenet_coords)} objects in config {current_config}")
        return frenet_coords
    except Exception as e:
        logger.warning(f"[FRENET] Exception: {e}")
        traceback.print_exc()
        return None


def make_d1_order_latex() -> str:
    """Return LaTeX describing the ordering in d1 (s-coordinate / x-coordinate) for all objects.
    For curved roads, uses Frenet s (arc length along road).
    For straight roads, uses Cartesian x-coordinate.
    """
    # Try to get Frenet coordinates for curved roads
    frenet_coords = _get_frenet_coordinates_for_ordering()
    
    entries: list[tuple[float, str]] = []
    for i, o_id in enumerate(sorted(all_points_plot.keys())):
        pts = all_points_plot[o_id]
        ts = all_vals_plot[o_id]
        label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
        
        # Use Frenet s-coordinate if available, otherwise Cartesian x
        if frenet_coords is not None and o_id in frenet_coords:
            coords_d1 = frenet_coords[o_id][:, 0]  # s-coordinate
        else:
            coords_d1 = pts[:, 0]  # x-coordinate
        
        for val, t in zip(coords_d1.tolist(), ts.tolist()):  # type: ignore[misc]
            lbl = _format_t_subscript(t)
            entries.append((float(val), rf"{label}_{{{lbl}}}"))

    if not entries:
        return r"d_1:"

    entries.sort(key=lambda it: it[0])
    tol = 1e-9
    out = [entries[0][1]]
    for i in range(1, len(entries)):
        prev_val = entries[i - 1][0]
        cur_val = entries[i][0]
        connector = " = " if abs(cur_val - prev_val) <= tol else " < "
        out.append(connector + entries[i][1])
    return r"d_1: " + "".join(out)


def make_d2_order_latex() -> str:
    """Return LaTeX describing the ordering in d2 (n-coordinate / y-coordinate) for all objects.
    For curved roads, uses Frenet n (lateral offset from centerline).
    For straight roads, uses Cartesian y-coordinate.
    """
    # Try to get Frenet coordinates for curved roads
    frenet_coords = _get_frenet_coordinates_for_ordering()
    
    entries: list[tuple[float, str]] = []
    for i, o_id in enumerate(sorted(all_points_plot.keys())):
        pts = all_points_plot[o_id]
        ts = all_vals_plot[o_id]
        label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
        
        # Use Frenet n-coordinate if available, otherwise Cartesian y
        if frenet_coords is not None and o_id in frenet_coords:
            coords_d2 = frenet_coords[o_id][:, 1]  # n-coordinate (lateral offset)
        else:
            coords_d2 = pts[:, 1]  # y-coordinate
        
        for val, t in zip(coords_d2.tolist(), ts.tolist()):  # type: ignore[misc]
            lbl = _format_t_subscript(t)
            entries.append((float(val), rf"{label}_{{{lbl}}}"))

    if not entries:
        return r"d_2:"

    entries.sort(key=lambda it: it[0])
    tol = 1e-9
    out = [entries[0][1]]
    for i in range(1, len(entries)):
        prev_val = entries[i - 1][0]
        cur_val = entries[i][0]
        connector = " = " if abs(cur_val - prev_val) <= tol else " < "
        out.append(connector + entries[i][1])
    return r"d_2: " + "".join(out)

def make_d1_order_latex_generated() -> str:
    """
    Return LaTeX order for d1 including the latest generated points.
    Uses primes and * markers to indicate generations from parents.
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

    # Use n_total_points as the total number of original points
    base_idx = int(parent_idx)
    if parent_idx >= n_total_points:
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - n_total_points)
        if 0 <= sidx < len(succ_list):
            base_idx = int(succ_list[sidx]["original_parent_idx"])

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
        return "*"

    # Determine parent label using helper function
    _, _, parent_label = get_object_info_for_flat_idx(base_idx)
    parent_t = get_timestamp_for_flat_idx(base_idx)
    lbl_parent = _format_t_subscript(float(parent_t))
    current_gen_count = generation_counts.get(base_idx, 0)
    label_gen_count = current_gen_count + (0 if in_search else 1)
    parent_primes = _prime_str(label_gen_count)
    entries.append((float(gen_pt[0]), rf"{parent_label}{parent_primes}_{{{lbl_parent}}}"))

    # Track the latest generated point for each original index
    latest_generated: dict[int, np.ndarray] = {}
    for sp in successful_points:
        orig_idx = int(sp["original_parent_idx"])
        latest_generated[orig_idx] = sp["point"]

    # All original points with possible generated replacements
    for flat_idx in range(n_total_points):
        if flat_idx == base_idx:
            continue
        _, _, label = get_object_info_for_flat_idx(flat_idx)
        t = get_timestamp_for_flat_idx(flat_idx)
        pt = get_point_for_flat_idx(flat_idx)
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(flat_idx, 0)
        primes = _prime_str(gen_cnt)
        if flat_idx in latest_generated:
            entries.append((float(latest_generated[flat_idx][0]), rf"{label}{primes}_{{{lbl}}}"))
        else:
            entries.append((float(pt[0]), rf"{label}{primes}_{{{lbl}}}"))

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

def make_d2_order_latex_generated() -> str:
    """
    Same as make_d1_order_latex_generated but for d2 (y-coordinate).
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

    base_idx = int(parent_idx)
    if parent_idx >= n_total_points:
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - n_total_points)
        if 0 <= sidx < len(succ_list):
            base_idx = int(succ_list[sidx]["original_parent_idx"])

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

    # Determine parent label using helper function
    _, _, parent_label = get_object_info_for_flat_idx(base_idx)
    parent_t = get_timestamp_for_flat_idx(base_idx)
    lbl_parent = _format_t_subscript(float(parent_t))
    current_gen_count = generation_counts.get(base_idx, 0)
    label_gen_count = current_gen_count + (0 if in_search else 1)
    parent_primes = _prime_str(label_gen_count)
    entries.append((float(gen_pt[1]), rf"{parent_label}{parent_primes}_{{{lbl_parent}}}"))

    latest_generated: dict[int, np.ndarray] = {}
    for sp in successful_points:
        orig_idx = int(sp["original_parent_idx"])
        latest_generated[orig_idx] = sp["point"]

    # All original points with possible generated replacements
    for flat_idx in range(n_total_points):
        if flat_idx == base_idx:
            continue
        _, _, label = get_object_info_for_flat_idx(flat_idx)
        t = get_timestamp_for_flat_idx(flat_idx)
        pt = get_point_for_flat_idx(flat_idx)
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(flat_idx, 0)
        primes = _prime_str(gen_cnt)
        if flat_idx in latest_generated:
            entries.append((float(latest_generated[flat_idx][1]), rf"{label}{primes}_{{{lbl}}}"))
        else:
            entries.append((float(pt[1]), rf"{label}{primes}_{{{lbl}}}"))

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

# ===== Helpers for order comparison (now using PDP inequality matrices) =====
# ===== Helpers for order comparison imported from pdp_utils.order_comparison =====
# strip_primes, extract_order_string, check_pdp_match, check_pdp_match_detailed

def check_pdp_match_legacy(original_k: np.ndarray, original_l: np.ndarray, 
                          generated_k: np.ndarray, generated_l: np.ndarray,
                          pdp_variant: str = "fundamental",
                          buffer_x: float = DEFAULT_BUFFER_X,
                          buffer_y: float = DEFAULT_BUFFER_Y,
                          rough_x: float = 0.0,
                          rough_y: float = 0.0,
                          debug: bool = False) -> tuple[bool, bool]:
    """Legacy wrapper that combines k and l arrays."""
    original_points = np.vstack([original_k, original_l]) if original_k.size > 0 and original_l.size > 0 else (original_k if original_k.size > 0 else original_l)
    generated_points = np.vstack([generated_k, generated_l]) if generated_k.size > 0 and generated_l.size > 0 else (generated_k if generated_k.size > 0 else generated_l)
    return check_pdp_match(original_points, generated_points, pdp_variant, buffer_x, buffer_y, rough_x, rough_y, debug)

# ===== Legacy: keep order string functions for display purposes =====

# ===== Central helper to store order match (using PDP) =====
def update_order_match_flags() -> None:
    """
    Compute and store d1/d2 order match booleans in session_state using PDP inequality matrices.
    
    This function now uses the same PDP logic as N_PDP.py for consistency.
    Works with any number of objects (not just k and l).
    Supports multi-point selection: checks ALL n selected points together.
    Also stores inequality matrices for heat map visualization.
    """
    # Get all current candidate points (multi-point support)
    anim_generated_points = st.session_state.get("anim_generated_points", {})
    gen_pt = st.session_state.get("anim_generated_point", None)
    
    # Need at least some generated point to check
    if not anim_generated_points and gen_pt is None:
        st.session_state["order_match_d1"] = False
        st.session_state["order_match_d2"] = False
        st.session_state["pdp_detailed_results"] = None
        return
    
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    
    # Build generated configuration from all objects
    generated_points = all_pts_flat.copy()
    
    # Track the latest generated point for each original index
    latest_generated: dict[int, np.ndarray] = {}
    for sp in successful_points:
        orig_idx = int(sp["original_parent_idx"])
        latest_generated[orig_idx] = sp["point"]
    
    # CRITICAL: Add ALL current candidate points we're testing (multi-point support)!
    if anim_generated_points:
        for idx, pt in anim_generated_points.items():
            latest_generated[int(idx)] = np.array(pt)
    elif gen_pt is not None:
        # Fallback for single point (backwards compatibility)
        parent_idx = int(st.session_state.get("anim_parent_idx", 0))
        if parent_idx < n_total_points:
            current_original_parent_idx = parent_idx
        else:
            sidx = parent_idx - n_total_points
            if 0 <= sidx < len(successful_points):
                current_original_parent_idx = int(successful_points[sidx]["original_parent_idx"])
            else:
                current_original_parent_idx = 0
        latest_generated[current_original_parent_idx] = np.array(gen_pt)
    
    # Apply all generated points (including current candidates) to the configuration
    for flat_idx in range(n_total_points):
        if flat_idx in latest_generated:
            generated_points[flat_idx] = latest_generated[flat_idx]
    
    # Get PDP variant parameters from session_state
    pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
    pdp_variant = pdp_variants_list[0] if pdp_variants_list else "fundamental"
    buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
    buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    # Get match threshold from session_state
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    match_threshold = pct_threshold if mode == "Percentage" else 1.0
    max_mismatches_param = max_mismatch_val if mode == "Max mismatches" else None
    
    # Use detailed PDP check - handle Frenet variant specially
    if pdp_variant == "frenet":
        # Get centerline from lane polylines or extract from data
        centerline = st.session_state.get("frenet_centerline", None)
        if centerline is None:
            # Try to extract from lane polylines
            lane_polylines = st.session_state.get("lane_polylines")
            if lane_polylines and "centerline" in lane_polylines:
                centerline = lane_polylines["centerline"]
            else:
                # Fallback: extract from current config data
                centerline = _extract_centerline_from_data(current_config)
            if centerline is not None:
                st.session_state["frenet_centerline"] = centerline
        
        if centerline is not None and len(centerline) >= 2:
            detailed_results: dict[str, Any] = check_pdp_match_frenet_detailed(  # type: ignore[misc]
                all_pts_flat,
                generated_points,
                centerline=centerline,
                pdp_variant="fundamental",  # Base variant for Frenet
                buffer_s=buffer_x,  # Use buffer_x as buffer_s
                buffer_n=buffer_y,  # Use buffer_y as buffer_n
                rough_s=rough_x,    # Use rough_x as rough_s
                rough_n=rough_y,    # Use rough_y as rough_n
                match_threshold=match_threshold,
                max_mismatches=max_mismatches_param
            )
            # Map Frenet results to d1/d2 naming (s->d1, n->d2)
            detailed_results["d1_match"] = detailed_results.pop("s_match")
            detailed_results["d2_match"] = detailed_results.pop("n_match")
            detailed_results["d1_percentage"] = detailed_results.pop("s_percentage")
            detailed_results["d2_percentage"] = detailed_results.pop("n_percentage")
            detailed_results["d1_mismatches"] = detailed_results.pop("s_mismatches")
            detailed_results["d2_mismatches"] = detailed_results.pop("n_mismatches")
            detailed_results["original_d1_matrix"] = detailed_results.pop("original_s_matrix")
            detailed_results["original_d2_matrix"] = detailed_results.pop("original_n_matrix")
            detailed_results["generated_d1_matrix"] = detailed_results.pop("generated_s_matrix")
            detailed_results["generated_d2_matrix"] = detailed_results.pop("generated_n_matrix")
        else:
            # Fallback to fundamental if no centerline
            detailed_results: dict[str, Any] = check_pdp_match_detailed(
                all_pts_flat, generated_points,
                pdp_variant="fundamental",
                match_threshold=match_threshold,
                max_mismatches=max_mismatches_param
            )
    else:
        # Only pass buffer/rough parameters if variant actually uses them
        if pdp_variant in ["buffer", "bufferrough", "realistic"]:
            detailed_results: dict[str, Any] = check_pdp_match_detailed(
                all_pts_flat,
                generated_points,
                pdp_variant=pdp_variant,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
                rough_x=rough_x if pdp_variant in ["rough", "bufferrough"] else 0.0,
                rough_y=rough_y if pdp_variant in ["rough", "bufferrough", "realistic"] else 0.0,
                match_threshold=match_threshold,
                max_mismatches=max_mismatches_param
            )
        else:
            # For fundamental and rough variants, pass appropriate parameters
            detailed_results: dict[str, Any] = check_pdp_match_detailed(
                all_pts_flat,
                generated_points,
                pdp_variant=pdp_variant,
                buffer_x=0.0,
                buffer_y=0.0,
                rough_x=rough_x if pdp_variant == "rough" else 0.0,
                rough_y=rough_y if pdp_variant == "rough" else 0.0,
                match_threshold=match_threshold,
                max_mismatches=max_mismatches_param
            )
    
    # Apply single-object logic: accept if EITHER d1 OR d2 matches
    num_unique_objects = len(all_points_plot.keys()) if 'all_points_plot' in globals() else 2
    if num_unique_objects <= 1:
        # For single object: if either dimension matches, consider both as matching
        if detailed_results.get("d1_match", False) or detailed_results.get("d2_match", False):
            st.session_state["order_match_d1"] = True
            st.session_state["order_match_d2"] = True
        else:
            st.session_state["order_match_d1"] = False
            st.session_state["order_match_d2"] = False
    else:
        # For multiple objects: use original logic
        st.session_state["order_match_d1"] = detailed_results.get("d1_match", False)
        st.session_state["order_match_d2"] = detailed_results.get("d2_match", False)
    
    st.session_state["pdp_detailed_results"] = detailed_results

# ============= Helper: Get match threshold from session_state ============
def get_threshold_settings() -> tuple[str, float, int]:
    """Get the threshold settings from session_state.
    
    Returns:
        tuple of (mode, percentage, max_mismatches)
        - mode: 'Percentage' or 'Max mismatches'
        - percentage: float 0.0-1.0 (only used if mode='Percentage')
        - max_mismatches: int (only used if mode='Max mismatches')
    """
    mode = st.session_state.get("cfg_threshold_mode", "Percentage")
    pct = st.session_state.get("cfg_threshold_pct", 100) / 100.0
    abs_val = st.session_state.get("cfg_threshold_abs", 0)
    return mode, pct, abs_val

def get_match_threshold() -> float:
    """Get the match threshold as a percentage (for backward compatibility)."""
    mode, pct, _ = get_threshold_settings()
    if mode == "Percentage":
        return pct
    else:
        # For absolute mode, return 1.0 (strict) - use max_mismatches for actual checking
        return 1.0

def get_threshold_params() -> tuple[float, int | None]:
    """Get both threshold parameters for check_pdp_match.
    
    Returns:
        tuple of (match_threshold, max_mismatches)
        - For percentage mode: (percentage, None)
        - For absolute mode: (1.0, max_mismatches_value)
    """
    mode, pct, abs_val = get_threshold_settings()
    if mode == "Percentage":
        return pct, None
    else:
        return 1.0, abs_val

def check_threshold_match(d1_pct: float, d2_pct: float, total_cells: int, d1_mismatches: int = 0, d2_mismatches: int = 0) -> tuple[bool, bool]:
    """Check if the match meets the threshold criteria.
    
    Args:
        d1_pct: d1 match percentage (0.0-1.0)
        d2_pct: d2 match percentage (0.0-1.0)
        total_cells: total number of comparable cells in the matrix
        d1_mismatches: number of mismatching cells in d1
        d2_mismatches: number of mismatching cells in d2
    
    Returns:
        tuple of (d1_match, d2_match)
    """
    mode, pct_threshold, max_mismatches = get_threshold_settings()
    
    if mode == "Percentage":
        avg_pct = (d1_pct + d2_pct) / 2.0
        match = avg_pct >= pct_threshold
        return match, match
    else:
        # Absolute mode: check if total mismatches <= max_mismatches
        total_mismatches = d1_mismatches + d2_mismatches
        match = total_mismatches <= max_mismatches
        return match, match

# ============= Helper: Binary search iteration ============
def run_binary_iteration(
    current_points: np.ndarray,
    successful_points: list[SuccessfulPoint],
    pdp_variant: str,
    buffer_x: float,
    buffer_y: float,
    rough_x: float,
    rough_y: float,
    match_threshold: float = 1.0,
    max_mismatches: int | None = None,
    max_binary_steps: int = MAX_BINARY_SEARCH_STEPS
) -> tuple[list[SuccessfulPoint], bool]:
    """
    Run one iteration of multi-point generation using binary search strategy.
    
    Binary search strategy:
    1. Start with a point at distance maxdist from the parent point
    2. If PDP matches → save as ok_point, try to go further by adding delta
    3. If PDP doesn't match → compute midpoint between ok_point and current point
    4. Repeat for 7 steps, halving delta each time
    5. Final placement is at the last ok_point
    
    Returns:
        (new_successful_points, success): Updated list and whether iteration succeeded
    """
    # Select which points to move this iteration
    selected_indices = select_points_for_iteration()
    if not selected_indices:
        return successful_points, False
    
    # Generate initial movement vectors for all selected points
    base_distance = maxdist
    initial_vectors = generate_movement_vectors(selected_indices, base_distance)
    
    # Get parent positions for each selected index
    def get_parent_position(idx: int) -> np.ndarray:
        """Get the most recent position for a point (either from successful_points or original)."""
        for sp in reversed(successful_points):
            if int(sp["original_parent_idx"]) == idx:
                return sp["point"]
        if 0 <= idx < len(current_points):
            return current_points[idx]
        return np.array([0.0, 0.0])
    
    # Build configuration with additional candidate positions for PDP checking
    def build_config_with_candidates(candidate_positions: dict[int, np.ndarray]) -> np.ndarray:
        """Build configuration from original + successful + candidate positions."""
        config = current_points.copy()
        # Apply successful points
        latest_by_idx: dict[int, np.ndarray] = {}
        for sp in successful_points:
            orig_idx = int(sp["original_parent_idx"])
            latest_by_idx[orig_idx] = sp["point"]
        # Apply candidate positions
        for idx, pt in candidate_positions.items():
            latest_by_idx[idx] = pt
        # Update config
        for idx, pt in latest_by_idx.items():
            if 0 <= idx < len(config):
                config[idx] = pt
        return config
    
    # Initialize binary search state for each selected point
    # ok_points: last known good positions (start at parent)
    # deltas: current movement vectors
    ok_points: dict[int, np.ndarray] = {}
    deltas: dict[int, np.ndarray] = {}
    
    for idx in selected_indices:
        parent_pt = get_parent_position(idx)
        ok_points[idx] = parent_pt.copy()
        dx, dy = initial_vectors[idx]
        deltas[idx] = np.array([dx, dy])
    
    # Track if we've ever found a full match (for diagnostics)
    had_full_match = False
    diag_rows: list[dict[str, Any]] = st.session_state.get("diag_rows", [])
    
    # Binary search: 7 steps
    for binary_step in range(max_binary_steps):
        # Compute candidate positions: ok_point + delta for each point
        candidate_positions: dict[int, np.ndarray] = {}  # Explicit type for Pylance
        for idx in selected_indices:
            ok_pt = ok_points[idx]
            delta = deltas[idx]
            new_x = np.clip(ok_pt[0] + delta[0], COORD_MIN_X, COORD_MAX_X)
            new_y = np.clip(ok_pt[1] + delta[1], COORD_MIN_Y, COORD_MAX_Y)
            candidate_positions[idx] = np.array([new_x, new_y])
        
        # Build config with candidates and check PDP
        test_config = build_config_with_candidates(candidate_positions)
        
        same_d1, same_d2 = check_pdp_match(
            all_pts_flat,
            test_config,
            pdp_variant=pdp_variant,
            buffer_x=buffer_x,
            buffer_y=buffer_y,
            rough_x=rough_x,
            rough_y=rough_y,
            match_threshold=match_threshold,
            max_mismatches=max_mismatches
        )
        
        # Record diagnostic row
        delta_magnitude = np.linalg.norm(list(deltas.values())[0]) if deltas else 0
        diag_rows.append({
            "n": binary_step + 1,
            "order_match_d1": same_d1,
            "order_match_d2": same_d2,
            "D_before_update": delta_magnitude,
            "delta": delta_magnitude / 2 if not (same_d1 and same_d2) else delta_magnitude,
        })
        
        if same_d1 and same_d2:
            # Match! Update ok_points to current candidates
            had_full_match = True
            for idx in selected_indices:
                ok_points[idx] = candidate_positions[idx].copy()
            # Stop early if using a relaxed threshold (< 100%) or absolute mismatch mode
            # This ensures the algorithm stops at the first valid configuration
            # rather than continuing to search for a "better" match
            if match_threshold < 1.0 or max_mismatches is not None:
                break
        else:
            # No match: halve delta (binary search narrowing)
            for idx in selected_indices:
                deltas[idx] = deltas[idx] / 2.0
    
    # Store diagnostics
    st.session_state["diag_rows"] = diag_rows
    st.session_state["anim_had_full_match"] = had_full_match
    
    # Final placement: use the last ok_points, with damping applied
    iteration_num = len(successful_points) // max(1, len(selected_indices))
    for idx in selected_indices:
        final_pt = ok_points[idx]
        parent_pt = get_parent_position(idx)
        # Apply random damping factor to reduce distance from parent
        damped_pt = apply_damping_factor(parent_pt, final_pt)
        sp: SuccessfulPoint = {
            "point": damped_pt,
            "parent_idx": idx,
            "parent_point": parent_pt,
            "original_parent_idx": idx,
            "iteration": iteration_num,
        }
        successful_points.append(sp)
    
    # Record iteration summary
    iter_log: list[dict[str, Any]] = st.session_state.get("binary_iteration_summary", [])
    current_config = int(st.session_state.get("anim_current_config", 1))
    iter_log.append({
        "config": current_config,
        "iteration": iteration_num,
        "match_d1": had_full_match,
        "match_d2": had_full_match,
    })
    st.session_state["binary_iteration_summary"] = iter_log
    
    return successful_points, True


# ============= Helper: Binary generation (non-animated) ============
def generate_binary_multipoint() -> None:
    """
    Multi-point aware version of non-animated binary generation.
    
    Uses the 7-step binary search strategy for each iteration.
    Supports multi-point selection and multi-variant generation.
    """
    # Get parameters
    default_iterations = int(st.session_state.get("cfg_iterations", 3))
    default_num_configs = int(st.session_state.get("cfg_num_configs", 1))
    num_iterations = int(st.session_state.get("anim_max_iterations", default_iterations))
    num_configs = int(st.session_state.get("anim_num_configs", default_num_configs))
    
    pdp_variants_list = st.session_state.get("anim_pdp_variants_list", ["fundamental"])
    buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
    buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    # Get threshold settings
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    match_threshold = pct_threshold if mode == "Percentage" else 1.0
    max_mismatches_param = max_mismatch_val if mode == "Max mismatches" else None
    
    all_configs: list[dict[str, Any]] = []
    current_points: np.ndarray = all_pts_flat.copy()
    
    # Reset diagnostics
    st.session_state["diag_rows"] = []
    st.session_state["binary_iteration_summary"] = []
    
    # Initialize successful_points to avoid 'possibly unbound' warning
    successful_points: list[SuccessfulPoint] = []
    
    # Process each variant
    for variant_idx, pdp_variant in enumerate(pdp_variants_list):
        st.session_state["anim_current_variant_idx"] = variant_idx
        st.session_state["anim_current_variant"] = pdp_variant
        
        # Generate configurations for this variant
        for config_num in range(1, num_configs + 1):
            st.session_state["anim_current_config"] = config_num
            successful_points = []
            
            # Reset diagnostics for each configuration
            st.session_state["diag_rows"] = []
            
            # Run iterations for this configuration using binary search
            for iteration in range(num_iterations):
                st.session_state["anim_completed_iterations"] = iteration
                
                successful_points, _ = run_binary_iteration(
                    current_points=current_points,
                    successful_points=successful_points,
                    pdp_variant=pdp_variant,
                    buffer_x=buffer_x,
                    buffer_y=buffer_y,
                    rough_x=rough_x,
                    rough_y=rough_y,
                    match_threshold=match_threshold,
                    max_mismatches=max_mismatches_param
                )
            
            # Store this configuration
            for sp in successful_points:
                sp["config_num"] = config_num  # type: ignore
            
            all_configs.append({
                "config_num": config_num,
                "points": list(successful_points),
                "pdp_variant": pdp_variant
            })
            
            st.session_state["anim_successful_points"] = successful_points
    
    # Store all configurations
    st.session_state["anim_all_configs"] = all_configs
    st.session_state["anim_running"] = False
    st.session_state["anim_completed_iterations"] = num_iterations
    st.session_state["anim_binary_mode"] = True
    
    # CRITICAL: Store all_pts for LaTeX order display functions
    st.session_state["anim_all_pts"] = all_pts_flat.copy()
    
    # Store generated points for display
    if successful_points:
        last_point = successful_points[-1]
        st.session_state["anim_generated_point"] = last_point["point"]
        st.session_state["anim_parent_idx"] = last_point.get("parent_idx", 0)
        update_order_match_flags()
    
    # Rerun to update the UI
    st.rerun()


# ============= Helper: detect stagnation for early stopping ============
def _check_stagnation(successful_points: list[SuccessfulPoint], n_points_per_iter: int) -> bool:
    """Check if the latest iteration produced zero real movement.

    Returns True when every point moved less than EARLY_STOP_EPSILON from its
    parent position, meaning the search fell back to placing points at their
    parent locations (no valid PDP-preserving direction found).
    """
    if n_points_per_iter <= 0 or len(successful_points) < n_points_per_iter:
        return False
    for sp in successful_points[-n_points_per_iter:]:
        parent = sp.get("parent_point")
        point = sp.get("point")
        if parent is None or point is None:
            return False
        if np.linalg.norm(np.asarray(point) - np.asarray(parent)) > EARLY_STOP_EPSILON:
            return False  # at least one point moved → not stagnant
    return True


# ============= Helper: Multi-point generation iteration ============
def run_multipoint_iteration(
    current_points: np.ndarray,
    successful_points: list[SuccessfulPoint],
    pdp_variant: str,
    buffer_x: float,
    buffer_y: float,
    rough_x: float,
    rough_y: float,
    max_search_steps: int = 10,
    pdp_checker: Optional[IncrementalPDPChecker] = None,
) -> tuple[list[SuccessfulPoint], bool]:
    """
    Run one iteration of multi-point generation.
    
    Selects points based on selection mode, generates movement vectors,
    and uses exponential search (halving vectors on failure) until PDP matches.
    
    Returns:
        (new_successful_points, success): Updated list and whether iteration succeeded
    """
    # Select which points to move this iteration
    selected_indices = select_points_for_iteration()
    if not selected_indices:
        return successful_points, False
    
    # Generate initial movement vectors
    base_distance = maxdist
    movement_vectors = generate_movement_vectors(selected_indices, base_distance)
    
    # Build current configuration with already-accepted points
    def build_current_config(additional_positions: dict[int, np.ndarray] = {}) -> np.ndarray:
        """Build configuration from original + successful + additional positions."""
        config = current_points.copy()
        # Apply successful points
        latest_by_idx: dict[int, np.ndarray] = {}
        for sp in successful_points:
            orig_idx = int(sp["original_parent_idx"])
            latest_by_idx[orig_idx] = sp["point"]
        # Apply additional positions (candidate points)
        for idx, pt in additional_positions.items():
            latest_by_idx[idx] = pt
        # Update config
        for idx, pt in latest_by_idx.items():
            if 0 <= idx < len(config):
                config[idx] = pt
        return config
    
    # Exponential search: try, halve on failure, repeat
    current_scale = 1.0
    parent_positions: dict[int, np.ndarray] = {}  # Initialize before loop to avoid unbound warning
    for search_step in range(max_search_steps):
        # Scale vectors
        scaled_vectors = scale_movement_vectors(movement_vectors, current_scale)
        
        # Get parent points (either from successful_points or original)
        parent_positions: dict[int, np.ndarray] = {}
        for idx in selected_indices:
            # Find most recent position for this index
            latest_pos = None
            for sp in reversed(successful_points):
                if int(sp["original_parent_idx"]) == idx:
                    latest_pos = sp["point"]
                    break
            if latest_pos is None:
                latest_pos = current_points[idx] if 0 <= idx < len(current_points) else np.array([0.0, 0.0])
            parent_positions[idx] = latest_pos
        
        # Apply movements from parent positions
        candidate_positions: dict[int, np.ndarray] = {}
        for idx, (dx, dy) in scaled_vectors.items():
            parent_pt = parent_positions.get(idx, current_points[idx])
            new_x = np.clip(parent_pt[0] + dx, COORD_MIN_X, COORD_MAX_X)
            new_y = np.clip(parent_pt[1] + dy, COORD_MIN_Y, COORD_MAX_Y)
            candidate_positions[idx] = np.array([new_x, new_y])
        
        # --- PDP Check ---
        if pdp_checker is not None:
            # Incremental path: O(k*N) instead of O(N²)
            _pdp_saved = pdp_checker.save_state_for_indices(list(candidate_positions.keys()))
            pdp_checker.update_points(candidate_positions)
            same_d1, same_d2 = pdp_checker.check_match()
            success = same_d1 and same_d2
            if not success:
                pdp_checker.restore_saved(_pdp_saved)
        else:
            # Original path: build full config and check O(N²)
            test_config = build_current_config(candidate_positions)
            _thresh, _max_mm = get_threshold_params()
            if pdp_variant in ["buffer", "bufferrough", "realistic"]:
                same_d1, same_d2 = check_pdp_match(
                    all_pts_flat,
                    test_config,
                    pdp_variant=pdp_variant,
                    buffer_x=buffer_x,
                    buffer_y=buffer_y,
                    rough_x=rough_x if pdp_variant in ["rough", "bufferrough"] else 0.0,
                    rough_y=rough_y if pdp_variant in ["rough", "bufferrough", "realistic"] else 0.0,
                    match_threshold=_thresh,
                    max_mismatches=_max_mm
                )
            else:
                same_d1, same_d2 = check_pdp_match(
                    all_pts_flat,
                    test_config,
                    pdp_variant=pdp_variant,
                    buffer_x=0.0,
                    buffer_y=0.0,
                    rough_x=rough_x if pdp_variant == "rough" else 0.0,
                    rough_y=rough_y if pdp_variant == "rough" else 0.0,
                    match_threshold=_thresh,
                    max_mismatches=_max_mm
                )
            success = same_d1 and same_d2
        
        if success:
            # Success! Add all candidate points to successful_points (with damping applied)
            if pdp_checker is not None:
                # Undo candidate positions, apply damped positions instead
                pdp_checker.restore_saved(_pdp_saved)
            iteration_num = len(successful_points) // max(1, len(selected_indices))
            damped_map: dict[int, np.ndarray] = {}
            for idx, new_pt in candidate_positions.items():
                parent_pt = parent_positions[idx]
                # Apply random damping factor to reduce distance from parent
                damped_pt = apply_damping_factor(parent_pt, new_pt)
                damped_map[idx] = damped_pt
                sp: SuccessfulPoint = {
                    "point": damped_pt,
                    "parent_idx": idx,  # Original index used as parent
                    "parent_point": parent_pt,
                    "original_parent_idx": idx,
                    "iteration": iteration_num,
                }
                successful_points.append(sp)
            if pdp_checker is not None:
                pdp_checker.update_points(damped_map)
            return successful_points, True
        
        # PDP check failed - halve the vectors
        current_scale *= 0.5
        
        # Also try random angle perturbation for "Same direction" mode
        movement_direction = st.session_state.get("cfg_movement_direction", "Same direction")
        if movement_direction == "Same direction" and search_step > 0:
            # Perturb the shared angle slightly
            angle_perturbation = float(np.random.uniform(-0.3, 0.3))
            # Regenerate vectors with new angle
            old_angle = np.arctan2(list(movement_vectors.values())[0][1], list(movement_vectors.values())[0][0])
            new_angle = old_angle + angle_perturbation
            new_base_dist = base_distance * current_scale * 2  # *2 because we just halved
            dx = new_base_dist * np.cos(new_angle)
            dy = new_base_dist * np.sin(new_angle)
            movement_vectors = {idx: (dx, dy) for idx in selected_indices}
    
    # Max search steps reached without finding PDP match — report failure.
    # Do NOT append zero-movement "successful" points; let the caller retry
    # with a different random point/direction selection.
    logger.debug("[DEBUG RUN_MULTIPOINT] Max search steps (%d) reached — no PDP-preserving move found", max_search_steps)
    return successful_points, False


def generate_exp_multipoint() -> None:
    """
    Multi-point aware version of non-animated exponential generation.
    
    Supports:
    - Single point (default, original behavior)
    - Multiple random points
    - Consecutive time stamps
    
    With movement direction options:
    - Same direction (coherent)
    - Random directions (independent)
    """
    # Get parameters
    default_iterations = int(st.session_state.get("cfg_iterations", 3))
    default_num_configs = int(st.session_state.get("cfg_num_configs", 1))
    num_iterations = int(st.session_state.get("anim_max_iterations", default_iterations))
    num_configs = int(st.session_state.get("anim_num_configs", default_num_configs))
    
    pdp_variants_list = st.session_state.get("anim_pdp_variants_list", ["fundamental"])
    buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
    buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    all_configs: list[dict[str, Any]] = []
    current_points: np.ndarray = all_pts_flat.copy()
    
    # Show progress for large point sets
    n_total = len(all_pts_flat)
    total_iters = len(pdp_variants_list) * num_configs * num_iterations
    progress_bar = st.progress(0, text=f"Generating... ({n_total} points × {num_iterations} iterations)")
    completed = 0
    
    # Process each variant
    for variant_idx, pdp_variant in enumerate(pdp_variants_list):
        st.session_state["anim_current_variant_idx"] = variant_idx
        st.session_state["anim_current_variant"] = pdp_variant
        
        # Generate configurations for this variant
        for config_num in range(1, num_configs + 1):
            st.session_state["anim_current_config"] = config_num
            successful_points: list[SuccessfulPoint] = []
            
            # Run iterations for this configuration
            for iteration in range(num_iterations):
                st.session_state["anim_completed_iterations"] = iteration
                completed += 1
                progress_bar.progress(
                    completed / total_iters,
                    text=f"Config {config_num}/{num_configs} · Iteration {iteration + 1}/{num_iterations} ({n_total} points)"
                )
                
                successful_points, success = run_multipoint_iteration(
                    current_points=current_points,
                    successful_points=successful_points,
                    pdp_variant=pdp_variant,
                    buffer_x=buffer_x,
                    buffer_y=buffer_y,
                    rough_x=rough_x,
                    rough_y=rough_y
                )
            
            # Store this configuration
            for sp in successful_points:
                sp["config_num"] = config_num  # type: ignore
            
            all_configs.append({
                "config_num": config_num,
                "points": list(successful_points),
                "pdp_variant": pdp_variant
            })
            
            st.session_state["anim_successful_points"] = successful_points
    
    progress_bar.empty()
    
    # Store all configurations
    st.session_state["anim_all_configs"] = all_configs
    st.session_state["anim_running"] = False
    st.session_state["anim_completed_iterations"] = num_iterations
    
    # CRITICAL: Store all_pts for LaTeX order display functions
    st.session_state["anim_all_pts"] = all_pts_flat.copy()
    
    # CRITICAL: Store generated points for display and order match calculation
    # Use the last successful point as the generated point to enable display
    if successful_points:
        last_point = successful_points[-1]
        st.session_state["anim_generated_point"] = last_point["point"]
        st.session_state["anim_parent_idx"] = last_point.get("parent_idx", 0)
        st.session_state["anim_successful_points"] = successful_points
        
        # Calculate and store order match flags
        update_order_match_flags()
    
    # Rerun to update the UI (especially the Reset button state)
    # This ensures the Reset button becomes enabled after generation completes
    st.rerun()


# ============= Helper: generate_exp (non-animated exponential) ============
def generate_exp() -> None:
    """
    Non-animated version of the exponential strategy.

    It:
    - Copies the logic of the exponential branch used in the animation,
    - Uses the same parameters stored in st.session_state,
    - Uses the *current radio-button values* as defaults for the number
      of iterations and the number of configurations,
    - Runs everything in one go (no time.sleep here).
    """
    max_loops = 100000
    loops = 0

    # Use radio button values as sane defaults if state is missing
    default_iterations = int(st.session_state.get("cfg_iterations", 3))
    default_num_configs = int(st.session_state.get("cfg_num_configs", 1))

    while st.session_state.get("anim_running", False) and loops < max_loops:
        loops += 1

        # Build current generated configuration
        gen_pt = st.session_state.get("anim_generated_point", None)
        if gen_pt is None:
            break
            
        successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        
        # Get parent info for the current candidate point
        parent_idx = int(st.session_state.get("anim_parent_idx", 0))
        
        # Determine the original parent index for the current candidate
        if parent_idx < n_total_points:
            current_original_parent_idx = parent_idx
        else:
            # Parent is a previously generated point - find its original parent
            sidx = parent_idx - n_total_points
            if 0 <= sidx < len(successful_points):
                current_original_parent_idx = int(successful_points[sidx]["original_parent_idx"])
            else:
                current_original_parent_idx = 0
        
        # Construct current generated configuration using all points
        generated_points = all_pts_flat.copy()
        
        # Track the latest generated point for each original index
        latest_generated: dict[int, np.ndarray] = {}
        for sp in successful_points:
            orig_idx = int(sp["original_parent_idx"])
            latest_generated[orig_idx] = sp["point"]
        
        # CRITICAL: Add the current candidate point we're testing!
        latest_generated[current_original_parent_idx] = np.array(gen_pt)
        
        # Apply all generated points (including current candidate) to the configuration
        for flat_idx in range(n_total_points):
            if flat_idx in latest_generated:
                generated_points[flat_idx] = latest_generated[flat_idx]
        
        # Get PDP variant parameters from session_state
        # Use the current variant being processed
        pdp_variant = st.session_state.get("anim_current_variant", "fundamental")
        buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        rough_x = st.session_state.get("cfg_rough_x", 0.0)
        rough_y = st.session_state.get("cfg_rough_y", 0.0)
        
        # Use PDP inequality matrix comparison with selected variant
        # Get both threshold parameters
        _thresh, _max_mm = get_threshold_params()
        
        # Apply buffer/rough parameters only for variants that use them
        if pdp_variant in ["buffer", "bufferrough", "realistic"]:
            same_d1, same_d2 = check_pdp_match(
                all_pts_flat,
                generated_points,
                pdp_variant=pdp_variant,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
                rough_x=rough_x if pdp_variant in ["rough", "bufferrough"] else 0.0,
                rough_y=rough_y if pdp_variant in ["rough", "bufferrough", "realistic"] else 0.0,
                match_threshold=_thresh,
                max_mismatches=_max_mm
            )
        else:
            # For fundamental and rough variants
            same_d1, same_d2 = check_pdp_match(
                all_pts_flat,
                generated_points,
                pdp_variant=pdp_variant,
                buffer_x=0.0,
                buffer_y=0.0,
                rough_x=rough_x if pdp_variant == "rough" else 0.0,
                rough_y=rough_y if pdp_variant == "rough" else 0.0,
                match_threshold=_thresh,
                max_mismatches=_max_mm
            )

        completed_iterations = int(st.session_state.get("anim_completed_iterations", 0))
        max_iterations = int(st.session_state.get("anim_max_iterations", default_iterations))
        search_steps = int(st.session_state.get("anim_search_steps", 0))
        max_search_steps = 7

        distance = float(st.session_state.get("anim_distance", maxdist))
        angle = float(st.session_state.get("anim_angle", 0.0))
        gen_pt = st.session_state.get("anim_generated_point", None)
        parent_idx = int(st.session_state.get("anim_parent_idx", 0))
        all_pts = st.session_state.get("anim_all_pts", np.array([]))
        successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        in_search = bool(st.session_state.get("anim_in_search", True))

        # For PDP, BOTH d1 AND d2 must match (regardless of number of objects)
        orders_match = same_d1 and same_d2 and gen_pt is not None

        # === Case 1: success (orders match) or distance collapsed to 0 ===
        if orders_match or (distance <= 0.0 and gen_pt is not None):
            if all_pts.size > 0 and parent_idx < n_total_points:
                parent_point_val = all_pts[parent_idx]
                original_parent_idx_val = parent_idx
            else:
                succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                sidx = int(parent_idx - n_total_points)
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

            # <<< hier: order match updaten voor deze plaatsing >>>
            update_order_match_flags()

            # Check if we finished all iterations for this configuration
            if completed_iterations + 1 >= max_iterations:
                current_config = int(st.session_state.get("anim_current_config", 1))
                num_configs = int(st.session_state.get("anim_num_configs", default_num_configs))

                # Get current PDP variant being processed
                current_variant = st.session_state.get("anim_current_variant", "fundamental")

                # Store this finished configuration with variant info
                all_configs: list[dict[str, Any]] = st.session_state.get("anim_all_configs", [])
                all_configs.append({
                    "config_num": current_config,
                    "points": list(successful_points),
                    "pdp_variant": current_variant  # Add variant info
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
                    
                    # CRITICAL: Reset successful_points for the new configuration
                    # so each config generates its own unique set of points
                    st.session_state["anim_successful_points"] = []

                    all_pts_reset = all_pts_flat.copy()
                    all_indices_reset = get_movable_indices()  # Only movable points
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
                        parent_idx_reset = n_total_points + youngest_success_idx_reset
                    else:
                        parent_idx_reset = chosen_idx_reset
                        parent_pt_reset = all_pts_reset[parent_idx_reset]

                    distance_new = maxdist
                    max_attempts = MAX_RESET_ATTEMPTS
                    angle_local: float = 0.0  # Initialize to avoid unbound warning
                    new_x: float = parent_pt_reset[0]  # Initialize to avoid unbound warning
                    new_y: float = parent_pt_reset[1]  # Initialize to avoid unbound warning
                    for _ in range(max_attempts):
                        angle_local = float(np.random.uniform(0, 2 * np.pi))
                        new_x = parent_pt_reset[0] + distance_new * np.cos(angle_local)
                        new_y = parent_pt_reset[1] + distance_new * np.sin(angle_local)
                        if COORD_MIN_X <= new_x <= COORD_MAX_X and COORD_MIN_Y <= new_y <= COORD_MAX_Y:
                            break
                    else:
                        new_x = np.clip(new_x, COORD_MIN_X, COORD_MAX_X)
                        new_y = np.clip(new_y, COORD_MIN_Y, COORD_MAX_Y)
                    new_gen_pt = np.array([new_x, new_y])

                    st.session_state["anim_parent_idx"] = parent_idx_reset
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = distance_new
                    st.session_state["anim_all_pts"] = all_pts_reset
                    st.session_state["anim_config_complete_wait"] = False
                    # Sync multi-point data for single-point mode consistency
                    st.session_state["anim_selected_indices"] = [int(parent_idx_reset)]
                    st.session_state["anim_generated_points"] = {int(parent_idx_reset): new_gen_pt}
                    st.session_state["anim_movement_vectors"] = {}
                else:
                    # All configurations for current variant completed
                    # Check if there are more variants to process
                    pdp_variants_list = st.session_state.get("anim_pdp_variants_list", ["fundamental"])
                    current_variant_idx = st.session_state.get("anim_current_variant_idx", 0)
                    
                    if current_variant_idx + 1 < len(pdp_variants_list):
                        # Move to next variant
                        next_variant_idx = current_variant_idx + 1
                        next_variant = pdp_variants_list[next_variant_idx]
                        
                        st.session_state["anim_current_variant_idx"] = next_variant_idx
                        st.session_state["anim_current_variant"] = next_variant
                        st.session_state["anim_current_config"] = 1
                        st.session_state["anim_completed_iterations"] = 0
                        st.session_state["anim_search_steps"] = 0
                        st.session_state["anim_running"] = True
                        st.session_state["anim_successful_points"] = []  # Reset for new variant
                        
                        # Initialize first point for new variant
                        all_pts_reset = all_pts_flat.copy()
                        all_indices_reset = get_movable_indices()  # Only movable points
                        if all_indices_reset:
                            chosen_idx_reset = int(np.random.choice(all_indices_reset))
                        else:
                            chosen_idx_reset = 0
                        
                        parent_idx_reset = chosen_idx_reset
                        parent_pt_reset = all_pts_reset[parent_idx_reset]
                        
                        distance_new = maxdist
                        max_attempts = MAX_RESET_ATTEMPTS
                        angle_local: float = 0.0  # Initialize to avoid unbound warning
                        new_x: float = parent_pt_reset[0]  # Initialize to avoid unbound warning
                        new_y: float = parent_pt_reset[1]  # Initialize to avoid unbound warning
                        for _ in range(max_attempts):
                            angle_local = float(np.random.uniform(0, 2 * np.pi))
                            new_x = parent_pt_reset[0] + distance_new * np.cos(angle_local)
                            new_y = parent_pt_reset[1] + distance_new * np.sin(angle_local)
                            if COORD_MIN_X <= new_x <= COORD_MAX_X and COORD_MIN_Y <= new_y <= COORD_MAX_Y:
                                break
                        else:
                            new_x = np.clip(new_x, COORD_MIN_X, COORD_MAX_X)
                            new_y = np.clip(new_y, COORD_MIN_Y, COORD_MAX_Y)
                        new_gen_pt = np.array([new_x, new_y])
                        
                        st.session_state["anim_parent_idx"] = parent_idx_reset
                        st.session_state["anim_angle"] = angle_local
                        st.session_state["anim_generated_point"] = new_gen_pt
                        st.session_state["anim_distance"] = distance_new
                        st.session_state["anim_all_pts"] = all_pts_reset
                        st.session_state["anim_config_complete_wait"] = False
                        # Sync multi-point data for single-point mode consistency
                        st.session_state["anim_selected_indices"] = [int(parent_idx_reset)]
                        st.session_state["anim_generated_points"] = {int(parent_idx_reset): new_gen_pt}
                        st.session_state["anim_movement_vectors"] = {}
                    else:
                        # All variants completed
                        st.session_state["anim_running"] = False
            else:
                # Prepare the next iteration for the same configuration
                all_indices = get_movable_indices()  # Only movable points
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
                    parent_idx_new = n_total_points + youngest_success_idx
                else:
                    parent_pt_new = get_point_for_flat_idx(chosen_idx)
                    parent_idx_new = chosen_idx

                distance_new = maxdist
                max_attempts = MAX_RESET_ATTEMPTS
                angle_local: float = 0.0  # Initialize to avoid unbound warning
                new_x: float = parent_pt_new[0]  # Initialize to avoid unbound warning
                new_y: float = parent_pt_new[1]  # Initialize to avoid unbound warning
                for _ in range(max_attempts):
                    angle_local = float(np.random.uniform(0, 2 * np.pi))
                    new_x = parent_pt_new[0] + distance_new * np.cos(angle_local)
                    new_y = parent_pt_new[1] + distance_new * np.sin(angle_local)
                    if COORD_MIN_X <= new_x <= COORD_MAX_X and COORD_MIN_Y <= new_y <= COORD_MAX_Y:
                        break
                else:
                    new_x = np.clip(new_x, COORD_MIN_X, COORD_MAX_X)
                    new_y = np.clip(new_y, COORD_MIN_Y, COORD_MAX_Y)
                new_gen_pt = np.array([new_x, new_y])

                st.session_state["anim_parent_idx"] = parent_idx_new
                st.session_state["anim_angle"] = angle_local
                st.session_state["anim_generated_point"] = new_gen_pt
                st.session_state["anim_distance"] = distance_new
                # Sync multi-point data for single-point mode consistency
                st.session_state["anim_selected_indices"] = [int(parent_idx_new)]
                st.session_state["anim_generated_points"] = {int(parent_idx_new): new_gen_pt}
                st.session_state["anim_movement_vectors"] = {}
        else:
            # === Case 2: keep searching (halve radius etc.) ===
            search_steps += 1
            st.session_state["anim_search_steps"] = search_steps

            if search_steps >= max_search_steps:
                # If search did not converge, snap back to parent
                if gen_pt is not None and all_pts.size > 0:
                    if parent_idx < n_total_points:
                        parent_pt_cur = all_pts[parent_idx]
                    else:
                        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                        sidx = int(parent_idx - n_total_points)
                        if 0 <= sidx < len(succ_list):
                            parent_pt_cur = succ_list[sidx]["point"]
                        else:
                            parent_pt_cur = np.array([0.0, 0.0])

                    st.session_state["anim_generated_point"] = parent_pt_cur.copy()
                    st.session_state["anim_distance"] = 0.0
                    st.session_state["anim_in_search"] = True
                    # Also clear multi-point data and sync selected_indices
                    st.session_state["anim_selected_indices"] = [int(parent_idx)]
                    st.session_state["anim_generated_points"] = {int(parent_idx): parent_pt_cur.copy()}
                    st.session_state["anim_movement_vectors"] = {}
            else:
                # Standard exponential search step: halve distance, tweak angle
                if gen_pt is not None and all_pts.size > 0:
                    if parent_idx < n_total_points:
                        parent_pt_cur = all_pts[parent_idx]
                    else:
                        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                        sidx = int(parent_idx - n_total_points)
                        if 0 <= sidx < len(succ_list):
                            parent_pt_cur = succ_list[sidx]["point"]
                        else:
                            parent_pt_cur = np.array([0.0, 0.0])

                    new_distance = distance / 2.0
                    min_distance = 1e-5
                    angle_local: float = angle  # Initialize with current angle
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

                    # Keep candidate inside coordinate bounds if possible
                    if not (COORD_MIN_X <= new_x <= COORD_MAX_X and COORD_MIN_Y <= new_y <= COORD_MAX_Y):
                        angle_local = (angle_local + np.pi) % (2 * np.pi)
                        new_x = parent_pt_cur[0] + new_distance * np.cos(angle_local)
                        new_y = parent_pt_cur[1] + new_distance * np.sin(angle_local)
                        new_gen_pt = np.array([new_x, new_y])

                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = new_distance
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_in_search"] = True
                    
                    # Update multi-point data: scale movement vectors by 0.5 and recalculate generated points
                    movement_vectors = st.session_state.get("anim_movement_vectors", {})
                    selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])
                    if movement_vectors:
                        # Scale all movement vectors by 0.5
                        scaled_vectors = scale_movement_vectors(movement_vectors, 0.5)
                        st.session_state["anim_movement_vectors"] = scaled_vectors
                        
                        # Recalculate generated points with scaled vectors
                        new_generated_points = {}
                        for idx in selected_indices:
                            if idx in scaled_vectors:
                                dx, dy = scaled_vectors[idx]
                                if idx < len(all_pts):
                                    base_pt = all_pts[idx]
                                else:
                                    base_pt = np.array([0.0, 0.0])
                                gx = base_pt[0] + dx
                                gy = base_pt[1] + dy
                                # Clip to bounds
                                gx = np.clip(gx, COORD_MIN_X, COORD_MAX_X)
                                gy = np.clip(gy, COORD_MIN_Y, COORD_MAX_Y)
                                new_generated_points[idx] = np.array([gx, gy])
                        st.session_state["anim_generated_points"] = new_generated_points

# ============= Animate button handler ============
if animate_btn:
    # Reset search diagnostics for a fresh animation run
    st.session_state["anim_delta"] = None
    
    # Clear animation history and redo stack when starting a new animation (for "Previous/Next" functionality)
    st.session_state["anim_state_history"] = []
    st.session_state["anim_state_redo"] = []
    
    # Use the same "Number of configurations" setting as batch generation
    num_anim_configs_val = int(num_configs)
    
    # Store animation mode (auto or manual modes)
    anim_mode_val = st.session_state.get("cfg_anim_mode", "Auto-advance")
    st.session_state["anim_manual_mode"] = anim_mode_val in ["Manual step-by-step", "Manual iteration-by-iteration", "Manual config-by-config"]
    st.session_state["anim_manual_step_mode"] = (anim_mode_val == "Manual step-by-step")
    st.session_state["anim_manual_iteration_mode"] = (anim_mode_val == "Manual iteration-by-iteration")
    st.session_state["anim_manual_config_mode"] = (anim_mode_val == "Manual config-by-config")

    if strategy == "exponential":
        num_configs_to_generate = num_anim_configs_val

        all_pts = all_pts_flat.copy()
        all_ts = all_ts_flat.copy()
        n_total = all_pts.shape[0]
        
        # Multi-point selection support
        selected_indices = select_points_for_iteration()
        if not selected_indices:
            movable_indices = get_movable_indices()
            selected_indices = [int(np.random.choice(movable_indices))] if movable_indices else [0]
        
        # For backwards compatibility, use first selected index as "parent_idx"
        parent_idx = selected_indices[0]
        parent_pt = all_pts[parent_idx]
        distance = maxdist
        
        # Generate movement vectors for all selected points
        movement_vectors = generate_movement_vectors(selected_indices, distance)
        
        # Calculate generated points for all selected indices and check if any is outside bounds
        # If outside bounds, try new random direction (like binary strategy does), not halving!
        max_direction_attempts = 10
        found_valid = False
        generated_points: dict[int, np.ndarray] = {}  # Initialize before loop
        for _ in range(max_direction_attempts):
            generated_points = {}
            all_within_bounds = True
            # Iterate directly over movement_vectors to ensure we use the correct keys
            for idx_int, (dx, dy) in movement_vectors.items():
                gen_x = all_pts[idx_int, 0] + dx
                gen_y = all_pts[idx_int, 1] + dy
                # Check if within bounds BEFORE clipping
                if not (XLIM[0] <= gen_x <= XLIM[1] and YLIM[0] <= gen_y <= YLIM[1]):
                    all_within_bounds = False
                # Clip to bounds for storage
                gen_x = np.clip(gen_x, XLIM[0], XLIM[1])
                gen_y = np.clip(gen_y, YLIM[0], YLIM[1])
                generated_points[idx_int] = np.array([gen_x, gen_y])
            
            if all_within_bounds:
                found_valid = True
                break
            # Generate new random directions (keep same maxdist distance!)
            movement_vectors = generate_movement_vectors(selected_indices, maxdist)
        
        if not found_valid:
            # If still out of bounds after max attempts, halve distance as fallback
            distance = maxdist / 2.0
            movement_vectors = generate_movement_vectors(selected_indices, distance)
            for idx_int, (dx, dy) in movement_vectors.items():
                gen_x = np.clip(all_pts[idx_int, 0] + dx, XLIM[0], XLIM[1])
                gen_y = np.clip(all_pts[idx_int, 1] + dy, YLIM[0], YLIM[1])
                generated_points[idx_int] = np.array([gen_x, gen_y])
        
        # For backwards compatibility, keep single generated_point as first one
        generated_point = generated_points.get(parent_idx, all_pts[parent_idx].copy())
        alfa = np.arctan2(generated_point[1] - parent_pt[1], generated_point[0] - parent_pt[0])

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
        st.session_state["anim_linear_mode"] = False
        st.session_state["anim_binary_step"] = 0
        st.session_state["diag_rows"] = []
        st.session_state["binary_iteration_summary"] = []
        st.session_state["anim_had_full_match"] = False
        
        # Multi-point animation support - store all selected indices and their generated points
        # Ensure all keys are Python ints for consistent lookup
        st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
        st.session_state["anim_generated_points"] = {int(k): v for k, v in generated_points.items()}
        st.session_state["anim_movement_vectors"] = {int(k): v for k, v in movement_vectors.items()}
        
        # Multi-variant support
        pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        st.session_state["anim_pdp_variants_list"] = pdp_variants_list
        st.session_state["anim_current_variant_idx"] = 0
        st.session_state["anim_current_variant"] = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        
        # Rerun to update button states immediately
        st.rerun()

    elif strategy == "binary":
        logger.debug(f"[DEBUG INIT BINARY] strategy={strategy}, setting anim_binary_mode=True")
        num_configs_to_generate = num_anim_configs_val

        all_pts = all_pts_flat.copy()
        all_ts = all_ts_flat.copy()
        n_total = all_pts.shape[0]
        # Multi-point selection support
        selected_indices = select_points_for_iteration()
        if not selected_indices:
            movable_indices = get_movable_indices()
            selected_indices = [int(np.random.choice(movable_indices))] if movable_indices else [0]
        
        # For backwards compatibility, use first selected index as "parent_idx"
        parent_idx = selected_indices[0]
        parent_pt = all_pts[parent_idx]

        # =============================================================
        # BINARY STRATEGY (according to specification):
        # =============================================================
        # Init:
        #   a-f: Choose parent(s), randomize direction, place points at maxdist
        #        Test if all points are on graph (within bounds), retry up to 10x
        #   g: correct_order = parent coordinates (for each selected point)
        #   h: WAIT, then halve to 0.5×maxdist BEFORE first test
        #
        # Steps n=1 to 7:
        #   - Test current positions for order match (ALL n points together!)
        #   - WAIT
        #   - If match: correct_order = current positions
        #               new_distance = current_distance + 0.5^(n+1) × maxdist
        #   - If no match: new_distance = current_distance - 0.5^(n+1) × maxdist
        #   - Move points and circles to new_distance
        #
        # End:
        #   - WAIT
        #   - Place points at correct_order
        #   - Circle radius = distance(correct_order, parent)
        # =============================================================
        
        # Generate movement vectors for all selected points (like exponential does)
        movement_vectors = generate_movement_vectors(selected_indices, maxdist)
        
        logger.debug(f"[DEBUG BINARY INIT] selected_indices={selected_indices}, maxdist={maxdist}")
        for idx in selected_indices:
            vec = movement_vectors.get(idx, (0.0, 0.0))
            vec_mag = np.sqrt(vec[0]**2 + vec[1]**2)
            logger.debug(f"[DEBUG BINARY INIT] idx={idx}, parent={all_pts[idx]}, movement_vec={vec}, magnitude={vec_mag:.4f}")
        
        # Step a-f: Check if all points are within bounds, retry with new directions if not
        # Even if no perfect direction is found, use the best attempt (some points may be at bounds)
        max_direction_attempts = 50  # Increase attempts for multi-point mode
        found_valid = False
        best_generated_points: dict[int, np.ndarray] = {}
        best_movement_vectors = movement_vectors.copy()
        generated_points: dict[int, np.ndarray] = {}  # Initialize before loop
        
        for attempt in range(max_direction_attempts):
            all_within_bounds = True
            generated_points = {}
            
            for idx in selected_indices:
                dx, dy = movement_vectors.get(idx, (0.0, 0.0))
                base_pt = all_pts[idx]
                candidate_x = base_pt[0] + dx
                candidate_y = base_pt[1] + dy
                
                # Check if within bounds
                if not (COORD_MIN_X <= candidate_x <= COORD_MAX_X and COORD_MIN_Y <= candidate_y <= COORD_MAX_Y):
                    all_within_bounds = False
                
                # Store unclipped position for visualization (points at maxdist)
                generated_points[idx] = np.array([candidate_x, candidate_y])
            
            # Keep track of first attempt as fallback
            if attempt == 0:
                best_generated_points = {k: v.copy() for k, v in generated_points.items()}
                best_movement_vectors = {k: v for k, v in movement_vectors.items()}
            
            if all_within_bounds:
                found_valid = True
                best_generated_points = generated_points
                best_movement_vectors = movement_vectors.copy()
                break
            
            # Regenerate movement vectors with new random directions
            movement_vectors = generate_movement_vectors(selected_indices, maxdist)
        
        # Use the best attempt (either valid or first attempt)
        generated_points = best_generated_points
        movement_vectors = best_movement_vectors
        
        # DEBUG: Verify distances
        logger.debug(f"[DEBUG BINARY INIT] found_valid={found_valid}")
        for idx in selected_indices:
            parent_pt_dbg = all_pts[idx]
            gen_pt_dbg = generated_points.get(idx, parent_pt_dbg)
            actual_dist = np.linalg.norm(gen_pt_dbg - parent_pt_dbg)
            logger.debug(f"[DEBUG BINARY INIT] idx={idx}, parent={parent_pt_dbg}, generated={gen_pt_dbg}, actual_distance={actual_dist:.4f}")
        
        # ALWAYS start at maxdist, even if some points may be outside bounds
        # The points will be visually shown at their calculated positions
        current_distance = maxdist
        if not found_valid:
            logger.warning(f"[DEBUG BINARY] No perfect direction found after {max_direction_attempts} attempts, using best attempt at maxdist")
        
        # Step g: Initialize correct_order with parent coordinates for all selected points
        correct_orders: dict[int, np.ndarray] = {idx: all_pts[idx].copy() for idx in selected_indices}
        
        # For backwards compatibility, single point values
        generated_point = generated_points.get(parent_idx, all_pts[parent_idx].copy())
        correct_order = correct_orders.get(parent_idx, parent_pt.copy())
        alfa = np.arctan2(
            generated_point[1] - parent_pt[1],
            generated_point[0] - parent_pt[0]
        ) if np.linalg.norm(generated_point - parent_pt) > 1e-9 else 0.0
        direction = np.array([np.cos(alfa), np.sin(alfa)])

        st.session_state["show_anim_circle"] = True
        st.session_state["anim_running"] = True
        st.session_state["anim_circle_idx"] = int(parent_idx)
        st.session_state["anim_distance"] = current_distance  # Circle radius = current distance
        st.session_state["anim_generated_point"] = generated_point
        st.session_state["anim_parent_idx"] = int(parent_idx)
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
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_current_config"] = 1
        st.session_state["anim_all_configs"] = []
        st.session_state["anim_search_steps"] = 0

        # Binary search state (new specification):
        st.session_state["anim_binary_mode"] = True
        st.session_state["anim_linear_mode"] = False  # Not linear
        st.session_state["anim_binary_step"] = 0  # 0 = showing maxdist, will halve first
        st.session_state["anim_binary_direction"] = direction.copy()  # Unit vector (for first point, backwards compat)
        st.session_state["anim_binary_current_distance"] = current_distance  # Current distance from parent
        st.session_state["anim_binary_correct_order"] = correct_order.copy()  # Last good position (first point)
        st.session_state["anim_binary_correct_orders"] = {int(k): v.copy() for k, v in correct_orders.items()}  # Multi-point
        st.session_state["anim_binary_initialized"] = False  # Will halve to 0.5×maxdist first
        st.session_state["diag_rows"] = []
        st.session_state["binary_iteration_summary"] = []
        st.session_state["anim_had_full_match"] = False
        
        # Legacy state (for compatibility with circle drawing)
        st.session_state["anim_delta"] = current_distance
        st.session_state["anim_ok_point"] = correct_order.copy()
        
        # Multi-point animation support
        st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
        st.session_state["anim_generated_points"] = {int(k): v for k, v in generated_points.items()}
        st.session_state["anim_movement_vectors"] = {int(k): v for k, v in movement_vectors.items()}
        
        # Multi-variant support
        pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        st.session_state["anim_pdp_variants_list"] = pdp_variants_list
        st.session_state["anim_current_variant_idx"] = 0
        st.session_state["anim_current_variant"] = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        
        # Rerun to update button states immediately
        st.rerun()

    elif strategy == "linear":
        # ============= LINEAR SEARCH STRATEGY INITIALIZATION =============
        # Same as binary but decreases by 0.1×maxdist per step instead of binary search
        logger.debug(f"[DEBUG INIT LINEAR] strategy={strategy}, setting anim_linear_mode=True")
        num_configs_to_generate = num_anim_configs_val

        all_pts = all_pts_flat.copy()
        all_ts = all_ts_flat.copy()
        n_total = all_pts.shape[0]
        
        # Multi-point selection support
        selected_indices = select_points_for_iteration()
        if not selected_indices:
            movable_indices = get_movable_indices()
            selected_indices = [int(np.random.choice(movable_indices))] if movable_indices else [0]
        
        # For backwards compatibility, use first selected index as "parent_idx"
        parent_idx = selected_indices[0]
        parent_pt = all_pts[parent_idx]

        # Use global maxdist (already calculated from point distances)
        # Same as exponential and binary strategies
        
        # Generate movement vectors for all selected points
        movement_vectors = generate_movement_vectors(selected_indices, maxdist)
        
        # Generate initial positions at maxdist for all selected points
        max_direction_attempts = 50
        found_valid = False
        best_generated_points: dict[int, np.ndarray] = {}
        best_movement_vectors: dict[int, tuple[float, float]] = {}
        generated_points: dict[int, np.ndarray] = {}  # Initialize before loop
        
        for attempt in range(max_direction_attempts):
            all_within_bounds = True
            generated_points = {}
            
            for idx in selected_indices:
                parent_pt_idx = all_pts[idx]
                mv = movement_vectors.get(idx, (0.0, 0.0))
                candidate_x = parent_pt_idx[0] + mv[0]
                candidate_y = parent_pt_idx[1] + mv[1]
                
                # Check if within bounds
                if not (COORD_MIN_X <= candidate_x <= COORD_MAX_X and COORD_MIN_Y <= candidate_y <= COORD_MAX_Y):
                    all_within_bounds = False
                
                generated_points[idx] = np.array([candidate_x, candidate_y])
            
            if attempt == 0:
                best_generated_points = {k: v.copy() for k, v in generated_points.items()}
                best_movement_vectors = {k: v for k, v in movement_vectors.items()}
            
            if all_within_bounds:
                found_valid = True
                best_generated_points = generated_points
                best_movement_vectors = movement_vectors.copy()
                break
            
            movement_vectors = generate_movement_vectors(selected_indices, maxdist)
        
        generated_points = best_generated_points
        movement_vectors = best_movement_vectors
        
        # DEBUG: Verify distances
        logger.debug(f"[DEBUG LINEAR INIT] found_valid={found_valid}")
        for idx in selected_indices:
            parent_pt_dbg = all_pts[idx]
            gen_pt_dbg = generated_points.get(idx, parent_pt_dbg)
            actual_dist = np.linalg.norm(gen_pt_dbg - parent_pt_dbg)
            logger.debug(f"[DEBUG LINEAR INIT] idx={idx}, parent={parent_pt_dbg}, generated={gen_pt_dbg}, actual_distance={actual_dist:.4f}")
        
        # Start at maxdist
        current_distance = maxdist
        
        # Initialize correct_order with parent coordinates for all selected points
        correct_orders: dict[int, np.ndarray] = {idx: all_pts[idx].copy() for idx in selected_indices}
        
        # For backwards compatibility
        generated_point = generated_points.get(parent_idx, all_pts[parent_idx].copy())
        correct_order = correct_orders.get(parent_idx, parent_pt.copy())
        alfa = np.arctan2(
            generated_point[1] - parent_pt[1],
            generated_point[0] - parent_pt[0]
        ) if np.linalg.norm(generated_point - parent_pt) > 1e-9 else 0.0
        direction = np.array([np.cos(alfa), np.sin(alfa)])

        st.session_state["show_anim_circle"] = True
        st.session_state["anim_running"] = True
        st.session_state["anim_circle_idx"] = int(parent_idx)
        st.session_state["anim_distance"] = current_distance
        st.session_state["anim_generated_point"] = generated_point
        st.session_state["anim_parent_idx"] = int(parent_idx)
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
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_current_config"] = 1
        st.session_state["anim_all_configs"] = []
        st.session_state["anim_search_steps"] = 0

        # Linear search state
        st.session_state["anim_binary_mode"] = False  # Not binary
        st.session_state["anim_linear_mode"] = True   # Linear mode
        st.session_state["anim_linear_step"] = 0      # Step counter
        st.session_state["anim_linear_current_distance"] = current_distance
        st.session_state["anim_linear_maxdist"] = maxdist
        st.session_state["anim_linear_step_size"] = maxdist * 0.1  # 10% of maxdist per step
        st.session_state["anim_binary_correct_order"] = correct_order.copy()
        st.session_state["anim_binary_correct_orders"] = {int(k): v.copy() for k, v in correct_orders.items()}
        st.session_state["anim_had_full_match"] = False
        st.session_state["diag_rows"] = []
        st.session_state["binary_iteration_summary"] = []
        
        # Multi-point animation support
        st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
        st.session_state["anim_generated_points"] = {int(k): v for k, v in generated_points.items()}
        st.session_state["anim_movement_vectors"] = {int(k): v for k, v in movement_vectors.items()}
        
        # Multi-variant support
        pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        st.session_state["anim_pdp_variants_list"] = pdp_variants_list
        st.session_state["anim_current_variant_idx"] = 0
        st.session_state["anim_current_variant"] = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        
        # Rerun to update button states immediately
        st.rerun()


# ============= Generate button handler (non-animated exponential) ============
if generate_btn:
    # Reset state for a fresh non-animated generation run
    st.session_state["anim_all_configs"] = []
    st.session_state["anim_successful_points"] = []
    st.session_state["anim_completed_iterations"] = 0
    st.session_state["anim_current_config"] = 1
    st.session_state["anim_search_steps"] = 0
    st.session_state["anim_binary_mode"] = False
    st.session_state["anim_linear_mode"] = False
    st.session_state["anim_binary_step"] = 0
    st.session_state["anim_delta"] = None
    st.session_state["diag_rows"] = []
    st.session_state["binary_iteration_summary"] = []
    
    # Multi-variant support - initialize variant tracking
    pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
    st.session_state["anim_pdp_variants_list"] = pdp_variants_list
    st.session_state["anim_current_variant_idx"] = 0
    st.session_state["anim_current_variant"] = pdp_variants_list[0] if pdp_variants_list else "fundamental"
    
    # Debug: Print the PDP configuration being used
    logger.debug(f"[DEBUG GENERATE] Selected variants: {pdp_variants_list}")
    logger.debug(f"[DEBUG GENERATE] Current variant: {st.session_state['anim_current_variant']}")
    logger.debug(f"[DEBUG GENERATE] buffer_x: {st.session_state.get('cfg_buffer_x', 'NOT SET')}")
    logger.debug(f"[DEBUG GENERATE] buffer_y: {st.session_state.get('cfg_buffer_y', 'NOT SET')}")
    logger.debug(f"[DEBUG GENERATE] rough_x: {st.session_state.get('cfg_rough_x', 'NOT SET')}")
    logger.debug(f"[DEBUG GENERATE] rough_y: {st.session_state.get('cfg_rough_y', 'NOT SET')}")
    logger.debug(f"[DEBUG GENERATE] Strategy selected: '{strategy}'")
    logger.debug(f"[DEBUG GENERATE] cfg_strategy from session_state: '{st.session_state.get('cfg_strategy', 'NOT SET')}'")

    if strategy == "exponential":
        # Set up parameters for multi-point generation
        st.session_state["anim_max_iterations"] = int(num_iterations)
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_running"] = True
        st.session_state["show_anim_circle"] = False  # no circle for generate
        
        # Debug: print point selection mode
        point_selection_mode = st.session_state.get("cfg_point_selection_mode", "Single point")
        movement_direction = st.session_state.get("cfg_movement_direction", "Same direction")
        logger.debug(f"[DEBUG GENERATE] Point selection mode: {point_selection_mode}")
        logger.debug(f"[DEBUG GENERATE] Movement direction: {movement_direction}")

        # Run the new multi-point aware generator
        generate_exp_multipoint()
    else:
        # Binary strategy
        st.session_state["anim_max_iterations"] = int(num_iterations)
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_running"] = True
        st.session_state["show_anim_circle"] = False  # no circle for generate
        
        # Debug: print point selection mode
        point_selection_mode = st.session_state.get("cfg_point_selection_mode", "Single point")
        movement_direction = st.session_state.get("cfg_movement_direction", "Same direction")
        logger.debug(f"[DEBUG GENERATE BINARY] Point selection mode: {point_selection_mode}")
        logger.debug(f"[DEBUG GENERATE BINARY] Movement direction: {movement_direction}")

        # Run the binary search generator
        generate_binary_multipoint()

# ============= Perpendicular variance helper ============
def _perpendicular_variance(
    all_points_plot: dict[int, np.ndarray],
    successful_points: list[dict[str, Any]],
) -> float:
    """Compute variance of perpendicular distances from generated points to the original path.

    For each object the original trajectory forms a polyline.  For every
    generated point the shortest (perpendicular) distance to that polyline is
    calculated.  The *variance* of all those distances (across all objects) is
    returned as a single scalar.

    Uses vectorised NumPy operations to avoid Python-level inner loops.
    """
    # Map global flat index → generated coordinate
    gen_map: dict[int, np.ndarray] = {}
    for sp in successful_points:
        gen_map[sp["original_parent_idx"]] = sp["point"]

    perp_dists: list[float] = []
    global_idx = 0
    for oid in sorted(all_points_plot.keys()):
        orig = all_points_plot[oid]
        n = orig.shape[0]

        # Collect generated points that belong to this object
        gen_pts_for_obj: list[np.ndarray] = []
        for li in range(n):
            gi = global_idx + li
            if gi in gen_map:
                gen_pts_for_obj.append(gen_map[gi])
        global_idx += n

        if not gen_pts_for_obj or n < 2:
            continue

        # Vectorised distance computation:
        #   gen_arr: (K, 2)  — generated points for this object
        #   A / B  : (M, 2)  — segment endpoints (M = n-1 segments)
        gen_arr = np.asarray(gen_pts_for_obj)          # (K, 2)
        A = orig[:-1]                                   # (M, 2)
        B = orig[1:]                                    # (M, 2)
        AB = B - A                                      # (M, 2)
        ab2 = np.sum(AB * AB, axis=1)                   # (M,)

        # AP[k, m, :] = gen_arr[k] - A[m]
        AP = gen_arr[:, None, :] - A[None, :, :]        # (K, M, 2)
        # Projection parameter t, clamped to [0, 1]
        t = np.sum(AP * AB[None, :, :], axis=2) / np.maximum(ab2[None, :], 1e-12)  # (K, M)
        t = np.clip(t, 0.0, 1.0)
        # Closest point on each segment
        closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]  # (K, M, 2)
        # Distance from each gen point to its closest point on each segment
        dists = np.linalg.norm(gen_arr[:, None, :] - closest, axis=2)  # (K, M)
        # Minimum across all segments
        min_dists = np.min(dists, axis=1)               # (K,)
        perp_dists.extend(min_dists.tolist())

    if len(perp_dists) < 2:
        return 0.0
    return float(np.var(perp_dists))

# ============= Advanced batch generation handlers ============
if generate_30_btn:
    # Store in session state that we want to generate
    st.session_state["_generate_30_requested"] = True

if generate_5000_btn:
    # Store in session state that we want to generate 5000
    st.session_state["_generate_5000_requested"] = True

if generate_50_btn:
    # Store in session state that we want to generate 50 configs (reinsertion)
    st.session_state["_generate_50_requested"] = True
    # Flag to apply reinsertion preset on next rerun (before widgets render)
    st.session_state["_reinsertion_preset_pending"] = True
    st.rerun()

if generate_ext30_btn:
    # Store in session state that we want to generate 100 configs x 2500 iterations
    st.session_state["_generate_ext30_requested"] = True
    # Ensure full timestamps: reset step to 1 and use all available timestamps
    _needs_rerun = False
    if int(st.session_state.get("_cfg_timestamp_step", 1)) != 1:
        st.session_state["_cfg_timestamp_step"] = 1
        _needs_rerun = True
    if int(st.session_state.get("cfg_k", 0)) != n_timepoints:
        st.session_state["_pending_cfg_k"] = n_timepoints
        _needs_rerun = True
    if _needs_rerun:
        st.rerun()

if generate_ext30_fe_btn:
    st.session_state["_generate_ext30_fe_requested"] = True
    _needs_rerun = False
    if int(st.session_state.get("_cfg_timestamp_step", 1)) != 1:
        st.session_state["_cfg_timestamp_step"] = 1
        _needs_rerun = True
    if int(st.session_state.get("cfg_k", 0)) != n_timepoints:
        st.session_state["_pending_cfg_k"] = n_timepoints
        _needs_rerun = True
    if _needs_rerun:
        st.rerun()

if generate_ext30_half_btn:
    # Same as ext30 but with step=2 (half timestamps)
    st.session_state["_generate_ext30_half_requested"] = True
    st.session_state["_ext30_half_preset_pending"] = True
    st.rerun()

if generate_c68_realistic_btn:
    st.session_state["_generate_c68r_requested"] = True
    st.session_state["_c68r_preset_pending"] = True
    st.rerun()

if generate_c68_fundamental_btn:
    st.session_state["_generate_c68f_requested"] = True
    st.session_state["_c68f_preset_pending"] = True
    st.rerun()

if generate_ext30_rough_btn:
    st.session_state["_generate_ext30_rough_requested"] = True
    _needs_rerun = False
    if int(st.session_state.get("_cfg_timestamp_step", 1)) != 1:
        st.session_state["_cfg_timestamp_step"] = 1
        _needs_rerun = True
    if int(st.session_state.get("cfg_k", 0)) != n_timepoints:
        st.session_state["_pending_cfg_k"] = n_timepoints
        _needs_rerun = True
    if _needs_rerun:
        st.rerun()

if generate_br_consec_btn:
    st.session_state["_generate_br_consec_requested"] = True
    _needs_rerun = False
    if int(st.session_state.get("_cfg_timestamp_step", 1)) != 1:
        st.session_state["_cfg_timestamp_step"] = 1
        _needs_rerun = True
    if int(st.session_state.get("cfg_k", 0)) != n_timepoints:
        st.session_state["_pending_cfg_k"] = n_timepoints
        _needs_rerun = True
    if _needs_rerun:
        st.rerun()

if generate_br_consec_rd_btn:
    st.session_state["_generate_br_consec_rd_requested"] = True
    _needs_rerun = False
    if int(st.session_state.get("_cfg_timestamp_step", 1)) != 1:
        st.session_state["_cfg_timestamp_step"] = 1
        _needs_rerun = True
    if int(st.session_state.get("cfg_k", 0)) != n_timepoints:
        st.session_state["_pending_cfg_k"] = n_timepoints
        _needs_rerun = True
    if _needs_rerun:
        st.rerun()

if generate_recursive_6event_btn:
    st.session_state["_generate_recursive_6event_requested"] = True
    st.session_state["_generate_recursive_6event_results"] = None

if generate_6ev_single_btn:
    st.session_state["_generate_6ev_single_requested"] = True
    st.session_state["_generate_6ev_single_results"] = None
    st.session_state.pop("_6evs_points_plot", None)
    st.session_state.pop("_6evs_vals_plot", None)

if generate_half_ts_btn:
    st.session_state["_generate_half_ts_requested"] = True
    st.session_state["_generate_half_ts_results"] = None
    st.session_state.pop("_hts_points_plot", None)
    st.session_state.pop("_hts_vals_plot", None)

if generate_quarter_ts_btn:
    st.session_state["_generate_quarter_ts_requested"] = True
    st.session_state["_generate_quarter_ts_results"] = None
    st.session_state.pop("_qts_points_plot", None)
    st.session_state.pop("_qts_vals_plot", None)

if generate_eighth_ts_btn:
    st.session_state["_generate_eighth_ts_requested"] = True
    st.session_state["_generate_eighth_ts_results"] = None
    st.session_state.pop("_ets_points_plot", None)
    st.session_state.pop("_ets_vals_plot", None)

if generate_sixteenth_ts_btn:
    st.session_state["_generate_sixteenth_ts_requested"] = True
    st.session_state["_generate_sixteenth_ts_results"] = None
    st.session_state.pop("_sts_points_plot", None)
    st.session_state.pop("_sts_vals_plot", None)

if generate_four_ts_btn:
    st.session_state["_generate_four_ts_requested"] = True
    st.session_state["_generate_four_ts_results"] = None
    st.session_state.pop("_fts_points_plot", None)
    st.session_state.pop("_fts_vals_plot", None)

if generate_two_ts_btn:
    st.session_state["_generate_two_ts_requested"] = True
    st.session_state["_generate_two_ts_results"] = None
    st.session_state.pop("_tts_points_plot", None)
    st.session_state.pop("_tts_vals_plot", None)

# Check if we have stored results or need to generate
if st.session_state.get("_generate_30_requested", False) and not st.session_state.get("_generate_30_results", None):
    st.markdown("---")
    st.markdown("### Generating 1000 Configurations...")
    st.caption("This may take several minutes. Progress is shown below.")
    
    # Store current settings
    current_iterations = int(st.session_state.get("cfg_iterations", 3))
    pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
    buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
    buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    # Get threshold settings
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
    
    # Generate 1000 configurations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_generated_configs: list[dict[str, Any]] = []
    
    for config_idx in range(MAX_GENERATION_ITERATIONS):
        status_text.text(f"Generating configuration {config_idx + 1}/{MAX_GENERATION_ITERATIONS}...")
        
        # Generate one configuration using the core logic
        current_points = all_pts_flat.copy()
        successful_points: list[SuccessfulPoint] = []
        
        # Use the first variant
        pdp_variant = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        
        # Run iterations
        for iteration in range(current_iterations):
            successful_points, success = run_multipoint_iteration(
                current_points=current_points,
                successful_points=successful_points,
                pdp_variant=pdp_variant,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
                rough_x=rough_x,
                rough_y=rough_y
            )
        
        # Store configuration
        if successful_points:
            config_data = {
                "successful_points": successful_points,
                "config_number": config_idx + 1,
                "pdp_variant": pdp_variant,
                "iterations": current_iterations,
                "buffer_x": buffer_x,
                "buffer_y": buffer_y,
                "rough_x": rough_x,
                "rough_y": rough_y,
                "threshold_mode": mode,
                "max_threshold": max_threshold
            }
            all_generated_configs.append(config_data)
        
        progress_bar.progress((config_idx + 1) / 1000)
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_30_requested"] = False
    else:
        st.success(f"Successfully generated {len(all_generated_configs)} configurations!")
        
        # Calculate perpendicular variance for each configuration
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        
        for config in all_generated_configs:
            successful_points = config.get("successful_points", [])
            pv = _perpendicular_variance(all_points_plot, successful_points)
            config_num = config.get("config_number", 0)
            deviations.append((config_num, pv, config))
        
        # Sort by perpendicular variance (descending) and take top 100
        deviations.sort(key=lambda x: x[1], reverse=True)
        top_100 = deviations[:100]
        
        # Store results in session state
        st.session_state["_generate_30_results"] = top_100
        st.rerun()

if st.session_state.get("_generate_5000_requested", False) and not st.session_state.get("_generate_5000_results", None):
    st.markdown("---")
    st.markdown("### Generating 5 Configurations...")
    st.caption("This may take several minutes. Progress is shown below.")
    
    # Store current settings (iterations forced to 50 for this button)
    current_iterations = 50
    pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
    buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
    buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    # Get threshold settings
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
    
    # Generate 5 configurations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_generated_configs: list[dict[str, Any]] = []
    st.session_state["_prefer_high_d2_change_sampling"] = False
    for config_idx in range(5):
        status_text.text(f"Generating configuration {config_idx + 1}/5 | iteration 0/{current_iterations}...")
        
        # Generate one configuration using the core logic
        current_points = all_pts_flat.copy()
        successful_points: list[SuccessfulPoint] = []
        
        # Use the first variant
        pdp_variant = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        
        # Run iterations
        for iteration in range(current_iterations):
            status_text.text(
                f"Generating configuration {config_idx + 1}/5 | iteration {iteration + 1}/{current_iterations}..."
            )
            successful_points, success = run_multipoint_iteration(
                current_points=current_points,
                successful_points=successful_points,
                pdp_variant=pdp_variant,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
                rough_x=rough_x,
                rough_y=rough_y
            )
        
        # Store configuration
        if successful_points:
            config_data = {
                "successful_points": successful_points,
                "config_number": config_idx + 1,
                "pdp_variant": pdp_variant,
                "iterations": current_iterations,
                "buffer_x": buffer_x,
                "buffer_y": buffer_y,
                "rough_x": rough_x,
                "rough_y": rough_y,
                "threshold_mode": mode,
                "max_threshold": max_threshold
            }
            all_generated_configs.append(config_data)
        
        progress_bar.progress((config_idx + 1) / 5)
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_5000_requested"] = False
    else:
        st.success(f"Successfully generated {len(all_generated_configs)} configurations!")
        
        # Calculate perpendicular variance for each configuration
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        
        for config in all_generated_configs:
            successful_points = config.get("successful_points", [])
            pv = _perpendicular_variance(all_points_plot, successful_points)
            config_num = config.get("config_number", 0)
            deviations.append((config_num, pv, config))
        
        # Sort by perpendicular variance (descending) and take top 500
        deviations.sort(key=lambda x: x[1], reverse=True)
        top_500 = deviations[:500]
        
        # Store results in session state
        st.session_state["_generate_5000_results"] = top_500
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: run full-timestamp batch generation (deduplicates ext30 / ext30_rough
#          / br_consec / br_consec_rd / ext30_fe blocks)
# ---------------------------------------------------------------------------
def _run_full_ts_generation(
    *,
    request_key: str,
    results_key: str,
    title: str,
    caption: str = "",
    status_prefix: str = "Generation",
    # Data construction
    all_objects_points: dict[int, tuple[np.ndarray, np.ndarray]],
    external_pts_for_window: np.ndarray,
    external_ts_for_window: np.ndarray,
    is_fixed_fn: Callable[[int, int, int], bool] | None = None,
    # PDP parameter overrides (None → read from sidebar session_state)
    pdp_variant_override: str | None = None,
    buffer_x_override: float | None = None,
    buffer_y_override: float | None = None,
    rough_x_override: float | None = None,
    rough_y_override: float | None = None,
    # Generation parameters
    num_configs: int = MAX_FILTER_CONFIGS,
    num_iterations: int = MAX_FILTER_ITERATIONS,
    top_n: int = 3,
    use_checker: bool = True,
    # Result storage keys (for filtered data that display sections need)
    full_points_key: str | None = None,
    full_vals_key: str | None = None,
    # Hooks for br_consec-style variants
    session_overrides: dict[str, Any] | None = None,
    per_config_hook: Callable[[int, list[int], dict[str, Any]], None] | None = None,
    cleanup_session_keys: list[str] | None = None,
    # Optional custom success message
    success_msg: str | None = None,
) -> None:
    """Generic batch generation using ALL timestamps from *all_objects_points*.

    Builds flattened data, swaps 8 globals, runs *num_configs* × *num_iterations*
    loop with optional ``IncrementalPDPChecker`` and early stopping, ranks results
    by perpendicular variance, and stores the top-*top_n* in session state.

    Parameters
    ----------
    is_fixed_fn : callable, optional
        ``(oid, local_idx, n_pts_for_obj) -> bool``.  Default: all ``False``.
    per_config_hook : callable, optional
        ``(config_idx, sorted_oids, hook_state) -> None``.  Called before each
        config's iteration loop.  *hook_state* is a dict that may contain:
        * ``"status_suffix"`` – appended to the status text
        * any other key – merged into the config result dict
    """
    global all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat
    global all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot

    if is_fixed_fn is None:
        is_fixed_fn = lambda _oid, _li, _n: False

    st.markdown("---")
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)

    # --- Step 1: Build full-timestamp data from all_objects_points (unfiltered) ---
    sorted_oids = sorted(all_objects_points.keys())
    full_points: dict[int, np.ndarray] = {}
    full_vals: dict[int, np.ndarray] = {}
    for oid in sorted_oids:
        pts, ts = all_objects_points[oid]
        full_points[oid] = pts
        full_vals[oid] = ts

    n_ts_per_obj = {oid: full_points[oid].shape[0] for oid in sorted_oids}
    n_ts_total = sum(n_ts_per_obj.values())
    ts_info = ", ".join(f"obj {oid}: {n} ts" for oid, n in n_ts_per_obj.items())

    # Count fixed endpoints for info message
    n_fixed = sum(
        sum(1 for li in range(n_ts_per_obj[oid]) if is_fixed_fn(oid, li, n_ts_per_obj[oid]))
        for oid in sorted_oids
    )
    if n_fixed > 0:
        st.info(
            f"Full timestamp data: {n_ts_total} total points ({ts_info}). "
            f"**{n_fixed} endpoints fixed** (first + last per object)."
        )
    else:
        st.info(f"Full timestamp data: {n_ts_total} total points ({ts_info})")

    # --- Step 2: Build flattened arrays ---
    pts_list: list[np.ndarray] = []
    ts_list: list[float] = []
    obj_ids: list[int] = []
    local_idx: list[int] = []
    is_fixed: list[bool] = []
    for oid in sorted_oids:
        n_pts = full_points[oid].shape[0]
        for li in range(n_pts):
            pts_list.append(full_points[oid][li])
            ts_list.append(float(full_vals[oid][li]))
            obj_ids.append(oid)
            local_idx.append(li)
            is_fixed.append(is_fixed_fn(oid, li, n_pts))
    for ext_pt, ext_t in zip(external_pts_for_window, external_ts_for_window):
        pts_list.append(ext_pt)
        ts_list.append(float(ext_t))
        obj_ids.append(-1)
        local_idx.append(len(pts_list) - 1)
        is_fixed.append(True)
    pts_flat = np.array(pts_list) if pts_list else np.array([]).reshape(0, 2)
    ts_flat = np.array(ts_list) if ts_list else np.array([])

    # --- Step 3: Temporarily swap global variables ---
    saved = (
        all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat,
        all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot,
    )
    all_pts_flat = pts_flat
    all_ts_flat = ts_flat
    all_obj_ids_flat = obj_ids
    all_local_idx_flat = local_idx
    all_is_fixed_flat = is_fixed
    n_total_points = pts_flat.shape[0]
    all_points_plot = full_points
    all_vals_plot = full_vals

    # Apply session overrides (e.g. consecutive-timestamp selection mode)
    if session_overrides:
        for k, v in session_overrides.items():
            st.session_state[k] = v

    try:
        # --- Resolve PDP parameters ---
        if pdp_variant_override is not None:
            pdp_variant = pdp_variant_override
        else:
            _variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
            pdp_variant = _variants_list[0] if _variants_list else "fundamental"
        buffer_x = buffer_x_override if buffer_x_override is not None else st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        buffer_y = buffer_y_override if buffer_y_override is not None else st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        rough_x = rough_x_override if rough_x_override is not None else st.session_state.get("cfg_rough_x", 0.0)
        rough_y = rough_y_override if rough_y_override is not None else st.session_state.get("cfg_rough_y", 0.0)
        mode, pct_threshold, max_mismatch_val = get_threshold_settings()
        max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val

        progress_bar = st.progress(0)
        status_text = st.empty()
        all_generated_configs: list[dict[str, Any]] = []

        # --- Optional incremental PDP checker ---
        checker = None
        if use_checker:
            _thresh_ck, _max_mm_ck = get_threshold_params()
            checker = IncrementalPDPChecker(
                all_pts_flat, pdp_variant,
                buffer_x=buffer_x, buffer_y=buffer_y,
                rough_x=rough_x, rough_y=rough_y,
                match_threshold=_thresh_ck, max_mismatches=_max_mm_ck,
            )

        _t_gen_start = time.perf_counter()
        _total_iters = 0
        _early_stops = 0

        for config_idx in range(num_configs):
            if checker:
                checker.reset_to_original()
            current_points = all_pts_flat.copy()
            successful_points: list[SuccessfulPoint] = []
            _stagnant = 0
            _iters_used = num_iterations

            # Per-config hook (e.g. br_consec sets random group size / object)
            hook_state: dict[str, Any] = {}
            if per_config_hook:
                per_config_hook(config_idx, sorted_oids, hook_state)
            _status_suffix = hook_state.pop("status_suffix", "")

            for iteration in range(num_iterations):
                _status = f"{status_prefix} — config {config_idx + 1}/{num_configs} | iter {iteration + 1}/{num_iterations}"
                if _status_suffix:
                    _status += f" {_status_suffix}"
                status_text.text(_status + "...")
                _sp_before = len(successful_points)
                successful_points, success = run_multipoint_iteration(
                    current_points=current_points,
                    successful_points=successful_points,
                    pdp_variant=pdp_variant,
                    buffer_x=buffer_x,
                    buffer_y=buffer_y,
                    rough_x=rough_x,
                    rough_y=rough_y,
                    pdp_checker=checker,
                )
                _n_added = len(successful_points) - _sp_before
                if _n_added > 0 and _check_stagnation(successful_points, _n_added):
                    _stagnant += 1
                else:
                    _stagnant = 0
                if _stagnant >= EARLY_STOP_PATIENCE:
                    _iters_used = iteration + 1
                    _early_stops += 1
                    break

            _total_iters += _iters_used
            if successful_points:
                config_entry: dict[str, Any] = {
                    "successful_points": successful_points,
                    "config_number": config_idx + 1,
                    "pdp_variant": pdp_variant,
                    "iterations": _iters_used,
                    "buffer_x": buffer_x,
                    "buffer_y": buffer_y,
                    "rough_x": rough_x,
                    "rough_y": rough_y,
                    "threshold_mode": mode,
                    "max_threshold": max_threshold,
                }
                # Merge any extra fields from per_config_hook
                config_entry.update(hook_state)
                all_generated_configs.append(config_entry)

            progress_bar.progress((config_idx + 1) / num_configs)

        progress_bar.empty()
        status_text.empty()
        _t_gen_elapsed = time.perf_counter() - _t_gen_start
        _max_possible = num_configs * num_iterations
        _saved_pct = (1 - _total_iters / max(1, _max_possible)) * 100
        st.info(
            f"⏱ Generation took {_t_gen_elapsed:.2f}s total "
            f"({_t_gen_elapsed / max(1, num_configs):.2f}s/config, "
            f"{_t_gen_elapsed / max(1, _total_iters) * 1000:.1f}ms/iter) | "
            f"Early stopped {_early_stops}/{num_configs} configs — "
            f"{_total_iters:,}/{_max_possible:,} iters used ({_saved_pct:.0f}% saved)"
        )

        if not all_generated_configs:
            st.error("No configurations were successfully generated.")
            st.session_state[request_key] = False
        else:
            _msg = success_msg or f"Successfully generated {len(all_generated_configs)} configurations!"
            st.success(_msg)

            deviations: list[tuple[int, float, dict[str, Any]]] = []
            for config in all_generated_configs:
                sp = config.get("successful_points", [])
                pv = _perpendicular_variance(all_points_plot, sp)
                deviations.append((config.get("config_number", 0), pv, config))

            deviations.sort(key=lambda x: x[1], reverse=True)
            top = deviations[:top_n]

            st.session_state[results_key] = top
            if full_points_key:
                st.session_state[full_points_key] = full_points
            if full_vals_key:
                st.session_state[full_vals_key] = full_vals
            st.rerun()
    finally:
        # --- Restore original globals ---
        (all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat,
         all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot) = saved
        # --- Remove override keys ---
        if cleanup_session_keys:
            for k in cleanup_session_keys:
                st.session_state.pop(k, None)


# ============= Generate 200 configs x 200 iterations (reinsertion) ============
if st.session_state.get("_generate_50_requested", False) and not st.session_state.get("_generate_50_results", None):
    st.markdown("---")
    st.markdown("### Generating 200 Configurations (reinsertion zone: t=131–138, 200 iterations each)...")
    st.caption("Using 8 timestamps from the reinsertion zone. This may take several minutes.")
    
    # Use current settings but force 200 iterations
    current_iterations = 200
    pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
    buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
    buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    # Get threshold settings
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
    
    # Generate 50 configurations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_generated_configs: list[dict[str, Any]] = []
    st.session_state["_prefer_high_d2_change_sampling"] = False
    for config_idx in range(200):
        status_text.text(f"Generating configuration {config_idx + 1}/200 | iteration 0/{current_iterations}...")
        
        # Generate one configuration using the core logic
        current_points = all_pts_flat.copy()
        successful_points: list[SuccessfulPoint] = []
        
        # Use the first variant
        pdp_variant = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        
        # Run iterations
        for iteration in range(current_iterations):
            status_text.text(
                f"Generating configuration {config_idx + 1}/200 | iteration {iteration + 1}/{current_iterations}..."
            )
            successful_points, success = run_multipoint_iteration(
                current_points=current_points,
                successful_points=successful_points,
                pdp_variant=pdp_variant,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
                rough_x=rough_x,
                rough_y=rough_y
            )
        
        # Store configuration
        if successful_points:
            config_data = {
                "successful_points": successful_points,
                "config_number": config_idx + 1,
                "pdp_variant": pdp_variant,
                "iterations": current_iterations,
                "buffer_x": buffer_x,
                "buffer_y": buffer_y,
                "rough_x": rough_x,
                "rough_y": rough_y,
                "threshold_mode": mode,
                "max_threshold": max_threshold
            }
            all_generated_configs.append(config_data)
        
        progress_bar.progress((config_idx + 1) / 200)
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_50_requested"] = False
    else:
        st.success(f"Successfully generated {len(all_generated_configs)} configurations!")
        
        # Calculate perpendicular variance for each configuration
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        
        for config in all_generated_configs:
            successful_points = config.get("successful_points", [])
            pv = _perpendicular_variance(all_points_plot, successful_points)
            config_num = config.get("config_number", 0)
            deviations.append((config_num, pv, config))
        
        # Sort by perpendicular variance (descending) and take top 25
        deviations.sort(key=lambda x: x[1], reverse=True)
        top_5 = deviations[:25]
        
        # Store results in session state
        st.session_state["_generate_50_results"] = top_5
        st.rerun()

# ============= Generate 100 configs × 2500 iterations (ext30) ============
if st.session_state.get("_generate_ext30_requested", False) and not st.session_state.get("_generate_ext30_results", None):
    _run_full_ts_generation(
        request_key="_generate_ext30_requested",
        results_key="_generate_ext30_results",
        title="Generating 100 Configurations (2500 iterations each)...",
        caption="Using ALL timestamps — building full-range dataset first.",
        status_prefix="ext30",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
        full_points_key="_ext30_full_points_plot",
        full_vals_key="_ext30_full_vals_plot",
    )

# ============= Generate 100 configs × 2500 iter — ROUGH d1=d2=0.30m (ext30_rough) ============
if st.session_state.get("_generate_ext30_rough_requested", False) and not st.session_state.get("_generate_ext30_rough_results", None):
    _run_full_ts_generation(
        request_key="_generate_ext30_rough_requested",
        results_key="_generate_ext30_rough_results",
        title="Generating 100 Configurations (2500 iter, rough d1=d2=0.30m)...",
        caption="Using ALL timestamps — rough d1=d2=0.30m geforceerd (ongeacht sidebar).",
        status_prefix="Rough d1=d2=0.30m",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
        pdp_variant_override="fundamental",
        buffer_x_override=0.0,
        buffer_y_override=0.0,
        rough_x_override=0.30,
        rough_y_override=0.30,
        full_points_key="_ext30_rough_full_points_plot",
        full_vals_key="_ext30_rough_full_vals_plot",
    )

# ============= Generate 100 configs × 2500 iter — BUFFERROUGH + CONSECUTIVE TS (br_consec) ============
_BRC_OVERRIDE_KEYS = [
    "_override_point_selection_mode", "_override_movement_direction",
    "_override_consecutive_object_id", "_override_consecutive_first_timestamp",
    "_override_group_num_timestamps",
]

def _br_consec_per_config(config_idx: int, sorted_oids: list[int], hook_state: dict[str, Any]) -> None:
    """Per-config hook for br_consec: random group size (2-3), cycle objects."""
    grp = int(np.random.choice([2, 3]))
    obj_id = sorted_oids[config_idx % len(sorted_oids)]
    st.session_state["_override_group_num_timestamps"] = grp
    st.session_state["_override_consecutive_object_id"] = obj_id
    hook_state["status_suffix"] = f"(obj {obj_id}, grp {grp})"
    hook_state["group_size"] = grp
    hook_state["object_id"] = obj_id

if st.session_state.get("_generate_br_consec_requested", False) and not st.session_state.get("_generate_br_consec_results", None):
    _run_full_ts_generation(
        request_key="_generate_br_consec_requested",
        results_key="_generate_br_consec_results",
        title="Generating 100 Configs (2500 iter, bufferrough, consecutive ts)...",
        caption="bufferrough (buffer x/y=1m, rough x/y=0.30m) | Consecutive timestamps (2–3) same direction | ALL timestamps",
        status_prefix="bufferrough consec",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
        pdp_variant_override="bufferrough",
        buffer_x_override=1.0,
        buffer_y_override=1.0,
        rough_x_override=0.30,
        rough_y_override=0.30,
        session_overrides={
            "_override_point_selection_mode": "Consecutive time stamps",
            "_override_movement_direction": "Same direction",
            "_override_consecutive_object_id": sorted(all_objects_points.keys())[0] if all_objects_points else 0,
            "_override_consecutive_first_timestamp": 0,
        },
        per_config_hook=_br_consec_per_config,
        cleanup_session_keys=_BRC_OVERRIDE_KEYS,
        full_points_key="_br_consec_full_points_plot",
        full_vals_key="_br_consec_full_vals_plot",
    )

# ============= Generate 100 configs × 2500 iter — BUFFERROUGH + CONSECUTIVE TS + RANDOM DIRS (br_consec_rd) ============
if st.session_state.get("_generate_br_consec_rd_requested", False) and not st.session_state.get("_generate_br_consec_rd_results", None):
    _run_full_ts_generation(
        request_key="_generate_br_consec_rd_requested",
        results_key="_generate_br_consec_rd_results",
        title="Generating 100 Configs (2500 iter, bufferrough, consec ts, random dirs)...",
        caption="bufferrough (buffer x/y=1m, rough x/y=0.30m) | Consecutive timestamps (2–3) RANDOM directions | ALL timestamps",
        status_prefix="bufferrough consec random dirs",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
        pdp_variant_override="bufferrough",
        buffer_x_override=1.0,
        buffer_y_override=1.0,
        rough_x_override=0.30,
        rough_y_override=0.30,
        session_overrides={
            "_override_point_selection_mode": "Consecutive time stamps",
            "_override_movement_direction": "Random directions",
            "_override_consecutive_object_id": sorted(all_objects_points.keys())[0] if all_objects_points else 0,
            "_override_consecutive_first_timestamp": 0,
        },
        per_config_hook=_br_consec_per_config,
        cleanup_session_keys=_BRC_OVERRIDE_KEYS,
        full_points_key="_br_consec_rd_full_points_plot",
        full_vals_key="_br_consec_rd_full_vals_plot",
    )

# ============= Generate 100 configs × 2500 iterations — FIXED ENDPOINTS (ext30_fe) ============
if st.session_state.get("_generate_ext30_fe_requested", False) and not st.session_state.get("_generate_ext30_fe_results", None):
    _run_full_ts_generation(
        request_key="_generate_ext30_fe_requested",
        results_key="_generate_ext30_fe_results",
        title="Generating 100 Configurations (2500 iter, fixed endpoints)...",
        caption="Eerste en laatste timestamp per object worden NIET verplaatst — ze fungeren als ankerpunten.",
        status_prefix="Fixed-EP",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
        is_fixed_fn=lambda _oid, _li, _n: (_li == 0 or _li == _n - 1),
        full_points_key="_ext30_fe_full_points_plot",
        full_vals_key="_ext30_fe_full_vals_plot",
        success_msg=None,  # uses default; original had custom text but functionally equivalent
    )

# ============= Generate 100 configs × 2500 iterations — half timestamps (ext30_half) ============
if st.session_state.get("_generate_ext30_half_requested", False) and not st.session_state.get("_generate_ext30_half_results", None):
    st.markdown("---")
    st.markdown("### Generating 100 Configurations (2500 iterations, ½ timestamps)...")
    st.caption("Zelfde als 'Generate 100 & Top 3' maar gebruikt om de andere timestamp (step=2).")

    pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
    buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
    buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
    _ext30h_iterations = 2500

    # Display number of timestamps after ½-filtering
    st.info(f"Aantal timestamps na filtering: {len(selected_ts_window)}")

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_generated_configs: list[dict[str, Any]] = []

    _ext30h_num_configs = MAX_FILTER_CONFIGS
    _t_gen_start = time.perf_counter()
    _total_iters = 0
    _early_stops = 0
    for config_idx in range(_ext30h_num_configs):
        current_points = all_pts_flat.copy()
        successful_points: list[SuccessfulPoint] = []
        pdp_variant = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        _stagnant = 0
        _iters_used = _ext30h_iterations

        for iteration in range(_ext30h_iterations):
            status_text.text(f"½-ts — config {config_idx + 1}/{_ext30h_num_configs} | iter {iteration + 1}/{_ext30h_iterations}")
            _sp_before = len(successful_points)
            successful_points, success = run_multipoint_iteration(
                current_points=current_points,
                successful_points=successful_points,
                pdp_variant=pdp_variant,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
                rough_x=rough_x,
                rough_y=rough_y,
            )
            _n_added = len(successful_points) - _sp_before
            if _n_added > 0 and _check_stagnation(successful_points, _n_added):
                _stagnant += 1
            else:
                _stagnant = 0
            if _stagnant >= EARLY_STOP_PATIENCE:
                _iters_used = iteration + 1
                _early_stops += 1
                break

        _total_iters += _iters_used
        if successful_points:
            all_generated_configs.append({
                "successful_points": successful_points,
                "config_number": config_idx + 1,
                "pdp_variant": pdp_variant,
                "iterations": _iters_used,
                "buffer_x": buffer_x,
                "buffer_y": buffer_y,
                "rough_x": rough_x,
                "rough_y": rough_y,
                "threshold_mode": mode,
                "max_threshold": max_threshold,
            })

        progress_bar.progress((config_idx + 1) / _ext30h_num_configs)

    progress_bar.empty()
    status_text.empty()
    _t_gen_elapsed = time.perf_counter() - _t_gen_start
    _max_possible = _ext30h_num_configs * _ext30h_iterations
    _saved_pct = (1 - _total_iters / max(1, _max_possible)) * 100
    st.info(
        f"⏱ Generation took {_t_gen_elapsed:.2f}s total "
        f"({_t_gen_elapsed / max(1, _ext30h_num_configs):.2f}s/config, "
        f"{_t_gen_elapsed / max(1, _total_iters) * 1000:.1f}ms/iter) | "
        f"Early stopped {_early_stops}/{_ext30h_num_configs} configs — "
        f"{_total_iters:,}/{_max_possible:,} iters used ({_saved_pct:.0f}% saved)"
    )
    # Reset timestamp step back to 1
    st.session_state["_cfg_timestamp_step"] = 1

    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_ext30_half_requested"] = False
    else:
        st.success(f"Successfully generated {len(all_generated_configs)} configurations!")

        deviations: list[tuple[int, float, dict[str, Any]]] = []
        for config in all_generated_configs:
            successful_points = config.get("successful_points", [])
            pv = _perpendicular_variance(all_points_plot, successful_points)
            config_num = config.get("config_number", 0)
            deviations.append((config_num, pv, config))

        deviations.sort(key=lambda x: x[1], reverse=True)
        top_3 = deviations[:3]

        st.session_state["_generate_ext30_half_results"] = top_3
        st.rerun()


# ============= Recursive 6-Event Overtake Generation ============
if st.session_state.get("_generate_recursive_6event_requested", False) and not st.session_state.get("_generate_recursive_6event_results", None):
    _6ev_detection = detect_overtake_events(all_objects_points)
    if _6ev_detection is None:
        st.error(
            "⚠️ Could not detect 6 overtake events — the trajectory data does not "
            "show a clear lane-change + return pattern. Make sure you are viewing "
            "an overtake scenario with at least 2 objects."
        )
        st.session_state["_generate_recursive_6event_requested"] = False
    else:
        _6ev_oid, _6ev_timestamps = _6ev_detection

        # --- Build 6-event filtered data for every object ---
        _6ev_ts_set = set(_6ev_timestamps)
        _6ev_sorted_oids = sorted(all_objects_points.keys())
        _6ev_points: dict[int, np.ndarray] = {}
        _6ev_vals: dict[int, np.ndarray] = {}
        for _oid in _6ev_sorted_oids:
            _pts, _ts = all_objects_points[_oid]
            # For each event timestamp find the closest available timestamp
            _keep_indices: list[int] = []
            for _evt_t in _6ev_timestamps:
                _closest_idx = int(np.argmin(np.abs(_ts - _evt_t)))
                if _closest_idx not in _keep_indices:
                    _keep_indices.append(_closest_idx)
            _keep_indices.sort()
            _6ev_points[_oid] = _pts[_keep_indices]
            _6ev_vals[_oid] = _ts[_keep_indices]

        # --- Show detection info ---
        st.markdown("---")
        st.markdown("### 🔄 Recursive 6-Event Overtake Generation")
        _ts_nice = ", ".join(f"{t:.0f}" for t in _6ev_timestamps)
        st.info(
            f"Overtaking vehicle: **object {_6ev_oid}** | "
            f"6 event timestamps: [{_ts_nice}]"
        )
        _n6_per = {_oid: _6ev_points[_oid].shape[0] for _oid in _6ev_sorted_oids}
        _n6_total = sum(_n6_per.values())
        st.caption(
            f"Points per object: {_n6_per} | total {_n6_total} points | "
            "Generating 1 config per round × 2500 iter, up to 20 recursive rounds"
        )

        # --- Save + swap globals (module-level: no `global` keyword needed) ---
        _6ev_saved = (
            all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat,
            all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot,
        )

        # Build flattened from the 6-event data
        def _build_6ev_flat(pts_dict: dict[int, np.ndarray], vals_dict: dict[int, np.ndarray]):
            _pl: list[np.ndarray] = []
            _tl: list[float] = []
            _oi: list[int] = []
            _li: list[int] = []
            _fi: list[bool] = []
            for _oid in _6ev_sorted_oids:
                _pts_d = pts_dict[_oid]
                for _idx in range(_pts_d.shape[0]):
                    _pl.append(_pts_d[_idx])
                    _tl.append(float(vals_dict[_oid][_idx]))
                    _oi.append(_oid)
                    _li.append(_idx)
                    _fi.append(False)
            # External points
            for _ep, _et in zip(external_pts_for_window, external_ts_for_window):
                _pl.append(_ep)
                _tl.append(float(_et))
                _oi.append(-1)
                _li.append(len(_pl) - 1)
                _fi.append(True)
            _pf = np.array(_pl) if _pl else np.array([]).reshape(0, 2)
            _tf = np.array(_tl) if _tl else np.array([])
            return _pf, _tf, _oi, _li, _fi

        # Resolve PDP parameters once
        _6ev_variants = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        _6ev_pdp = _6ev_variants[0] if _6ev_variants else "fundamental"
        _6ev_bx = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        _6ev_by = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        _6ev_rx = st.session_state.get("cfg_rough_x", 0.0)
        _6ev_ry = st.session_state.get("cfg_rough_y", 0.0)
        _6ev_mode, _6ev_pct, _6ev_maxmm = get_threshold_settings()

        # --- Recursive loop ---
        MAX_RECURSIVE_ROUNDS = 20
        NUM_CONFIGS_PER_ROUND = 1
        NUM_ITER_PER_CONFIG = MAX_FILTER_ITERATIONS

        _round_results: list[dict[str, Any]] = []
        _cur_points = {_oid: _6ev_points[_oid].copy() for _oid in _6ev_sorted_oids}
        _cur_vals = {_oid: _6ev_vals[_oid].copy() for _oid in _6ev_sorted_oids}

        # The ORIGINAL 6-event flat is needed to build the PDP checker reference
        _orig_flat, _orig_ts_flat, _, _, _ = _build_6ev_flat(_6ev_points, _6ev_vals)

        _progress = st.progress(0)
        _status = st.empty()
        _t_rec_start = time.perf_counter()
        _total_rec_iters = 0

        try:
          for _round_idx in range(MAX_RECURSIVE_ROUNDS):
            _status.text(f"Recursive round {_round_idx + 1}/{MAX_RECURSIVE_ROUNDS}...")
            # Build flat arrays from current positions
            _pf, _tf, _oi, _li, _fi = _build_6ev_flat(_cur_points, _cur_vals)

            # Swap globals
            all_pts_flat = _pf
            all_ts_flat = _tf
            all_obj_ids_flat = _oi
            all_local_idx_flat = _li
            all_is_fixed_flat = _fi
            n_total_points = _pf.shape[0]
            all_points_plot = _cur_points
            all_vals_plot = _cur_vals

            # PDP checker: compare generated configs against the ORIGINAL ordering
            _thresh_ck, _maxmm_ck = get_threshold_params()
            _checker = IncrementalPDPChecker(
                _orig_flat, _6ev_pdp,
                buffer_x=_6ev_bx, buffer_y=_6ev_by,
                rough_x=_6ev_rx, rough_y=_6ev_ry,
                match_threshold=_thresh_ck, max_mismatches=_maxmm_ck,
            )

            # Run a batch of configs starting from _cur_points
            _round_configs: list[dict[str, Any]] = []
            _round_early = 0
            for _ci in range(NUM_CONFIGS_PER_ROUND):
                _checker.reset_to_original()
                _cur_pts_copy = _pf.copy()
                _sp: list[SuccessfulPoint] = []
                _stag = 0
                _used = NUM_ITER_PER_CONFIG
                for _it in range(NUM_ITER_PER_CONFIG):
                    _status.text(
                        f"Round {_round_idx + 1}/{MAX_RECURSIVE_ROUNDS} — "
                        f"iter {_it + 1}/{NUM_ITER_PER_CONFIG}"
                    )
                    _sp_before = len(_sp)
                    _sp, _ok = run_multipoint_iteration(
                        current_points=_cur_pts_copy,
                        successful_points=_sp,
                        pdp_variant=_6ev_pdp,
                        buffer_x=_6ev_bx,
                        buffer_y=_6ev_by,
                        rough_x=_6ev_rx,
                        rough_y=_6ev_ry,
                        pdp_checker=_checker,
                    )
                    _na = len(_sp) - _sp_before
                    if _na > 0 and _check_stagnation(_sp, _na):
                        _stag += 1
                    else:
                        _stag = 0
                    if _stag >= EARLY_STOP_PATIENCE:
                        _used = _it + 1
                        _round_early += 1
                        break
                _total_rec_iters += _used
                if _sp:
                    _round_configs.append({
                        "successful_points": _sp,
                        "config_number": _ci + 1,
                        "iterations": _used,
                    })
                _progress.progress(
                    (_round_idx * NUM_CONFIGS_PER_ROUND + _ci + 1)
                    / (MAX_RECURSIVE_ROUNDS * NUM_CONFIGS_PER_ROUND)
                )

            if not _round_configs:
                st.warning(f"Round {_round_idx + 1}: no valid configs generated — stopping recursion.")
                break

            # Rank by perpendicular variance against the 6-event original
            _devs: list[tuple[int, float, dict[str, Any]]] = []
            for _cfg in _round_configs:
                _pv = _perpendicular_variance(_6ev_points, _cfg["successful_points"])
                _devs.append((_cfg["config_number"], _pv, _cfg))
            _devs.sort(key=lambda x: x[1], reverse=True)
            _best = _devs[0]
            _round_results.append({
                "round": _round_idx + 1,
                "best_config": _best[0],
                "best_variance": _best[1],
                "n_configs": len(_round_configs),
                "early_stopped": _round_early,
                "top3": _devs[:5],
            })

            # Reconstruct best config's final positions → new starting data
            _best_sp = _best[2]["successful_points"]
            _latest: dict[int, np.ndarray] = {}
            for _sp_entry in _best_sp:
                _latest[int(_sp_entry["original_parent_idx"])] = _sp_entry["point"]
            # Rebuild _cur_points from flat mapping
            _gi = 0
            for _oid in _6ev_sorted_oids:
                _n_obj = _cur_points[_oid].shape[0]
                for _idx_local in range(_n_obj):
                    if _gi in _latest:
                        _cur_points[_oid][_idx_local] = _latest[_gi]
                    _gi += 1

            st.info(
                f"Round {_round_idx + 1}: best variance = {_best[1]:.6f} "
                f"(config #{_best[0]}, {len(_round_configs)} valid configs, "
                f"{_round_early} early-stopped)"
            )

          # --- After all rounds ---
          _progress.empty()
          _status.empty()
          _t_rec_elapsed = time.perf_counter() - _t_rec_start
          st.success(
              f"✅ Recursive generation complete — {len(_round_results)} rounds in {_t_rec_elapsed:.1f}s "
              f"({_total_rec_iters:,} total iterations)"
          )

          # Store results: list of round summaries + final top 3 from last round
          if _round_results:
              _final_top3 = _round_results[-1]["top3"]
              st.session_state["_generate_recursive_6event_results"] = _final_top3
              st.session_state["_recursive_6event_rounds"] = _round_results
              st.session_state["_recursive_6event_points_plot"] = _6ev_points
              st.session_state["_recursive_6event_vals_plot"] = _6ev_vals
              st.session_state["_recursive_6event_timestamps"] = _6ev_timestamps
              st.session_state["_recursive_6event_overtaker"] = _6ev_oid
          else:
              st.session_state["_generate_recursive_6event_requested"] = False
        finally:
          # Restore globals no matter what
          (all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat,
           all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot) = _6ev_saved

        if _round_results:
            st.rerun()


# ---------------------------------------------------------------------------
# Helper: run filtered-timestamp generation (deduplicates ½/¼/⅛/1∕16 blocks)
# ---------------------------------------------------------------------------
def _run_filtered_ts_generation(
    *,
    step: int,
    label: str,
    session_prefix: str,
    state_prefix: str,
    all_objects_points: dict[int, tuple[np.ndarray, np.ndarray]],
    external_pts_for_window: np.ndarray,
    external_ts_for_window: np.ndarray,
    num_configs: int = MAX_FILTER_CONFIGS,
    num_iterations: int = MAX_FILTER_ITERATIONS,
) -> None:
    """Generate *num_configs* configurations using only every *step*-th timestamp.

    The last timestamp is always kept regardless of divisibility.
    Results (top-10 by perpendicular variance) are stored in
    ``st.session_state[f"_generate_{session_prefix}_ts_results"]``.
    Filtered data is stored at ``st.session_state[f"_{state_prefix}_points_plot"]``
    and ``st.session_state[f"_{state_prefix}_vals_plot"]``.
    """
    global all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat
    global all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot

    st.markdown("---")
    st.markdown(f"### Generating {num_configs} Configurations ({label} timestamps, filtered)...")

    # --- Step 1: Filter timestamps — keep every step-th + always keep the last ---
    sorted_oids = sorted(all_objects_points.keys())
    filt_points: dict[int, np.ndarray] = {}
    filt_vals: dict[int, np.ndarray] = {}

    for oid in sorted_oids:
        orig_pts, orig_ts = all_objects_points[oid]
        keep_mask = (orig_ts % step == 0)
        last_mask = np.zeros(len(orig_ts), dtype=bool)
        last_mask[-1] = True
        filt_points[oid] = orig_pts[keep_mask | last_mask]
        filt_vals[oid] = orig_ts[keep_mask | last_mask]

    # --- Step 2: Count effective timestamps ---
    n_ts_per_obj = {oid: filt_points[oid].shape[0] for oid in sorted_oids}
    n_ts_total = sum(n_ts_per_obj.values())
    ts_info = ", ".join(f"obj {oid}: {n} timestamps" for oid, n in n_ts_per_obj.items())
    st.caption(f"{num_configs} configs × {num_iterations} iter | exponential | PDP fundamental | {ts_info}")
    st.info(f"Aantal timestamps na filtering: {n_ts_total} totaal ({ts_info})")

    # --- Step 3: Build flattened points from filtered data ---
    pts_list: list[np.ndarray] = []
    ts_list: list[float] = []
    obj_ids_flat: list[int] = []
    local_idx_flat: list[int] = []
    is_fixed_flat: list[bool] = []

    for oid in sorted_oids:
        for li in range(filt_points[oid].shape[0]):
            pts_list.append(filt_points[oid][li])
            ts_list.append(float(filt_vals[oid][li]))
            obj_ids_flat.append(oid)
            local_idx_flat.append(li)
            is_fixed_flat.append(False)
    for ext_pt, ext_t in zip(external_pts_for_window, external_ts_for_window):
        pts_list.append(ext_pt)
        ts_list.append(float(ext_t))
        obj_ids_flat.append(-1)
        local_idx_flat.append(len(pts_list) - 1)
        is_fixed_flat.append(True)

    pts_flat = np.array(pts_list) if pts_list else np.array([]).reshape(0, 2)
    ts_flat = np.array(ts_list) if ts_list else np.array([])

    # --- Step 4: Temporarily swap global variables ---
    saved = (
        all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat,
        all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot,
    )
    all_pts_flat = pts_flat
    all_ts_flat = ts_flat
    all_obj_ids_flat = obj_ids_flat
    all_local_idx_flat = local_idx_flat
    all_is_fixed_flat = is_fixed_flat
    n_total_points = pts_flat.shape[0]
    all_points_plot = filt_points
    all_vals_plot = filt_vals

    try:
        pdp_variant = "fundamental"
        buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        rough_x = st.session_state.get("cfg_rough_x", 0.0)
        rough_y = st.session_state.get("cfg_rough_y", 0.0)
        mode, pct_threshold, max_mismatch_val = get_threshold_settings()
        max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val

        progress_bar = st.progress(0)
        status_text = st.empty()
        all_generated_configs: list[dict[str, Any]] = []

        # --- Create incremental PDP checker for O(k·N) checks ---
        _thresh_ck, _max_mm_ck = get_threshold_params()
        _filt_checker = IncrementalPDPChecker(
            pts_flat, pdp_variant,
            buffer_x=buffer_x, buffer_y=buffer_y,
            rough_x=rough_x, rough_y=rough_y,
            match_threshold=_thresh_ck, max_mismatches=_max_mm_ck,
        )
        _t_gen_start = time.perf_counter()
        _total_iters = 0
        _early_stops = 0

        for config_idx in range(num_configs):
            _filt_checker.reset_to_original()
            current_points = pts_flat.copy()
            successful_points: list[SuccessfulPoint] = []
            _stagnant = 0
            _iters_used = num_iterations
            for iteration in range(num_iterations):
                status_text.text(
                    f"{label}-ts filtered — config {config_idx + 1}/{num_configs} "
                    f"| iter {iteration + 1}/{num_iterations}"
                )
                _sp_before = len(successful_points)
                successful_points, success = run_multipoint_iteration(
                    current_points=current_points,
                    successful_points=successful_points,
                    pdp_variant=pdp_variant,
                    buffer_x=buffer_x,
                    buffer_y=buffer_y,
                    rough_x=rough_x,
                    rough_y=rough_y,
                    pdp_checker=_filt_checker,
                )
                _n_added = len(successful_points) - _sp_before
                if _n_added > 0 and _check_stagnation(successful_points, _n_added):
                    _stagnant += 1
                else:
                    _stagnant = 0
                if _stagnant >= EARLY_STOP_PATIENCE:
                    _iters_used = iteration + 1
                    _early_stops += 1
                    break
            _total_iters += _iters_used
            if successful_points:
                all_generated_configs.append({
                    "successful_points": successful_points,
                    "config_number": config_idx + 1,
                    "pdp_variant": pdp_variant,
                    "iterations": _iters_used,
                    "buffer_x": buffer_x, "buffer_y": buffer_y,
                    "rough_x": rough_x, "rough_y": rough_y,
                    "threshold_mode": mode, "max_threshold": max_threshold,
                })
            progress_bar.progress((config_idx + 1) / num_configs)

        progress_bar.empty()
        status_text.empty()
        _t_gen_elapsed = time.perf_counter() - _t_gen_start
        _max_possible = num_configs * num_iterations
        _saved_pct = (1 - _total_iters / max(1, _max_possible)) * 100
        st.info(
            f"⏱ Generation took {_t_gen_elapsed:.2f}s total "
            f"({_t_gen_elapsed / max(1, num_configs):.2f}s/config, "
            f"{_t_gen_elapsed / max(1, _total_iters) * 1000:.1f}ms/iter) | "
            f"Early stopped {_early_stops}/{num_configs} configs — "
            f"{_total_iters:,}/{_max_possible:,} iters used ({_saved_pct:.0f}% saved)"
        )
    finally:
        (all_pts_flat, all_ts_flat, all_obj_ids_flat, all_local_idx_flat,
         all_is_fixed_flat, n_total_points, all_points_plot, all_vals_plot) = saved

    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state[f"_generate_{session_prefix}_ts_requested"] = False
    else:
        st.success(f"Successfully generated {len(all_generated_configs)} configurations!")
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        for config in all_generated_configs:
            sp = config.get("successful_points", [])
            pv = _perpendicular_variance(filt_points, sp)
            deviations.append((config.get("config_number", 0), pv, config))
        deviations.sort(key=lambda x: x[1], reverse=True)
        top_10 = deviations[:10]
        st.session_state[f"_generate_{session_prefix}_ts_results"] = top_10
        st.session_state[f"_{state_prefix}_points_plot"] = filt_points
        st.session_state[f"_{state_prefix}_vals_plot"] = filt_vals
        st.rerun()


# ============= Generate filtered-timestamp configurations (½, ¼, ⅛, 1/16) ============
if st.session_state.get("_generate_half_ts_requested", False) and not st.session_state.get("_generate_half_ts_results", None):
    _run_filtered_ts_generation(
        step=2, label="½", session_prefix="half", state_prefix="hts",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
    )

if st.session_state.get("_generate_quarter_ts_requested", False) and not st.session_state.get("_generate_quarter_ts_results", None):
    _run_filtered_ts_generation(
        step=4, label="¼", session_prefix="quarter", state_prefix="qts",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
    )

if st.session_state.get("_generate_eighth_ts_requested", False) and not st.session_state.get("_generate_eighth_ts_results", None):
    _run_filtered_ts_generation(
        step=8, label="⅛", session_prefix="eighth", state_prefix="ets",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
    )

if st.session_state.get("_generate_sixteenth_ts_requested", False) and not st.session_state.get("_generate_sixteenth_ts_results", None):
    _run_filtered_ts_generation(
        step=16, label="1/16", session_prefix="sixteenth", state_prefix="sts",
        all_objects_points=all_objects_points,
        external_pts_for_window=external_pts_for_window,
        external_ts_for_window=external_ts_for_window,
    )

# ============= 6-Event Single Iteration (ts 0, 34, 67, 183, 213, 249) ============
if st.session_state.get("_generate_6ev_single_requested", False) and not st.session_state.get("_generate_6ev_single_results", None):
    st.markdown("---")
    _6evs_sorted_oids = sorted(all_objects_points.keys())
    _6evs_points_plot: dict[int, np.ndarray] = {}
    _6evs_vals_plot: dict[int, np.ndarray] = {}
    _6evs_keep_set = {0.0, 34.0, 67.0, 183.0, 213.0, 249.0}

    for _6evs_oid in _6evs_sorted_oids:
        _6evs_orig_pts, _6evs_orig_ts = all_objects_points[_6evs_oid]
        _6evs_keep_mask = np.isin(_6evs_orig_ts, list(_6evs_keep_set))
        _6evs_points_plot[_6evs_oid] = _6evs_orig_pts[_6evs_keep_mask]
        _6evs_vals_plot[_6evs_oid] = _6evs_orig_ts[_6evs_keep_mask]

    st.markdown("### 6-Event Single Iteration (timestamps: 0, 34, 67, 183, 213, 249)")

    _6evs_n_ts_per_obj = {oid: _6evs_points_plot[oid].shape[0] for oid in _6evs_sorted_oids}
    _6evs_n_ts_total = sum(_6evs_n_ts_per_obj.values())
    _6evs_ts_info = ", ".join(f"obj {oid}: {n} timestamps" for oid, n in _6evs_n_ts_per_obj.items())

    # ------ User-tuneable generation parameters ------
    with st.expander("⚙️ Generation settings", expanded=False):
        _6evs_pcol1, _6evs_pcol2, _6evs_pcol3 = st.columns([1, 1, 1], gap="small")
        with _6evs_pcol1:
            _6evs_pdp_variant = st.selectbox(
                "PDP variant",
                options=["fundamental", "realistic", "buffer", "rough", "bufferrough"],
                index=0,  # default = fundamental (strict order match)
                key="_6evs_pdp_variant",
                help="PDP variant used to accept/reject moves.\n\n"
                     "• **fundamental**: strict ordering, no tolerance\n"
                     "• **realistic**: buffer on d1 + roughness on d2 (recommended for traffic)\n"
                     "• **buffer/rough/bufferrough**: manual control",
            )
        with _6evs_pcol2:
            _6evs_buffer_x = st.number_input(
                "Buffer X (d1)",
                min_value=0.0, max_value=100.0,
                value=st.session_state.get("_6evs_buffer_x", 1.5),
                step=0.1, format="%.1f",
                key="_6evs_buffer_x",
                help="Tolerance on longitudinal (driving-direction) ordering. "
                     "Points within this distance are considered equivalent in d1. "
                     "Only used by buffer/bufferrough/realistic variants.",
            )
        with _6evs_pcol3:
            _6evs_rough_y = st.number_input(
                "Rough Y (d2)",
                min_value=0.0, max_value=100.0,
                value=st.session_state.get("_6evs_rough_y", 0.4),
                step=0.05, format="%.2f",
                key="_6evs_rough_y",
                help="Equality tolerance on lateral ordering. "
                     "Points within this distance are treated as equal in d2. "
                     "Only used by rough/bufferrough/realistic variants.",
            )

    _6evs_settings = {
        "PDP variant": _6evs_pdp_variant,
        "Buffer X": _6evs_buffer_x,
        "Rough Y": _6evs_rough_y,
        "Strategy": "exponential",
        "Timestamps": [0, 34, 67, 183, 213, 249],
        "Point selection": "Single point (random)",
        "Y-axis range": "[-10, +10]",
        "X-axis": "data range + 20% margin",
        "Equal aspect": False,
        "External points": [list(p) for p in external_points_list] if external_points_list else "none",
    }
    st.caption(
        f"1 iteration per click | exponential | PDP {_6evs_pdp_variant}"
        f"{' | buf_x=' + str(_6evs_buffer_x) if _6evs_pdp_variant in ('buffer','bufferrough','realistic') else ''}"
        f"{' | rough_y=' + str(_6evs_rough_y) if _6evs_pdp_variant in ('rough','bufferrough','realistic') else ''}"
        f" | {_6evs_ts_info}"
    )
    with st.expander("📋 Settings summary", expanded=False):
        st.json(_6evs_settings)

    # ------ Build flat arrays ------
    _6evs_pts_list: list[np.ndarray] = []
    _6evs_ts_list: list[float] = []
    for _6evs_oid in _6evs_sorted_oids:
        for _6evs_li in range(_6evs_points_plot[_6evs_oid].shape[0]):
            _6evs_pts_list.append(_6evs_points_plot[_6evs_oid][_6evs_li])
            _6evs_ts_list.append(float(_6evs_vals_plot[_6evs_oid][_6evs_li]))
    for _6evs_ext_pt, _6evs_ext_t in zip(external_pts_for_window, external_ts_for_window):
        _6evs_pts_list.append(_6evs_ext_pt)
        _6evs_ts_list.append(float(_6evs_ext_t))
    _6evs_pts_flat = np.array(_6evs_pts_list) if _6evs_pts_list else np.array([]).reshape(0, 2)
    _6evs_ts_flat = np.array(_6evs_ts_list) if _6evs_ts_list else np.array([])

    _6evs_obj_ids_flat: list[int] = []
    _6evs_local_idx_flat: list[int] = []
    _6evs_is_fixed_flat: list[bool] = []
    for _6evs_oid in _6evs_sorted_oids:
        for _6evs_li in range(_6evs_points_plot[_6evs_oid].shape[0]):
            _6evs_obj_ids_flat.append(_6evs_oid)
            _6evs_local_idx_flat.append(_6evs_li)
            _6evs_is_fixed_flat.append(False)
    for _6evs_ei in range(len(external_pts_for_window)):
        _6evs_obj_ids_flat.append(-1)
        _6evs_local_idx_flat.append(_6evs_ei)
        _6evs_is_fixed_flat.append(True)

    # ------ Swap globals ------
    # NOTE: maxdist is intentionally NOT swapped — it is computed from ALL 249
    # timestamps of both objects (line ~3389) which gives a realistic step size.
    # The 6-event points are a sparse subset; computing maxdist from them gives
    # unrealistically large values.
    _save_all_pts_flat = all_pts_flat
    _save_all_ts_flat = all_ts_flat
    _save_all_obj_ids_flat = all_obj_ids_flat
    _save_all_local_idx_flat = all_local_idx_flat
    _save_all_is_fixed_flat = all_is_fixed_flat
    _save_n_total_points = n_total_points
    _save_all_points_plot = all_points_plot
    _save_all_vals_plot = all_vals_plot

    all_pts_flat = _6evs_pts_flat
    all_ts_flat = _6evs_ts_flat
    all_obj_ids_flat = _6evs_obj_ids_flat
    all_local_idx_flat = _6evs_local_idx_flat
    all_is_fixed_flat = _6evs_is_fixed_flat
    n_total_points = _6evs_pts_flat.shape[0]
    all_points_plot = _6evs_points_plot
    all_vals_plot = _6evs_vals_plot

    try:
        pdp_variant = st.session_state.get("_6evs_pdp_variant", "fundamental")
        buffer_x = st.session_state.get("_6evs_buffer_x", 1.5) if pdp_variant in ("buffer", "bufferrough", "realistic") else 0.0
        buffer_y = 0.0
        rough_x = 0.0
        rough_y = st.session_state.get("_6evs_rough_y", 0.4) if pdp_variant in ("rough", "bufferrough", "realistic") else 0.0
        mode, pct_threshold, max_mismatch_val = get_threshold_settings()
        max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val

        status_text = st.empty()
        progress_bar = st.progress(0)

        # Check if we are continuing from a previous config
        _6evs_prev = st.session_state.pop("_6evs_continue_from", None)
        _6evs_existing_results: list[tuple] = []
        if _6evs_prev is not None:
            _prev_cnum, _prev_pv, _prev_cfg = _6evs_prev
            # Recover previous results list so we can append
            _6evs_existing_results = list(st.session_state.get("_6evs_prev_results_backup", []))
            # Build starting points from the previous config's generated points
            _prev_sp = _prev_cfg.get("successful_points", [])
            _prev_gen_map: dict[int, np.ndarray] = {}
            for sp in _prev_sp:
                _prev_gen_map[int(sp["original_parent_idx"])] = sp["point"]
            current_points = _6evs_pts_flat.copy()
            for fidx, pt in _prev_gen_map.items():
                if 0 <= fidx < len(current_points):
                    current_points[fidx] = pt
            # Carry over the accumulated successful_points from the previous config
            successful_points: list[SuccessfulPoint] = list(_prev_sp)
            _next_cnum = _prev_cnum + 1
        else:
            current_points = _6evs_pts_flat.copy()
            successful_points: list[SuccessfulPoint] = []
            _next_cnum = 1

        # How many iterations to run in this batch (default 1)
        _6evs_batch_count = st.session_state.pop("_6evs_batch_count", 1)

        # Create IncrementalPDPChecker once for O(k·N) PDP checks instead of O(N²)
        _6evs_pdp_checker = IncrementalPDPChecker(
            original_points=_6evs_pts_flat,
            pdp_variant=pdp_variant,
            buffer_x=buffer_x if pdp_variant in ("buffer", "bufferrough", "realistic") else 0.0,
            buffer_y=buffer_y if pdp_variant in ("buffer", "bufferrough") else 0.0,
            rough_x=rough_x if pdp_variant in ("rough", "bufferrough") else 0.0,
            rough_y=rough_y if pdp_variant in ("rough", "bufferrough", "realistic") else 0.0,
            match_threshold=pct_threshold,
            max_mismatches=max_mismatch_val if mode != "Percentage" else None,
        )
        # Apply already-accepted successful_points to the checker base state
        if successful_points:
            _sp_base_map: dict[int, np.ndarray] = {}
            for _sp_item in successful_points:
                _sp_base_map[int(_sp_item["original_parent_idx"])] = _sp_item["point"]
            _6evs_pdp_checker.update_points(_sp_base_map)

        _6evs_iters_completed = 0
        _6evs_iters_failed = 0
        for _6evs_cfg_i in range(_6evs_batch_count):
            status_text.text(
                f"6-event — iteration #{_next_cnum} "
                f"{'step ' + str(_6evs_cfg_i + 1) + '/' + str(_6evs_batch_count) + ' | ' if _6evs_batch_count > 1 else ''}"
                f"trying …"
            )
            progress_bar.progress((_6evs_cfg_i + 1) / _6evs_batch_count if _6evs_batch_count > 1 else 0.5)

            # ONE attempt: pick random point + direction, max 10 halvings.
            # If PDP fails, point stays at start.
            successful_points_candidate, _6evs_ok = run_multipoint_iteration(
                current_points=current_points,
                successful_points=list(successful_points),
                pdp_variant=pdp_variant,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
                rough_x=rough_x,
                rough_y=rough_y,
                pdp_checker=_6evs_pdp_checker,
            )

            if _6evs_ok:
                successful_points = successful_points_candidate
            else:
                _6evs_iters_failed += 1

            # Always record the iteration (successful or not)
            _6evs_iters_completed += 1
            config_data = {
                "successful_points": list(successful_points),
                "config_number": _next_cnum,
                "pdp_variant": pdp_variant,
                "iterations": _next_cnum,
                "buffer_x": buffer_x, "buffer_y": buffer_y,
                "rough_x": rough_x, "rough_y": rough_y,
                "threshold_mode": mode, "max_threshold": max_threshold,
                "move_succeeded": _6evs_ok,
            }
            pv = _perpendicular_variance(_6evs_points_plot, successful_points)
            _6evs_existing_results.append((_next_cnum, pv, config_data))

            # Prepare for next iteration in the batch: update current_points
            _prev_gen_map_batch: dict[int, np.ndarray] = {}
            for _sp in successful_points:
                _prev_gen_map_batch[int(_sp["original_parent_idx"])] = _sp["point"]
            current_points = _6evs_pts_flat.copy()
            for _fidx, _pt in _prev_gen_map_batch.items():
                if 0 <= _fidx < len(current_points):
                    current_points[_fidx] = _pt
            _next_cnum += 1

        progress_bar.empty()
        status_text.empty()

        # Store generation diagnostic log for display near the buttons
        st.session_state["_6evs_gen_log"] = {
            "success": True,
            "iters_completed": _6evs_iters_completed,
            "iters_failed": _6evs_iters_failed,
            "batch_requested": _6evs_batch_count,
            "accumulated_sp": len(successful_points),
            "maxdist_used": maxdist,
        }

        st.session_state["_generate_6ev_single_results"] = _6evs_existing_results
        st.session_state["_6evs_prev_results_backup"] = list(_6evs_existing_results)
        st.session_state["_6evs_points_plot"] = _6evs_points_plot
        st.session_state["_6evs_vals_plot"] = _6evs_vals_plot
        # Jump to the last generated config (+1 because index 0 = original)
        _new_browse = len(_6evs_existing_results)
        st.session_state["_6evs_browse_idx"] = _new_browse
        st.rerun()
    finally:
        all_pts_flat = _save_all_pts_flat
        all_ts_flat = _save_all_ts_flat
        all_obj_ids_flat = _save_all_obj_ids_flat
        all_local_idx_flat = _save_all_local_idx_flat
        all_is_fixed_flat = _save_all_is_fixed_flat
        n_total_points = _save_n_total_points
        all_points_plot = _save_all_points_plot
        all_vals_plot = _save_all_vals_plot
if st.session_state.get("_generate_four_ts_requested", False) and not st.session_state.get("_generate_four_ts_results", None):
    st.markdown("---")
    _fts_sorted_oids = sorted(all_objects_points.keys())
    _fts_points_plot: dict[int, np.ndarray] = {}
    _fts_vals_plot: dict[int, np.ndarray] = {}
    _fts_keep_set = {0.0, 46.0, 92.0, 136.0}

    for _fts_oid in _fts_sorted_oids:
        _fts_orig_pts, _fts_orig_ts = all_objects_points[_fts_oid]
        _fts_keep_mask = np.isin(_fts_orig_ts, list(_fts_keep_set))
        _fts_points_plot[_fts_oid] = _fts_orig_pts[_fts_keep_mask]
        _fts_vals_plot[_fts_oid] = _fts_orig_ts[_fts_keep_mask]

    st.markdown("### Generating 100 Configurations (4 timestamps: 0, 46, 92, 136)...")

    _fts_n_ts_per_obj = {oid: _fts_points_plot[oid].shape[0] for oid in _fts_sorted_oids}
    _fts_n_ts_total = sum(_fts_n_ts_per_obj.values())
    _fts_ts_info = ", ".join(f"obj {oid}: {n} timestamps" for oid, n in _fts_n_ts_per_obj.items())
    st.caption(f"100 configs × 2500 iter | exponential | PDP fundamental | {_fts_ts_info}")
    st.info(f"Aantal timestamps na filtering: {_fts_n_ts_total} totaal ({_fts_ts_info})")

    _fts_pts_list: list[np.ndarray] = []
    _fts_ts_list: list[float] = []
    for _fts_oid in _fts_sorted_oids:
        for _fts_li in range(_fts_points_plot[_fts_oid].shape[0]):
            _fts_pts_list.append(_fts_points_plot[_fts_oid][_fts_li])
            _fts_ts_list.append(float(_fts_vals_plot[_fts_oid][_fts_li]))
    for _fts_ext_pt, _fts_ext_t in zip(external_pts_for_window, external_ts_for_window):
        _fts_pts_list.append(_fts_ext_pt)
        _fts_ts_list.append(float(_fts_ext_t))
    _fts_pts_flat = np.array(_fts_pts_list) if _fts_pts_list else np.array([]).reshape(0, 2)
    _fts_ts_flat = np.array(_fts_ts_list) if _fts_ts_list else np.array([])

    _fts_obj_ids_flat: list[int] = []
    _fts_local_idx_flat: list[int] = []
    _fts_is_fixed_flat: list[bool] = []
    for _fts_oid in _fts_sorted_oids:
        for _fts_li in range(_fts_points_plot[_fts_oid].shape[0]):
            _fts_obj_ids_flat.append(_fts_oid)
            _fts_local_idx_flat.append(_fts_li)
            _fts_is_fixed_flat.append(False)
    for _fts_ei in range(len(external_pts_for_window)):
        _fts_obj_ids_flat.append(-1)
        _fts_local_idx_flat.append(_fts_ei)
        _fts_is_fixed_flat.append(True)

    _save_all_pts_flat = all_pts_flat
    _save_all_ts_flat = all_ts_flat
    _save_all_obj_ids_flat = all_obj_ids_flat
    _save_all_local_idx_flat = all_local_idx_flat
    _save_all_is_fixed_flat = all_is_fixed_flat
    _save_n_total_points = n_total_points
    _save_all_points_plot = all_points_plot
    _save_all_vals_plot = all_vals_plot

    all_pts_flat = _fts_pts_flat
    all_ts_flat = _fts_ts_flat
    all_obj_ids_flat = _fts_obj_ids_flat
    all_local_idx_flat = _fts_local_idx_flat
    all_is_fixed_flat = _fts_is_fixed_flat
    n_total_points = _fts_pts_flat.shape[0]
    all_points_plot = _fts_points_plot
    all_vals_plot = _fts_vals_plot

    try:
        pdp_variant = "fundamental"
        buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        rough_x = st.session_state.get("cfg_rough_x", 0.0)
        rough_y = st.session_state.get("cfg_rough_y", 0.0)
        mode, pct_threshold, max_mismatch_val = get_threshold_settings()
        max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
        _fts_iterations = 2500
        _fts_num_configs = MAX_FILTER_CONFIGS

        progress_bar = st.progress(0)
        status_text = st.empty()
        all_generated_configs: list[dict[str, Any]] = []

        for config_idx in range(_fts_num_configs):
            current_points = _fts_pts_flat.copy()
            successful_points: list[SuccessfulPoint] = []
            for iteration in range(_fts_iterations):
                status_text.text(f"4-ts filtered — config {config_idx + 1}/{_fts_num_configs} | iter {iteration + 1}/{_fts_iterations}")
                successful_points, success = run_multipoint_iteration(
                    current_points=current_points,
                    successful_points=successful_points,
                    pdp_variant=pdp_variant,
                    buffer_x=buffer_x,
                    buffer_y=buffer_y,
                    rough_x=rough_x,
                    rough_y=rough_y,
                )
            if successful_points:
                all_generated_configs.append({
                    "successful_points": successful_points,
                    "config_number": config_idx + 1,
                    "pdp_variant": pdp_variant,
                    "iterations": _fts_iterations,
                    "buffer_x": buffer_x, "buffer_y": buffer_y,
                    "rough_x": rough_x, "rough_y": rough_y,
                    "threshold_mode": mode, "max_threshold": max_threshold,
                })
            progress_bar.progress((config_idx + 1) / _fts_num_configs)

        progress_bar.empty()
        status_text.empty()
    finally:
        # --- Restore global variables (guaranteed even on exception) ---
        all_pts_flat = _save_all_pts_flat
        all_ts_flat = _save_all_ts_flat
        all_obj_ids_flat = _save_all_obj_ids_flat
        all_local_idx_flat = _save_all_local_idx_flat
        all_is_fixed_flat = _save_all_is_fixed_flat
        n_total_points = _save_n_total_points
        all_points_plot = _save_all_points_plot
        all_vals_plot = _save_all_vals_plot

    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_four_ts_requested"] = False
    else:
        st.success(f"Successfully generated {len(all_generated_configs)} configurations!")
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        for config in all_generated_configs:
            successful_points = config.get("successful_points", [])
            pv = _perpendicular_variance(_fts_points_plot, successful_points)
            config_num = config.get("config_number", 0)
            deviations.append((config_num, pv, config))
        deviations.sort(key=lambda x: x[1], reverse=True)
        top_10 = deviations[:10]
        st.session_state["_generate_four_ts_results"] = top_10
        st.session_state["_fts_points_plot"] = _fts_points_plot
        st.session_state["_fts_vals_plot"] = _fts_vals_plot
        st.rerun()

# ============= Generate 100 configs × 2500 iterations — 2 timestamps (0, 136) ============
if st.session_state.get("_generate_two_ts_requested", False) and not st.session_state.get("_generate_two_ts_results", None):
    st.markdown("---")
    _tts_sorted_oids = sorted(all_objects_points.keys())
    _tts_points_plot: dict[int, np.ndarray] = {}
    _tts_vals_plot: dict[int, np.ndarray] = {}
    _tts_keep_set = {0.0, 136.0}

    for _tts_oid in _tts_sorted_oids:
        _tts_orig_pts, _tts_orig_ts = all_objects_points[_tts_oid]
        _tts_keep_mask = np.isin(_tts_orig_ts, list(_tts_keep_set))
        _tts_points_plot[_tts_oid] = _tts_orig_pts[_tts_keep_mask]
        _tts_vals_plot[_tts_oid] = _tts_orig_ts[_tts_keep_mask]

    st.markdown("### Generating 100 Configurations (2 timestamps: 0, 136)...")

    _tts_n_ts_per_obj = {oid: _tts_points_plot[oid].shape[0] for oid in _tts_sorted_oids}
    _tts_n_ts_total = sum(_tts_n_ts_per_obj.values())
    _tts_ts_info = ", ".join(f"obj {oid}: {n} timestamps" for oid, n in _tts_n_ts_per_obj.items())
    st.caption(f"100 configs × 2500 iter | exponential | PDP fundamental | {_tts_ts_info}")
    st.info(f"Aantal timestamps na filtering: {_tts_n_ts_total} totaal ({_tts_ts_info})")

    _tts_pts_list: list[np.ndarray] = []
    _tts_ts_list: list[float] = []
    for _tts_oid in _tts_sorted_oids:
        for _tts_li in range(_tts_points_plot[_tts_oid].shape[0]):
            _tts_pts_list.append(_tts_points_plot[_tts_oid][_tts_li])
            _tts_ts_list.append(float(_tts_vals_plot[_tts_oid][_tts_li]))
    for _tts_ext_pt, _tts_ext_t in zip(external_pts_for_window, external_ts_for_window):
        _tts_pts_list.append(_tts_ext_pt)
        _tts_ts_list.append(float(_tts_ext_t))
    _tts_pts_flat = np.array(_tts_pts_list) if _tts_pts_list else np.array([]).reshape(0, 2)
    _tts_ts_flat = np.array(_tts_ts_list) if _tts_ts_list else np.array([])

    _tts_obj_ids_flat: list[int] = []
    _tts_local_idx_flat: list[int] = []
    _tts_is_fixed_flat: list[bool] = []
    for _tts_oid in _tts_sorted_oids:
        for _tts_li in range(_tts_points_plot[_tts_oid].shape[0]):
            _tts_obj_ids_flat.append(_tts_oid)
            _tts_local_idx_flat.append(_tts_li)
            _tts_is_fixed_flat.append(False)
    for _tts_ei in range(len(external_pts_for_window)):
        _tts_obj_ids_flat.append(-1)
        _tts_local_idx_flat.append(_tts_ei)
        _tts_is_fixed_flat.append(True)

    _save_all_pts_flat = all_pts_flat
    _save_all_ts_flat = all_ts_flat
    _save_all_obj_ids_flat = all_obj_ids_flat
    _save_all_local_idx_flat = all_local_idx_flat
    _save_all_is_fixed_flat = all_is_fixed_flat
    _save_n_total_points = n_total_points
    _save_all_points_plot = all_points_plot
    _save_all_vals_plot = all_vals_plot

    all_pts_flat = _tts_pts_flat
    all_ts_flat = _tts_ts_flat
    all_obj_ids_flat = _tts_obj_ids_flat
    all_local_idx_flat = _tts_local_idx_flat
    all_is_fixed_flat = _tts_is_fixed_flat
    n_total_points = _tts_pts_flat.shape[0]
    all_points_plot = _tts_points_plot
    all_vals_plot = _tts_vals_plot

    try:
        pdp_variant = "fundamental"
        buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        rough_x = st.session_state.get("cfg_rough_x", 0.0)
        rough_y = st.session_state.get("cfg_rough_y", 0.0)
        mode, pct_threshold, max_mismatch_val = get_threshold_settings()
        max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
        _tts_iterations = 2500
        _tts_num_configs = MAX_FILTER_CONFIGS

        progress_bar = st.progress(0)
        status_text = st.empty()
        all_generated_configs: list[dict[str, Any]] = []

        for config_idx in range(_tts_num_configs):
            current_points = _tts_pts_flat.copy()
            successful_points: list[SuccessfulPoint] = []
            for iteration in range(_tts_iterations):
                status_text.text(f"2-ts filtered — config {config_idx + 1}/{_tts_num_configs} | iter {iteration + 1}/{_tts_iterations}")
                successful_points, success = run_multipoint_iteration(
                    current_points=current_points,
                    successful_points=successful_points,
                    pdp_variant=pdp_variant,
                    buffer_x=buffer_x,
                    buffer_y=buffer_y,
                    rough_x=rough_x,
                    rough_y=rough_y,
                )
            if successful_points:
                all_generated_configs.append({
                    "successful_points": successful_points,
                    "config_number": config_idx + 1,
                    "pdp_variant": pdp_variant,
                    "iterations": _tts_iterations,
                    "buffer_x": buffer_x, "buffer_y": buffer_y,
                    "rough_x": rough_x, "rough_y": rough_y,
                    "threshold_mode": mode, "max_threshold": max_threshold,
                })
            progress_bar.progress((config_idx + 1) / _tts_num_configs)

        progress_bar.empty()
        status_text.empty()
    finally:
        # --- Restore global variables (guaranteed even on exception) ---
        all_pts_flat = _save_all_pts_flat
        all_ts_flat = _save_all_ts_flat
        all_obj_ids_flat = _save_all_obj_ids_flat
        all_local_idx_flat = _save_all_local_idx_flat
        all_is_fixed_flat = _save_all_is_fixed_flat
        n_total_points = _save_n_total_points
        all_points_plot = _save_all_points_plot
        all_vals_plot = _save_all_vals_plot

    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_two_ts_requested"] = False
    else:
        st.success(f"Successfully generated {len(all_generated_configs)} configurations!")
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        for config in all_generated_configs:
            successful_points = config.get("successful_points", [])
            pv = _perpendicular_variance(_tts_points_plot, successful_points)
            config_num = config.get("config_number", 0)
            deviations.append((config_num, pv, config))
        deviations.sort(key=lambda x: x[1], reverse=True)
        top_10 = deviations[:10]
        st.session_state["_generate_two_ts_results"] = top_10
        st.session_state["_tts_points_plot"] = _tts_points_plot
        st.session_state["_tts_vals_plot"] = _tts_vals_plot
        st.rerun()

# ============= Generate 100 configs × 2500 iterations — Config 68 Realistic ============
if st.session_state.get("_generate_c68r_requested", False) and not st.session_state.get("_generate_c68r_results", None):
    st.markdown("---")
    st.markdown("### Generating 100 Configs (C68 Realistic, 2500 iter each)...")
    st.caption("Config 68 | t=82..160 step 2 | Multiple points same direction | realistic (d1 buf 5m, d2 rough 0.30m) | External pts")

    _c68r_pdp = "realistic"
    _c68r_bx, _c68r_by = 5.0, 0.0
    _c68r_rx, _c68r_ry = 0.0, 0.30
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    _c68r_max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
    _c68r_iters = 2500

    # Freeze first and last timestamp per object
    _frozen = set()
    _gi = 0
    for oid in sorted(all_points_plot.keys()):
        n_pts = all_points_plot[oid].shape[0]
        if n_pts > 0:
            _frozen.add(_gi)          # first timestamp
            _frozen.add(_gi + n_pts - 1)  # last timestamp
        _gi += n_pts
    st.session_state["_frozen_endpoints"] = _frozen

    progress_bar = st.progress(0)
    status_text = st.empty()
    all_generated_configs: list[dict[str, Any]] = []

    _t_gen_start = time.perf_counter()
    _total_iters = 0
    _early_stops = 0
    for config_idx in range(MAX_FILTER_CONFIGS):
        current_points = all_pts_flat.copy()
        successful_points: list[SuccessfulPoint] = []
        _stagnant = 0
        _iters_used = _c68r_iters
        for iteration in range(_c68r_iters):
            status_text.text(f"C68 Realistic — config {config_idx + 1}/{MAX_FILTER_CONFIGS} | iter {iteration + 1}/{_c68r_iters}")
            _sp_before = len(successful_points)
            successful_points, success = run_multipoint_iteration(
                current_points=current_points,
                successful_points=successful_points,
                pdp_variant=_c68r_pdp,
                buffer_x=_c68r_bx, buffer_y=_c68r_by,
                rough_x=_c68r_rx, rough_y=_c68r_ry,
            )
            _n_added = len(successful_points) - _sp_before
            if _n_added > 0 and _check_stagnation(successful_points, _n_added):
                _stagnant += 1
            else:
                _stagnant = 0
            if _stagnant >= EARLY_STOP_PATIENCE:
                _iters_used = iteration + 1
                _early_stops += 1
                break
        _total_iters += _iters_used
        if successful_points:
            all_generated_configs.append({
                "successful_points": successful_points,
                "config_number": config_idx + 1,
                "pdp_variant": _c68r_pdp,
                "iterations": _iters_used,
                "buffer_x": _c68r_bx, "buffer_y": _c68r_by,
                "rough_x": _c68r_rx, "rough_y": _c68r_ry,
                "threshold_mode": mode, "max_threshold": _c68r_max_threshold,
            })
        progress_bar.progress((config_idx + 1) / MAX_FILTER_CONFIGS)

    progress_bar.empty()
    status_text.empty()
    _t_gen_elapsed = time.perf_counter() - _t_gen_start
    _max_possible = MAX_FILTER_CONFIGS * _c68r_iters
    _saved_pct = (1 - _total_iters / max(1, _max_possible)) * 100
    st.info(
        f"⏱ Generation took {_t_gen_elapsed:.2f}s total "
        f"({_t_gen_elapsed / max(1, MAX_FILTER_CONFIGS):.2f}s/config, "
        f"{_t_gen_elapsed / max(1, _total_iters) * 1000:.1f}ms/iter) | "
        f"Early stopped {_early_stops}/{MAX_FILTER_CONFIGS} configs — "
        f"{_total_iters:,}/{_max_possible:,} iters used ({_saved_pct:.0f}% saved)"
    )
    st.session_state.pop("_frozen_endpoints", None)  # unfreeze

    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_c68r_requested"] = False
    else:
        st.success(f"Generated {len(all_generated_configs)} configs!")
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        for config in all_generated_configs:
            pv = _perpendicular_variance(all_points_plot, config.get("successful_points", []))
            deviations.append((config.get("config_number", 0), pv, config))
        deviations.sort(key=lambda x: x[1], reverse=True)
        st.session_state["_generate_c68r_results"] = deviations[:10]
        st.rerun()

# ============= Generate 100 configs × 2500 iterations — Config 68 Fundamental ============
if st.session_state.get("_generate_c68f_requested", False) and not st.session_state.get("_generate_c68f_results", None):
    st.markdown("---")
    st.markdown("### Generating 100 Configs (C68 Fundamental, 2500 iter each)...")
    st.caption("Config 68 | t=82..160 step 2 | Single point | fundamental | External pts at lane centers")

    _c68f_pdp = "fundamental"
    _c68f_bx, _c68f_by = 0.0, 0.0
    _c68f_rx, _c68f_ry = 0.0, 0.0
    mode, pct_threshold, max_mismatch_val = get_threshold_settings()
    _c68f_max_threshold = pct_threshold if mode == "Percentage" else max_mismatch_val
    _c68f_iters = 2500

    # Freeze first and last timestamp per object
    _frozen = set()
    _gi = 0
    for oid in sorted(all_points_plot.keys()):
        n_pts = all_points_plot[oid].shape[0]
        if n_pts > 0:
            _frozen.add(_gi)          # first timestamp
            _frozen.add(_gi + n_pts - 1)  # last timestamp
        _gi += n_pts
    st.session_state["_frozen_endpoints"] = _frozen

    progress_bar = st.progress(0)
    status_text = st.empty()
    all_generated_configs: list[dict[str, Any]] = []

    _t_gen_start = time.perf_counter()
    _total_iters = 0
    _early_stops = 0
    for config_idx in range(MAX_FILTER_CONFIGS):
        current_points = all_pts_flat.copy()
        successful_points: list[SuccessfulPoint] = []
        _stagnant = 0
        _iters_used = _c68f_iters
        for iteration in range(_c68f_iters):
            status_text.text(f"C68 Fundamental — config {config_idx + 1}/{MAX_FILTER_CONFIGS} | iter {iteration + 1}/{_c68f_iters}")
            _sp_before = len(successful_points)
            successful_points, success = run_multipoint_iteration(
                current_points=current_points,
                successful_points=successful_points,
                pdp_variant=_c68f_pdp,
                buffer_x=_c68f_bx, buffer_y=_c68f_by,
                rough_x=_c68f_rx, rough_y=_c68f_ry,
            )
            _n_added = len(successful_points) - _sp_before
            if _n_added > 0 and _check_stagnation(successful_points, _n_added):
                _stagnant += 1
            else:
                _stagnant = 0
            if _stagnant >= EARLY_STOP_PATIENCE:
                _iters_used = iteration + 1
                _early_stops += 1
                break
        _total_iters += _iters_used
        if successful_points:
            all_generated_configs.append({
                "successful_points": successful_points,
                "config_number": config_idx + 1,
                "pdp_variant": _c68f_pdp,
                "iterations": _iters_used,
                "buffer_x": _c68f_bx, "buffer_y": _c68f_by,
                "rough_x": _c68f_rx, "rough_y": _c68f_ry,
                "threshold_mode": mode, "max_threshold": _c68f_max_threshold,
            })
        progress_bar.progress((config_idx + 1) / MAX_FILTER_CONFIGS)

    progress_bar.empty()
    status_text.empty()
    _t_gen_elapsed = time.perf_counter() - _t_gen_start
    _max_possible = MAX_FILTER_CONFIGS * _c68f_iters
    _saved_pct = (1 - _total_iters / max(1, _max_possible)) * 100
    st.info(
        f"⏱ Generation took {_t_gen_elapsed:.2f}s total "
        f"({_t_gen_elapsed / max(1, MAX_FILTER_CONFIGS):.2f}s/config, "
        f"{_t_gen_elapsed / max(1, _total_iters) * 1000:.1f}ms/iter) | "
        f"Early stopped {_early_stops}/{MAX_FILTER_CONFIGS} configs — "
        f"{_total_iters:,}/{_max_possible:,} iters used ({_saved_pct:.0f}% saved)"
    )
    st.session_state.pop("_frozen_endpoints", None)  # unfreeze

    if not all_generated_configs:
        st.error("No configurations were successfully generated.")
        st.session_state["_generate_c68f_requested"] = False
    else:
        st.success(f"Generated {len(all_generated_configs)} configs!")
        deviations: list[tuple[int, float, dict[str, Any]]] = []
        for config in all_generated_configs:
            pv = _perpendicular_variance(all_points_plot, config.get("successful_points", []))
            deviations.append((config.get("config_number", 0), pv, config))
        deviations.sort(key=lambda x: x[1], reverse=True)
        st.session_state["_generate_c68f_results"] = deviations[:10]
        st.rerun()

# Display results if they exist
if st.session_state.get("_generate_30_results", None):
    top_100 = st.session_state["_generate_30_results"]
    
    st.markdown("---")
    st.markdown("### Top 100 Most Deviating Configurations (from 1000 generated)")
    st.markdown("""These configurations exhibit the largest spatial deviations from the original while maintaining the PDP inequality pattern.
    
**Deviation Metrics (calculated per configuration):**
- **Perpendicular Variance (m²)**: Variance of the perpendicular (shortest) distances from each generated point to the original trajectory polyline. Higher values indicate more uneven lateral path deviation.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps. Values range from 0° (parallel) to 180° (opposite direction).
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps. This captures variations in vehicle speed or trajectory compression/expansion.

Configurations are ranked by perpendicular variance (highest first). Each visualization shows a focused 9-timestamp window around the largest deviation, with original and generated trajectories aligned on the same axes.""")
    
    # Store metrics for summary chart
    config_metrics: list[dict[str, Any]] = []
    
    # Display each of the top 100
    for rank, (config_num, deviation, config) in enumerate(top_100, 1):
            st.markdown(f"#### Rank {rank}: Configuration #{config_num} (Perp. variance: {deviation:.4f} m²)")
            
            # Get configuration characteristics
            pdp_variant = config.get("pdp_variant", "fundamental")
            iterations = config.get("iterations", "N/A")
            threshold_mode = config.get("threshold_mode", "Percentage")
            max_threshold = config.get("max_threshold", 0.0)
            
            # Calculate angle and distance deviations
            successful_points = config.get("successful_points", [])
            
            # Create mapping from original_parent_idx to generated coordinates
            generated_coords_map: dict[int, np.ndarray] = {}
            for sp in successful_points:
                orig_idx = sp["original_parent_idx"]
                gen_coord = sp["point"]
                generated_coords_map[orig_idx] = gen_coord
            
            max_angle_deviation = 0.0
            max_distance_deviation = 0.0
            
            # Process each object
            global_idx = 0
            for oid in sorted(all_points_plot.keys()):
                n_pts = all_points_plot[oid].shape[0]
                original_pts = all_points_plot[oid]
                
                # Build generated points for this object
                generated_pts_list = []
                for local_idx in range(n_pts):
                    if global_idx in generated_coords_map:
                        coord = generated_coords_map[global_idx]
                    else:
                        coord = original_pts[local_idx]
                    generated_pts_list.append(coord)
                    global_idx += 1
                
                generated_pts = np.array(generated_pts_list)
                
                # Calculate deviations for consecutive points
                for i in range(n_pts - 1):
                    # Original angle and distance
                    orig_dx = original_pts[i+1, 0] - original_pts[i, 0]
                    orig_dy = original_pts[i+1, 1] - original_pts[i, 1]
                    orig_angle = np.degrees(np.arctan2(orig_dy, orig_dx))
                    orig_dist = np.sqrt(orig_dx**2 + orig_dy**2)
                    
                    # Generated angle and distance
                    gen_dx = generated_pts[i+1, 0] - generated_pts[i, 0]
                    gen_dy = generated_pts[i+1, 1] - generated_pts[i, 1]
                    gen_angle = np.degrees(np.arctan2(gen_dy, gen_dx))
                    gen_dist = np.sqrt(gen_dx**2 + gen_dy**2)
                    
                    # Angle deviation (handle wraparound)
                    angle_diff = abs(gen_angle - orig_angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    max_angle_deviation = max(max_angle_deviation, angle_diff)
                    
                    # Distance deviation
                    dist_diff = abs(gen_dist - orig_dist)
                    max_distance_deviation = max(max_distance_deviation, dist_diff)
            
            # Store metrics
            config_metrics.append({
                "config_num": config_num,
                "rank": rank,
                "perp_variance": deviation,
                "max_angle_dev": max_angle_deviation,
                "max_dist_dev": max_distance_deviation
            })
            
            # Format threshold display
            if threshold_mode == "Percentage":
                threshold_display = f"{max_threshold:.1%}"
            else:
                threshold_display = str(int(max_threshold))
            
            # Build generated coordinate map
            successful_points = config.get("successful_points", [])
            generated_coords_map: dict[int, np.ndarray] = {}
            for sp in successful_points:
                orig_idx = int(sp["original_parent_idx"])
                generated_coords_map[orig_idx] = sp["point"]

            # Focus window: 9 timestamps around the point with largest deviation from original
            max_dev_flat_idx: int | None = None
            max_dev_value = -1.0
            for flat_idx, gen_coord in generated_coords_map.items():
                if 0 <= flat_idx < len(all_pts_flat):
                    dev = float(np.linalg.norm(np.array(gen_coord) - np.array(all_pts_flat[flat_idx])))
                    if dev > max_dev_value:
                        max_dev_value = dev
                        max_dev_flat_idx = flat_idx

            sorted_obj_ids = sorted(all_points_plot.keys())
            if max_dev_flat_idx is not None and 0 <= max_dev_flat_idx < len(all_obj_ids_flat):
                focus_object_id = all_obj_ids_flat[max_dev_flat_idx]
                focus_local_idx = all_local_idx_flat[max_dev_flat_idx]
            else:
                focus_object_id = sorted_obj_ids[0] if sorted_obj_ids else 0
                focus_local_idx = 0

            focus_n = all_points_plot.get(focus_object_id, np.empty((0, 2))).shape[0]
            focus_window_size = min(9, focus_n) if focus_n > 0 else 0
            if focus_window_size > 0:
                focus_start = max(0, focus_local_idx - focus_window_size // 2)
                focus_end = min(focus_n, focus_start + focus_window_size)
                focus_start = max(0, focus_end - focus_window_size)
                focus_ts = all_vals_plot[focus_object_id][focus_start:focus_end]
                focus_ts_set = set(float(t) for t in focus_ts)
            else:
                focus_start = 0
                focus_end = 0
                focus_ts_set = set(float(t) for t in all_ts_flat.tolist())

            # Build focused original and generated trajectories using the same timestamp window
            original_points_dict: dict[int, list[tuple[np.ndarray, float]]] = {}
            generated_points_dict: dict[int, list[tuple[np.ndarray, float]]] = {}
            global_idx = 0
            for oid in sorted_obj_ids:
                n_pts = all_points_plot[oid].shape[0]
                vals = all_vals_plot[oid]
                for local_idx in range(n_pts):
                    t_val = float(vals[local_idx])
                    orig_coord = all_points_plot[oid][local_idx]
                    gen_coord = generated_coords_map.get(global_idx, orig_coord)
                    if t_val in focus_ts_set:
                        original_points_dict.setdefault(oid, []).append((orig_coord, t_val))
                        generated_points_dict.setdefault(oid, []).append((np.array(gen_coord), t_val))
                    global_idx += 1

            # Shared focused axis limits for original/generated alignment
            x_all: list[float] = []
            y_all: list[float] = []
            for points_dict in (original_points_dict, generated_points_dict):
                for pts_list in points_dict.values():
                    for coord, _ in pts_list:
                        x_all.append(float(coord[0]))
                        y_all.append(float(coord[1]))

            if x_all and y_all:
                x_min, x_max = min(x_all), max(x_all)
                y_min, y_max = min(y_all), max(y_all)
                x_margin = max(1.0, 0.15 * (x_max - x_min))
                y_margin = max(1.0, 0.15 * (y_max - y_min))
                focused_xlim = (x_min - x_margin, x_max + x_margin)
                focused_ylim = (y_min - y_margin, y_max + y_margin)
            else:
                focused_xlim = XLIM
                focused_ylim = YLIM

            # Create side-by-side visualization with aligned focused window
            fig = Figure(figsize=(12, 5.5), dpi=120)
            canvas = FigureCanvas(fig)
            ax_left = fig.add_subplot(121)
            ax_right = fig.add_subplot(122)

            for ax in (ax_left, ax_right):
                ax.set_xlim(*focused_xlim)
                ax.set_ylim(*focused_ylim)
                ax.set_aspect("equal", adjustable="box")
                for axis_spine in ax.spines.values():
                    axis_spine.set_linewidth(0.9)
                    axis_spine.set_color("#222")
                ax.tick_params(axis="both", labelsize=9, width=0.8, color="#222")
                ax.set_xlabel("d1", fontsize=11, labelpad=8)
                ax.set_ylabel("d2", fontsize=11, labelpad=8)

            ax_left.set_title("Original (Focused)", fontsize=12, fontweight='bold')
            ax_right.set_title(f"Generated (Config #{config_num}, Focused)", fontsize=12, fontweight='bold')

            banner_line1 = f"Variant 1/1 ({pdp_variant}) | Config {config_num} | Iteration {iterations}"
            if focus_n > 0 and focus_end > focus_start:
                focus_t0 = all_vals_plot[focus_object_id][focus_start]
                focus_t1 = all_vals_plot[focus_object_id][focus_end - 1]
                banner_line2 = f"Max threshold {threshold_display} | Focus window: t={focus_t0:g}..{focus_t1:g} ({focus_end - focus_start} timestamps)"
            else:
                banner_line2 = f"Max threshold {threshold_display}"
            banner_text = f"{banner_line1}\n{banner_line2}"
            ax_right.text(
                0.5,
                0.97,
                banner_text,
                transform=ax_right.transAxes,
                ha='center',
                va='top',
                fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5DEB3', edgecolor='black', linewidth=1.5)
            )

            obj_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
            avg_y = float(np.mean(y_all)) if y_all else 0.0
            if not _is_custom_upload:
                lane_width = 3.0
                lane_offsets = [-lane_width, 0.0, lane_width]
                for ax in (ax_left, ax_right):
                    for offset in lane_offsets:
                        lane_y = avg_y + offset
                        ax.axhline(y=lane_y, color='black', linewidth=0.8, linestyle='-' if offset in [-lane_width, lane_width] else '--')

            # Draw both original and generated using the same focused timestamps
            offsets = [(3, 3), (3, -8), (-8, 3)]
            for i, o_id in enumerate(sorted_obj_ids):
                original_list = sorted(original_points_dict.get(o_id, []), key=lambda x: x[1])
                generated_list = sorted(generated_points_dict.get(o_id, []), key=lambda x: x[1])
                if not original_list or not generated_list:
                    continue

                original_pts = np.array([p[0] for p in original_list])
                original_vals = np.array([p[1] for p in original_list])
                generated_pts = np.array([p[0] for p in generated_list])
                generated_vals = np.array([p[1] for p in generated_list])

                color = obj_colors[i % len(obj_colors)]
                label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
                ax_left.plot(original_pts[:, 0], original_pts[:, 1], '-', color=color, linewidth=1.5, alpha=0.7)
                ax_right.plot(generated_pts[:, 0], generated_pts[:, 1], '-', color=color, linewidth=1.5, alpha=0.7, label=label)

                for j, ((x, y), tval) in enumerate(zip(original_pts, original_vals)):
                    ax_left.scatter([x], [y], s=25, zorder=10, color=color, marker='o')
                    off = offsets[j % len(offsets)]
                    tnum = float(tval)
                    lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
                    if label and lbl:
                        try:
                            ax_left.annotate(
                                f"$\\mathit{{{label}}}_{{{lbl}}}$",
                                xy=(x, y),
                                xytext=off,
                                textcoords="offset points",
                                fontsize=8,
                                color=color,
                                ha="center",
                                va="center",
                            )
                        except Exception as e:
                            logger.warning(f"[RENDER] LaTeX annotation failed for {label}_{lbl}: {e}")

                for j, ((x, y), tval) in enumerate(zip(generated_pts, generated_vals)):
                    ax_right.scatter([x], [y], s=25, zorder=10, color=color, marker='o')
                    off = offsets[j % len(offsets)]
                    tnum = float(tval)
                    lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
                    if label and lbl:
                        try:
                            ax_right.annotate(
                                f"$\\mathit{{{label}}}_{{{lbl}}}$",
                                xy=(x, y),
                                xytext=off,
                                textcoords="offset points",
                                fontsize=8,
                                color=color,
                                ha="center",
                                va="center",
                            )
                        except Exception as e:
                            logger.warning(f"[RENDER] LaTeX annotation failed for {label}_{lbl}: {e}")
            
            try:
                fig.tight_layout()
            except Exception as e:
                logger.warning(f"[RENDER] tight_layout failed: {e}")
            
            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            
            # Display image
            st.image(buf, use_container_width=True)
            
            # Download button
            buf.seek(0)
            st.download_button(
                label=f"📥 Download Configuration #{config_num}",
                data=buf,
                file_name=f"config_{config_num}_deviation_{deviation:.3f}m.png",
                mime="image/png",
                key=f"download_config_{config_num}"
            )
            
            st.markdown("---")
    
    # Calculate statistics
    angle_devs = [m['max_angle_dev'] for m in config_metrics]
    dist_devs = [m['max_dist_dev'] for m in config_metrics]
    pv_vals = [m['perp_variance'] for m in config_metrics]
    
    mean_angle = float(np.mean(angle_devs))
    std_angle = float(np.std(angle_devs))
    mean_dist = float(np.mean(dist_devs))
    std_dist = float(np.std(dist_devs))
    mean_pv = float(np.mean(pv_vals))
    std_pv = float(np.std(pv_vals))
    
    # Display statistics summary
    st.markdown("### Statistics Summary (Top 100 configurations)")
    st.caption("Mean ± standard deviation calculated across the top 100 most deviating configurations. Lower standard deviation indicates consistent behavior across these high-deviation cases.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Perp. Variance", f"{mean_pv:.4f} m² ± {std_pv:.4f}")
    with col2:
        st.metric("Max Angle Dev", f"{mean_angle:.1f}° ± {std_angle:.1f}°")
    with col3:
        st.metric("Max Distance Dev", f"{mean_dist:.2f}m ± {std_dist:.2f}m")
    
    st.markdown("---")
    
    # Display data table
    st.markdown("### Top 100 Configurations - Detailed Metrics")
    st.caption("""Complete metrics for the 100 most deviating configurations (selected from 1000 generated). Select cells and copy (Ctrl+C) to paste into Excel, PowerPoint, or other applications.
    
- **Rank**: Position in descending order of perpendicular variance (1 = highest variance)
- **Config #**: Unique configuration identifier from the generation batch (1-1000)
- **Perp. Variance (m²)**: Variance of perpendicular distances from generated points to the original path
- **Max Angle Dev (°)**: Largest trajectory angle change between consecutive timestamps (per configuration)
- **Max Distance Dev (m)**: Largest inter-point spacing change between consecutive timestamps (per configuration)""")
    
    # Create DataFrame for display
    df_metrics = pd.DataFrame(config_metrics)
    df_metrics = df_metrics[['rank', 'config_num', 'perp_variance', 'max_angle_dev', 'max_dist_dev']]
    df_metrics.columns = ['Rank', 'Config #', 'Perp. Variance (m²)', 'Max Angle Dev (°)', 'Max Distance Dev (m)']
    
    # Format numeric columns
    df_metrics['Perp. Variance (m²)'] = df_metrics['Perp. Variance (m²)'].apply(lambda x: f"{x:.4f}")
    df_metrics['Max Angle Dev (°)'] = df_metrics['Max Angle Dev (°)'].apply(lambda x: f"{x:.1f}")
    df_metrics['Max Distance Dev (m)'] = df_metrics['Max Distance Dev (m)'].apply(lambda x: f"{x:.2f}")
    
    # Display table
    st.dataframe(df_metrics, use_container_width=True, height=400)
    
    # Add download button for CSV
    csv = df_metrics.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="top_100_configurations_metrics.csv",
        mime="text/csv",
        key="download_metrics_csv",
        help="Download all metrics as CSV file for further analysis in Excel, Python, R, etc."
    )
    
    st.markdown("---")
    
    # Display summary metrics chart
    st.markdown("### Maximum Deviations Summary (Top 100 from 1000 generated)")
    st.caption("""Visual comparison of maximum deviations across the top 100 configurations. Red dashed line indicates the mean value calculated from these 100 configurations.
    
- **Left chart**: Maximum angle deviation shows the largest directional change in any trajectory segment
- **Right chart**: Maximum distance deviation shows the largest speed/spacing variation in any trajectory segment

These metrics help identify configurations with extreme local variations, even if their perpendicular variance is moderate.""")
    
    # Create bar chart
    fig_metrics = Figure(figsize=(12, 5), dpi=100)
    canvas_metrics = FigureCanvas(fig_metrics)
    
    ax1 = fig_metrics.add_subplot(121)
    ax2 = fig_metrics.add_subplot(122)
    
    # Extract data
    config_labels = [f"#{m['config_num']}" for m in config_metrics]
    x_positions = range(len(config_labels))
    
    # Angle deviations bar chart
    ax1.bar(x_positions, angle_devs, color='#FF7F0E', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Configuration Rank", fontsize=11)
    ax1.set_ylabel("Max Angle Deviation (degrees)", fontsize=11)
    ax1.set_title("Maximum Angle Deviation (Top 100)", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=mean_angle, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_angle:.1f}°')
    ax1.legend()
    
    # Only show x-tick labels for every 10th config
    ax1.set_xticks([i for i in range(0, len(config_labels), 10)])
    ax1.set_xticklabels([config_labels[i] for i in range(0, len(config_labels), 10)], rotation=45, ha='right')
    
    # Distance deviations bar chart
    ax2.bar(x_positions, dist_devs, color='#1F77B4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel("Configuration Rank", fontsize=11)
    ax2.set_ylabel("Max Distance Deviation (m)", fontsize=11)
    ax2.set_title("Maximum Distance Deviation (Top 100)", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}m')
    ax2.legend()
    
    # Only show x-tick labels for every 10th config
    ax2.set_xticks([i for i in range(0, len(config_labels), 10)])
    ax2.set_xticklabels([config_labels[i] for i in range(0, len(config_labels), 10)], rotation=45, ha='right')
    
    fig_metrics.tight_layout()
    
    # Display chart
    buf_metrics = io.BytesIO()
    fig_metrics.savefig(buf_metrics, format='png', dpi=100, bbox_inches='tight')
    buf_metrics.seek(0)
    st.image(buf_metrics, use_container_width=True)
    
    st.markdown("---")
    
    # Add a clear button
    if st.button("Clear Results & Cache", key="clear_top5_results"):
        st.session_state["_generate_30_requested"] = False
        st.session_state["_generate_30_results"] = None
        st.cache_data.clear()
        st.rerun()

# Display results for 5000-config generation if they exist
if st.session_state.get("_generate_5000_results", None):
    top_500 = st.session_state["_generate_5000_results"]
    
    st.markdown("---")
    st.markdown("### Top 500 Most Deviating Configurations (from 5 generated)")
    st.markdown("""These configurations exhibit the largest spatial deviations from the original while maintaining the PDP inequality pattern.
    
**Deviation Metrics (calculated per configuration):**
- **Perpendicular Variance (m²)**: Variance of the perpendicular (shortest) distances from each generated point to the original trajectory polyline. Higher values indicate more uneven lateral path deviation.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps. Values range from 0° (parallel) to 180° (opposite direction).
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps. This captures variations in vehicle speed or trajectory compression/expansion.

Configurations are ranked by perpendicular variance (highest first). Each visualization shows the complete generated trajectory with lane markings for context.""")
    
    # Store metrics for summary chart
    config_metrics: list[dict[str, Any]] = []
    
    # Display each of the top 500
    for rank, (config_num, deviation, config) in enumerate(top_500, 1):
            st.markdown(f"#### Rank {rank}: Configuration #{config_num} (Perp. variance: {deviation:.4f} m²)")
            
            # Get configuration characteristics
            pdp_variant = config.get("pdp_variant", "fundamental")
            iterations = config.get("iterations", "N/A")
            threshold_mode = config.get("threshold_mode", "Percentage")
            max_threshold = config.get("max_threshold", 0.0)
            
            # Calculate angle and distance deviations
            successful_points = config.get("successful_points", [])
            
            # Create mapping from original_parent_idx to generated coordinates
            generated_coords_map: dict[int, np.ndarray] = {}
            for sp in successful_points:
                orig_idx = sp["original_parent_idx"]
                gen_coord = sp["point"]
                generated_coords_map[orig_idx] = gen_coord
            
            max_angle_deviation = 0.0
            max_distance_deviation = 0.0
            
            # Process each object
            global_idx = 0
            for oid in sorted(all_points_plot.keys()):
                n_pts = all_points_plot[oid].shape[0]
                original_pts = all_points_plot[oid]
                
                # Build generated points for this object
                generated_pts_list = []
                for local_idx in range(n_pts):
                    if global_idx in generated_coords_map:
                        coord = generated_coords_map[global_idx]
                    else:
                        coord = original_pts[local_idx]
                    generated_pts_list.append(coord)
                    global_idx += 1
                
                generated_pts = np.array(generated_pts_list)
                
                # Calculate deviations for consecutive points
                for i in range(n_pts - 1):
                    # Original angle and distance
                    orig_dx = original_pts[i+1, 0] - original_pts[i, 0]
                    orig_dy = original_pts[i+1, 1] - original_pts[i, 1]
                    orig_angle = np.degrees(np.arctan2(orig_dy, orig_dx))
                    orig_dist = np.sqrt(orig_dx**2 + orig_dy**2)
                    
                    # Generated angle and distance
                    gen_dx = generated_pts[i+1, 0] - generated_pts[i, 0]
                    gen_dy = generated_pts[i+1, 1] - generated_pts[i, 1]
                    gen_angle = np.degrees(np.arctan2(gen_dy, gen_dx))
                    gen_dist = np.sqrt(gen_dx**2 + gen_dy**2)
                    
                    # Angle deviation (handle wraparound)
                    angle_diff = abs(gen_angle - orig_angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    max_angle_deviation = max(max_angle_deviation, angle_diff)
                    
                    # Distance deviation
                    dist_diff = abs(gen_dist - orig_dist)
                    max_distance_deviation = max(max_distance_deviation, dist_diff)
            
            # Store metrics
            config_metrics.append({
                "config_num": config_num,
                "rank": rank,
                "perp_variance": deviation,
                "max_angle_dev": max_angle_deviation,
                "max_dist_dev": max_distance_deviation
            })
            
            # Format threshold display
            if threshold_mode == "Percentage":
                threshold_display = f"{max_threshold:.1%}"
            else:
                threshold_display = str(int(max_threshold))
            
            # Create visualization - only right plot for download
            fig = Figure(figsize=(6, 5.5), dpi=120)
            canvas = FigureCanvas(fig)
            
            # Single subplot: generated (right)
            ax_right = fig.add_subplot(111)
            
            # Setup axis
            ax_right.set_xlim(*XLIM)
            ax_right.set_ylim(*YLIM)
            ax_right.set_aspect("equal", adjustable="box")
            for sp in ax_right.spines.values():
                sp.set_linewidth(0.9)
                sp.set_color("#222")
            ax_right.tick_params(axis="both", labelsize=9, width=0.8, color="#222")
            ax_right.set_xlabel("d1", fontsize=11, labelpad=8)
            ax_right.set_ylabel("d2", fontsize=11, labelpad=8)
            
            # Add banner text inside the top of right subplot (two lines)
            banner_line1 = f"Variant 1/1 ({pdp_variant}) | Config {config_num} | Iteration {iterations}"
            banner_line2 = f"Max threshold {threshold_display}"
            banner_text = f"{banner_line1}\n{banner_line2}"
            ax_right.text(0.5, 0.97, banner_text, 
                        transform=ax_right.transAxes,
                        ha='center', va='top', 
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5DEB3', edgecolor='black', linewidth=1.5))
            
            # Define colors locally
            obj_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
            
            # Calculate average vehicle y-position for lane positioning
            all_y_coords = []
            for o_id in sorted(all_points_plot.keys()):
                pts = all_points_plot[o_id]
                if pts.shape[0] > 0:
                    all_y_coords.extend(pts[:, 1].tolist())
            
            avg_y = float(np.mean(all_y_coords)) if all_y_coords else 0.0
            
            # Draw generated on right
            ax_right.set_title(f"Generated (Config #{config_num})", fontsize=12, fontweight='bold')
            
            # Draw lanes positioned at vehicle location (skip for custom uploads)
            if not _is_custom_upload:
                lane_width = 3.0
                lane_offsets = [-lane_width, 0.0, lane_width]
                for offset in lane_offsets:
                    lane_y = avg_y + offset
                    ax_right.axhline(y=lane_y, color='black', linewidth=0.8, linestyle='-' if offset in [-lane_width, lane_width] else '--')
            
            # Build the generated configuration
            successful_points = config.get("successful_points", [])
            
            # Create a mapping from original_parent_idx to generated coordinates
            generated_coords_map: dict[int, np.ndarray] = {}
            for sp in successful_points:
                orig_idx = sp["original_parent_idx"]
                gen_coord = sp["point"]
                generated_coords_map[orig_idx] = gen_coord
            
            # Build complete point set for visualization (all objects, all timestamps)
            generated_points_dict: dict[int, list[tuple[np.ndarray, float]]] = {}
            
            global_idx = 0
            for oid in sorted(all_points_plot.keys()):
                n_pts = all_points_plot[oid].shape[0]
                vals = all_vals_plot[oid]
                
                if oid not in generated_points_dict:
                    generated_points_dict[oid] = []
                
                for local_idx in range(n_pts):
                    # Use generated coordinate if available, otherwise use original
                    if global_idx in generated_coords_map:
                        coord = generated_coords_map[global_idx]
                    else:
                        coord = all_points_plot[oid][local_idx]
                    
                    t_val = float(vals[local_idx])
                    generated_points_dict[oid].append((coord, t_val))
                    global_idx += 1
            
            # Draw generated trajectories
            for i, o_id in enumerate(sorted(generated_points_dict.keys())):
                points_list = generated_points_dict[o_id]
                # Sort by timestamp
                points_list.sort(key=lambda x: x[1])
                pts_array = np.array([p[0] for p in points_list])
                vals_array = np.array([p[1] for p in points_list])
                
                color = obj_colors[i % len(obj_colors)]
                label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
                ax_right.plot(pts_array[:, 0], pts_array[:, 1], '-', color=color, linewidth=1.5, alpha=0.7, label=label)
                
                # Add point annotations
                offsets = [(3, 3), (3, -8), (-8, 3)]
                for j, ((x, y), tval) in enumerate(zip(pts_array, vals_array)):
                    ax_right.scatter([x], [y], s=25, zorder=10, color=color, marker='o')
                    off = offsets[j % len(offsets)]
                    try:
                        tnum = float(tval)
                    except (TypeError, ValueError):
                        tnum = float(np.array(tval, dtype=float))
                    lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
                    # Only add label if both label and lbl are valid
                    if label and lbl:
                        try:
                            label_text = f"$\\mathit{{{label}}}_{{{lbl}}}$"
                            ax_right.annotate(
                                label_text,
                                xy=(x, y),
                                xytext=off,
                                textcoords="offset points",
                                fontsize=8,
                                color=color,
                                ha="center",
                                va="center",
                            )
                        except Exception as e:
                            logger.warning(f"[RENDER] LaTeX annotation failed for {label}_{lbl}: {e}")
            
            try:
                fig.tight_layout()
            except Exception as e:
                logger.warning(f"[RENDER] tight_layout failed: {e}")
            
            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            
            # Display image
            st.image(buf, use_container_width=True)
            
            # Download button
            buf.seek(0)
            st.download_button(
                label=f"📥 Download Configuration #{config_num}",
                data=buf,
                file_name=f"config_{config_num}_deviation_{deviation:.1f}m.png",
                mime="image/png",
                key=f"download_config_{config_num}_top300"
            )
            
            st.markdown("---")
    
    # Calculate statistics
    angle_devs = [m['max_angle_dev'] for m in config_metrics]
    dist_devs = [m['max_dist_dev'] for m in config_metrics]
    pv_vals = [m['perp_variance'] for m in config_metrics]
    
    mean_angle = float(np.mean(angle_devs))
    std_angle = float(np.std(angle_devs))
    mean_dist = float(np.mean(dist_devs))
    std_dist = float(np.std(dist_devs))
    mean_pv = float(np.mean(pv_vals))
    std_pv = float(np.std(pv_vals))
    
    # Display statistics summary
    st.markdown("### Statistics Summary (Top 500 configurations)")
    st.caption("Mean ± standard deviation calculated across the top 500 most deviating configurations. Lower standard deviation indicates consistent behavior across these high-deviation cases.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Perp. Variance", f"{mean_pv:.4f} m² ± {std_pv:.4f}")
    with col2:
        st.metric("Max Angle Dev", f"{mean_angle:.3f}° ± {std_angle:.3f}°")
    with col3:
        st.metric("Max Distance Dev", f"{mean_dist:.3f}m ± {std_dist:.3f}m")
    
    st.markdown("---")
    
    # Display data table
    st.markdown("### Top 500 Configurations - Detailed Metrics")
    st.caption("""Complete metrics for the most deviating configurations (selected from 5 generated). Select cells and copy (Ctrl+C) to paste into Excel, PowerPoint, or other applications.
    
- **Rank**: Position in descending order of perpendicular variance (1 = highest variance)
- **Config #**: Unique configuration identifier from the generation batch (1-5)
- **Perp. Variance (m²)**: Variance of perpendicular distances from generated points to the original path
- **Max Angle Dev (°)**: Largest trajectory angle change between consecutive timestamps (per configuration)
- **Max Distance Dev (m)**: Largest inter-point spacing change between consecutive timestamps (per configuration)""")
    
    # Create DataFrame for display
    df_metrics = pd.DataFrame(config_metrics)
    df_metrics = df_metrics[['rank', 'config_num', 'perp_variance', 'max_angle_dev', 'max_dist_dev']]
    df_metrics.columns = ['Rank', 'Config #', 'Perp. Variance (m²)', 'Max Angle Dev (°)', 'Max Distance Dev (m)']
    
    # Format numeric columns
    df_metrics['Perp. Variance (m²)'] = df_metrics['Perp. Variance (m²)'].apply(lambda x: f"{x:.4f}")
    df_metrics['Max Angle Dev (°)'] = df_metrics['Max Angle Dev (°)'].apply(lambda x: f"{x:.3f}")
    df_metrics['Max Distance Dev (m)'] = df_metrics['Max Distance Dev (m)'].apply(lambda x: f"{x:.3f}")
    
    # Display table
    st.dataframe(df_metrics, use_container_width=True, height=400)
    
    # Add download button for CSV
    csv = df_metrics.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="top_500_configurations_metrics.csv",
        mime="text/csv",
        key="download_metrics_csv_top500",
        help="Download all metrics as CSV file for further analysis in Excel, Python, R, etc."
    )
    
    st.markdown("---")
    
    # Display summary metrics chart
    st.markdown("### Maximum Deviations Summary (Top 500 from 5 generated)")
    st.caption("""Visual comparison of maximum deviations across the top 500 configurations. Red dashed line indicates the mean value calculated from these 500 configurations.
    
- **Left chart**: Maximum angle deviation shows the largest directional change in any trajectory segment
- **Right chart**: Maximum distance deviation shows the largest speed/spacing variation in any trajectory segment

These metrics help identify configurations with extreme local variations, even if their perpendicular variance is moderate.""")
    
    # Create bar chart
    fig_metrics = Figure(figsize=(12, 5), dpi=100)
    canvas_metrics = FigureCanvas(fig_metrics)
    
    ax1 = fig_metrics.add_subplot(121)
    ax2 = fig_metrics.add_subplot(122)
    
    # Extract data
    config_labels = [f"#{m['config_num']}" for m in config_metrics]
    x_positions = range(len(config_labels))
    
    # Angle deviations bar chart
    ax1.bar(x_positions, angle_devs, color='#FF7F0E', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Configuration Rank", fontsize=11)
    ax1.set_ylabel("Max Angle Deviation (degrees)", fontsize=11)
    ax1.set_title("Maximum Angle Deviation (Top 500)", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=mean_angle, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_angle:.1f}°')
    ax1.legend()
    
    # Only show x-tick labels for every 30th config
    ax1.set_xticks([i for i in range(0, len(config_labels), 50)])
    ax1.set_xticklabels([config_labels[i] for i in range(0, len(config_labels), 50)], rotation=45, ha='right')
    
    # Distance deviations bar chart
    ax2.bar(x_positions, dist_devs, color='#1F77B4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel("Configuration Rank", fontsize=11)
    ax2.set_ylabel("Max Distance Deviation (m)", fontsize=11)
    ax2.set_title("Maximum Distance Deviation (Top 500)", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}m')
    ax2.legend()
    
    # Only show x-tick labels for every 30th config
    ax2.set_xticks([i for i in range(0, len(config_labels), 50)])
    ax2.set_xticklabels([config_labels[i] for i in range(0, len(config_labels), 50)], rotation=45, ha='right')
    
    fig_metrics.tight_layout()
    
    # Display chart
    buf_metrics = io.BytesIO()
    fig_metrics.savefig(buf_metrics, format='png', dpi=100, bbox_inches='tight')
    buf_metrics.seek(0)
    st.image(buf_metrics, use_container_width=True)
    
    st.markdown("---")
    
    # Add a clear button
    if st.button("Clear Results & Cache", key="clear_top500_results"):
        st.session_state["_generate_5000_requested"] = False
        st.session_state["_generate_5000_results"] = None
        st.cache_data.clear()
        st.rerun()

# Display results for 50-config reinsertion generation if they exist
if st.session_state.get("_generate_50_results", None):
    top_5 = st.session_state["_generate_50_results"]
    
    st.markdown("---")
    st.markdown("### Top 25 Most Deviating Configurations (from 200 generated, 200 iterations each)")
    st.markdown("""These configurations exhibit the largest spatial deviations from the original while maintaining the PDP inequality pattern.
    
**Generation settings**: 200 configurations × 200 iterations, focused on the reinsertion zone.

**Deviation Metrics (calculated per configuration):**
- **Perpendicular Variance (m²)**: Variance of the perpendicular (shortest) distances from each generated point to the original trajectory polyline. Higher values indicate more uneven lateral path deviation.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps.
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps.

Configurations are ranked by perpendicular variance (highest first).""")
    
    # Store metrics for summary chart
    config_metrics: list[dict[str, Any]] = []
    
    # Display each of the top 5
    for rank, (config_num, deviation, config) in enumerate(top_5, 1):
            st.markdown(f"#### Rank {rank}: Configuration #{config_num} (Perp. variance: {deviation:.4f} m²)")
            
            # Get configuration characteristics
            pdp_variant = config.get("pdp_variant", "fundamental")
            iterations = config.get("iterations", "N/A")
            threshold_mode = config.get("threshold_mode", "Percentage")
            max_threshold = config.get("max_threshold", 0.0)
            
            # Calculate angle and distance deviations
            successful_points = config.get("successful_points", [])
            
            # Create mapping from original_parent_idx to generated coordinates
            generated_coords_map: dict[int, np.ndarray] = {}
            for sp in successful_points:
                orig_idx = sp["original_parent_idx"]
                gen_coord = sp["point"]
                generated_coords_map[orig_idx] = gen_coord
            
            max_angle_deviation = 0.0
            max_distance_deviation = 0.0
            
            # Process each object
            global_idx = 0
            for oid in sorted(all_points_plot.keys()):
                n_pts = all_points_plot[oid].shape[0]
                original_pts = all_points_plot[oid]
                
                # Build generated trajectory for this object
                gen_pts = original_pts.copy()
                for local_i in range(n_pts):
                    gi = global_idx + local_i
                    if gi in generated_coords_map:
                        gen_pts[local_i] = generated_coords_map[gi]
                
                # Calculate angle deviations between consecutive timestamps
                for i in range(1, n_pts):
                    orig_dx = original_pts[i, 0] - original_pts[i-1, 0]
                    orig_dy = original_pts[i, 1] - original_pts[i-1, 1]
                    gen_dx = gen_pts[i, 0] - gen_pts[i-1, 0]
                    gen_dy = gen_pts[i, 1] - gen_pts[i-1, 1]
                    
                    orig_angle = np.arctan2(orig_dy, orig_dx)
                    gen_angle = np.arctan2(gen_dy, gen_dx)
                    angle_diff = abs(np.degrees(orig_angle - gen_angle))
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    max_angle_deviation = max(max_angle_deviation, angle_diff)
                    
                    orig_dist = np.linalg.norm([orig_dx, orig_dy])
                    gen_dist = np.linalg.norm([gen_dx, gen_dy])
                    dist_diff = abs(gen_dist - orig_dist)
                    max_distance_deviation = max(max_distance_deviation, dist_diff)
                
                global_idx += n_pts
            
            # Store metrics
            config_metrics.append({
                "config_num": config_num,
                "rank": rank,
                "perp_variance": deviation,
                "max_angle_deviation": max_angle_deviation,
                "max_distance_deviation": max_distance_deviation
            })
            
            # Display summary metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Perp. Variance", f"{deviation:.4f} m²")
            mc2.metric("Max Angle Δ", f"{max_angle_deviation:.1f}°")
            mc3.metric("Max Distance Δ", f"{max_distance_deviation:.2f}m")
            mc4.metric("Iterations", f"{iterations}")
            
            # Draw trajectory plot — zoomed into most deviant area
            fig_gen = Figure(figsize=(12, 4))
            ax_gen = fig_gen.add_subplot(111)
            
            # First pass: find the most deviant point to center the view
            max_dev_dist = 0.0
            max_dev_orig = np.array([0.0, 0.0])
            max_dev_gen = np.array([0.0, 0.0])
            all_plot_data: list[tuple[np.ndarray, np.ndarray]] = []  # (original, generated) per object
            
            global_idx = 0
            for oid in sorted(all_points_plot.keys()):
                n_pts = all_points_plot[oid].shape[0]
                original_pts = all_points_plot[oid]
                
                # Build generated trajectory
                gen_pts = original_pts.copy()
                for local_i in range(n_pts):
                    gi = global_idx + local_i
                    if gi in generated_coords_map:
                        gen_pts[local_i] = generated_coords_map[gi]
                
                all_plot_data.append((original_pts, gen_pts))
                
                # Track most deviant point
                for local_i in range(n_pts):
                    d = np.linalg.norm(gen_pts[local_i] - original_pts[local_i])
                    if d > max_dev_dist:
                        max_dev_dist = d
                        max_dev_orig = original_pts[local_i].copy()
                        max_dev_gen = gen_pts[local_i].copy()
                
                global_idx += n_pts
            
            # Compute zoom window centered on the most deviant point
            center_x = (max_dev_orig[0] + max_dev_gen[0]) / 2
            center_y = (max_dev_orig[1] + max_dev_gen[1]) / 2
            # Show ±150m in x around the most deviant point
            x_half = 150.0
            x_lo, x_hi = center_x - x_half, center_x + x_half
            
            # Compute y range from all data within the x window, then add padding
            y_vals_in_window: list[float] = []
            for orig_pts, g_pts in all_plot_data:
                for pts in [orig_pts, g_pts]:
                    mask = (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
                    if mask.any():
                        y_vals_in_window.extend(pts[mask, 1].tolist())
            
            if y_vals_in_window:
                y_min_data = min(y_vals_in_window)
                y_max_data = max(y_vals_in_window)
                y_pad = max((y_max_data - y_min_data) * 0.3, 1.0)  # at least 1m padding
                y_lo = y_min_data - y_pad
                y_hi = y_max_data + y_pad
            else:
                y_lo, y_hi = -10.0, 0.0
            
            # Second pass: draw trajectories
            for idx_oid, oid in enumerate(sorted(all_points_plot.keys())):
                original_pts, gen_pts = all_plot_data[idx_oid]
                label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
                color = f"C{oid}"
                
                # Draw original
                ax_gen.plot(original_pts[:, 0], original_pts[:, 1],
                           linewidth=1.0, color=color, alpha=0.3, linestyle='--',
                           label=f"{label} original")
                # Draw generated
                ax_gen.plot(gen_pts[:, 0], gen_pts[:, 1],
                           linewidth=1.5, color=color, alpha=1.0,
                           label=f"{label} generated")
            
            # Mark most deviant point
            ax_gen.annotate(
                f"max Δ={max_dev_dist:.2f}m",
                xy=(max_dev_gen[0], max_dev_gen[1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=7, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
            )
            
            ax_gen.set_xlim(x_lo, x_hi)
            ax_gen.set_ylim(y_lo, y_hi)
            ax_gen.legend(fontsize=7, loc='upper left')
            ax_gen.set_xlabel("x (m)")
            ax_gen.set_ylabel("y (m)")
            ax_gen.set_title(f"Config #{config_num} — Perp. variance: {deviation:.4f} m² (zoomed to max deviation)")
            fig_gen.tight_layout()
            
            buf = io.BytesIO()
            fig_gen.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            
            st.markdown("---")
    
    # Summary metrics table
    if config_metrics:
        st.markdown("#### Summary")
        metrics_df = pd.DataFrame(config_metrics)
        st.dataframe(metrics_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={
                "rank": "Rank",
                "config_num": "Config #",
                "perp_variance": "Perp. Variance (m²)",
                "max_angle_deviation": "Max Angle Δ (°)",
                "max_distance_deviation": "Max Distance Δ (m)"
            }
        ), use_container_width=True)
    
    # Clear button
    if st.button("Clear Results & Cache", key="clear_top5_results"):
        st.session_state["_generate_50_requested"] = False
        st.session_state["_generate_50_results"] = None
        st.cache_data.clear()
        st.rerun()

# ============= Display results for ext30 (30 configs x 600 iterations, top 3, with GIF) ============
if st.session_state.get("_generate_ext30_results", None):
    _ext30_top3 = st.session_state["_generate_ext30_results"]
    # Use full-timestamp data stored during generation (not sidebar-filtered globals)
    _ext30_display_points = st.session_state.get("_ext30_full_points_plot", all_points_plot)
    _ext30_display_vals = st.session_state.get("_ext30_full_vals_plot", all_vals_plot)

    st.markdown("---")
    _ext30_n_ts = sum(_ext30_display_points[oid].shape[0] for oid in sorted(_ext30_display_points.keys()))
    _ext30_n_ts_per_obj = {oid: _ext30_display_points[oid].shape[0] for oid in sorted(_ext30_display_points.keys())}
    st.markdown(f"### Top 3 Most Deviating Configurations (from {MAX_FILTER_CONFIGS} generated, 2500 iterations each) — {list(_ext30_n_ts_per_obj.values())[0]} timestamps per object")
    st.markdown(f"""
**Generation settings**: {MAX_FILTER_CONFIGS} configurations × 2500 iterations | **{_ext30_n_ts} total points** ({', '.join(f'object {oid}: {n}' for oid, n in _ext30_n_ts_per_obj.items())}).

**Deviation Metrics:**
- **Perpendicular Variance (m²)**: Variance of perpendicular distances from generated points to the original trajectory.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps.
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps.

Each configuration includes an **animated GIF** download showing the trajectory building up over timestamps.""")

    _ext30_config_metrics: list[dict[str, Any]] = []

    for _ext30_rank, (_ext30_cnum, _ext30_dev, _ext30_cfg) in enumerate(_ext30_top3, 1):
        st.markdown(f"#### Rank {_ext30_rank}: Configuration #{_ext30_cnum} (Perp. variance: {_ext30_dev:.4f} m²)")

        _ext30_pdp_variant = _ext30_cfg.get("pdp_variant", "fundamental")
        _ext30_iterations = _ext30_cfg.get("iterations", "N/A")
        _ext30_threshold_mode = _ext30_cfg.get("threshold_mode", "Percentage")
        _ext30_max_threshold = _ext30_cfg.get("max_threshold", 0.0)

        # Build generated coordinate map
        _ext30_sp = _ext30_cfg.get("successful_points", [])
        _ext30_gen_map: dict[int, np.ndarray] = {}
        for sp in _ext30_sp:
            _ext30_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        # Calculate angle and distance deviations
        _ext30_max_angle = 0.0
        _ext30_max_dist = 0.0
        _ext30_sorted_oids = sorted(_ext30_display_points.keys())
        _ext30_all_plot_data: list[tuple[np.ndarray, np.ndarray]] = []  # (original, generated) per object
        _ext30_ts_data: dict[int, np.ndarray | None] = {}  # filtered timestamps per object
        filtered_counts = {}
        _ext30_gi = 0
        for oid in _ext30_sorted_oids:
            orig_pts = _ext30_display_points[oid]
            orig_ts = _ext30_display_vals[oid] if oid in _ext30_display_vals else None
            n_pts = orig_pts.shape[0]
            gen_pts = orig_pts.copy()
            for li in range(n_pts):
                gi = _ext30_gi + li
                if gi in _ext30_gen_map:
                    gen_pts[li] = _ext30_gen_map[gi]
            # Geen filtering — gebruik alle timestamps
            _ext30_all_plot_data.append((orig_pts, gen_pts))
            _ext30_ts_data[oid] = orig_ts
            filtered_counts[oid] = n_pts

            # Afwijkingsberekeningen op alle punten
            if orig_pts.shape[0] > 1:
                for i in range(1, orig_pts.shape[0]):
                    orig_dx = orig_pts[i, 0] - orig_pts[i-1, 0]
                    orig_dy = orig_pts[i, 1] - orig_pts[i-1, 1]
                    gen_dx = gen_pts[i, 0] - gen_pts[i-1, 0]
                    gen_dy = gen_pts[i, 1] - gen_pts[i-1, 1]
                    angle_diff = abs(np.degrees(np.arctan2(gen_dy, gen_dx) - np.arctan2(orig_dy, orig_dx)))
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    _ext30_max_angle = max(_ext30_max_angle, angle_diff)
                    dist_diff = abs(np.linalg.norm([gen_dx, gen_dy]) - np.linalg.norm([orig_dx, orig_dy]))
                    _ext30_max_dist = max(_ext30_max_dist, dist_diff)
            _ext30_gi += n_pts

        _ext30_config_metrics.append({
            "config_num": _ext30_cnum,
            "rank": _ext30_rank,
            "perp_variance": _ext30_dev,
            "max_angle_deviation": _ext30_max_angle,
            "max_distance_deviation": _ext30_max_dist,
        })

        # Display summary metrics
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Perp. Variance", f"{_ext30_dev:.4f} m²")
        _mc2.metric("Max Angle Δ", f"{_ext30_max_angle:.1f}°")
        _mc3.metric("Max Distance Δ", f"{_ext30_max_dist:.2f}m")
        _mc4.metric("Iterations", f"{_ext30_iterations}")

        # ---------- Static trajectory plot (zoomed to max deviation) ----------
        _ext30_max_dev_dist = 0.0
        _ext30_max_dev_orig = np.array([0.0, 0.0])
        _ext30_max_dev_gen = np.array([0.0, 0.0])

        for idx_oid, oid in enumerate(_ext30_sorted_oids):
            orig_pts, g_pts = _ext30_all_plot_data[idx_oid]
            for li in range(orig_pts.shape[0]):
                d = float(np.linalg.norm(g_pts[li] - orig_pts[li]))
                if d > _ext30_max_dev_dist:
                    _ext30_max_dev_dist = d
                    _ext30_max_dev_orig = orig_pts[li].copy()
                    _ext30_max_dev_gen = g_pts[li].copy()

        # Compute tight axis limits from actual data (all objects, orig + gen)
        _ext30_all_x: list[float] = []
        _ext30_all_y: list[float] = []
        for orig_pts, g_pts in _ext30_all_plot_data:
            _ext30_all_x.extend(orig_pts[:, 0].tolist())
            _ext30_all_x.extend(g_pts[:, 0].tolist())
            _ext30_all_y.extend(orig_pts[:, 1].tolist())
            _ext30_all_y.extend(g_pts[:, 1].tolist())

        _ext30_x_lo = min(_ext30_all_x) - max((max(_ext30_all_x) - min(_ext30_all_x)) * 0.05, 1.0)
        _ext30_x_hi = max(_ext30_all_x) + max((max(_ext30_all_x) - min(_ext30_all_x)) * 0.05, 1.0)
        _ext30_y_lo = min(_ext30_all_y) - max((max(_ext30_all_y) - min(_ext30_all_y)) * 0.08, 0.5)
        _ext30_y_hi = max(_ext30_all_y) + max((max(_ext30_all_y) - min(_ext30_all_y)) * 0.08, 0.5)

        # Pad the shorter axis so the plot ratio is at most ~1.5:1 (like the reference image)
        _ext30_dx = _ext30_x_hi - _ext30_x_lo
        _ext30_dy = _ext30_y_hi - _ext30_y_lo
        _ext30_target_ratio = 1.3  # width / height similar to the reference Configuration 1 plot
        if _ext30_dx > _ext30_dy * _ext30_target_ratio:
            # x range is too wide relative to y — expand y symmetrically
            _ext30_needed_dy = _ext30_dx / _ext30_target_ratio
            _ext30_pad = (_ext30_needed_dy - _ext30_dy) / 2
            _ext30_y_lo -= _ext30_pad
            _ext30_y_hi += _ext30_pad
            _ext30_dy = _ext30_needed_dy
        elif _ext30_dy * _ext30_target_ratio > _ext30_dx:
            # y range is too tall relative to x — expand x symmetrically
            _ext30_needed_dx = _ext30_dy * _ext30_target_ratio
            _ext30_pad = (_ext30_needed_dx - _ext30_dx) / 2
            _ext30_x_lo -= _ext30_pad
            _ext30_x_hi += _ext30_pad
            _ext30_dx = _ext30_needed_dx

        # Figure size: 10" wide (PPT-friendly), height from data ratio, equal scale guaranteed by matching fig ratio to data ratio
        _ext30_fig_w = 10.0
        _ext30_fig_h = _ext30_fig_w * (_ext30_dy / _ext30_dx) if _ext30_dx > 0 else 7.5
        _ext30_fig_h = max(3.0, min(_ext30_fig_h, 7.5))

        fig_static = Figure(figsize=(_ext30_fig_w, _ext30_fig_h), dpi=150)
        ax_s = fig_static.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_oid, oid in enumerate(_ext30_sorted_oids):
            orig_pts, g_pts = _ext30_all_plot_data[idx_oid]
            label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
            color = f"C{oid}"
            orig_label = f"{label} original" if idx_oid == 0 else None
            gen_label = f"{label} generated" if idx_oid == 0 else None
            # Timestamps per object (geen filtering)
            ts_filt = _ext30_ts_data.get(oid, None)
            expected_dt = 1
            # Plot alle opeenvolgende segmenten
            if orig_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(orig_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [orig_pts[i, 0], orig_pts[i+1, 0]]
                        yseg = [orig_pts[i, 1], orig_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.0, color=color, alpha=0.3, linestyle='--', label=orig_label if i == 0 else None)
            if g_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(g_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [g_pts[i, 0], g_pts[i+1, 0]]
                        yseg = [g_pts[i, 1], g_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.5, color=color, alpha=1.0, label=gen_label if i == 0 else None)
        ax_s.annotate(f"max Δ={_ext30_max_dev_dist:.2f}m", xy=(_ext30_max_dev_gen[0], _ext30_max_dev_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_ext30_x_lo, _ext30_x_hi)
        ax_s.set_ylim(_ext30_y_lo, _ext30_y_hi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)")
        ax_s.set_ylabel("d2 / y-as (m)")
        # Toon het juiste aantal timestamps per object (na filtering)
        _ext30_ts_per_o = filtered_counts if 'filtered_counts' in locals() else {oid: _ext30_display_points[oid].shape[0] for oid in _ext30_sorted_oids}
        _ext30_ts_info = ", ".join(f"obj{oid}:{n}" for oid, n in _ext30_ts_per_o.items())
        ax_s.set_title(f"Config #{_ext30_cnum} — PV={_ext30_dev:.4f} m² — {_ext30_ts_info} timestamps")
        fig_static.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        _buf_s = io.BytesIO()
        fig_static.savefig(_buf_s, format='png', dpi=150)
        _buf_s.seek(0)
        st.image(_buf_s, use_container_width=True)

        # ---------- Animated GIF: trajectory building up over timestamps ----------
        # Collect all timestamps across all objects (sorted)
        _ext30_all_ts: list[float] = []
        for oid in _ext30_sorted_oids:
            _ext30_all_ts.extend(_ext30_display_vals[oid].tolist())
        _ext30_unique_ts = sorted(set(_ext30_all_ts))
        _ext30_n_frames = len(_ext30_unique_ts)

        # Compute global axis limits from all data (original + generated)
        _ext30_all_x: list[float] = []
        _ext30_all_y: list[float] = []
        for orig_pts, g_pts in _ext30_all_plot_data:
            _ext30_all_x.extend(orig_pts[:, 0].tolist())
            _ext30_all_x.extend(g_pts[:, 0].tolist())
            _ext30_all_y.extend(orig_pts[:, 1].tolist())
            _ext30_all_y.extend(g_pts[:, 1].tolist())
        _ext30_gif_xmin = min(_ext30_all_x) - max((max(_ext30_all_x) - min(_ext30_all_x)) * 0.05, 1.0)
        _ext30_gif_xmax = max(_ext30_all_x) + max((max(_ext30_all_x) - min(_ext30_all_x)) * 0.05, 1.0)
        _ext30_gif_ymin = min(_ext30_all_y) - max((max(_ext30_all_y) - min(_ext30_all_y)) * 0.1, 0.5)
        _ext30_gif_ymax = max(_ext30_all_y) + max((max(_ext30_all_y) - min(_ext30_all_y)) * 0.1, 0.5)

        # Pad shorter axis for GIF (same logic as static plot)
        _ext30_gif_dx = _ext30_gif_xmax - _ext30_gif_xmin
        _ext30_gif_dy = _ext30_gif_ymax - _ext30_gif_ymin
        if _ext30_gif_dx > _ext30_gif_dy * _ext30_target_ratio:
            _g_pad = (_ext30_gif_dx / _ext30_target_ratio - _ext30_gif_dy) / 2
            _ext30_gif_ymin -= _g_pad
            _ext30_gif_ymax += _g_pad
            _ext30_gif_dy = _ext30_gif_ymax - _ext30_gif_ymin
        elif _ext30_gif_dy * _ext30_target_ratio > _ext30_gif_dx:
            _g_pad = (_ext30_gif_dy * _ext30_target_ratio - _ext30_gif_dx) / 2
            _ext30_gif_xmin -= _g_pad
            _ext30_gif_xmax += _g_pad
            _ext30_gif_dx = _ext30_gif_xmax - _ext30_gif_xmin

        _ext30_gif_fw = 10.0
        _ext30_gif_fh = _ext30_gif_fw * (_ext30_gif_dy / _ext30_gif_dx) if _ext30_gif_dx > 0 else 7.5
        _ext30_gif_fh = max(3.0, min(_ext30_gif_fh, 7.5))

        # Build per-object ordered (coord, t) lists for original and generated
        _ext30_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _ext30_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _ext30_gi2 = 0
        for oid in _ext30_sorted_oids:
            n_pts = _ext30_display_points[oid].shape[0]
            vals = _ext30_display_vals[oid]
            _ext30_orig_by_obj[oid] = []
            _ext30_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_val = float(vals[li])
                orig_coord = _ext30_display_points[oid][li]
                gen_coord = _ext30_gen_map.get(_ext30_gi2 + li, orig_coord)
                _ext30_orig_by_obj[oid].append((t_val, orig_coord))
                _ext30_gen_by_obj[oid].append((t_val, np.array(gen_coord)))
            _ext30_gi2 += n_pts

        _ext30_gif_frames: list[PILImage.Image] = []
        _ext30_gif_progress = st.progress(0)
        _ext30_gif_status = st.empty()

        for frame_idx, t_cutoff in enumerate(_ext30_unique_ts):
            _ext30_gif_status.text(f"Rendering GIF frame {frame_idx + 1}/{_ext30_n_frames} (t ≤ {t_cutoff:g})...")
            fig_frame = Figure(figsize=(_ext30_gif_fw, _ext30_gif_fh), dpi=150)
            ax_f = fig_frame.add_subplot(111)
            ax_f.set_xlim(_ext30_gif_xmin, _ext30_gif_xmax)
            ax_f.set_ylim(_ext30_gif_ymin, _ext30_gif_ymax)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            _ext30_ts_per_o2 = {oid: _ext30_display_points[oid].shape[0] for oid in _ext30_sorted_oids}
            _ext30_ts_info2 = ", ".join(f"obj{oid}:{n}" for oid, n in _ext30_ts_per_o2.items())
            ax_f.set_title(f"Config #{_ext30_cnum} — t ≤ {t_cutoff:g} — {_ext30_ts_info2} ts  (PV={_ext30_dev:.4f} m²)", fontsize=9)

            for idx_oid, oid in enumerate(_ext30_sorted_oids):
                label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
                color = f"C{oid}"
                # Only plot consecutive points for this object, never connect last to first, never connect between objects
                orig_pts_up = [coord for (t_val, coord) in _ext30_orig_by_obj[oid] if t_val <= t_cutoff]
                if len(orig_pts_up) > 1:
                    orig_arr = np.array(orig_pts_up)
                    ax_f.plot(orig_arr[:, 0], orig_arr[:, 1], linewidth=1.0, color=color, alpha=0.35, linestyle='--', label=f"{label} orig")
                # Generated trajectory up to t_cutoff
                gen_pts_up = [coord for (t_val, coord) in _ext30_gen_by_obj[oid] if t_val <= t_cutoff]
                if len(gen_pts_up) > 1:
                    gen_arr = np.array(gen_pts_up)
                    ax_f.plot(gen_arr[:, 0], gen_arr[:, 1], linewidth=1.8, color=color, alpha=1.0, label=f"{label} gen")

            ax_f.legend(fontsize=7, loc='upper left')
            fig_frame.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)

            # Render to PIL image
            _buf_frame = io.BytesIO()
            fig_frame.savefig(_buf_frame, format='png', dpi=150)
            _buf_frame.seek(0)
            _ext30_gif_frames.append(PILImage.open(_buf_frame).copy())
            plt.close(fig_frame)

            _ext30_gif_progress.progress((frame_idx + 1) / _ext30_n_frames)

        _ext30_gif_progress.empty()
        _ext30_gif_status.empty()

        # Assemble GIF
        if _ext30_gif_frames:
            # Add a longer pause on the last frame
            _ext30_gif_buf = io.BytesIO()
            _ext30_durations = [GIF_FRAME_DURATION_MS] * len(_ext30_gif_frames)  # 200ms per frame
            if _ext30_durations:
                _ext30_durations[-1] = GIF_LAST_FRAME_PAUSE_MS  # Pause 1.5s on last frame
            _ext30_gif_frames[0].save(
                _ext30_gif_buf,
                format='GIF',
                save_all=True,
                append_images=_ext30_gif_frames[1:],
                duration=_ext30_durations,
                loop=0,
            )
            _ext30_gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_ext30_cnum}",
                data=_ext30_gif_buf,
                file_name=f"config_{_ext30_cnum}_animation.gif",
                mime="image/gif",
                key=f"dl_gif_ext30_{_ext30_rank}",
            )

        st.markdown("---")

    # Summary table
    if _ext30_config_metrics:
        st.markdown("#### Summary")
        _ext30_df = pd.DataFrame(_ext30_config_metrics)
        st.dataframe(_ext30_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={
                "rank": "Rank",
                "config_num": "Config #",
                "perp_variance": "Perp. Variance (m²)",
                "max_angle_deviation": "Max Angle Δ (°)",
                "max_distance_deviation": "Max Distance Δ (m)",
            }
        ), use_container_width=True)

    # Clear button
    if st.button("Clear Results & Cache", key="clear_ext30_results"):
        st.session_state["_generate_ext30_requested"] = False
        st.session_state["_generate_ext30_results"] = None
        st.session_state.pop("_ext30_full_points_plot", None)
        st.session_state.pop("_ext30_full_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# ============= Display results for ext30_rough (rough d1=d2=0.30m, top 3, with GIF) ============
if st.session_state.get("_generate_ext30_rough_results", None):
    _r30_top3 = st.session_state["_generate_ext30_rough_results"]
    _r30_display_points = st.session_state.get("_ext30_rough_full_points_plot", all_points_plot)
    _r30_display_vals = st.session_state.get("_ext30_rough_full_vals_plot", all_vals_plot)

    st.markdown("---")
    _r30_n_ts = sum(_r30_display_points[oid].shape[0] for oid in sorted(_r30_display_points.keys()))
    _r30_n_ts_per_obj = {oid: _r30_display_points[oid].shape[0] for oid in sorted(_r30_display_points.keys())}
    st.markdown(f"### Top 3 Most Deviating (rough d1=d2=0.30m) — {MAX_FILTER_CONFIGS} configs, 2500 iter — {list(_r30_n_ts_per_obj.values())[0]} timestamps per object")
    st.markdown(f"""
**Generation settings**: {MAX_FILTER_CONFIGS} configurations × 2500 iterations | **{_r30_n_ts} total points** ({', '.join(f'object {oid}: {n}' for oid, n in _r30_n_ts_per_obj.items())}).
**Forced PDP settings**: fundamental + rough d1=0.30m, d2=0.30m (buffer=0).

**Deviation Metrics:**
- **Perpendicular Variance (m²)**: Variance of perpendicular distances from generated points to the original trajectory.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps.
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps.

Each configuration includes an **animated GIF** download showing the trajectory building up over timestamps.""")

    _r30_config_metrics: list[dict[str, Any]] = []

    for _r30_rank, (_r30_cnum, _r30_dev, _r30_cfg) in enumerate(_r30_top3, 1):
        st.markdown(f"#### Rank {_r30_rank}: Configuration #{_r30_cnum} (Perp. variance: {_r30_dev:.4f} m²)")

        _r30_pdp_variant = _r30_cfg.get("pdp_variant", "fundamental")
        _r30_iterations = _r30_cfg.get("iterations", "N/A")
        _r30_threshold_mode = _r30_cfg.get("threshold_mode", "Percentage")
        _r30_max_threshold = _r30_cfg.get("max_threshold", 0.0)

        # Build generated coordinate map
        _r30_sp = _r30_cfg.get("successful_points", [])
        _r30_gen_map: dict[int, np.ndarray] = {}
        for sp in _r30_sp:
            _r30_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        # Calculate angle and distance deviations
        _r30_max_angle = 0.0
        _r30_max_dist = 0.0
        _r30_sorted_oids = sorted(_r30_display_points.keys())
        _r30_all_plot_data: list[tuple[np.ndarray, np.ndarray]] = []
        _r30_ts_data: dict[int, np.ndarray | None] = {}
        _r30_filtered_counts = {}
        _r30_gi = 0
        for oid in _r30_sorted_oids:
            orig_pts = _r30_display_points[oid]
            orig_ts = _r30_display_vals[oid] if oid in _r30_display_vals else None
            n_pts = orig_pts.shape[0]
            gen_pts = orig_pts.copy()
            for li in range(n_pts):
                gi = _r30_gi + li
                if gi in _r30_gen_map:
                    gen_pts[li] = _r30_gen_map[gi]
            _r30_all_plot_data.append((orig_pts, gen_pts))
            _r30_ts_data[oid] = orig_ts
            _r30_filtered_counts[oid] = n_pts

            if orig_pts.shape[0] > 1:
                for i in range(1, orig_pts.shape[0]):
                    orig_dx = orig_pts[i, 0] - orig_pts[i-1, 0]
                    orig_dy = orig_pts[i, 1] - orig_pts[i-1, 1]
                    gen_dx = gen_pts[i, 0] - gen_pts[i-1, 0]
                    gen_dy = gen_pts[i, 1] - gen_pts[i-1, 1]
                    angle_diff = abs(np.degrees(np.arctan2(gen_dy, gen_dx) - np.arctan2(orig_dy, orig_dx)))
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    _r30_max_angle = max(_r30_max_angle, angle_diff)
                    dist_diff = abs(np.linalg.norm([gen_dx, gen_dy]) - np.linalg.norm([orig_dx, orig_dy]))
                    _r30_max_dist = max(_r30_max_dist, dist_diff)
            _r30_gi += n_pts

        _r30_config_metrics.append({
            "config_num": _r30_cnum,
            "rank": _r30_rank,
            "perp_variance": _r30_dev,
            "max_angle_deviation": _r30_max_angle,
            "max_distance_deviation": _r30_max_dist,
        })

        # Display summary metrics
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Perp. Variance", f"{_r30_dev:.4f} m²")
        _mc2.metric("Max Angle Δ", f"{_r30_max_angle:.1f}°")
        _mc3.metric("Max Distance Δ", f"{_r30_max_dist:.2f}m")
        _mc4.metric("Iterations", f"{_r30_iterations}")

        # ---------- Static trajectory plot (zoomed to max deviation) ----------
        _r30_max_dev_dist = 0.0
        _r30_max_dev_orig = np.array([0.0, 0.0])
        _r30_max_dev_gen = np.array([0.0, 0.0])

        for idx_oid, oid in enumerate(_r30_sorted_oids):
            orig_pts, g_pts = _r30_all_plot_data[idx_oid]
            for li in range(orig_pts.shape[0]):
                d = float(np.linalg.norm(g_pts[li] - orig_pts[li]))
                if d > _r30_max_dev_dist:
                    _r30_max_dev_dist = d
                    _r30_max_dev_orig = orig_pts[li].copy()
                    _r30_max_dev_gen = g_pts[li].copy()

        _r30_all_x: list[float] = []
        _r30_all_y: list[float] = []
        for orig_pts, g_pts in _r30_all_plot_data:
            _r30_all_x.extend(orig_pts[:, 0].tolist())
            _r30_all_x.extend(g_pts[:, 0].tolist())
            _r30_all_y.extend(orig_pts[:, 1].tolist())
            _r30_all_y.extend(g_pts[:, 1].tolist())

        _r30_x_lo = min(_r30_all_x) - max((max(_r30_all_x) - min(_r30_all_x)) * 0.05, 1.0)
        _r30_x_hi = max(_r30_all_x) + max((max(_r30_all_x) - min(_r30_all_x)) * 0.05, 1.0)
        _r30_y_lo = min(_r30_all_y) - max((max(_r30_all_y) - min(_r30_all_y)) * 0.08, 0.5)
        _r30_y_hi = max(_r30_all_y) + max((max(_r30_all_y) - min(_r30_all_y)) * 0.08, 0.5)

        _r30_dx = _r30_x_hi - _r30_x_lo
        _r30_dy = _r30_y_hi - _r30_y_lo
        _r30_target_ratio = 1.3
        if _r30_dx > _r30_dy * _r30_target_ratio:
            _r30_needed_dy = _r30_dx / _r30_target_ratio
            _r30_pad = (_r30_needed_dy - _r30_dy) / 2
            _r30_y_lo -= _r30_pad
            _r30_y_hi += _r30_pad
            _r30_dy = _r30_needed_dy
        elif _r30_dy * _r30_target_ratio > _r30_dx:
            _r30_needed_dx = _r30_dy * _r30_target_ratio
            _r30_pad = (_r30_needed_dx - _r30_dx) / 2
            _r30_x_lo -= _r30_pad
            _r30_x_hi += _r30_pad
            _r30_dx = _r30_needed_dx

        _r30_fig_w = 10.0
        _r30_fig_h = _r30_fig_w * (_r30_dy / _r30_dx) if _r30_dx > 0 else 7.5
        _r30_fig_h = max(3.0, min(_r30_fig_h, 7.5))

        fig_static = Figure(figsize=(_r30_fig_w, _r30_fig_h), dpi=150)
        ax_s = fig_static.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_oid, oid in enumerate(_r30_sorted_oids):
            orig_pts, g_pts = _r30_all_plot_data[idx_oid]
            label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
            color = f"C{oid}"
            orig_label = f"{label} original" if idx_oid == 0 else None
            gen_label = f"{label} generated" if idx_oid == 0 else None
            ts_filt = _r30_ts_data.get(oid, None)
            expected_dt = 1
            if orig_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(orig_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [orig_pts[i, 0], orig_pts[i+1, 0]]
                        yseg = [orig_pts[i, 1], orig_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.0, color=color, alpha=0.3, linestyle='--', label=orig_label if i == 0 else None)
            if g_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(g_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [g_pts[i, 0], g_pts[i+1, 0]]
                        yseg = [g_pts[i, 1], g_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.5, color=color, alpha=1.0, label=gen_label if i == 0 else None)
        ax_s.annotate(f"max Δ={_r30_max_dev_dist:.2f}m", xy=(_r30_max_dev_gen[0], _r30_max_dev_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_r30_x_lo, _r30_x_hi)
        ax_s.set_ylim(_r30_y_lo, _r30_y_hi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)")
        ax_s.set_ylabel("d2 / y-as (m)")
        _r30_ts_per_o = _r30_filtered_counts
        _r30_ts_info = ", ".join(f"obj{oid}:{n}" for oid, n in _r30_ts_per_o.items())
        ax_s.set_title(f"Config #{_r30_cnum} — PV={_r30_dev:.4f} m² — {_r30_ts_info} timestamps (rough d1=d2=0.30m)")
        fig_static.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        _buf_s = io.BytesIO()
        fig_static.savefig(_buf_s, format='png', dpi=150)
        _buf_s.seek(0)
        st.image(_buf_s, use_container_width=True)

        # ---------- Animated GIF: trajectory building up over timestamps ----------
        _r30_all_ts: list[float] = []
        for oid in _r30_sorted_oids:
            _r30_all_ts.extend(_r30_display_vals[oid].tolist())
        _r30_unique_ts = sorted(set(_r30_all_ts))
        _r30_n_frames = len(_r30_unique_ts)

        _r30_all_x_gif: list[float] = []
        _r30_all_y_gif: list[float] = []
        for orig_pts, g_pts in _r30_all_plot_data:
            _r30_all_x_gif.extend(orig_pts[:, 0].tolist())
            _r30_all_x_gif.extend(g_pts[:, 0].tolist())
            _r30_all_y_gif.extend(orig_pts[:, 1].tolist())
            _r30_all_y_gif.extend(g_pts[:, 1].tolist())
        _r30_gif_xmin = min(_r30_all_x_gif) - max((max(_r30_all_x_gif) - min(_r30_all_x_gif)) * 0.05, 1.0)
        _r30_gif_xmax = max(_r30_all_x_gif) + max((max(_r30_all_x_gif) - min(_r30_all_x_gif)) * 0.05, 1.0)
        _r30_gif_ymin = min(_r30_all_y_gif) - max((max(_r30_all_y_gif) - min(_r30_all_y_gif)) * 0.1, 0.5)
        _r30_gif_ymax = max(_r30_all_y_gif) + max((max(_r30_all_y_gif) - min(_r30_all_y_gif)) * 0.1, 0.5)

        _r30_gif_dx = _r30_gif_xmax - _r30_gif_xmin
        _r30_gif_dy = _r30_gif_ymax - _r30_gif_ymin
        if _r30_gif_dx > _r30_gif_dy * _r30_target_ratio:
            _g_pad = (_r30_gif_dx / _r30_target_ratio - _r30_gif_dy) / 2
            _r30_gif_ymin -= _g_pad
            _r30_gif_ymax += _g_pad
            _r30_gif_dy = _r30_gif_ymax - _r30_gif_ymin
        elif _r30_gif_dy * _r30_target_ratio > _r30_gif_dx:
            _g_pad = (_r30_gif_dy * _r30_target_ratio - _r30_gif_dx) / 2
            _r30_gif_xmin -= _g_pad
            _r30_gif_xmax += _g_pad
            _r30_gif_dx = _r30_gif_xmax - _r30_gif_xmin

        _r30_gif_fw = 10.0
        _r30_gif_fh = _r30_gif_fw * (_r30_gif_dy / _r30_gif_dx) if _r30_gif_dx > 0 else 7.5
        _r30_gif_fh = max(3.0, min(_r30_gif_fh, 7.5))

        _r30_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _r30_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _r30_gi2 = 0
        for oid in _r30_sorted_oids:
            n_pts = _r30_display_points[oid].shape[0]
            vals = _r30_display_vals[oid]
            _r30_orig_by_obj[oid] = []
            _r30_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_val = float(vals[li])
                orig_coord = _r30_display_points[oid][li]
                gen_coord = _r30_gen_map.get(_r30_gi2 + li, orig_coord)
                _r30_orig_by_obj[oid].append((t_val, orig_coord))
                _r30_gen_by_obj[oid].append((t_val, np.array(gen_coord)))
            _r30_gi2 += n_pts

        _r30_gif_frames: list[PILImage.Image] = []
        _r30_gif_progress = st.progress(0)
        _r30_gif_status = st.empty()

        for frame_idx, t_cutoff in enumerate(_r30_unique_ts):
            _r30_gif_status.text(f"Rendering GIF frame {frame_idx + 1}/{_r30_n_frames} (t ≤ {t_cutoff:g})...")
            fig_frame = Figure(figsize=(_r30_gif_fw, _r30_gif_fh), dpi=150)
            ax_f = fig_frame.add_subplot(111)
            ax_f.set_xlim(_r30_gif_xmin, _r30_gif_xmax)
            ax_f.set_ylim(_r30_gif_ymin, _r30_gif_ymax)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            _r30_ts_per_o2 = {oid: _r30_display_points[oid].shape[0] for oid in _r30_sorted_oids}
            _r30_ts_info2 = ", ".join(f"obj{oid}:{n}" for oid, n in _r30_ts_per_o2.items())
            ax_f.set_title(f"Config #{_r30_cnum} — t ≤ {t_cutoff:g} — {_r30_ts_info2} ts  (PV={_r30_dev:.4f} m², rough 0.30m)", fontsize=9)

            for idx_oid, oid in enumerate(_r30_sorted_oids):
                label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
                color = f"C{oid}"
                orig_pts_up = [coord for (t_val, coord) in _r30_orig_by_obj[oid] if t_val <= t_cutoff]
                if len(orig_pts_up) > 1:
                    orig_arr = np.array(orig_pts_up)
                    ax_f.plot(orig_arr[:, 0], orig_arr[:, 1], linewidth=1.0, color=color, alpha=0.35, linestyle='--', label=f"{label} orig")
                gen_pts_up = [coord for (t_val, coord) in _r30_gen_by_obj[oid] if t_val <= t_cutoff]
                if len(gen_pts_up) > 1:
                    gen_arr = np.array(gen_pts_up)
                    ax_f.plot(gen_arr[:, 0], gen_arr[:, 1], linewidth=1.8, color=color, alpha=1.0, label=f"{label} gen")

            ax_f.legend(fontsize=7, loc='upper left')
            fig_frame.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)

            _buf_frame = io.BytesIO()
            fig_frame.savefig(_buf_frame, format='png', dpi=150)
            _buf_frame.seek(0)
            _r30_gif_frames.append(PILImage.open(_buf_frame).copy())
            plt.close(fig_frame)

            _r30_gif_progress.progress((frame_idx + 1) / _r30_n_frames)

        _r30_gif_progress.empty()
        _r30_gif_status.empty()

        if _r30_gif_frames:
            _r30_gif_buf = io.BytesIO()
            _r30_durations = [GIF_FRAME_DURATION_MS] * len(_r30_gif_frames)
            if _r30_durations:
                _r30_durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            _r30_gif_frames[0].save(
                _r30_gif_buf,
                format='GIF',
                save_all=True,
                append_images=_r30_gif_frames[1:],
                duration=_r30_durations,
                loop=0,
            )
            _r30_gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_r30_cnum} (rough 0.30m)",
                data=_r30_gif_buf,
                file_name=f"config_{_r30_cnum}_rough030_animation.gif",
                mime="image/gif",
                key=f"dl_gif_r30_{_r30_rank}",
            )

        st.markdown("---")

    # Summary table
    if _r30_config_metrics:
        st.markdown("#### Summary (rough d1=d2=0.30m)")
        _r30_df = pd.DataFrame(_r30_config_metrics)
        st.dataframe(_r30_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={
                "rank": "Rank",
                "config_num": "Config #",
                "perp_variance": "Perp. Variance (m²)",
                "max_angle_deviation": "Max Angle Δ (°)",
                "max_distance_deviation": "Max Distance Δ (m)",
            }
        ), use_container_width=True)

    # Clear button
    if st.button("Clear Results & Cache (rough)", key="clear_ext30_rough_results"):
        st.session_state["_generate_ext30_rough_requested"] = False
        st.session_state["_generate_ext30_rough_results"] = None
        st.session_state.pop("_ext30_rough_full_points_plot", None)
        st.session_state.pop("_ext30_rough_full_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# ============= Display results for br_consec (bufferrough + consecutive ts, top 3, with GIF) ============
if st.session_state.get("_generate_br_consec_results", None):
    _brc_top3 = st.session_state["_generate_br_consec_results"]
    _brc_display_points = st.session_state.get("_br_consec_full_points_plot", all_points_plot)
    _brc_display_vals = st.session_state.get("_br_consec_full_vals_plot", all_vals_plot)

    st.markdown("---")
    _brc_n_ts = sum(_brc_display_points[oid].shape[0] for oid in sorted(_brc_display_points.keys()))
    _brc_n_ts_per_obj = {oid: _brc_display_points[oid].shape[0] for oid in sorted(_brc_display_points.keys())}
    st.markdown(f"### Top 3 Most Deviating (bufferrough, consec ts) — {MAX_FILTER_CONFIGS} configs, 2500 iter — {list(_brc_n_ts_per_obj.values())[0]} timestamps per object")
    st.markdown(f"""
**Generation settings**: {MAX_FILTER_CONFIGS} configurations × 2500 iterations | **{_brc_n_ts} total points** ({', '.join(f'object {oid}: {n}' for oid, n in _brc_n_ts_per_obj.items())}).
**Forced PDP settings**: bufferrough | buffer x/y=1.0m | rough x/y=0.30m | Consecutive timestamps (2–3) same direction.

**Deviation Metrics:**
- **Perpendicular Variance (m²)**: Variance of perpendicular distances from generated points to the original trajectory.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps.
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps.

Each configuration includes an **animated GIF** download showing the trajectory building up over timestamps.""")

    _brc_config_metrics: list[dict[str, Any]] = []

    for _brc_rank, (_brc_cnum, _brc_dev, _brc_cfg) in enumerate(_brc_top3, 1):
        st.markdown(f"#### Rank {_brc_rank}: Configuration #{_brc_cnum} (Perp. variance: {_brc_dev:.4f} m²)")

        _brc_pdp_variant = _brc_cfg.get("pdp_variant", "bufferrough")
        _brc_iterations = _brc_cfg.get("iterations", "N/A")
        _brc_threshold_mode = _brc_cfg.get("threshold_mode", "Percentage")
        _brc_max_threshold = _brc_cfg.get("max_threshold", 0.0)
        _brc_group_size = _brc_cfg.get("group_size", "?")
        _brc_obj_id = _brc_cfg.get("object_id", "?")

        # Build generated coordinate map
        _brc_sp = _brc_cfg.get("successful_points", [])
        _brc_gen_map: dict[int, np.ndarray] = {}
        for sp in _brc_sp:
            _brc_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        # Calculate angle and distance deviations
        _brc_max_angle = 0.0
        _brc_max_dist = 0.0
        _brc_sorted_oids = sorted(_brc_display_points.keys())
        _brc_all_plot_data: list[tuple[np.ndarray, np.ndarray]] = []
        _brc_ts_data: dict[int, np.ndarray | None] = {}
        _brc_filtered_counts = {}
        _brc_gi = 0
        for oid in _brc_sorted_oids:
            orig_pts = _brc_display_points[oid]
            orig_ts = _brc_display_vals[oid] if oid in _brc_display_vals else None
            n_pts = orig_pts.shape[0]
            gen_pts = orig_pts.copy()
            for li in range(n_pts):
                gi = _brc_gi + li
                if gi in _brc_gen_map:
                    gen_pts[li] = _brc_gen_map[gi]
            _brc_all_plot_data.append((orig_pts, gen_pts))
            _brc_ts_data[oid] = orig_ts
            _brc_filtered_counts[oid] = n_pts

            if orig_pts.shape[0] > 1:
                for i in range(1, orig_pts.shape[0]):
                    orig_dx = orig_pts[i, 0] - orig_pts[i-1, 0]
                    orig_dy = orig_pts[i, 1] - orig_pts[i-1, 1]
                    gen_dx = gen_pts[i, 0] - gen_pts[i-1, 0]
                    gen_dy = gen_pts[i, 1] - gen_pts[i-1, 1]
                    angle_diff = abs(np.degrees(np.arctan2(gen_dy, gen_dx) - np.arctan2(orig_dy, orig_dx)))
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    _brc_max_angle = max(_brc_max_angle, angle_diff)
                    dist_diff = abs(np.linalg.norm([gen_dx, gen_dy]) - np.linalg.norm([orig_dx, orig_dy]))
                    _brc_max_dist = max(_brc_max_dist, dist_diff)
            _brc_gi += n_pts

        _brc_config_metrics.append({
            "config_num": _brc_cnum,
            "rank": _brc_rank,
            "perp_variance": _brc_dev,
            "max_angle_deviation": _brc_max_angle,
            "max_distance_deviation": _brc_max_dist,
            "group_size": _brc_group_size,
            "object_id": _brc_obj_id,
        })

        # Display summary metrics
        _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
        _mc1.metric("Perp. Variance", f"{_brc_dev:.4f} m²")
        _mc2.metric("Max Angle Δ", f"{_brc_max_angle:.1f}°")
        _mc3.metric("Max Distance Δ", f"{_brc_max_dist:.2f}m")
        _mc4.metric("Iterations", f"{_brc_iterations}")
        _mc5.metric("Group/Obj", f"{_brc_group_size} ts / obj {_brc_obj_id}")

        # ---------- Static trajectory plot (zoomed to max deviation) ----------
        _brc_max_dev_dist = 0.0
        _brc_max_dev_orig = np.array([0.0, 0.0])
        _brc_max_dev_gen = np.array([0.0, 0.0])

        for idx_oid, oid in enumerate(_brc_sorted_oids):
            orig_pts, g_pts = _brc_all_plot_data[idx_oid]
            for li in range(orig_pts.shape[0]):
                d = float(np.linalg.norm(g_pts[li] - orig_pts[li]))
                if d > _brc_max_dev_dist:
                    _brc_max_dev_dist = d
                    _brc_max_dev_orig = orig_pts[li].copy()
                    _brc_max_dev_gen = g_pts[li].copy()

        _brc_all_x: list[float] = []
        _brc_all_y: list[float] = []
        for orig_pts, g_pts in _brc_all_plot_data:
            _brc_all_x.extend(orig_pts[:, 0].tolist())
            _brc_all_x.extend(g_pts[:, 0].tolist())
            _brc_all_y.extend(orig_pts[:, 1].tolist())
            _brc_all_y.extend(g_pts[:, 1].tolist())

        _brc_x_lo = min(_brc_all_x) - max((max(_brc_all_x) - min(_brc_all_x)) * 0.05, 1.0)
        _brc_x_hi = max(_brc_all_x) + max((max(_brc_all_x) - min(_brc_all_x)) * 0.05, 1.0)
        _brc_y_lo = min(_brc_all_y) - max((max(_brc_all_y) - min(_brc_all_y)) * 0.08, 0.5)
        _brc_y_hi = max(_brc_all_y) + max((max(_brc_all_y) - min(_brc_all_y)) * 0.08, 0.5)

        _brc_dx = _brc_x_hi - _brc_x_lo
        _brc_dy = _brc_y_hi - _brc_y_lo
        _brc_target_ratio = 1.3
        if _brc_dx > _brc_dy * _brc_target_ratio:
            _brc_needed_dy = _brc_dx / _brc_target_ratio
            _brc_pad = (_brc_needed_dy - _brc_dy) / 2
            _brc_y_lo -= _brc_pad
            _brc_y_hi += _brc_pad
            _brc_dy = _brc_needed_dy
        elif _brc_dy * _brc_target_ratio > _brc_dx:
            _brc_needed_dx = _brc_dy * _brc_target_ratio
            _brc_pad = (_brc_needed_dx - _brc_dx) / 2
            _brc_x_lo -= _brc_pad
            _brc_x_hi += _brc_pad
            _brc_dx = _brc_needed_dx

        _brc_fig_w = 10.0
        _brc_fig_h = _brc_fig_w * (_brc_dy / _brc_dx) if _brc_dx > 0 else 7.5
        _brc_fig_h = max(3.0, min(_brc_fig_h, 7.5))

        fig_static = Figure(figsize=(_brc_fig_w, _brc_fig_h), dpi=150)
        ax_s = fig_static.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_oid, oid in enumerate(_brc_sorted_oids):
            orig_pts, g_pts = _brc_all_plot_data[idx_oid]
            label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
            color = f"C{oid}"
            orig_label = f"{label} original" if idx_oid == 0 else None
            gen_label = f"{label} generated" if idx_oid == 0 else None
            ts_filt = _brc_ts_data.get(oid, None)
            expected_dt = 1
            if orig_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(orig_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [orig_pts[i, 0], orig_pts[i+1, 0]]
                        yseg = [orig_pts[i, 1], orig_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.0, color=color, alpha=0.3, linestyle='--', label=orig_label if i == 0 else None)
            if g_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(g_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [g_pts[i, 0], g_pts[i+1, 0]]
                        yseg = [g_pts[i, 1], g_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.5, color=color, alpha=1.0, label=gen_label if i == 0 else None)
        ax_s.annotate(f"max Δ={_brc_max_dev_dist:.2f}m", xy=(_brc_max_dev_gen[0], _brc_max_dev_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_brc_x_lo, _brc_x_hi)
        ax_s.set_ylim(_brc_y_lo, _brc_y_hi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)")
        ax_s.set_ylabel("d2 / y-as (m)")
        _brc_ts_per_o = _brc_filtered_counts
        _brc_ts_info = ", ".join(f"obj{oid}:{n}" for oid, n in _brc_ts_per_o.items())
        ax_s.set_title(f"Config #{_brc_cnum} — PV={_brc_dev:.4f} m² — {_brc_ts_info} timestamps (bufferrough, consec ts)")
        fig_static.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        _buf_s = io.BytesIO()
        fig_static.savefig(_buf_s, format='png', dpi=150)
        _buf_s.seek(0)
        st.image(_buf_s, use_container_width=True)

        # ---------- Animated GIF: trajectory building up over timestamps ----------
        _brc_all_ts: list[float] = []
        for oid in _brc_sorted_oids:
            _brc_all_ts.extend(_brc_display_vals[oid].tolist())
        _brc_unique_ts = sorted(set(_brc_all_ts))
        _brc_n_frames = len(_brc_unique_ts)

        _brc_all_x_gif: list[float] = []
        _brc_all_y_gif: list[float] = []
        for orig_pts, g_pts in _brc_all_plot_data:
            _brc_all_x_gif.extend(orig_pts[:, 0].tolist())
            _brc_all_x_gif.extend(g_pts[:, 0].tolist())
            _brc_all_y_gif.extend(orig_pts[:, 1].tolist())
            _brc_all_y_gif.extend(g_pts[:, 1].tolist())
        _brc_gif_xmin = min(_brc_all_x_gif) - max((max(_brc_all_x_gif) - min(_brc_all_x_gif)) * 0.05, 1.0)
        _brc_gif_xmax = max(_brc_all_x_gif) + max((max(_brc_all_x_gif) - min(_brc_all_x_gif)) * 0.05, 1.0)
        _brc_gif_ymin = min(_brc_all_y_gif) - max((max(_brc_all_y_gif) - min(_brc_all_y_gif)) * 0.1, 0.5)
        _brc_gif_ymax = max(_brc_all_y_gif) + max((max(_brc_all_y_gif) - min(_brc_all_y_gif)) * 0.1, 0.5)

        _brc_gif_dx = _brc_gif_xmax - _brc_gif_xmin
        _brc_gif_dy = _brc_gif_ymax - _brc_gif_ymin
        if _brc_gif_dx > _brc_gif_dy * _brc_target_ratio:
            _g_pad = (_brc_gif_dx / _brc_target_ratio - _brc_gif_dy) / 2
            _brc_gif_ymin -= _g_pad
            _brc_gif_ymax += _g_pad
            _brc_gif_dy = _brc_gif_ymax - _brc_gif_ymin
        elif _brc_gif_dy * _brc_target_ratio > _brc_gif_dx:
            _g_pad = (_brc_gif_dy * _brc_target_ratio - _brc_gif_dx) / 2
            _brc_gif_xmin -= _g_pad
            _brc_gif_xmax += _g_pad
            _brc_gif_dx = _brc_gif_xmax - _brc_gif_xmin

        _brc_gif_fw = 10.0
        _brc_gif_fh = _brc_gif_fw * (_brc_gif_dy / _brc_gif_dx) if _brc_gif_dx > 0 else 7.5
        _brc_gif_fh = max(3.0, min(_brc_gif_fh, 7.5))

        _brc_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _brc_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _brc_gi2 = 0
        for oid in _brc_sorted_oids:
            n_pts = _brc_display_points[oid].shape[0]
            vals = _brc_display_vals[oid]
            _brc_orig_by_obj[oid] = []
            _brc_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_val = float(vals[li])
                orig_coord = _brc_display_points[oid][li]
                gen_coord = _brc_gen_map.get(_brc_gi2 + li, orig_coord)
                _brc_orig_by_obj[oid].append((t_val, orig_coord))
                _brc_gen_by_obj[oid].append((t_val, np.array(gen_coord)))
            _brc_gi2 += n_pts

        _brc_gif_frames: list[PILImage.Image] = []
        _brc_gif_progress = st.progress(0)
        _brc_gif_status = st.empty()

        for frame_idx, t_cutoff in enumerate(_brc_unique_ts):
            _brc_gif_status.text(f"Rendering GIF frame {frame_idx + 1}/{_brc_n_frames} (t ≤ {t_cutoff:g})...")
            fig_frame = Figure(figsize=(_brc_gif_fw, _brc_gif_fh), dpi=150)
            ax_f = fig_frame.add_subplot(111)
            ax_f.set_xlim(_brc_gif_xmin, _brc_gif_xmax)
            ax_f.set_ylim(_brc_gif_ymin, _brc_gif_ymax)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            _brc_ts_per_o2 = {oid: _brc_display_points[oid].shape[0] for oid in _brc_sorted_oids}
            _brc_ts_info2 = ", ".join(f"obj{oid}:{n}" for oid, n in _brc_ts_per_o2.items())
            ax_f.set_title(f"Config #{_brc_cnum} — t ≤ {t_cutoff:g} — {_brc_ts_info2} ts  (PV={_brc_dev:.4f} m², bufferrough consec)", fontsize=9)

            for idx_oid, oid in enumerate(_brc_sorted_oids):
                label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
                color = f"C{oid}"
                orig_pts_up = [coord for (t_val, coord) in _brc_orig_by_obj[oid] if t_val <= t_cutoff]
                if len(orig_pts_up) > 1:
                    orig_arr = np.array(orig_pts_up)
                    ax_f.plot(orig_arr[:, 0], orig_arr[:, 1], linewidth=1.0, color=color, alpha=0.35, linestyle='--', label=f"{label} orig")
                gen_pts_up = [coord for (t_val, coord) in _brc_gen_by_obj[oid] if t_val <= t_cutoff]
                if len(gen_pts_up) > 1:
                    gen_arr = np.array(gen_pts_up)
                    ax_f.plot(gen_arr[:, 0], gen_arr[:, 1], linewidth=1.8, color=color, alpha=1.0, label=f"{label} gen")

            ax_f.legend(fontsize=7, loc='upper left')
            fig_frame.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)

            _buf_frame = io.BytesIO()
            fig_frame.savefig(_buf_frame, format='png', dpi=150)
            _buf_frame.seek(0)
            _brc_gif_frames.append(PILImage.open(_buf_frame).copy())
            plt.close(fig_frame)

            _brc_gif_progress.progress((frame_idx + 1) / _brc_n_frames)

        _brc_gif_progress.empty()
        _brc_gif_status.empty()

        if _brc_gif_frames:
            _brc_gif_buf = io.BytesIO()
            _brc_durations = [GIF_FRAME_DURATION_MS] * len(_brc_gif_frames)
            if _brc_durations:
                _brc_durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            _brc_gif_frames[0].save(
                _brc_gif_buf,
                format='GIF',
                save_all=True,
                append_images=_brc_gif_frames[1:],
                duration=_brc_durations,
                loop=0,
            )
            _brc_gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_brc_cnum} (bufferrough consec)",
                data=_brc_gif_buf,
                file_name=f"config_{_brc_cnum}_bufferrough_consec_animation.gif",
                mime="image/gif",
                key=f"dl_gif_brc_{_brc_rank}",
            )

        st.markdown("---")

    # Summary table
    if _brc_config_metrics:
        st.markdown("#### Summary (bufferrough, consecutive timestamps)")
        _brc_df = pd.DataFrame(_brc_config_metrics)
        st.dataframe(_brc_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation", "group_size", "object_id"]].rename(
            columns={
                "rank": "Rank",
                "config_num": "Config #",
                "perp_variance": "Perp. Variance (m²)",
                "max_angle_deviation": "Max Angle Δ (°)",
                "max_distance_deviation": "Max Distance Δ (m)",
                "group_size": "Group Size",
                "object_id": "Object",
            }
        ), use_container_width=True)

    # Clear button
    if st.button("Clear Results & Cache (bufferrough consec)", key="clear_br_consec_results"):
        st.session_state["_generate_br_consec_requested"] = False
        st.session_state["_generate_br_consec_results"] = None
        st.session_state.pop("_br_consec_full_points_plot", None)
        st.session_state.pop("_br_consec_full_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# ============= Display results for br_consec_rd (bufferrough + consecutive ts + random dirs, top 3, with GIF) ============
if st.session_state.get("_generate_br_consec_rd_results", None):
    _brcrd_top3 = st.session_state["_generate_br_consec_rd_results"]
    _brcrd_display_points = st.session_state.get("_br_consec_rd_full_points_plot", all_points_plot)
    _brcrd_display_vals = st.session_state.get("_br_consec_rd_full_vals_plot", all_vals_plot)

    st.markdown("---")
    _brcrd_n_ts = sum(_brcrd_display_points[oid].shape[0] for oid in sorted(_brcrd_display_points.keys()))
    _brcrd_n_ts_per_obj = {oid: _brcrd_display_points[oid].shape[0] for oid in sorted(_brcrd_display_points.keys())}
    st.markdown(f"### Top 3 Most Deviating (bufferrough, consec ts, random dirs) — {MAX_FILTER_CONFIGS} configs, 2500 iter — {list(_brcrd_n_ts_per_obj.values())[0]} timestamps per object")
    st.markdown(f"""
**Generation settings**: {MAX_FILTER_CONFIGS} configurations × 2500 iterations | **{_brcrd_n_ts} total points** ({', '.join(f'object {oid}: {n}' for oid, n in _brcrd_n_ts_per_obj.items())}).
**Forced PDP settings**: bufferrough | buffer x/y=1.0m | rough x/y=0.30m | Consecutive timestamps (2–3) RANDOM directions.

**Deviation Metrics:**
- **Perpendicular Variance (m²)**: Variance of perpendicular distances from generated points to the original trajectory.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps.
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps.

Each configuration includes an **animated GIF** download showing the trajectory building up over timestamps.""")

    _brcrd_config_metrics: list[dict[str, Any]] = []

    for _brcrd_rank, (_brcrd_cnum, _brcrd_dev, _brcrd_cfg) in enumerate(_brcrd_top3, 1):
        st.markdown(f"#### Rank {_brcrd_rank}: Configuration #{_brcrd_cnum} (Perp. variance: {_brcrd_dev:.4f} m²)")

        _brcrd_pdp_variant = _brcrd_cfg.get("pdp_variant", "bufferrough")
        _brcrd_iterations = _brcrd_cfg.get("iterations", "N/A")
        _brcrd_threshold_mode = _brcrd_cfg.get("threshold_mode", "Percentage")
        _brcrd_max_threshold = _brcrd_cfg.get("max_threshold", 0.0)
        _brcrd_group_size = _brcrd_cfg.get("group_size", "?")
        _brcrd_obj_id = _brcrd_cfg.get("object_id", "?")

        # Build generated coordinate map
        _brcrd_sp = _brcrd_cfg.get("successful_points", [])
        _brcrd_gen_map: dict[int, np.ndarray] = {}
        for sp in _brcrd_sp:
            _brcrd_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        # Calculate angle and distance deviations
        _brcrd_max_angle = 0.0
        _brcrd_max_dist = 0.0
        _brcrd_sorted_oids = sorted(_brcrd_display_points.keys())
        _brcrd_all_plot_data: list[tuple[np.ndarray, np.ndarray]] = []
        _brcrd_ts_data: dict[int, np.ndarray | None] = {}
        _brcrd_filtered_counts = {}
        _brcrd_gi = 0
        for oid in _brcrd_sorted_oids:
            orig_pts = _brcrd_display_points[oid]
            orig_ts = _brcrd_display_vals[oid] if oid in _brcrd_display_vals else None
            n_pts = orig_pts.shape[0]
            gen_pts = orig_pts.copy()
            for li in range(n_pts):
                gi = _brcrd_gi + li
                if gi in _brcrd_gen_map:
                    gen_pts[li] = _brcrd_gen_map[gi]
            _brcrd_all_plot_data.append((orig_pts, gen_pts))
            _brcrd_ts_data[oid] = orig_ts
            _brcrd_filtered_counts[oid] = n_pts

            if orig_pts.shape[0] > 1:
                for i in range(1, orig_pts.shape[0]):
                    orig_dx = orig_pts[i, 0] - orig_pts[i-1, 0]
                    orig_dy = orig_pts[i, 1] - orig_pts[i-1, 1]
                    gen_dx = gen_pts[i, 0] - gen_pts[i-1, 0]
                    gen_dy = gen_pts[i, 1] - gen_pts[i-1, 1]
                    angle_diff = abs(np.degrees(np.arctan2(gen_dy, gen_dx) - np.arctan2(orig_dy, orig_dx)))
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    _brcrd_max_angle = max(_brcrd_max_angle, angle_diff)
                    dist_diff = abs(np.linalg.norm([gen_dx, gen_dy]) - np.linalg.norm([orig_dx, orig_dy]))
                    _brcrd_max_dist = max(_brcrd_max_dist, dist_diff)
            _brcrd_gi += n_pts

        _brcrd_config_metrics.append({
            "config_num": _brcrd_cnum,
            "rank": _brcrd_rank,
            "perp_variance": _brcrd_dev,
            "max_angle_deviation": _brcrd_max_angle,
            "max_distance_deviation": _brcrd_max_dist,
            "group_size": _brcrd_group_size,
            "object_id": _brcrd_obj_id,
        })

        # Display summary metrics
        _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
        _mc1.metric("Perp. Variance", f"{_brcrd_dev:.4f} m²")
        _mc2.metric("Max Angle Δ", f"{_brcrd_max_angle:.1f}°")
        _mc3.metric("Max Distance Δ", f"{_brcrd_max_dist:.2f}m")
        _mc4.metric("Iterations", f"{_brcrd_iterations}")
        _mc5.metric("Group/Obj", f"{_brcrd_group_size} ts / obj {_brcrd_obj_id}")

        # ---------- Static trajectory plot (zoomed to max deviation) ----------
        _brcrd_max_dev_dist = 0.0
        _brcrd_max_dev_orig = np.array([0.0, 0.0])
        _brcrd_max_dev_gen = np.array([0.0, 0.0])

        for idx_oid, oid in enumerate(_brcrd_sorted_oids):
            orig_pts, g_pts = _brcrd_all_plot_data[idx_oid]
            for li in range(orig_pts.shape[0]):
                d = float(np.linalg.norm(g_pts[li] - orig_pts[li]))
                if d > _brcrd_max_dev_dist:
                    _brcrd_max_dev_dist = d
                    _brcrd_max_dev_orig = orig_pts[li].copy()
                    _brcrd_max_dev_gen = g_pts[li].copy()

        _brcrd_all_x: list[float] = []
        _brcrd_all_y: list[float] = []
        for orig_pts, g_pts in _brcrd_all_plot_data:
            _brcrd_all_x.extend(orig_pts[:, 0].tolist())
            _brcrd_all_x.extend(g_pts[:, 0].tolist())
            _brcrd_all_y.extend(orig_pts[:, 1].tolist())
            _brcrd_all_y.extend(g_pts[:, 1].tolist())

        _brcrd_x_lo = min(_brcrd_all_x) - max((max(_brcrd_all_x) - min(_brcrd_all_x)) * 0.05, 1.0)
        _brcrd_x_hi = max(_brcrd_all_x) + max((max(_brcrd_all_x) - min(_brcrd_all_x)) * 0.05, 1.0)
        _brcrd_y_lo = min(_brcrd_all_y) - max((max(_brcrd_all_y) - min(_brcrd_all_y)) * 0.08, 0.5)
        _brcrd_y_hi = max(_brcrd_all_y) + max((max(_brcrd_all_y) - min(_brcrd_all_y)) * 0.08, 0.5)

        _brcrd_dx = _brcrd_x_hi - _brcrd_x_lo
        _brcrd_dy = _brcrd_y_hi - _brcrd_y_lo
        _brcrd_target_ratio = 1.3
        if _brcrd_dx > _brcrd_dy * _brcrd_target_ratio:
            _brcrd_needed_dy = _brcrd_dx / _brcrd_target_ratio
            _brcrd_pad = (_brcrd_needed_dy - _brcrd_dy) / 2
            _brcrd_y_lo -= _brcrd_pad
            _brcrd_y_hi += _brcrd_pad
            _brcrd_dy = _brcrd_needed_dy
        elif _brcrd_dy * _brcrd_target_ratio > _brcrd_dx:
            _brcrd_needed_dx = _brcrd_dy * _brcrd_target_ratio
            _brcrd_pad = (_brcrd_needed_dx - _brcrd_dx) / 2
            _brcrd_x_lo -= _brcrd_pad
            _brcrd_x_hi += _brcrd_pad
            _brcrd_dx = _brcrd_needed_dx

        _brcrd_fig_w = 10.0
        _brcrd_fig_h = _brcrd_fig_w * (_brcrd_dy / _brcrd_dx) if _brcrd_dx > 0 else 7.5
        _brcrd_fig_h = max(3.0, min(_brcrd_fig_h, 7.5))

        fig_static = Figure(figsize=(_brcrd_fig_w, _brcrd_fig_h), dpi=150)
        ax_s = fig_static.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_oid, oid in enumerate(_brcrd_sorted_oids):
            orig_pts, g_pts = _brcrd_all_plot_data[idx_oid]
            label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
            color = f"C{oid}"
            orig_label = f"{label} original" if idx_oid == 0 else None
            gen_label = f"{label} generated" if idx_oid == 0 else None
            ts_filt = _brcrd_ts_data.get(oid, None)
            expected_dt = 1
            if orig_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(orig_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [orig_pts[i, 0], orig_pts[i+1, 0]]
                        yseg = [orig_pts[i, 1], orig_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.0, color=color, alpha=0.3, linestyle='--', label=orig_label if i == 0 else None)
            if g_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(g_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        xseg = [g_pts[i, 0], g_pts[i+1, 0]]
                        yseg = [g_pts[i, 1], g_pts[i+1, 1]]
                        ax_s.plot(xseg, yseg, linewidth=1.5, color=color, alpha=1.0, label=gen_label if i == 0 else None)
        ax_s.annotate(f"max Δ={_brcrd_max_dev_dist:.2f}m", xy=(_brcrd_max_dev_gen[0], _brcrd_max_dev_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_brcrd_x_lo, _brcrd_x_hi)
        ax_s.set_ylim(_brcrd_y_lo, _brcrd_y_hi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)")
        ax_s.set_ylabel("d2 / y-as (m)")
        _brcrd_ts_per_o = _brcrd_filtered_counts
        _brcrd_ts_info = ", ".join(f"obj{oid}:{n}" for oid, n in _brcrd_ts_per_o.items())
        ax_s.set_title(f"Config #{_brcrd_cnum} — PV={_brcrd_dev:.4f} m² — {_brcrd_ts_info} timestamps (bufferrough, consec ts, random dirs)")
        fig_static.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        _buf_s = io.BytesIO()
        fig_static.savefig(_buf_s, format='png', dpi=150)
        _buf_s.seek(0)
        st.image(_buf_s, use_container_width=True)

        # ---------- Animated GIF: trajectory building up over timestamps ----------
        _brcrd_all_ts: list[float] = []
        for oid in _brcrd_sorted_oids:
            _brcrd_all_ts.extend(_brcrd_display_vals[oid].tolist())
        _brcrd_unique_ts = sorted(set(_brcrd_all_ts))
        _brcrd_n_frames = len(_brcrd_unique_ts)

        _brcrd_all_x_gif: list[float] = []
        _brcrd_all_y_gif: list[float] = []
        for orig_pts, g_pts in _brcrd_all_plot_data:
            _brcrd_all_x_gif.extend(orig_pts[:, 0].tolist())
            _brcrd_all_x_gif.extend(g_pts[:, 0].tolist())
            _brcrd_all_y_gif.extend(orig_pts[:, 1].tolist())
            _brcrd_all_y_gif.extend(g_pts[:, 1].tolist())
        _brcrd_gif_xmin = min(_brcrd_all_x_gif) - max((max(_brcrd_all_x_gif) - min(_brcrd_all_x_gif)) * 0.05, 1.0)
        _brcrd_gif_xmax = max(_brcrd_all_x_gif) + max((max(_brcrd_all_x_gif) - min(_brcrd_all_x_gif)) * 0.05, 1.0)
        _brcrd_gif_ymin = min(_brcrd_all_y_gif) - max((max(_brcrd_all_y_gif) - min(_brcrd_all_y_gif)) * 0.1, 0.5)
        _brcrd_gif_ymax = max(_brcrd_all_y_gif) + max((max(_brcrd_all_y_gif) - min(_brcrd_all_y_gif)) * 0.1, 0.5)

        _brcrd_gif_dx = _brcrd_gif_xmax - _brcrd_gif_xmin
        _brcrd_gif_dy = _brcrd_gif_ymax - _brcrd_gif_ymin
        if _brcrd_gif_dx > _brcrd_gif_dy * _brcrd_target_ratio:
            _g_pad = (_brcrd_gif_dx / _brcrd_target_ratio - _brcrd_gif_dy) / 2
            _brcrd_gif_ymin -= _g_pad
            _brcrd_gif_ymax += _g_pad
            _brcrd_gif_dy = _brcrd_gif_ymax - _brcrd_gif_ymin
        elif _brcrd_gif_dy * _brcrd_target_ratio > _brcrd_gif_dx:
            _g_pad = (_brcrd_gif_dy * _brcrd_target_ratio - _brcrd_gif_dx) / 2
            _brcrd_gif_xmin -= _g_pad
            _brcrd_gif_xmax += _g_pad
            _brcrd_gif_dx = _brcrd_gif_xmax - _brcrd_gif_xmin

        _brcrd_gif_fw = 10.0
        _brcrd_gif_fh = _brcrd_gif_fw * (_brcrd_gif_dy / _brcrd_gif_dx) if _brcrd_gif_dx > 0 else 7.5
        _brcrd_gif_fh = max(3.0, min(_brcrd_gif_fh, 7.5))

        _brcrd_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _brcrd_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _brcrd_gi2 = 0
        for oid in _brcrd_sorted_oids:
            n_pts = _brcrd_display_points[oid].shape[0]
            vals = _brcrd_display_vals[oid]
            _brcrd_orig_by_obj[oid] = []
            _brcrd_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_val = float(vals[li])
                orig_coord = _brcrd_display_points[oid][li]
                gen_coord = _brcrd_gen_map.get(_brcrd_gi2 + li, orig_coord)
                _brcrd_orig_by_obj[oid].append((t_val, orig_coord))
                _brcrd_gen_by_obj[oid].append((t_val, np.array(gen_coord)))
            _brcrd_gi2 += n_pts

        _brcrd_gif_frames: list[PILImage.Image] = []
        _brcrd_gif_progress = st.progress(0)
        _brcrd_gif_status = st.empty()

        for frame_idx, t_cutoff in enumerate(_brcrd_unique_ts):
            _brcrd_gif_status.text(f"Rendering GIF frame {frame_idx + 1}/{_brcrd_n_frames} (t ≤ {t_cutoff:g})...")
            fig_frame = Figure(figsize=(_brcrd_gif_fw, _brcrd_gif_fh), dpi=150)
            ax_f = fig_frame.add_subplot(111)
            ax_f.set_xlim(_brcrd_gif_xmin, _brcrd_gif_xmax)
            ax_f.set_ylim(_brcrd_gif_ymin, _brcrd_gif_ymax)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            _brcrd_ts_per_o2 = {oid: _brcrd_display_points[oid].shape[0] for oid in _brcrd_sorted_oids}
            _brcrd_ts_info2 = ", ".join(f"obj{oid}:{n}" for oid, n in _brcrd_ts_per_o2.items())
            ax_f.set_title(f"Config #{_brcrd_cnum} — t ≤ {t_cutoff:g} — {_brcrd_ts_info2} ts  (PV={_brcrd_dev:.4f} m², bufferrough consec rand)", fontsize=9)

            for idx_oid, oid in enumerate(_brcrd_sorted_oids):
                label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
                color = f"C{oid}"
                orig_pts_up = [coord for (t_val, coord) in _brcrd_orig_by_obj[oid] if t_val <= t_cutoff]
                if len(orig_pts_up) > 1:
                    orig_arr = np.array(orig_pts_up)
                    ax_f.plot(orig_arr[:, 0], orig_arr[:, 1], linewidth=1.0, color=color, alpha=0.35, linestyle='--', label=f"{label} orig")
                gen_pts_up = [coord for (t_val, coord) in _brcrd_gen_by_obj[oid] if t_val <= t_cutoff]
                if len(gen_pts_up) > 1:
                    gen_arr = np.array(gen_pts_up)
                    ax_f.plot(gen_arr[:, 0], gen_arr[:, 1], linewidth=1.8, color=color, alpha=1.0, label=f"{label} gen")

            ax_f.legend(fontsize=7, loc='upper left')
            fig_frame.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)

            _buf_frame = io.BytesIO()
            fig_frame.savefig(_buf_frame, format='png', dpi=150)
            _buf_frame.seek(0)
            _brcrd_gif_frames.append(PILImage.open(_buf_frame).copy())
            plt.close(fig_frame)

            _brcrd_gif_progress.progress((frame_idx + 1) / _brcrd_n_frames)

        _brcrd_gif_progress.empty()
        _brcrd_gif_status.empty()

        if _brcrd_gif_frames:
            _brcrd_gif_buf = io.BytesIO()
            _brcrd_durations = [GIF_FRAME_DURATION_MS] * len(_brcrd_gif_frames)
            if _brcrd_durations:
                _brcrd_durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            _brcrd_gif_frames[0].save(
                _brcrd_gif_buf,
                format='GIF',
                save_all=True,
                append_images=_brcrd_gif_frames[1:],
                duration=_brcrd_durations,
                loop=0,
            )
            _brcrd_gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_brcrd_cnum} (bufferrough consec rand dirs)",
                data=_brcrd_gif_buf,
                file_name=f"config_{_brcrd_cnum}_bufferrough_consec_randdirs_animation.gif",
                mime="image/gif",
                key=f"dl_gif_brcrd_{_brcrd_rank}",
            )

        st.markdown("---")

    # Summary table
    if _brcrd_config_metrics:
        st.markdown("#### Summary (bufferrough, consecutive timestamps, random directions)")
        _brcrd_df = pd.DataFrame(_brcrd_config_metrics)
        st.dataframe(_brcrd_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation", "group_size", "object_id"]].rename(
            columns={
                "rank": "Rank",
                "config_num": "Config #",
                "perp_variance": "Perp. Variance (m²)",
                "max_angle_deviation": "Max Angle Δ (°)",
                "max_distance_deviation": "Max Distance Δ (m)",
                "group_size": "Group Size",
                "object_id": "Object",
            }
        ), use_container_width=True)

    # Clear button
    if st.button("Clear Results & Cache (bufferrough consec rand dirs)", key="clear_br_consec_rd_results"):
        st.session_state["_generate_br_consec_rd_requested"] = False
        st.session_state["_generate_br_consec_rd_results"] = None
        st.session_state.pop("_br_consec_rd_full_points_plot", None)
        st.session_state.pop("_br_consec_rd_full_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# ============= Display results for ext30_fe (fixed endpoints, top 3, with GIF) ============
if st.session_state.get("_generate_ext30_fe_results", None):
    _fe_top3 = st.session_state["_generate_ext30_fe_results"]
    _fe_display_points = st.session_state.get("_ext30_fe_full_points_plot", all_points_plot)
    _fe_display_vals = st.session_state.get("_ext30_fe_full_vals_plot", all_vals_plot)

    st.markdown("---")
    _fe_n_ts = sum(_fe_display_points[oid].shape[0] for oid in sorted(_fe_display_points.keys()))
    _fe_n_ts_per_obj = {oid: _fe_display_points[oid].shape[0] for oid in sorted(_fe_display_points.keys())}
    _fe_n_fixed = sum(2 if n > 1 else 1 for n in _fe_n_ts_per_obj.values())
    st.markdown(f"### Top 3 — Fixed Endpoints ({MAX_FILTER_CONFIGS} configs, 2500 iter) — {list(_fe_n_ts_per_obj.values())[0]} timestamps per object")
    st.markdown(f"""
**Generation settings**: {MAX_FILTER_CONFIGS} configurations × 2500 iterations | **{_fe_n_ts} total points** ({', '.join(f'object {oid}: {n}' for oid, n in _fe_n_ts_per_obj.items())}).
**Fixed endpoints**: {_fe_n_fixed} punten (eerste + laatste per object) zijn NIET verplaatst — ze fungeren als ankerpunten.

**Deviation Metrics:**
- **Perpendicular Variance (m²)**: Variance of perpendicular distances from generated points to the original trajectory.
- **Max Angle Deviation (°)**: Maximum angular difference in trajectory direction between consecutive timestamps.
- **Max Distance Deviation (m)**: Maximum change in inter-point spacing between consecutive timestamps.

Each configuration includes an **animated GIF** download showing the trajectory building up over timestamps.""")

    _fe_config_metrics: list[dict[str, Any]] = []

    for _fe_rank, (_fe_cnum, _fe_dev, _fe_cfg) in enumerate(_fe_top3, 1):
        st.markdown(f"#### Rank {_fe_rank}: Configuration #{_fe_cnum} (Perp. variance: {_fe_dev:.4f} m²)")

        _fe_pdp_variant = _fe_cfg.get("pdp_variant", "fundamental")
        _fe_iterations = _fe_cfg.get("iterations", "N/A")
        _fe_threshold_mode = _fe_cfg.get("threshold_mode", "Percentage")
        _fe_max_threshold = _fe_cfg.get("max_threshold", 0.0)

        _fe_sp = _fe_cfg.get("successful_points", [])
        _fe_gen_map: dict[int, np.ndarray] = {}
        for sp in _fe_sp:
            _fe_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        _fe_max_angle = 0.0
        _fe_max_dist = 0.0
        _fe_sorted_oids = sorted(_fe_display_points.keys())
        _fe_all_plot_data: list[tuple[np.ndarray, np.ndarray]] = []
        _fe_ts_data: dict[int, np.ndarray | None] = {}
        _fe_filtered_counts = {}
        _fe_gi = 0
        for oid in _fe_sorted_oids:
            orig_pts = _fe_display_points[oid]
            orig_ts = _fe_display_vals[oid] if oid in _fe_display_vals else None
            n_pts = orig_pts.shape[0]
            gen_pts = orig_pts.copy()
            for li in range(n_pts):
                gi = _fe_gi + li
                if gi in _fe_gen_map:
                    gen_pts[li] = _fe_gen_map[gi]
            _fe_all_plot_data.append((orig_pts, gen_pts))
            _fe_ts_data[oid] = orig_ts
            _fe_filtered_counts[oid] = n_pts

            if orig_pts.shape[0] > 1:
                for i in range(1, orig_pts.shape[0]):
                    orig_dx = orig_pts[i, 0] - orig_pts[i-1, 0]
                    orig_dy = orig_pts[i, 1] - orig_pts[i-1, 1]
                    gen_dx = gen_pts[i, 0] - gen_pts[i-1, 0]
                    gen_dy = gen_pts[i, 1] - gen_pts[i-1, 1]
                    angle_diff = abs(np.degrees(np.arctan2(gen_dy, gen_dx) - np.arctan2(orig_dy, orig_dx)))
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    _fe_max_angle = max(_fe_max_angle, angle_diff)
                    dist_diff = abs(np.linalg.norm([gen_dx, gen_dy]) - np.linalg.norm([orig_dx, orig_dy]))
                    _fe_max_dist = max(_fe_max_dist, dist_diff)
            _fe_gi += n_pts

        _fe_config_metrics.append({
            "config_num": _fe_cnum,
            "rank": _fe_rank,
            "perp_variance": _fe_dev,
            "max_angle_deviation": _fe_max_angle,
            "max_distance_deviation": _fe_max_dist,
        })

        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Perp. Variance", f"{_fe_dev:.4f} m²")
        _mc2.metric("Max Angle Δ", f"{_fe_max_angle:.1f}°")
        _mc3.metric("Max Distance Δ", f"{_fe_max_dist:.2f}m")
        _mc4.metric("Iterations", f"{_fe_iterations}")

        # ---------- Static trajectory plot ----------
        _fe_max_dev_dist = 0.0
        _fe_max_dev_orig = np.array([0.0, 0.0])
        _fe_max_dev_gen = np.array([0.0, 0.0])
        for idx_oid, oid in enumerate(_fe_sorted_oids):
            orig_pts, g_pts = _fe_all_plot_data[idx_oid]
            for li in range(orig_pts.shape[0]):
                d = float(np.linalg.norm(g_pts[li] - orig_pts[li]))
                if d > _fe_max_dev_dist:
                    _fe_max_dev_dist = d
                    _fe_max_dev_orig = orig_pts[li].copy()
                    _fe_max_dev_gen = g_pts[li].copy()

        _fe_all_x: list[float] = []
        _fe_all_y: list[float] = []
        for orig_pts, g_pts in _fe_all_plot_data:
            _fe_all_x.extend(orig_pts[:, 0].tolist())
            _fe_all_x.extend(g_pts[:, 0].tolist())
            _fe_all_y.extend(orig_pts[:, 1].tolist())
            _fe_all_y.extend(g_pts[:, 1].tolist())

        _fe_x_lo = min(_fe_all_x) - max((max(_fe_all_x) - min(_fe_all_x)) * 0.05, 1.0)
        _fe_x_hi = max(_fe_all_x) + max((max(_fe_all_x) - min(_fe_all_x)) * 0.05, 1.0)
        _fe_y_lo = min(_fe_all_y) - max((max(_fe_all_y) - min(_fe_all_y)) * 0.08, 0.5)
        _fe_y_hi = max(_fe_all_y) + max((max(_fe_all_y) - min(_fe_all_y)) * 0.08, 0.5)

        _fe_dx = _fe_x_hi - _fe_x_lo
        _fe_dy = _fe_y_hi - _fe_y_lo
        _fe_target_ratio = 1.3
        if _fe_dx > _fe_dy * _fe_target_ratio:
            _fe_needed_dy = _fe_dx / _fe_target_ratio
            _fe_pad = (_fe_needed_dy - _fe_dy) / 2
            _fe_y_lo -= _fe_pad
            _fe_y_hi += _fe_pad
            _fe_dy = _fe_needed_dy
        elif _fe_dy * _fe_target_ratio > _fe_dx:
            _fe_needed_dx = _fe_dy * _fe_target_ratio
            _fe_pad = (_fe_needed_dx - _fe_dx) / 2
            _fe_x_lo -= _fe_pad
            _fe_x_hi += _fe_pad
            _fe_dx = _fe_needed_dx

        _fe_fig_w = 10.0
        _fe_fig_h = _fe_fig_w * (_fe_dy / _fe_dx) if _fe_dx > 0 else 7.5
        _fe_fig_h = max(3.0, min(_fe_fig_h, 7.5))

        fig_static = Figure(figsize=(_fe_fig_w, _fe_fig_h), dpi=150)
        ax_s = fig_static.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_oid, oid in enumerate(_fe_sorted_oids):
            orig_pts, g_pts = _fe_all_plot_data[idx_oid]
            label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
            color = f"C{oid}"
            ts_filt = _fe_ts_data.get(oid, None)
            expected_dt = 1
            if orig_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(orig_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        ax_s.plot([orig_pts[i, 0], orig_pts[i+1, 0]], [orig_pts[i, 1], orig_pts[i+1, 1]],
                                  linewidth=1.0, color=color, alpha=0.3, linestyle='--',
                                  label=(f"{label} original" if i == 0 and idx_oid == 0 else None))
            if g_pts.shape[0] > 1 and ts_filt is not None:
                for i in range(g_pts.shape[0] - 1):
                    dt = ts_filt[i+1] - ts_filt[i]
                    if dt == expected_dt:
                        ax_s.plot([g_pts[i, 0], g_pts[i+1, 0]], [g_pts[i, 1], g_pts[i+1, 1]],
                                  linewidth=1.5, color=color, alpha=1.0,
                                  label=(f"{label} generated" if i == 0 and idx_oid == 0 else None))
            # Mark fixed endpoints with a distinctive marker
            if orig_pts.shape[0] > 0:
                ax_s.scatter([orig_pts[0, 0], orig_pts[-1, 0]], [orig_pts[0, 1], orig_pts[-1, 1]],
                             marker='D', s=60, color=color, edgecolors='black', linewidths=0.8,
                             zorder=5, label=(f"{label} fixed EP" if idx_oid == 0 else None))
        ax_s.annotate(f"max Δ={_fe_max_dev_dist:.2f}m", xy=(_fe_max_dev_gen[0], _fe_max_dev_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_fe_x_lo, _fe_x_hi)
        ax_s.set_ylim(_fe_y_lo, _fe_y_hi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)")
        ax_s.set_ylabel("d2 / y-as (m)")
        _fe_ts_per_o = _fe_filtered_counts
        _fe_ts_info = ", ".join(f"obj{oid}:{n}" for oid, n in _fe_ts_per_o.items())
        ax_s.set_title(f"Config #{_fe_cnum} (fixed EP) — PV={_fe_dev:.4f} m² — {_fe_ts_info} timestamps")
        fig_static.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        _buf_s = io.BytesIO()
        fig_static.savefig(_buf_s, format='png', dpi=150)
        _buf_s.seek(0)
        st.image(_buf_s, use_container_width=True)

        # ---------- Animated GIF ----------
        _fe_all_ts: list[float] = []
        for oid in _fe_sorted_oids:
            _fe_all_ts.extend(_fe_display_vals[oid].tolist())
        _fe_unique_ts = sorted(set(_fe_all_ts))
        _fe_n_frames = len(_fe_unique_ts)

        _fe_gif_xmin = _fe_x_lo
        _fe_gif_xmax = _fe_x_hi
        _fe_gif_ymin = _fe_y_lo
        _fe_gif_ymax = _fe_y_hi
        _fe_gif_dx = _fe_dx
        _fe_gif_dy = _fe_dy
        _fe_gif_fw = _fe_fig_w
        _fe_gif_fh = _fe_fig_h

        _fe_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _fe_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _fe_gi2 = 0
        for oid in _fe_sorted_oids:
            n_pts = _fe_display_points[oid].shape[0]
            vals = _fe_display_vals[oid]
            _fe_orig_by_obj[oid] = []
            _fe_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_val = float(vals[li])
                orig_coord = _fe_display_points[oid][li]
                gen_coord = _fe_gen_map.get(_fe_gi2 + li, orig_coord)
                _fe_orig_by_obj[oid].append((t_val, orig_coord))
                _fe_gen_by_obj[oid].append((t_val, np.array(gen_coord)))
            _fe_gi2 += n_pts

        _fe_gif_frames: list[PILImage.Image] = []
        _fe_gif_progress = st.progress(0)
        _fe_gif_status = st.empty()

        for frame_idx, t_cutoff in enumerate(_fe_unique_ts):
            _fe_gif_status.text(f"Rendering GIF frame {frame_idx + 1}/{_fe_n_frames} (t ≤ {t_cutoff:g})...")
            fig_frame = Figure(figsize=(_fe_gif_fw, _fe_gif_fh), dpi=150)
            ax_f = fig_frame.add_subplot(111)
            ax_f.set_xlim(_fe_gif_xmin, _fe_gif_xmax)
            ax_f.set_ylim(_fe_gif_ymin, _fe_gif_ymax)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            _fe_ts_per_o2 = {oid: _fe_display_points[oid].shape[0] for oid in _fe_sorted_oids}
            _fe_ts_info2 = ", ".join(f"obj{oid}:{n}" for oid, n in _fe_ts_per_o2.items())
            ax_f.set_title(f"Config #{_fe_cnum} (fixed EP) — t ≤ {t_cutoff:g} — {_fe_ts_info2} ts  (PV={_fe_dev:.4f} m²)", fontsize=9)

            for idx_oid, oid in enumerate(_fe_sorted_oids):
                label = OBJECT_LABELS[oid % len(OBJECT_LABELS)]
                color = f"C{oid}"
                orig_pts_up = [coord for (t_val, coord) in _fe_orig_by_obj[oid] if t_val <= t_cutoff]
                if len(orig_pts_up) > 1:
                    orig_arr = np.array(orig_pts_up)
                    ax_f.plot(orig_arr[:, 0], orig_arr[:, 1], linewidth=1.0, color=color, alpha=0.35, linestyle='--', label=f"{label} orig")
                gen_pts_up = [coord for (t_val, coord) in _fe_gen_by_obj[oid] if t_val <= t_cutoff]
                if len(gen_pts_up) > 1:
                    gen_arr = np.array(gen_pts_up)
                    ax_f.plot(gen_arr[:, 0], gen_arr[:, 1], linewidth=1.8, color=color, alpha=1.0, label=f"{label} gen")
                # Show fixed endpoints as diamonds in every frame
                _fe_ep_pts = _fe_display_points[oid]
                if _fe_ep_pts.shape[0] > 0:
                    _fe_ep_show = []
                    if float(_fe_display_vals[oid][0]) <= t_cutoff:
                        _fe_ep_show.append(_fe_ep_pts[0])
                    if float(_fe_display_vals[oid][-1]) <= t_cutoff:
                        _fe_ep_show.append(_fe_ep_pts[-1])
                    if _fe_ep_show:
                        _fe_ep_arr = np.array(_fe_ep_show)
                        ax_f.scatter(_fe_ep_arr[:, 0], _fe_ep_arr[:, 1], marker='D', s=50, color=color,
                                     edgecolors='black', linewidths=0.7, zorder=5)

            ax_f.legend(fontsize=7, loc='upper left')
            fig_frame.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)

            _buf_frame = io.BytesIO()
            fig_frame.savefig(_buf_frame, format='png', dpi=150)
            _buf_frame.seek(0)
            _fe_gif_frames.append(PILImage.open(_buf_frame).copy())
            plt.close(fig_frame)

            _fe_gif_progress.progress((frame_idx + 1) / _fe_n_frames)

        _fe_gif_progress.empty()
        _fe_gif_status.empty()

        if _fe_gif_frames:
            _fe_gif_buf = io.BytesIO()
            _fe_durations = [GIF_FRAME_DURATION_MS] * len(_fe_gif_frames)
            if _fe_durations:
                _fe_durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            _fe_gif_frames[0].save(
                _fe_gif_buf,
                format='GIF',
                save_all=True,
                append_images=_fe_gif_frames[1:],
                duration=_fe_durations,
                loop=0,
            )
            _fe_gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_fe_cnum} (fixed EP)",
                data=_fe_gif_buf,
                file_name=f"config_{_fe_cnum}_fixed_ep_animation.gif",
                mime="image/gif",
                key=f"dl_gif_ext30_fe_{_fe_rank}",
            )

        st.markdown("---")

    if _fe_config_metrics:
        st.markdown("#### Summary (Fixed Endpoints)")
        _fe_df = pd.DataFrame(_fe_config_metrics)
        st.dataframe(_fe_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={
                "rank": "Rank",
                "config_num": "Config #",
                "perp_variance": "Perp. Variance (m²)",
                "max_angle_deviation": "Max Angle Δ (°)",
                "max_distance_deviation": "Max Distance Δ (m)",
            }
        ), use_container_width=True)

    if st.button("Clear Fixed-EP Results", key="clear_ext30_fe_results"):
        st.session_state["_generate_ext30_fe_requested"] = False
        st.session_state["_generate_ext30_fe_results"] = None
        st.session_state.pop("_ext30_fe_full_points_plot", None)
        st.session_state.pop("_ext30_fe_full_vals_plot", None)
        st.cache_data.clear()
        st.rerun()


def _display_top_n_with_gif(
    results: list[tuple[int, float, dict[str, Any]]],
    section_title: str,
    section_description: str,
    clear_key: str,
    requested_key: str,
    results_key: str,
) -> None:
    """Display top-N most deviating configs with static plot + animated GIF download."""
    st.markdown("---")
    if not all_points_plot:
        st.warning("No data available to display.")
        return
    _dn_n_ts_per_obj = {oid: all_points_plot[oid].shape[0] for oid in sorted(all_points_plot.keys())}
    _dn_n_ts_total = sum(_dn_n_ts_per_obj.values())
    _dn_ts_display = list(_dn_n_ts_per_obj.values())[0] if _dn_n_ts_per_obj else 0
    st.markdown(f"### {section_title} — {_dn_ts_display} timestamps per object")
    st.markdown(f"{section_description} | **{_dn_n_ts_total} total points** ({', '.join(f'obj {oid}: {n}' for oid, n in _dn_n_ts_per_obj.items())})")

    _dn_config_metrics: list[dict[str, Any]] = []
    _dn_sorted_oids = sorted(all_points_plot.keys())

    for _dn_rank, (_dn_cnum, _dn_dev, _dn_cfg) in enumerate(results, 1):
        st.markdown(f"#### Rank {_dn_rank}: Configuration #{_dn_cnum} (Perp. variance: {_dn_dev:.4f} m²)")

        _dn_iters = _dn_cfg.get("iterations", "N/A")
        _dn_sp = _dn_cfg.get("successful_points", [])
        _dn_gen_map: dict[int, np.ndarray] = {}
        for sp in _dn_sp:
            _dn_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        # Build per-object original and generated arrays
        _dn_max_angle = 0.0
        _dn_max_dist = 0.0
        _dn_plot_data: list[tuple[np.ndarray, np.ndarray]] = []
        _dn_gi = 0
        for oid in _dn_sorted_oids:
            n_pts = all_points_plot[oid].shape[0]
            orig = all_points_plot[oid]
            gen = orig.copy()
            for li in range(n_pts):
                gi = _dn_gi + li
                if gi in _dn_gen_map:
                    gen[li] = _dn_gen_map[gi]
            _dn_plot_data.append((orig, gen))
            for i in range(1, n_pts):
                o_dx, o_dy = orig[i, 0] - orig[i-1, 0], orig[i, 1] - orig[i-1, 1]
                g_dx, g_dy = gen[i, 0] - gen[i-1, 0], gen[i, 1] - gen[i-1, 1]
                ad = abs(np.degrees(np.arctan2(g_dy, g_dx) - np.arctan2(o_dy, o_dx)))
                if ad > 180: ad = 360 - ad
                _dn_max_angle = max(_dn_max_angle, ad)
                _dn_max_dist = max(_dn_max_dist, abs(np.linalg.norm([g_dx, g_dy]) - np.linalg.norm([o_dx, o_dy])))
            _dn_gi += n_pts

        _dn_config_metrics.append({
            "config_num": _dn_cnum, "rank": _dn_rank, "perp_variance": _dn_dev,
            "max_angle_deviation": _dn_max_angle, "max_distance_deviation": _dn_max_dist,
        })

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Perp. Variance", f"{_dn_dev:.4f} m²")
        mc2.metric("Max Angle Δ", f"{_dn_max_angle:.1f}°")
        mc3.metric("Max Distance Δ", f"{_dn_max_dist:.2f}m")
        mc4.metric("Iterations", f"{_dn_iters}")

        # --- Static plot ---
        _dn_all_x: list[float] = []
        _dn_all_y: list[float] = []
        _dn_max_dd = 0.0
        _dn_md_orig = np.array([0.0, 0.0])
        _dn_md_gen = np.array([0.0, 0.0])
        for orig, gen in _dn_plot_data:
            _dn_all_x.extend(orig[:, 0].tolist()); _dn_all_x.extend(gen[:, 0].tolist())
            _dn_all_y.extend(orig[:, 1].tolist()); _dn_all_y.extend(gen[:, 1].tolist())
            for li in range(orig.shape[0]):
                d = float(np.linalg.norm(gen[li] - orig[li]))
                if d > _dn_max_dd:
                    _dn_max_dd = d; _dn_md_orig = orig[li].copy(); _dn_md_gen = gen[li].copy()

        _dn_xlo = min(_dn_all_x) - max((max(_dn_all_x) - min(_dn_all_x)) * 0.05, 1.0)
        _dn_xhi = max(_dn_all_x) + max((max(_dn_all_x) - min(_dn_all_x)) * 0.05, 1.0)
        _dn_ylo = min(_dn_all_y) - max((max(_dn_all_y) - min(_dn_all_y)) * 0.08, 0.5)
        _dn_yhi = max(_dn_all_y) + max((max(_dn_all_y) - min(_dn_all_y)) * 0.08, 0.5)

        _dn_dx = _dn_xhi - _dn_xlo; _dn_dy = _dn_yhi - _dn_ylo
        _dn_tr = 1.3
        if _dn_dx > _dn_dy * _dn_tr:
            pad = (_dn_dx / _dn_tr - _dn_dy) / 2; _dn_ylo -= pad; _dn_yhi += pad; _dn_dy = _dn_yhi - _dn_ylo
        elif _dn_dy * _dn_tr > _dn_dx:
            pad = (_dn_dy * _dn_tr - _dn_dx) / 2; _dn_xlo -= pad; _dn_xhi += pad; _dn_dx = _dn_xhi - _dn_xlo
        _dn_fw = 10.0
        _dn_fh = max(3.0, min(7.5, _dn_fw * (_dn_dy / _dn_dx))) if _dn_dx > 0 else 5.0

        fig_s = Figure(figsize=(_dn_fw, _dn_fh), dpi=150)
        ax_s = fig_s.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_o, oid in enumerate(_dn_sorted_oids):
            orig, gen = _dn_plot_data[idx_o]
            lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
            ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.3, ls='--', label=f"{lbl} original")
            ax_s.plot(gen[:, 0], gen[:, 1], lw=1.5, color=clr, alpha=1.0, label=f"{lbl} generated")
        ax_s.annotate(f"max Δ={_dn_max_dd:.2f}m", xy=(_dn_md_gen[0], _dn_md_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_dn_xlo, _dn_xhi); ax_s.set_ylim(_dn_ylo, _dn_yhi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)"); ax_s.set_ylabel("d2 / y-as (m)")
        _dn_ts_info = ", ".join(f"obj{oid}:{all_points_plot[oid].shape[0]}" for oid in _dn_sorted_oids)
        ax_s.set_title(f"Config #{_dn_cnum} — PV={_dn_dev:.4f} m² — {_dn_ts_info} timestamps")
        fig_s.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', dpi=150); buf_s.seek(0)
        st.image(buf_s, use_container_width=True)

        # --- Animated GIF ---
        _dn_all_ts: list[float] = []
        for oid in _dn_sorted_oids:
            _dn_all_ts.extend(all_vals_plot[oid].tolist())
        _dn_unique_ts = sorted(set(_dn_all_ts))
        _dn_n_frames = len(_dn_unique_ts)

        # GIF axis limits (same padding logic)
        _dn_gif_xmin = _dn_xlo; _dn_gif_xmax = _dn_xhi
        _dn_gif_ymin = _dn_ylo; _dn_gif_ymax = _dn_yhi
        _dn_gif_fw = _dn_fw; _dn_gif_fh = _dn_fh

        # Build per-object (t, coord) lists
        _dn_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _dn_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _dn_gi2 = 0
        for oid in _dn_sorted_oids:
            n_pts = all_points_plot[oid].shape[0]
            vals = all_vals_plot[oid]
            _dn_orig_by_obj[oid] = []; _dn_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_v = float(vals[li])
                oc = all_points_plot[oid][li]
                gc = _dn_gen_map.get(_dn_gi2 + li, oc)
                _dn_orig_by_obj[oid].append((t_v, oc))
                _dn_gen_by_obj[oid].append((t_v, np.array(gc)))
            _dn_gi2 += n_pts

        gif_frames: list[PILImage.Image] = []
        gif_progress = st.progress(0)
        gif_status = st.empty()

        for fi, tc in enumerate(_dn_unique_ts):
            gif_status.text(f"Rendering GIF frame {fi + 1}/{_dn_n_frames} (t ≤ {tc:g})...")
            fig_f = Figure(figsize=(_dn_gif_fw, _dn_gif_fh), dpi=150)
            ax_f = fig_f.add_subplot(111)
            ax_f.set_xlim(_dn_gif_xmin, _dn_gif_xmax)
            ax_f.set_ylim(_dn_gif_ymin, _dn_gif_ymax)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            ax_f.set_title(f"Config #{_dn_cnum} — t ≤ {tc:g}  (PV={_dn_dev:.4f} m²)", fontsize=9)
            for idx_o, oid in enumerate(_dn_sorted_oids):
                lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
                opu = [c for (t, c) in _dn_orig_by_obj[oid] if t <= tc]
                if len(opu) > 1:
                    oa = np.array(opu)
                    ax_f.plot(oa[:, 0], oa[:, 1], lw=1.0, color=clr, alpha=0.35, ls='--', label=f"{lbl} orig")
                gpu = [c for (t, c) in _dn_gen_by_obj[oid] if t <= tc]
                if len(gpu) > 1:
                    ga = np.array(gpu)
                    ax_f.plot(ga[:, 0], ga[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
                    # Endpoint marker on last generated point
                    ax_f.plot(ga[-1, 0], ga[-1, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_f.legend(fontsize=7, loc='upper left')
            fig_f.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
            buf_f = io.BytesIO(); fig_f.savefig(buf_f, format='png', dpi=150); buf_f.seek(0)
            gif_frames.append(PILImage.open(buf_f).copy())
            plt.close(fig_f)
            gif_progress.progress((fi + 1) / _dn_n_frames)

        gif_progress.empty(); gif_status.empty()

        if gif_frames:
            gif_buf = io.BytesIO()
            durations = [GIF_FRAME_DURATION_MS] * len(gif_frames)
            if durations:
                durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            gif_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
            gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_dn_cnum}",
                data=gif_buf, file_name=f"config_{_dn_cnum}_animation.gif", mime="image/gif",
                key=f"dl_gif_{clear_key}_{_dn_rank}",
            )

        st.markdown("---")

    # Summary table
    if _dn_config_metrics:
        st.markdown("#### Summary")
        _dn_df = pd.DataFrame(_dn_config_metrics)
        st.dataframe(_dn_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={"rank": "Rank", "config_num": "Config #", "perp_variance": "Perp. Variance (m²)",
                     "max_angle_deviation": "Max Angle Δ (°)", "max_distance_deviation": "Max Distance Δ (m)"}
        ), use_container_width=True)

    if st.button("Clear Results & Cache", key=clear_key):
        st.session_state[requested_key] = False
        st.session_state[results_key] = None
        # Reset timestamp step back to 1
        st.session_state["_cfg_timestamp_step"] = 1
        st.cache_data.clear()
        st.rerun()


# Display ext30-half results (½ timestamps)
if st.session_state.get("_generate_ext30_half_results", None):
    _display_top_n_with_gif(
        results=st.session_state["_generate_ext30_half_results"],
        section_title="Top 3 — 100 configs × 2500 iter (½ timestamps)",
        section_description="Same settings as 'Generate 100 & Top 3' but every other timestamp (step=2)",
        clear_key="clear_ext30_half_results",
        requested_key="_generate_ext30_half_requested",
        results_key="_generate_ext30_half_results",
    )

# Display ½ timestamps filtered results (new button)
if st.session_state.get("_generate_half_ts_results", None):
    _hts_results = st.session_state["_generate_half_ts_results"]
    _hts_pp = st.session_state.get("_hts_points_plot", all_points_plot)
    _hts_vp = st.session_state.get("_hts_vals_plot", all_vals_plot)
    _hts_sorted_oids = sorted(_hts_pp.keys())

    st.markdown("---")
    _hts_n_per_obj = {oid: _hts_pp[oid].shape[0] for oid in _hts_sorted_oids}
    _hts_n_total = sum(_hts_n_per_obj.values())
    st.markdown(f"### Top 10 — 100 configs × 2500 iter (½ timestamps, filtered)")
    st.markdown(f"Exponential | PDP fundamental | **{_hts_n_total} total pts** ({', '.join(f'obj {oid}: {n}' for oid, n in _hts_n_per_obj.items())})")

    _hts_config_metrics: list[dict[str, Any]] = []

    for _hts_rank, (_hts_cnum, _hts_dev, _hts_cfg) in enumerate(_hts_results, 1):
        st.markdown(f"#### Rank {_hts_rank}: Configuration #{_hts_cnum} (PV={_hts_dev:.4f} m²)")

        _hts_iters = _hts_cfg.get("iterations", "N/A")
        _hts_sp = _hts_cfg.get("successful_points", [])
        _hts_gen_map: dict[int, np.ndarray] = {}
        for sp in _hts_sp:
            _hts_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        # Build per-object original and generated arrays
        _hts_max_angle = 0.0
        _hts_max_dist = 0.0
        _hts_plot_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (orig, gen, ts)
        _hts_gi = 0
        for oid in _hts_sorted_oids:
            n_pts = _hts_pp[oid].shape[0]
            orig = _hts_pp[oid]
            ts = _hts_vp[oid]
            gen = orig.copy()
            for li in range(n_pts):
                gi = _hts_gi + li
                if gi in _hts_gen_map:
                    gen[li] = _hts_gen_map[gi]
            _hts_plot_data.append((orig, gen, ts))
            for i in range(1, n_pts):
                o_dx, o_dy = orig[i, 0] - orig[i-1, 0], orig[i, 1] - orig[i-1, 1]
                g_dx, g_dy = gen[i, 0] - gen[i-1, 0], gen[i, 1] - gen[i-1, 1]
                ad = abs(np.degrees(np.arctan2(g_dy, g_dx) - np.arctan2(o_dy, o_dx)))
                if ad > 180: ad = 360 - ad
                _hts_max_angle = max(_hts_max_angle, ad)
                _hts_max_dist = max(_hts_max_dist, abs(np.linalg.norm([g_dx, g_dy]) - np.linalg.norm([o_dx, o_dy])))
            _hts_gi += n_pts

        _hts_config_metrics.append({
            "config_num": _hts_cnum, "rank": _hts_rank, "perp_variance": _hts_dev,
            "max_angle_deviation": _hts_max_angle, "max_distance_deviation": _hts_max_dist,
        })

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Perp. Variance", f"{_hts_dev:.4f} m²")
        mc2.metric("Max Angle Δ", f"{_hts_max_angle:.1f}°")
        mc3.metric("Max Distance Δ", f"{_hts_max_dist:.2f}m")
        mc4.metric("Iterations", f"{_hts_iters}")

        # --- Compute axis limits ---
        _hts_all_x: list[float] = []
        _hts_all_y: list[float] = []
        _hts_max_dd = 0.0
        _hts_md_orig = np.array([0.0, 0.0])
        _hts_md_gen = np.array([0.0, 0.0])
        for orig, gen, ts in _hts_plot_data:
            _hts_all_x.extend(orig[:, 0].tolist()); _hts_all_x.extend(gen[:, 0].tolist())
            _hts_all_y.extend(orig[:, 1].tolist()); _hts_all_y.extend(gen[:, 1].tolist())
            for li in range(orig.shape[0]):
                d = float(np.linalg.norm(gen[li] - orig[li]))
                if d > _hts_max_dd:
                    _hts_max_dd = d; _hts_md_orig = orig[li].copy(); _hts_md_gen = gen[li].copy()

        _hts_xlo = min(_hts_all_x) - max((max(_hts_all_x) - min(_hts_all_x)) * 0.05, 1.0)
        _hts_xhi = max(_hts_all_x) + max((max(_hts_all_x) - min(_hts_all_x)) * 0.05, 1.0)
        _hts_ylo = min(_hts_all_y) - max((max(_hts_all_y) - min(_hts_all_y)) * 0.08, 0.5)
        _hts_yhi = max(_hts_all_y) + max((max(_hts_all_y) - min(_hts_all_y)) * 0.08, 0.5)

        _hts_dx = _hts_xhi - _hts_xlo; _hts_dy = _hts_yhi - _hts_ylo
        _hts_tr = 1.3
        if _hts_dx > _hts_dy * _hts_tr:
            pad = (_hts_dx / _hts_tr - _hts_dy) / 2; _hts_ylo -= pad; _hts_yhi += pad; _hts_dy = _hts_yhi - _hts_ylo
        elif _hts_dy * _hts_tr > _hts_dx:
            pad = (_hts_dy * _hts_tr - _hts_dx) / 2; _hts_xlo -= pad; _hts_xhi += pad; _hts_dx = _hts_xhi - _hts_xlo
        _hts_fw = 10.0
        _hts_fh = max(3.0, min(7.5, _hts_fw * (_hts_dy / _hts_dx))) if _hts_dx > 0 else 5.0

        # --- Static plot: draw each object SEPARATELY ---
        fig_s = Figure(figsize=(_hts_fw, _hts_fh), dpi=150)
        ax_s = fig_s.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_o, oid in enumerate(_hts_sorted_oids):
            orig, gen, ts = _hts_plot_data[idx_o]
            lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
            # Draw each object's orig (dashed) and gen (solid) — NOT connecting between objects
            ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.3, ls='--', label=f"{lbl} orig")
            ax_s.plot(gen[:, 0], gen[:, 1], lw=1.5, color=clr, alpha=1.0, label=f"{lbl} gen")
            # Endpoint marker
            ax_s.plot(gen[-1, 0], gen[-1, 1], 'o', color=clr, markersize=6, zorder=5)
        ax_s.annotate(f"max Δ={_hts_max_dd:.2f}m", xy=(_hts_md_gen[0], _hts_md_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_hts_xlo, _hts_xhi); ax_s.set_ylim(_hts_ylo, _hts_yhi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)"); ax_s.set_ylabel("d2 / y-as (m)")
        ax_s.set_title(f"Config #{_hts_cnum}  (PV={_hts_dev:.4f} m²)")
        fig_s.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', dpi=150); buf_s.seek(0)
        st.image(buf_s, use_container_width=True)

        # --- Animated GIF: build up per object over timestamps ---
        _hts_all_ts_vals: list[float] = []
        for oid in _hts_sorted_oids:
            _hts_all_ts_vals.extend(_hts_vp[oid].tolist())
        _hts_unique_ts = sorted(set(_hts_all_ts_vals))
        _hts_n_frames = len(_hts_unique_ts)

        # Build per-object (t, coord) lists
        _hts_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _hts_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _hts_gi2 = 0
        for oid in _hts_sorted_oids:
            n_pts = _hts_pp[oid].shape[0]
            vals = _hts_vp[oid]
            _hts_orig_by_obj[oid] = []; _hts_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_v = float(vals[li])
                oc = _hts_pp[oid][li]
                gc = _hts_gen_map.get(_hts_gi2 + li, oc)
                _hts_orig_by_obj[oid].append((t_v, oc))
                _hts_gen_by_obj[oid].append((t_v, np.array(gc)))
            _hts_gi2 += n_pts

        gif_frames: list[PILImage.Image] = []
        gif_progress = st.progress(0)
        gif_status = st.empty()

        for fi, tc in enumerate(_hts_unique_ts):
            gif_status.text(f"Rendering GIF frame {fi + 1}/{_hts_n_frames} (t ≤ {tc:g})...")
            fig_f = Figure(figsize=(_hts_fw, _hts_fh), dpi=150)
            ax_f = fig_f.add_subplot(111)
            ax_f.set_xlim(_hts_xlo, _hts_xhi)
            ax_f.set_ylim(_hts_ylo, _hts_yhi)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            ax_f.set_title(f"Config #{_hts_cnum} — t ≤ {tc:g}  (PV={_hts_dev:.4f} m²)", fontsize=9)
            for idx_o, oid in enumerate(_hts_sorted_oids):
                lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
                opu = [c for (t, c) in _hts_orig_by_obj[oid] if t <= tc]
                if len(opu) > 1:
                    oa = np.array(opu)
                    ax_f.plot(oa[:, 0], oa[:, 1], lw=1.0, color=clr, alpha=0.35, ls='--', label=f"{lbl} orig")
                gpu = [c for (t, c) in _hts_gen_by_obj[oid] if t <= tc]
                if len(gpu) > 1:
                    ga = np.array(gpu)
                    ax_f.plot(ga[:, 0], ga[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
                    # Endpoint marker on last generated point
                    ax_f.plot(ga[-1, 0], ga[-1, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_f.legend(fontsize=7, loc='upper left')
            fig_f.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
            buf_f = io.BytesIO(); fig_f.savefig(buf_f, format='png', dpi=150); buf_f.seek(0)
            gif_frames.append(PILImage.open(buf_f).copy())
            plt.close(fig_f)
            gif_progress.progress((fi + 1) / _hts_n_frames)

        gif_progress.empty(); gif_status.empty()

        if gif_frames:
            gif_buf = io.BytesIO()
            durations = [GIF_FRAME_DURATION_MS] * len(gif_frames)
            if durations:
                durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            gif_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
            gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_hts_cnum}",
                data=gif_buf, file_name=f"config_{_hts_cnum}_half_ts_animation.gif", mime="image/gif",
                key=f"dl_gif_half_ts_{_hts_rank}",
            )

        st.markdown("---")

    # Summary table
    if _hts_config_metrics:
        st.markdown("#### Summary")
        _hts_df = pd.DataFrame(_hts_config_metrics)
        st.dataframe(_hts_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={"rank": "Rank", "config_num": "Config #", "perp_variance": "Perp. Variance (m²)",
                     "max_angle_deviation": "Max Angle Δ (°)", "max_distance_deviation": "Max Distance Δ (m)"}
        ), use_container_width=True)

    if st.button("Clear Results & Cache", key="clear_half_ts_results"):
        st.session_state["_generate_half_ts_requested"] = False
        st.session_state["_generate_half_ts_results"] = None
        st.session_state.pop("_hts_points_plot", None)
        st.session_state.pop("_hts_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# Display ¼ timestamps filtered results (new button)
if st.session_state.get("_generate_quarter_ts_results", None):
    _qts_results = st.session_state["_generate_quarter_ts_results"]
    _qts_pp = st.session_state.get("_qts_points_plot", all_points_plot)
    _qts_vp = st.session_state.get("_qts_vals_plot", all_vals_plot)
    _qts_sorted_oids = sorted(_qts_pp.keys())

    st.markdown("---")
    _qts_n_per_obj = {oid: _qts_pp[oid].shape[0] for oid in _qts_sorted_oids}
    _qts_n_total = sum(_qts_n_per_obj.values())
    st.markdown(f"### Top 10 — 100 configs × 2500 iter (¼ timestamps, filtered)")
    st.markdown(f"Exponential | PDP fundamental | **{_qts_n_total} total pts** ({', '.join(f'obj {oid}: {n}' for oid, n in _qts_n_per_obj.items())})")

    _qts_config_metrics: list[dict[str, Any]] = []

    for _qts_rank, (_qts_cnum, _qts_dev, _qts_cfg) in enumerate(_qts_results, 1):
        st.markdown(f"#### Rank {_qts_rank}: Configuration #{_qts_cnum} (PV={_qts_dev:.4f} m²)")

        _qts_iters = _qts_cfg.get("iterations", "N/A")
        _qts_sp = _qts_cfg.get("successful_points", [])
        _qts_gen_map: dict[int, np.ndarray] = {}
        for sp in _qts_sp:
            _qts_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        # Build per-object original and generated arrays
        _qts_max_angle = 0.0
        _qts_max_dist = 0.0
        _qts_plot_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (orig, gen, ts)
        _qts_gi = 0
        for oid in _qts_sorted_oids:
            n_pts = _qts_pp[oid].shape[0]
            orig = _qts_pp[oid]
            ts = _qts_vp[oid]
            gen = orig.copy()
            for li in range(n_pts):
                gi = _qts_gi + li
                if gi in _qts_gen_map:
                    gen[li] = _qts_gen_map[gi]
            _qts_plot_data.append((orig, gen, ts))
            for i in range(1, n_pts):
                o_dx, o_dy = orig[i, 0] - orig[i-1, 0], orig[i, 1] - orig[i-1, 1]
                g_dx, g_dy = gen[i, 0] - gen[i-1, 0], gen[i, 1] - gen[i-1, 1]
                ad = abs(np.degrees(np.arctan2(g_dy, g_dx) - np.arctan2(o_dy, o_dx)))
                if ad > 180: ad = 360 - ad
                _qts_max_angle = max(_qts_max_angle, ad)
                _qts_max_dist = max(_qts_max_dist, abs(np.linalg.norm([g_dx, g_dy]) - np.linalg.norm([o_dx, o_dy])))
            _qts_gi += n_pts

        _qts_config_metrics.append({
            "config_num": _qts_cnum, "rank": _qts_rank, "perp_variance": _qts_dev,
            "max_angle_deviation": _qts_max_angle, "max_distance_deviation": _qts_max_dist,
        })

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Perp. Variance", f"{_qts_dev:.4f} m²")
        mc2.metric("Max Angle Δ", f"{_qts_max_angle:.1f}°")
        mc3.metric("Max Distance Δ", f"{_qts_max_dist:.2f}m")
        mc4.metric("Iterations", f"{_qts_iters}")

        # --- Compute axis limits ---
        _qts_all_x: list[float] = []
        _qts_all_y: list[float] = []
        _qts_max_dd = 0.0
        _qts_md_orig = np.array([0.0, 0.0])
        _qts_md_gen = np.array([0.0, 0.0])
        for orig, gen, ts in _qts_plot_data:
            _qts_all_x.extend(orig[:, 0].tolist()); _qts_all_x.extend(gen[:, 0].tolist())
            _qts_all_y.extend(orig[:, 1].tolist()); _qts_all_y.extend(gen[:, 1].tolist())
            for li in range(orig.shape[0]):
                d = float(np.linalg.norm(gen[li] - orig[li]))
                if d > _qts_max_dd:
                    _qts_max_dd = d; _qts_md_orig = orig[li].copy(); _qts_md_gen = gen[li].copy()

        _qts_xlo = min(_qts_all_x) - max((max(_qts_all_x) - min(_qts_all_x)) * 0.05, 1.0)
        _qts_xhi = max(_qts_all_x) + max((max(_qts_all_x) - min(_qts_all_x)) * 0.05, 1.0)
        _qts_ylo = min(_qts_all_y) - max((max(_qts_all_y) - min(_qts_all_y)) * 0.08, 0.5)
        _qts_yhi = max(_qts_all_y) + max((max(_qts_all_y) - min(_qts_all_y)) * 0.08, 0.5)

        _qts_dx = _qts_xhi - _qts_xlo; _qts_dy = _qts_yhi - _qts_ylo
        _qts_tr = 1.3
        if _qts_dx > _qts_dy * _qts_tr:
            pad = (_qts_dx / _qts_tr - _qts_dy) / 2; _qts_ylo -= pad; _qts_yhi += pad; _qts_dy = _qts_yhi - _qts_ylo
        elif _qts_dy * _qts_tr > _qts_dx:
            pad = (_qts_dy * _qts_tr - _qts_dx) / 2; _qts_xlo -= pad; _qts_xhi += pad; _qts_dx = _qts_xhi - _qts_xlo
        _qts_fw = 10.0
        _qts_fh = max(3.0, min(7.5, _qts_fw * (_qts_dy / _qts_dx))) if _qts_dx > 0 else 5.0

        # --- Static plot: draw each object SEPARATELY ---
        fig_s = Figure(figsize=(_qts_fw, _qts_fh), dpi=150)
        ax_s = fig_s.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_o, oid in enumerate(_qts_sorted_oids):
            orig, gen, ts = _qts_plot_data[idx_o]
            lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
            # Draw each object's orig (dashed) and gen (solid) — NOT connecting between objects
            ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.3, ls='--', label=f"{lbl} orig")
            ax_s.plot(gen[:, 0], gen[:, 1], lw=1.5, color=clr, alpha=1.0, label=f"{lbl} gen")
            # Endpoint marker
            ax_s.plot(gen[-1, 0], gen[-1, 1], 'o', color=clr, markersize=6, zorder=5)
        ax_s.annotate(f"max Δ={_qts_max_dd:.2f}m", xy=(_qts_md_gen[0], _qts_md_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_qts_xlo, _qts_xhi); ax_s.set_ylim(_qts_ylo, _qts_yhi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)"); ax_s.set_ylabel("d2 / y-as (m)")
        ax_s.set_title(f"Config #{_qts_cnum}  (PV={_qts_dev:.4f} m²)")
        fig_s.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', dpi=150); buf_s.seek(0)
        st.image(buf_s, use_container_width=True)

        # --- Animated GIF: build up per object over timestamps ---
        _qts_all_ts_vals: list[float] = []
        for oid in _qts_sorted_oids:
            _qts_all_ts_vals.extend(_qts_vp[oid].tolist())
        _qts_unique_ts = sorted(set(_qts_all_ts_vals))
        _qts_n_frames = len(_qts_unique_ts)

        # Build per-object (t, coord) lists
        _qts_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _qts_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _qts_gi2 = 0
        for oid in _qts_sorted_oids:
            n_pts = _qts_pp[oid].shape[0]
            vals = _qts_vp[oid]
            _qts_orig_by_obj[oid] = []; _qts_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_v = float(vals[li])
                oc = _qts_pp[oid][li]
                gc = _qts_gen_map.get(_qts_gi2 + li, oc)
                _qts_orig_by_obj[oid].append((t_v, oc))
                _qts_gen_by_obj[oid].append((t_v, np.array(gc)))
            _qts_gi2 += n_pts

        gif_frames: list[PILImage.Image] = []
        gif_progress = st.progress(0)
        gif_status = st.empty()

        for fi, tc in enumerate(_qts_unique_ts):
            gif_status.text(f"Rendering GIF frame {fi + 1}/{_qts_n_frames} (t ≤ {tc:g})...")
            fig_f = Figure(figsize=(_qts_fw, _qts_fh), dpi=150)
            ax_f = fig_f.add_subplot(111)
            ax_f.set_xlim(_qts_xlo, _qts_xhi)
            ax_f.set_ylim(_qts_ylo, _qts_yhi)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            ax_f.set_title(f"Config #{_qts_cnum} — t ≤ {tc:g}  (PV={_qts_dev:.4f} m²)", fontsize=9)
            for idx_o, oid in enumerate(_qts_sorted_oids):
                lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
                opu = [c for (t, c) in _qts_orig_by_obj[oid] if t <= tc]
                if len(opu) > 1:
                    oa = np.array(opu)
                    ax_f.plot(oa[:, 0], oa[:, 1], lw=1.0, color=clr, alpha=0.35, ls='--', label=f"{lbl} orig")
                gpu = [c for (t, c) in _qts_gen_by_obj[oid] if t <= tc]
                if len(gpu) > 1:
                    ga = np.array(gpu)
                    ax_f.plot(ga[:, 0], ga[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
                    # Endpoint marker on last generated point
                    ax_f.plot(ga[-1, 0], ga[-1, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_f.legend(fontsize=7, loc='upper left')
            fig_f.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
            buf_f = io.BytesIO(); fig_f.savefig(buf_f, format='png', dpi=150); buf_f.seek(0)
            gif_frames.append(PILImage.open(buf_f).copy())
            plt.close(fig_f)
            gif_progress.progress((fi + 1) / _qts_n_frames)

        gif_progress.empty(); gif_status.empty()

        if gif_frames:
            gif_buf = io.BytesIO()
            durations = [GIF_FRAME_DURATION_MS] * len(gif_frames)
            if durations:
                durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            gif_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
            gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_qts_cnum}",
                data=gif_buf, file_name=f"config_{_qts_cnum}_quarter_ts_animation.gif", mime="image/gif",
                key=f"dl_gif_quarter_ts_{_qts_rank}",
            )

        st.markdown("---")

    # Summary table
    if _qts_config_metrics:
        st.markdown("#### Summary")
        _qts_df = pd.DataFrame(_qts_config_metrics)
        st.dataframe(_qts_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={"rank": "Rank", "config_num": "Config #", "perp_variance": "Perp. Variance (m²)",
                     "max_angle_deviation": "Max Angle Δ (°)", "max_distance_deviation": "Max Distance Δ (m)"}
        ), use_container_width=True)

    if st.button("Clear Results & Cache", key="clear_quarter_ts_results"):
        st.session_state["_generate_quarter_ts_requested"] = False
        st.session_state["_generate_quarter_ts_results"] = None
        st.session_state.pop("_qts_points_plot", None)
        st.session_state.pop("_qts_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# Display ⅛ timestamps filtered results
if st.session_state.get("_generate_eighth_ts_results", None):
    _ets_results = st.session_state["_generate_eighth_ts_results"]
    _ets_pp = st.session_state.get("_ets_points_plot", all_points_plot)
    _ets_vp = st.session_state.get("_ets_vals_plot", all_vals_plot)
    _ets_sorted_oids = sorted(_ets_pp.keys())

    st.markdown("---")
    _ets_n_per_obj = {oid: _ets_pp[oid].shape[0] for oid in _ets_sorted_oids}
    _ets_n_total = sum(_ets_n_per_obj.values())
    st.markdown(f"### Top 10 — 100 configs × 2500 iter (⅛ timestamps, filtered)")
    st.markdown(f"Exponential | PDP fundamental | **{_ets_n_total} total pts** ({', '.join(f'obj {oid}: {n}' for oid, n in _ets_n_per_obj.items())})")

    _ets_config_metrics: list[dict[str, Any]] = []

    for _ets_rank, (_ets_cnum, _ets_dev, _ets_cfg) in enumerate(_ets_results, 1):
        st.markdown(f"#### Rank {_ets_rank}: Configuration #{_ets_cnum} (PV={_ets_dev:.4f} m²)")

        _ets_iters = _ets_cfg.get("iterations", "N/A")
        _ets_sp = _ets_cfg.get("successful_points", [])
        _ets_gen_map: dict[int, np.ndarray] = {}
        for sp in _ets_sp:
            _ets_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        _ets_max_angle = 0.0
        _ets_max_dist = 0.0
        _ets_plot_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        _ets_gi = 0
        for oid in _ets_sorted_oids:
            n_pts = _ets_pp[oid].shape[0]
            orig = _ets_pp[oid]
            ts = _ets_vp[oid]
            gen = orig.copy()
            for li in range(n_pts):
                gi = _ets_gi + li
                if gi in _ets_gen_map:
                    gen[li] = _ets_gen_map[gi]
            _ets_plot_data.append((orig, gen, ts))
            for i in range(1, n_pts):
                o_dx, o_dy = orig[i, 0] - orig[i-1, 0], orig[i, 1] - orig[i-1, 1]
                g_dx, g_dy = gen[i, 0] - gen[i-1, 0], gen[i, 1] - gen[i-1, 1]
                ad = abs(np.degrees(np.arctan2(g_dy, g_dx) - np.arctan2(o_dy, o_dx)))
                if ad > 180: ad = 360 - ad
                _ets_max_angle = max(_ets_max_angle, ad)
                _ets_max_dist = max(_ets_max_dist, abs(np.linalg.norm([g_dx, g_dy]) - np.linalg.norm([o_dx, o_dy])))
            _ets_gi += n_pts

        _ets_config_metrics.append({
            "config_num": _ets_cnum, "rank": _ets_rank, "perp_variance": _ets_dev,
            "max_angle_deviation": _ets_max_angle, "max_distance_deviation": _ets_max_dist,
        })

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Perp. Variance", f"{_ets_dev:.4f} m²")
        mc2.metric("Max Angle Δ", f"{_ets_max_angle:.1f}°")
        mc3.metric("Max Distance Δ", f"{_ets_max_dist:.2f}m")
        mc4.metric("Iterations", f"{_ets_iters}")

        _ets_all_x: list[float] = []
        _ets_all_y: list[float] = []
        _ets_max_dd = 0.0
        _ets_md_orig = np.array([0.0, 0.0])
        _ets_md_gen = np.array([0.0, 0.0])
        for orig, gen, ts in _ets_plot_data:
            _ets_all_x.extend(orig[:, 0].tolist()); _ets_all_x.extend(gen[:, 0].tolist())
            _ets_all_y.extend(orig[:, 1].tolist()); _ets_all_y.extend(gen[:, 1].tolist())
            for li in range(orig.shape[0]):
                d = float(np.linalg.norm(gen[li] - orig[li]))
                if d > _ets_max_dd:
                    _ets_max_dd = d; _ets_md_orig = orig[li].copy(); _ets_md_gen = gen[li].copy()

        _ets_xlo = min(_ets_all_x) - max((max(_ets_all_x) - min(_ets_all_x)) * 0.05, 1.0)
        _ets_xhi = max(_ets_all_x) + max((max(_ets_all_x) - min(_ets_all_x)) * 0.05, 1.0)
        _ets_ylo = min(_ets_all_y) - max((max(_ets_all_y) - min(_ets_all_y)) * 0.08, 0.5)
        _ets_yhi = max(_ets_all_y) + max((max(_ets_all_y) - min(_ets_all_y)) * 0.08, 0.5)

        _ets_dx = _ets_xhi - _ets_xlo; _ets_dy = _ets_yhi - _ets_ylo
        _ets_tr = 1.3
        if _ets_dx > _ets_dy * _ets_tr:
            pad = (_ets_dx / _ets_tr - _ets_dy) / 2; _ets_ylo -= pad; _ets_yhi += pad; _ets_dy = _ets_yhi - _ets_ylo
        elif _ets_dy * _ets_tr > _ets_dx:
            pad = (_ets_dy * _ets_tr - _ets_dx) / 2; _ets_xlo -= pad; _ets_xhi += pad; _ets_dx = _ets_xhi - _ets_xlo
        _ets_fw = 10.0
        _ets_fh = max(3.0, min(7.5, _ets_fw * (_ets_dy / _ets_dx))) if _ets_dx > 0 else 5.0

        fig_s = Figure(figsize=(_ets_fw, _ets_fh), dpi=150)
        ax_s = fig_s.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_o, oid in enumerate(_ets_sorted_oids):
            orig, gen, ts = _ets_plot_data[idx_o]
            lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
            ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.3, ls='--', label=f"{lbl} orig")
            ax_s.plot(gen[:, 0], gen[:, 1], lw=1.5, color=clr, alpha=1.0, label=f"{lbl} gen")
            ax_s.plot(gen[-1, 0], gen[-1, 1], 'o', color=clr, markersize=6, zorder=5)
        ax_s.annotate(f"max Δ={_ets_max_dd:.2f}m", xy=(_ets_md_gen[0], _ets_md_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_ets_xlo, _ets_xhi); ax_s.set_ylim(_ets_ylo, _ets_yhi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)"); ax_s.set_ylabel("d2 / y-as (m)")
        ax_s.set_title(f"Config #{_ets_cnum}  (PV={_ets_dev:.4f} m²)")
        fig_s.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', dpi=150); buf_s.seek(0)
        st.image(buf_s, use_container_width=True)

        _ets_all_ts_vals: list[float] = []
        for oid in _ets_sorted_oids:
            _ets_all_ts_vals.extend(_ets_vp[oid].tolist())
        _ets_unique_ts = sorted(set(_ets_all_ts_vals))
        _ets_n_frames = len(_ets_unique_ts)

        _ets_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _ets_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _ets_gi2 = 0
        for oid in _ets_sorted_oids:
            n_pts = _ets_pp[oid].shape[0]
            vals = _ets_vp[oid]
            _ets_orig_by_obj[oid] = []; _ets_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_v = float(vals[li])
                oc = _ets_pp[oid][li]
                gc = _ets_gen_map.get(_ets_gi2 + li, oc)
                _ets_orig_by_obj[oid].append((t_v, oc))
                _ets_gen_by_obj[oid].append((t_v, np.array(gc)))
            _ets_gi2 += n_pts

        gif_frames: list[PILImage.Image] = []
        gif_progress = st.progress(0)
        gif_status = st.empty()

        for fi, tc in enumerate(_ets_unique_ts):
            gif_status.text(f"Rendering GIF frame {fi + 1}/{_ets_n_frames} (t ≤ {tc:g})...")
            fig_f = Figure(figsize=(_ets_fw, _ets_fh), dpi=150)
            ax_f = fig_f.add_subplot(111)
            ax_f.set_xlim(_ets_xlo, _ets_xhi)
            ax_f.set_ylim(_ets_ylo, _ets_yhi)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            ax_f.set_title(f"Config #{_ets_cnum} — t ≤ {tc:g}  (PV={_ets_dev:.4f} m²)", fontsize=9)
            for idx_o, oid in enumerate(_ets_sorted_oids):
                lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
                opu = [c for (t, c) in _ets_orig_by_obj[oid] if t <= tc]
                if len(opu) > 1:
                    oa = np.array(opu)
                    ax_f.plot(oa[:, 0], oa[:, 1], lw=1.0, color=clr, alpha=0.35, ls='--', label=f"{lbl} orig")
                gpu = [c for (t, c) in _ets_gen_by_obj[oid] if t <= tc]
                if len(gpu) > 1:
                    ga = np.array(gpu)
                    ax_f.plot(ga[:, 0], ga[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
                    ax_f.plot(ga[-1, 0], ga[-1, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_f.legend(fontsize=7, loc='upper left')
            fig_f.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
            buf_f = io.BytesIO(); fig_f.savefig(buf_f, format='png', dpi=150); buf_f.seek(0)
            gif_frames.append(PILImage.open(buf_f).copy())
            plt.close(fig_f)
            gif_progress.progress((fi + 1) / _ets_n_frames)

        gif_progress.empty(); gif_status.empty()

        if gif_frames:
            gif_buf = io.BytesIO()
            durations = [GIF_FRAME_DURATION_MS] * len(gif_frames)
            if durations:
                durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            gif_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
            gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_ets_cnum}",
                data=gif_buf, file_name=f"config_{_ets_cnum}_eighth_ts_animation.gif", mime="image/gif",
                key=f"dl_gif_eighth_ts_{_ets_rank}",
            )

        st.markdown("---")

    if _ets_config_metrics:
        st.markdown("#### Summary")
        _ets_df = pd.DataFrame(_ets_config_metrics)
        st.dataframe(_ets_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={"rank": "Rank", "config_num": "Config #", "perp_variance": "Perp. Variance (m²)",
                     "max_angle_deviation": "Max Angle Δ (°)", "max_distance_deviation": "Max Distance Δ (m)"}
        ), use_container_width=True)

    if st.button("Clear Results & Cache", key="clear_eighth_ts_results"):
        st.session_state["_generate_eighth_ts_requested"] = False
        st.session_state["_generate_eighth_ts_results"] = None
        st.session_state.pop("_ets_points_plot", None)
        st.session_state.pop("_ets_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# Display 1/16 timestamps filtered results
if st.session_state.get("_generate_sixteenth_ts_results", None):
    _sts_results = st.session_state["_generate_sixteenth_ts_results"]
    _sts_pp = st.session_state.get("_sts_points_plot", all_points_plot)
    _sts_vp = st.session_state.get("_sts_vals_plot", all_vals_plot)
    _sts_sorted_oids = sorted(_sts_pp.keys())

    st.markdown("---")
    _sts_n_per_obj = {oid: _sts_pp[oid].shape[0] for oid in _sts_sorted_oids}
    _sts_n_total = sum(_sts_n_per_obj.values())
    st.markdown(f"### Top 10 — 100 configs × 2500 iter (1/16 timestamps, filtered)")
    st.markdown(f"Exponential | PDP fundamental | **{_sts_n_total} total pts** ({', '.join(f'obj {oid}: {n}' for oid, n in _sts_n_per_obj.items())})")

    _sts_config_metrics: list[dict[str, Any]] = []

    for _sts_rank, (_sts_cnum, _sts_dev, _sts_cfg) in enumerate(_sts_results, 1):
        st.markdown(f"#### Rank {_sts_rank}: Configuration #{_sts_cnum} (PV={_sts_dev:.4f} m²)")

        _sts_iters = _sts_cfg.get("iterations", "N/A")
        _sts_sp = _sts_cfg.get("successful_points", [])
        _sts_gen_map: dict[int, np.ndarray] = {}
        for sp in _sts_sp:
            _sts_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        _sts_max_angle = 0.0
        _sts_max_dist = 0.0
        _sts_plot_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        _sts_gi = 0
        for oid in _sts_sorted_oids:
            n_pts = _sts_pp[oid].shape[0]
            orig = _sts_pp[oid]
            ts = _sts_vp[oid]
            gen = orig.copy()
            for li in range(n_pts):
                gi = _sts_gi + li
                if gi in _sts_gen_map:
                    gen[li] = _sts_gen_map[gi]
            _sts_plot_data.append((orig, gen, ts))
            for i in range(1, n_pts):
                o_dx, o_dy = orig[i, 0] - orig[i-1, 0], orig[i, 1] - orig[i-1, 1]
                g_dx, g_dy = gen[i, 0] - gen[i-1, 0], gen[i, 1] - gen[i-1, 1]
                ad = abs(np.degrees(np.arctan2(g_dy, g_dx) - np.arctan2(o_dy, o_dx)))
                if ad > 180: ad = 360 - ad
                _sts_max_angle = max(_sts_max_angle, ad)
                _sts_max_dist = max(_sts_max_dist, abs(np.linalg.norm([g_dx, g_dy]) - np.linalg.norm([o_dx, o_dy])))
            _sts_gi += n_pts

        _sts_config_metrics.append({
            "config_num": _sts_cnum, "rank": _sts_rank, "perp_variance": _sts_dev,
            "max_angle_deviation": _sts_max_angle, "max_distance_deviation": _sts_max_dist,
        })

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Perp. Variance", f"{_sts_dev:.4f} m²")
        mc2.metric("Max Angle Δ", f"{_sts_max_angle:.1f}°")
        mc3.metric("Max Distance Δ", f"{_sts_max_dist:.2f}m")
        mc4.metric("Iterations", f"{_sts_iters}")

        _sts_all_x: list[float] = []
        _sts_all_y: list[float] = []
        _sts_max_dd = 0.0
        _sts_md_orig = np.array([0.0, 0.0])
        _sts_md_gen = np.array([0.0, 0.0])
        for orig, gen, ts in _sts_plot_data:
            _sts_all_x.extend(orig[:, 0].tolist()); _sts_all_x.extend(gen[:, 0].tolist())
            _sts_all_y.extend(orig[:, 1].tolist()); _sts_all_y.extend(gen[:, 1].tolist())
            for li in range(orig.shape[0]):
                d = float(np.linalg.norm(gen[li] - orig[li]))
                if d > _sts_max_dd:
                    _sts_max_dd = d; _sts_md_orig = orig[li].copy(); _sts_md_gen = gen[li].copy()

        _sts_xlo = min(_sts_all_x) - max((max(_sts_all_x) - min(_sts_all_x)) * 0.05, 1.0)
        _sts_xhi = max(_sts_all_x) + max((max(_sts_all_x) - min(_sts_all_x)) * 0.05, 1.0)
        _sts_ylo = min(_sts_all_y) - max((max(_sts_all_y) - min(_sts_all_y)) * 0.08, 0.5)
        _sts_yhi = max(_sts_all_y) + max((max(_sts_all_y) - min(_sts_all_y)) * 0.08, 0.5)

        _sts_dx = _sts_xhi - _sts_xlo; _sts_dy = _sts_yhi - _sts_ylo
        _sts_tr = 1.3
        if _sts_dx > _sts_dy * _sts_tr:
            pad = (_sts_dx / _sts_tr - _sts_dy) / 2; _sts_ylo -= pad; _sts_yhi += pad; _sts_dy = _sts_yhi - _sts_ylo
        elif _sts_dy * _sts_tr > _sts_dx:
            pad = (_sts_dy * _sts_tr - _sts_dx) / 2; _sts_xlo -= pad; _sts_xhi += pad; _sts_dx = _sts_xhi - _sts_xlo
        _sts_fw = 10.0
        _sts_fh = max(3.0, min(7.5, _sts_fw * (_sts_dy / _sts_dx))) if _sts_dx > 0 else 5.0

        fig_s = Figure(figsize=(_sts_fw, _sts_fh), dpi=150)
        ax_s = fig_s.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_o, oid in enumerate(_sts_sorted_oids):
            orig, gen, ts = _sts_plot_data[idx_o]
            lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
            ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.3, ls='--', label=f"{lbl} orig")
            ax_s.plot(gen[:, 0], gen[:, 1], lw=1.5, color=clr, alpha=1.0, label=f"{lbl} gen")
            ax_s.plot(gen[-1, 0], gen[-1, 1], 'o', color=clr, markersize=6, zorder=5)
        ax_s.annotate(f"max Δ={_sts_max_dd:.2f}m", xy=(_sts_md_gen[0], _sts_md_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_sts_xlo, _sts_xhi); ax_s.set_ylim(_sts_ylo, _sts_yhi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)"); ax_s.set_ylabel("d2 / y-as (m)")
        ax_s.set_title(f"Config #{_sts_cnum}  (PV={_sts_dev:.4f} m²)")
        fig_s.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', dpi=150); buf_s.seek(0)
        st.image(buf_s, use_container_width=True)

        _sts_all_ts_vals: list[float] = []
        for oid in _sts_sorted_oids:
            _sts_all_ts_vals.extend(_sts_vp[oid].tolist())
        _sts_unique_ts = sorted(set(_sts_all_ts_vals))
        _sts_n_frames = len(_sts_unique_ts)

        _sts_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _sts_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _sts_gi2 = 0
        for oid in _sts_sorted_oids:
            n_pts = _sts_pp[oid].shape[0]
            vals = _sts_vp[oid]
            _sts_orig_by_obj[oid] = []; _sts_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_v = float(vals[li])
                oc = _sts_pp[oid][li]
                gc = _sts_gen_map.get(_sts_gi2 + li, oc)
                _sts_orig_by_obj[oid].append((t_v, oc))
                _sts_gen_by_obj[oid].append((t_v, np.array(gc)))
            _sts_gi2 += n_pts

        gif_frames: list[PILImage.Image] = []
        gif_progress = st.progress(0)
        gif_status = st.empty()

        for fi, tc in enumerate(_sts_unique_ts):
            gif_status.text(f"Rendering GIF frame {fi + 1}/{_sts_n_frames} (t ≤ {tc:g})...")
            fig_f = Figure(figsize=(_sts_fw, _sts_fh), dpi=150)
            ax_f = fig_f.add_subplot(111)
            ax_f.set_xlim(_sts_xlo, _sts_xhi)
            ax_f.set_ylim(_sts_ylo, _sts_yhi)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            ax_f.set_title(f"Config #{_sts_cnum} — t ≤ {tc:g}  (PV={_sts_dev:.4f} m²)", fontsize=9)
            for idx_o, oid in enumerate(_sts_sorted_oids):
                lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
                opu = [c for (t, c) in _sts_orig_by_obj[oid] if t <= tc]
                if len(opu) > 1:
                    oa = np.array(opu)
                    ax_f.plot(oa[:, 0], oa[:, 1], lw=1.0, color=clr, alpha=0.35, ls='--', label=f"{lbl} orig")
                gpu = [c for (t, c) in _sts_gen_by_obj[oid] if t <= tc]
                if len(gpu) > 1:
                    ga = np.array(gpu)
                    ax_f.plot(ga[:, 0], ga[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
                    ax_f.plot(ga[-1, 0], ga[-1, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_f.legend(fontsize=7, loc='upper left')
            fig_f.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
            buf_f = io.BytesIO(); fig_f.savefig(buf_f, format='png', dpi=150); buf_f.seek(0)
            gif_frames.append(PILImage.open(buf_f).copy())
            plt.close(fig_f)
            gif_progress.progress((fi + 1) / _sts_n_frames)

        gif_progress.empty(); gif_status.empty()

        if gif_frames:
            gif_buf = io.BytesIO()
            durations = [GIF_FRAME_DURATION_MS] * len(gif_frames)
            if durations:
                durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            gif_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
            gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_sts_cnum}",
                data=gif_buf, file_name=f"config_{_sts_cnum}_sixteenth_ts_animation.gif", mime="image/gif",
                key=f"dl_gif_sixteenth_ts_{_sts_rank}",
            )

        st.markdown("---")

    if _sts_config_metrics:
        st.markdown("#### Summary")
        _sts_df = pd.DataFrame(_sts_config_metrics)
        st.dataframe(_sts_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={"rank": "Rank", "config_num": "Config #", "perp_variance": "Perp. Variance (m²)",
                     "max_angle_deviation": "Max Angle Δ (°)", "max_distance_deviation": "Max Distance Δ (m)"}
        ), use_container_width=True)

    if st.button("Clear Results & Cache", key="clear_sixteenth_ts_results"):
        st.session_state["_generate_sixteenth_ts_requested"] = False
        st.session_state["_generate_sixteenth_ts_results"] = None
        st.session_state.pop("_sts_points_plot", None)
        st.session_state.pop("_sts_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# ============= Display 6-Event Single Iteration results ============
if st.session_state.get("_generate_6ev_single_results", None):
    _6evs_results_raw = st.session_state["_generate_6ev_single_results"]
    # Prepend the original (unmodified) configuration as index 0
    _6evs_original_entry = (0, 0.0, {"successful_points": [], "move_succeeded": True})
    _6evs_results = [_6evs_original_entry] + list(_6evs_results_raw)
    _6evs_pp = st.session_state.get("_6evs_points_plot", all_points_plot)
    _6evs_vp = st.session_state.get("_6evs_vals_plot", all_vals_plot)
    _6evs_sorted_oids = sorted(_6evs_pp.keys())

    st.markdown("---")
    _6evs_n_per_obj = {oid: _6evs_pp[oid].shape[0] for oid in _6evs_sorted_oids}
    _6evs_n_total = sum(_6evs_n_per_obj.values())
    st.markdown(f"### 6-Event Single Iteration (timestamps: 0, 34, 67, 183, 213, 249)")
    _6evs_disp_variant = st.session_state.get("_6evs_pdp_variant", "fundamental")
    _6evs_disp_bx = st.session_state.get("_6evs_buffer_x", 1.5)
    _6evs_disp_ry = st.session_state.get("_6evs_rough_y", 0.4)
    st.markdown(
        f"Exponential | PDP {_6evs_disp_variant}"
        f"{' | buf_x=' + str(_6evs_disp_bx) if _6evs_disp_variant in ('buffer','bufferrough','realistic') else ''}"
        f"{' | rough_y=' + str(_6evs_disp_ry) if _6evs_disp_variant in ('rough','bufferrough','realistic') else ''}"
        f" | **{_6evs_n_total} total pts** "
        f"({', '.join(f'obj {oid}: {n}' for oid, n in _6evs_n_per_obj.items())})"
    )

    # Settings summary (including lane settings)
    with st.expander("⚙️ Chosen settings", expanded=False):
        _6evs_show_lanes = st.checkbox("Show lanes", value=st.session_state.get("_6evs_show_lanes", True), key="_6evs_show_lanes")
        _lane_col1, _lane_col2 = st.columns(2)
        with _lane_col1:
            _6evs_lane1_center = st.number_input(
                "Lane 1 center (y)", value=st.session_state.get("_6evs_lane1_center", 0.0),
                step=0.5, format="%.1f", key="_6evs_lane1_center",
            )
        with _lane_col2:
            _6evs_lane2_center = st.number_input(
                "Lane 2 center (y)", value=st.session_state.get("_6evs_lane2_center", -3.0),
                step=0.5, format="%.1f", key="_6evs_lane2_center",
            )
        _6evs_lane_width = 3.0  # fixed 3 m per lane
        st.caption(f"Lane width: {_6evs_lane_width} m | Lane 1: [{_6evs_lane1_center - _6evs_lane_width/2:.1f}, {_6evs_lane1_center + _6evs_lane_width/2:.1f}] | Lane 2: [{_6evs_lane2_center - _6evs_lane_width/2:.1f}, {_6evs_lane2_center + _6evs_lane_width/2:.1f}]")
        _6evs_settings_display = {
            "PDP variant": st.session_state.get("_6evs_pdp_variant", "fundamental"),
            "Buffer X": st.session_state.get("_6evs_buffer_x", 1.5),
            "Rough Y": st.session_state.get("_6evs_rough_y", 0.4),
            "Strategy": "exponential",
            "Timestamps": [0, 34, 67, 183, 213, 249],
            "Point selection": "Single point (random)",
            "Y-axis range": "[-10, +10]",
            "X-axis": "data range + 20% margin",
            "Equal aspect": False,
            "Lanes": {
                "show": _6evs_show_lanes,
                "lane_width": _6evs_lane_width,
                "lane_1_center_y": _6evs_lane1_center,
                "lane_2_center_y": _6evs_lane2_center,
            },
        }
        st.json(_6evs_settings_display)

    # ---- Quick-browse navigation ----
    _6evs_n_iters = len(_6evs_results)
    if "_6evs_browse_idx" not in st.session_state:
        st.session_state["_6evs_browse_idx"] = 0
    # Clamp to valid range
    _6evs_browse_idx = int(st.session_state["_6evs_browse_idx"])
    if _6evs_browse_idx >= _6evs_n_iters:
        _6evs_browse_idx = _6evs_n_iters - 1
    if _6evs_browse_idx < 0:
        _6evs_browse_idx = 0

    # ---- Check if animation just finished generating; activate playback ----
    if st.session_state.pop("_6evs_anim_pending", False):
        st.session_state["_6evs_anim_active"] = True
        st.session_state["_6evs_anim_paused"] = False
        _anim_start_idx = st.session_state.get("_6evs_anim_start_idx", 0)
        st.session_state["_6evs_anim_end_idx"] = _6evs_n_iters - 1
        # Jump to first frame of the animation (the one after the start)
        _first_anim = min(_anim_start_idx + 1, _6evs_n_iters - 1)
        st.session_state["_6evs_browse_idx"] = _first_anim
        _6evs_browse_idx = _first_anim

    # ---- Animation auto-advance ----
    _6evs_anim_active = st.session_state.get("_6evs_anim_active", False)
    if _6evs_anim_active and not st.session_state.get("_6evs_anim_paused", False):
        _anim_end_idx = st.session_state.get("_6evs_anim_end_idx", _6evs_n_iters - 1)
        if _6evs_browse_idx < _anim_end_idx:
            _anim_wait = st.session_state.get("_6evs_anim_speed_val", 10.0)
            time.sleep(_anim_wait)
            st.session_state["_6evs_browse_idx"] = _6evs_browse_idx + 1
            st.rerun()
        else:
            # Reached the end: stop animation
            st.session_state["_6evs_anim_active"] = False
            st.session_state["_6evs_anim_paused"] = False

    _6evs_n_generated = _6evs_n_iters - 1  # exclude the original entry
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 3, 1, 1])
    # Sync widget value with browse index BEFORE rendering the number_input
    st.session_state["_6evs_nav_num_input"] = _6evs_browse_idx
    with nav_col1:
        if st.button("⏮ First", key="_6evs_nav_first", disabled=(_6evs_browse_idx == 0)):
            st.session_state["_6evs_browse_idx"] = 0
            st.rerun()
    with nav_col2:
        if st.button("◀ Prev", key="_6evs_nav_prev", disabled=(_6evs_browse_idx == 0)):
            st.session_state["_6evs_browse_idx"] = _6evs_browse_idx - 1
            st.rerun()
    with nav_col3:
        _6evs_new_idx = st.number_input(
            "Config", min_value=0, max_value=_6evs_n_iters - 1,
            value=_6evs_browse_idx, step=1, key="_6evs_nav_num_input",
            label_visibility="collapsed",
        )
        if int(_6evs_new_idx) != _6evs_browse_idx:
            st.session_state["_6evs_browse_idx"] = int(_6evs_new_idx)
            st.rerun()
        _nav_label = "**Original**" if _6evs_browse_idx == 0 else f"Iteration {_6evs_browse_idx}"
        st.caption(f"{_nav_label}  /  {_6evs_n_generated} iterations")
    with nav_col4:
        if st.button("Next ▶", key="_6evs_nav_next", disabled=(_6evs_browse_idx >= _6evs_n_iters - 1)):
            st.session_state["_6evs_browse_idx"] = _6evs_browse_idx + 1
            st.rerun()
    with nav_col5:
        if st.button("Last ⏭", key="_6evs_nav_last", disabled=(_6evs_browse_idx >= _6evs_n_iters - 1)):
            st.session_state["_6evs_browse_idx"] = _6evs_n_iters - 1
            st.rerun()

    # ---- Display the selected iteration ----
    _6evs_cnum, _6evs_dev, _6evs_cfg = _6evs_results[_6evs_browse_idx]
    _6evs_sp = _6evs_cfg.get("successful_points", [])
    _6evs_n_moves = len(_6evs_sp)
    # Count unique point indices that have been moved
    _6evs_unique_moved = len({int(sp["original_parent_idx"]) for sp in _6evs_sp})

    # --- Determine which object/timestamp was moved in THIS iteration ---
    # Build flat-index → (object_id, timestamp) mapping
    _6evs_flat_to_obj_ts: list[tuple[int, float]] = []
    _6evs_gi_map = 0
    for _oid_map in _6evs_sorted_oids:
        for _li_map in range(_6evs_pp[_oid_map].shape[0]):
            _6evs_flat_to_obj_ts.append((_oid_map, float(_6evs_vp[_oid_map][_li_map])))
        _6evs_gi_map += _6evs_pp[_oid_map].shape[0]
    _6evs_move_succeeded = _6evs_cfg.get("move_succeeded", True)
    # The last sp entry in the cumulative list is the point moved in this iteration
    _6evs_moved_info = ""
    if _6evs_sp and _6evs_move_succeeded:
        _6evs_last_sp = _6evs_sp[-1]
        _6evs_moved_fidx = int(_6evs_last_sp["original_parent_idx"])
        if 0 <= _6evs_moved_fidx < len(_6evs_flat_to_obj_ts):
            _mv_oid, _mv_ts = _6evs_flat_to_obj_ts[_6evs_moved_fidx]
            _mv_lbl = OBJECT_LABELS[int(_mv_oid) % len(OBJECT_LABELS)]
            _mv_parent = _6evs_last_sp.get("parent_point")
            _mv_new = _6evs_last_sp.get("point")
            _mv_dist = float(np.linalg.norm(np.asarray(_mv_new) - np.asarray(_mv_parent))) if _mv_parent is not None and _mv_new is not None else 0.0
            _6evs_moved_info = f"🔄 Iteration #{_6evs_cnum}: moved **{_mv_lbl}** at **t={int(_mv_ts)}** (flat idx {_6evs_moved_fidx}) — Δ = {_mv_dist:.4f} m"
    elif not _6evs_move_succeeded:
        _6evs_moved_info = f"❌ Iteration #{_6evs_cnum}: move **mislukt** — punt bleef op startpositie (7 halveringen zonder geldige PDP-move)"

    if _6evs_browse_idx == 0:
        st.markdown("#### Original configuration  (PV = 0.000000 m²)")
    else:
        st.markdown(
            f"#### Iteration {_6evs_cnum} / {_6evs_n_generated}  "
            f"(PV={_6evs_dev:.6f} m²)  \n"
            f"**{_6evs_n_moves}** point(s) moved — "
            f"**{_6evs_unique_moved}** unique points"
        )
    if _6evs_moved_info:
        if _6evs_move_succeeded:
            st.info(_6evs_moved_info)
        else:
            st.warning(_6evs_moved_info)

    _6evs_gen_map: dict[int, np.ndarray] = {}
    for sp in _6evs_sp:
        _6evs_gen_map[int(sp["original_parent_idx"])] = sp["point"]

    _6evs_plot_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    _6evs_gi = 0
    for oid in _6evs_sorted_oids:
        n_pts = _6evs_pp[oid].shape[0]
        orig = _6evs_pp[oid]
        ts = _6evs_vp[oid]
        gen = orig.copy()
        for li in range(n_pts):
            gi = _6evs_gi + li
            if gi in _6evs_gen_map:
                gen[li] = _6evs_gen_map[gi]
        _6evs_plot_data.append((orig, gen, ts))
        _6evs_gi += n_pts

    # ---- Compute axis ranges ----
    _6evs_all_x: list[float] = []
    _6evs_all_y: list[float] = []
    for orig, gen, ts in _6evs_plot_data:
        _6evs_all_x.extend(orig[:, 0].tolist())
        _6evs_all_x.extend(gen[:, 0].tolist())
        _6evs_all_y.extend(orig[:, 1].tolist())
        _6evs_all_y.extend(gen[:, 1].tolist())

    # X-axis: data range + 20% margin on each side
    _6evs_x_data_min = min(_6evs_all_x) if _6evs_all_x else 0
    _6evs_x_data_max = max(_6evs_all_x) if _6evs_all_x else 1
    _6evs_x_range = _6evs_x_data_max - _6evs_x_data_min
    _6evs_x_margin = max(_6evs_x_range * 0.20, 5.0)
    _6evs_xlo = _6evs_x_data_min - _6evs_x_margin
    _6evs_xhi = _6evs_x_data_max + _6evs_x_margin
    # Y-axis: fixed at [-10, +10]
    _6evs_ylo = -10.0
    _6evs_yhi = 10.0

    # ---- Matplotlib static plot (NO equal aspect) ----
    _6evs_fw = 14.0
    _6evs_fh = 5.0
    fig_s = Figure(figsize=(_6evs_fw, _6evs_fh), dpi=150)
    ax_s = fig_s.add_subplot(111)
    # Deliberately NOT setting equal aspect

    # ---- Draw horizontal lanes (if enabled) ----
    _6evs_show_lanes_val = st.session_state.get("_6evs_show_lanes", True)
    if _6evs_show_lanes_val:
        _6evs_lw = 3.0  # lane width in m
        _6evs_l1c = st.session_state.get("_6evs_lane1_center", 0.0)
        _6evs_l2c = st.session_state.get("_6evs_lane2_center", -3.0)
        # Road surface: both lanes in dark-gray (#A9A9A9)
        ax_s.axhspan(_6evs_l1c - _6evs_lw / 2, _6evs_l1c + _6evs_lw / 2,
                     color='#A9A9A9', alpha=0.35, zorder=0)
        ax_s.axhspan(_6evs_l2c - _6evs_lw / 2, _6evs_l2c + _6evs_lw / 2,
                     color='#A9A9A9', alpha=0.35, zorder=0)
        # Outer road edges: solid black lines
        _6evs_road_top = max(_6evs_l1c + _6evs_lw / 2, _6evs_l2c + _6evs_lw / 2)
        _6evs_road_bot = min(_6evs_l1c - _6evs_lw / 2, _6evs_l2c - _6evs_lw / 2)
        ax_s.axhline(_6evs_road_top, color='black', linewidth=1.0, linestyle='-', zorder=1)
        ax_s.axhline(_6evs_road_bot, color='black', linewidth=1.0, linestyle='-', zorder=1)
        # Dashed white center line between the two lanes
        _6evs_divider_y = (_6evs_l1c + _6evs_l2c) / 2.0
        ax_s.axhline(_6evs_divider_y, color='white', linewidth=1.2, linestyle='--', zorder=1)
    for idx_o, oid in enumerate(_6evs_sorted_oids):
        orig, gen, ts = _6evs_plot_data[idx_o]
        lbl = OBJECT_LABELS[int(oid) % len(OBJECT_LABELS)]
        clr = f"C{int(oid)}"
        # Original: dashed + point labels
        ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.4, ls='--', label=f"{lbl} orig")
        for li in range(orig.shape[0]):
            ax_s.plot(orig[li, 0], orig[li, 1], 'x', color=clr, markersize=5, alpha=0.4)
            ax_s.annotate(f"t={int(ts[li])}", xy=(orig[li, 0], orig[li, 1]),
                          fontsize=5, color=clr, alpha=0.5,
                          xytext=(3, 3), textcoords='offset points')
        # Generated: solid + markers
        ax_s.plot(gen[:, 0], gen[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
        for li in range(gen.shape[0]):
            ax_s.plot(gen[li, 0], gen[li, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_s.annotate(f"t={int(ts[li])}", xy=(gen[li, 0], gen[li, 1]),
                          fontsize=5, color=clr, alpha=0.9,
                          xytext=(3, -8), textcoords='offset points')
    ax_s.set_xlim(_6evs_xlo, _6evs_xhi)
    ax_s.set_ylim(_6evs_ylo, _6evs_yhi)
    ax_s.legend(fontsize=7, loc='upper left')
    ax_s.set_xlabel("d1 / x-as (m)")
    ax_s.set_ylabel("d2 / y-as (m)")
    _6evs_plot_title = "6-Event — Original" if _6evs_browse_idx == 0 else f"6-Event — Iteration {_6evs_cnum}/{_6evs_n_generated}  |  {_6evs_n_moves} pts moved, {_6evs_unique_moved} unique  (PV={_6evs_dev:.6f} m²)"
    ax_s.set_title(_6evs_plot_title)
    ax_s.grid(True, alpha=0.3)
    fig_s.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.12)
    buf_s = io.BytesIO()
    fig_s.savefig(buf_s, format='png', dpi=150)
    buf_s.seek(0)
    st.image(buf_s, use_container_width=True)
    plt.close(fig_s)

    # ---- Sanity checks: road boundary, proximity & trajectory angle alerts ----
    _6evs_alerts: list[str] = []

    # 1) Road-edge proximity — flag points within 1.2 m of the outer road edge
    _6evs_edge_margin = 1.2  # metres
    if _6evs_show_lanes_val:
        _6evs_lw_chk = 3.0
        _6evs_l1c_chk = st.session_state.get("_6evs_lane1_center", 0.0)
        _6evs_l2c_chk = st.session_state.get("_6evs_lane2_center", -3.0)
        _6evs_road_top_chk = max(_6evs_l1c_chk + _6evs_lw_chk / 2, _6evs_l2c_chk + _6evs_lw_chk / 2)
        _6evs_road_bot_chk = min(_6evs_l1c_chk - _6evs_lw_chk / 2, _6evs_l2c_chk - _6evs_lw_chk / 2)
        _6evs_near_edge: list[str] = []
        for idx_chk, oid_chk in enumerate(_6evs_sorted_oids):
            _orig_chk, _gen_chk, _ts_chk = _6evs_plot_data[idx_chk]
            lbl_chk = OBJECT_LABELS[int(oid_chk) % len(OBJECT_LABELS)]
            for li_chk in range(_gen_chk.shape[0]):
                y_chk = float(_gen_chk[li_chk, 1])
                dist_top = _6evs_road_top_chk - y_chk
                dist_bot = y_chk - _6evs_road_bot_chk
                min_edge_dist = min(dist_top, dist_bot)
                if min_edge_dist < _6evs_edge_margin:
                    side = "boven" if dist_top < dist_bot else "onder"
                    _6evs_near_edge.append(
                        f"**{lbl_chk}** t={int(_ts_chk[li_chk])}: y={y_chk:.2f}m — "
                        f"{min_edge_dist:.2f}m van {side}rand "
                        f"(weg: [{_6evs_road_bot_chk:.1f}, {_6evs_road_top_chk:.1f}])"
                    )
        if _6evs_near_edge:
            _6evs_alerts.append(
                f"🚧 **Punt(en) dicht bij wegrand!** (<{_6evs_edge_margin}m van buitenrand)\n- "
                + "\n- ".join(_6evs_near_edge)
            )

    # 2) d1/d2 proximity check — flag when objects are too close at the same timestamp
    #    Condition: |d1| < 5 m AND |d2| < 2.2 m simultaneously
    _6evs_prox_d1_lim = 5.0   # metres along driving direction (x)
    _6evs_prox_d2_lim = 2.2   # metres perpendicular to driving direction (y)
    if len(_6evs_sorted_oids) >= 2:
        # Build per-timestamp generated positions for each object
        _obj0_idx = 0
        _obj1_idx = 1
        _gen0 = _6evs_plot_data[_obj0_idx][1]  # generated points object 0
        _ts0 = _6evs_plot_data[_obj0_idx][2]
        _gen1 = _6evs_plot_data[_obj1_idx][1]  # generated points object 1
        _ts1 = _6evs_plot_data[_obj1_idx][2]
        _lbl0 = OBJECT_LABELS[int(_6evs_sorted_oids[_obj0_idx]) % len(OBJECT_LABELS)]
        _lbl1 = OBJECT_LABELS[int(_6evs_sorted_oids[_obj1_idx]) % len(OBJECT_LABELS)]
        _6evs_prox: list[str] = []
        # Match timestamps between both objects
        _ts0_list = [float(t) for t in _ts0]
        _ts1_list = [float(t) for t in _ts1]
        for _pi0, _t0 in enumerate(_ts0_list):
            if _t0 in _ts1_list:
                _pi1 = _ts1_list.index(_t0)
                _dx = abs(float(_gen0[_pi0, 0]) - float(_gen1[_pi1, 0]))
                _dy = abs(float(_gen0[_pi0, 1]) - float(_gen1[_pi1, 1]))
                if _dx < _6evs_prox_d1_lim and _dy < _6evs_prox_d2_lim:
                    _6evs_prox.append(
                        f"t={int(_t0)}: d1={_dx:.2f}m, d2={_dy:.2f}m "
                        f"(limiet: d1<{_6evs_prox_d1_lim}m EN d2<{_6evs_prox_d2_lim}m)"
                    )
        if _6evs_prox:
            _6evs_alerts.append(
                f"⚠️ **Objecten te dicht bij!** {_lbl0} en {_lbl1} bevinden zich "
                f"op de volgende timestamps te dicht bij elkaar:\n- "
                + "\n- ".join(_6evs_prox)
            )

    # 3) Sharp-angle check — detect unrealistically sharp turns in overtaking
    #    Compute the deflection angle at each intermediate point of each generated
    #    trajectory.  A deflection near 90° (or equivalently an interior angle of
    #    90° / 270°) means the vehicle turns nearly perpendicular — completely
    #    unrealistic for a smooth lane-change manoeuvre.
    _6evs_sharp_min_deg = 70.0   # flag deflection angles in [70°, 110°]
    _6evs_sharp_max_deg = 110.0  # i.e. "near 90°"
    _6evs_sharp: list[str] = []
    for idx_chk, oid_chk in enumerate(_6evs_sorted_oids):
        _orig_chk, _gen_chk, _ts_chk = _6evs_plot_data[idx_chk]
        lbl_chk = OBJECT_LABELS[int(oid_chk) % len(OBJECT_LABELS)]
        n_gen = _gen_chk.shape[0]
        if n_gen < 3:
            continue
        for li_chk in range(1, n_gen - 1):
            v1 = _gen_chk[li_chk] - _gen_chk[li_chk - 1]      # incoming segment
            v2 = _gen_chk[li_chk + 1] - _gen_chk[li_chk]        # outgoing segment
            len1 = float(np.linalg.norm(v1))
            len2 = float(np.linalg.norm(v2))
            if len1 < 1e-9 or len2 < 1e-9:
                continue  # skip degenerate (zero-length) segments
            cos_angle = float(np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0))
            # deflection: 0° = straight ahead, 90° = right-angle turn, 180° = U-turn
            deflection_deg = float(np.degrees(np.arccos(cos_angle)))
            if _6evs_sharp_min_deg <= deflection_deg <= _6evs_sharp_max_deg:
                _6evs_sharp.append(
                    f"**{lbl_chk}** at t={int(_ts_chk[li_chk])}: "
                    f"deflection ≈ {deflection_deg:.1f}° (near 90° — sharp turn)"
                )
    if _6evs_sharp:
        _6evs_alerts.append(
            f"📐 **Sharp turn detected!** Deflection angles near 90° "
            f"({_6evs_sharp_min_deg:.0f}°–{_6evs_sharp_max_deg:.0f}°) found — "
            f"this indicates an unrealistically sharp lane-change:\n- "
            + "\n- ".join(_6evs_sharp)
        )

    # Show all alerts
    for _alert_msg in _6evs_alerts:
        st.warning(_alert_msg)

    # ---- Show generation log (if any) ----
    _6evs_gen_log = st.session_state.pop("_6evs_gen_log", None)
    if _6evs_gen_log is not None:
        _6evs_log_failed = _6evs_gen_log.get('iters_failed', 0)
        _6evs_log_completed = _6evs_gen_log.get('iters_completed', 0)
        _6evs_log_succeeded = _6evs_log_completed - _6evs_log_failed
        if _6evs_log_failed == 0:
            st.success(
                f"✅ **{_6evs_log_completed}** iteratie(s) voltooid — "
                f"alle moves geslaagd "
                f"({_6evs_gen_log['accumulated_sp']} totaal verplaatste punten, "
                f"maxdist={_6evs_gen_log.get('maxdist_used', '?'):.4f}m)"
            )
        else:
            st.warning(
                f"⚠️ **{_6evs_log_completed}** iteratie(s) voltooid — "
                f"**{_6evs_log_succeeded}** geslaagd, **{_6evs_log_failed}** mislukt "
                f"(PDP of veiligheidscontrole gefaald — punt bleef op startpositie). "
                f"{_6evs_gen_log['accumulated_sp']} totaal verplaatste punten, "
                f"maxdist={_6evs_gen_log.get('maxdist_used', '?'):.4f}m"
            )

    # ---- Generate Next / Generate N buttons (right below graph) ----
    _6evs_btn_col1, _6evs_btn_col2, _6evs_btn_col3 = st.columns([2, 1, 2])
    with _6evs_btn_col1:
        if st.button("🔄 Next iteration (+1)", key="_6evs_gen_next", type="primary",
                     help="Pick a random point and move it (1 iteration), starting from the current state"):
            _6evs_cur = _6evs_results[_6evs_browse_idx]
            st.session_state["_6evs_continue_from"] = _6evs_cur
            st.session_state["_6evs_batch_count"] = 1
            st.session_state["_generate_6ev_single_requested"] = True
            st.session_state["_generate_6ev_single_results"] = None
            st.rerun()
    with _6evs_btn_col2:
        _6evs_n_gen = st.number_input(
            "N", min_value=2, max_value=5000, value=10, step=10,
            key="_6evs_gen_n_input", label_visibility="collapsed",
        )
    with _6evs_btn_col3:
        if st.button(f"⚡ {_6evs_n_gen} more iterations", key="_6evs_gen_batch",
                     help=f"Run {_6evs_n_gen} iterations in one go, each moving 1 random point"):
            _6evs_cur = _6evs_results[_6evs_browse_idx]
            st.session_state["_6evs_continue_from"] = _6evs_cur
            st.session_state["_6evs_batch_count"] = int(_6evs_n_gen)
            st.session_state["_generate_6ev_single_requested"] = True
            st.session_state["_generate_6ev_single_results"] = None
            st.rerun()

    # ---- Animation row ----
    _6evs_anim_active = st.session_state.get("_6evs_anim_active", False)
    st.markdown("---")
    _anim_col1, _anim_col2, _anim_col3 = st.columns([1, 1, 1], gap="small")
    with _anim_col1:
        _6evs_anim_n = st.number_input(
            "Anim. iterations", min_value=2, max_value=5000, value=100, step=10,
            key="_6evs_anim_n_input", label_visibility="collapsed",
            help="Number of iterations to generate and then animate through.",
            disabled=_6evs_anim_active,
        )
    with _anim_col2:
        _6evs_anim_speed = st.number_input(
            "Sec/frame", min_value=0.5, max_value=30.0, value=10.0, step=0.5,
            format="%.1f", key="_6evs_anim_speed", label_visibility="collapsed",
            help="Seconds to display each iteration before advancing.",
            disabled=_6evs_anim_active,
        )
    with _anim_col3:
        if not _6evs_anim_active:
            if st.button(f"\u25b6 Animate {int(_6evs_anim_n)} iters ({_6evs_anim_speed}s/frame)",
                         key="_6evs_anim_start",
                         help="Generate iterations in batch, then play back one by one."):
                # Generate the batch first
                _6evs_cur = _6evs_results[_6evs_browse_idx]
                st.session_state["_6evs_continue_from"] = _6evs_cur
                st.session_state["_6evs_batch_count"] = int(_6evs_anim_n)
                st.session_state["_6evs_anim_pending"] = True  # flag to start anim after generation
                st.session_state["_6evs_anim_speed_val"] = float(_6evs_anim_speed)
                st.session_state["_6evs_anim_start_idx"] = _6evs_browse_idx  # first frame of animation
                st.session_state["_generate_6ev_single_requested"] = True
                st.session_state["_generate_6ev_single_results"] = None
                st.rerun()

    # Animation transport controls (shown when active)
    if _6evs_anim_active:
        _ac1, _ac2, _ac3, _ac4 = st.columns([1, 1, 1, 2], gap="small")
        with _ac1:
            if st.button("\u23ee Back", key="_6evs_anim_back"):
                _new = max(0, _6evs_browse_idx - 1)
                st.session_state["_6evs_browse_idx"] = _new
                st.session_state["_6evs_anim_paused"] = True  # pause after manual back
                st.rerun()
        with _ac2:
            _paused = st.session_state.get("_6evs_anim_paused", False)
            if _paused:
                if st.button("\u25b6 Resume", key="_6evs_anim_resume", type="primary"):
                    st.session_state["_6evs_anim_paused"] = False
                    st.rerun()
            else:
                if st.button("\u23f8 Pause", key="_6evs_anim_pause"):
                    st.session_state["_6evs_anim_paused"] = True
                    st.rerun()
        with _ac3:
            if st.button("\u23f9 Stop", key="_6evs_anim_stop"):
                st.session_state["_6evs_anim_active"] = False
                st.session_state["_6evs_anim_paused"] = False
                st.rerun()
        with _ac4:
            _anim_end = st.session_state.get("_6evs_anim_end_idx", _6evs_n_iters - 1)
            _anim_start = st.session_state.get("_6evs_anim_start_idx", 0)
            _anim_frame = _6evs_browse_idx - _anim_start
            _anim_total = _anim_end - _anim_start
            _anim_status = "\u23f8 gepauzeerd" if st.session_state.get("_6evs_anim_paused", False) else "\u25b6 speelt"
            st.caption(f"Animatie: frame {_anim_frame + 1} / {_anim_total + 1}  |  {_anim_status}")

    # ---- PDP Inequality Matrices (d1 & d2, original vs generated) ----
    # Build flat original and generated point arrays for this config
    _6evs_orig_flat_list: list[np.ndarray] = []
    _6evs_gen_flat_list: list[np.ndarray] = []
    for orig, gen, ts in _6evs_plot_data:
        for li in range(orig.shape[0]):
            _6evs_orig_flat_list.append(orig[li])
            _6evs_gen_flat_list.append(gen[li])
    _6evs_orig_flat = np.array(_6evs_orig_flat_list) if _6evs_orig_flat_list else np.array([]).reshape(0, 2)
    _6evs_gen_flat = np.array(_6evs_gen_flat_list) if _6evs_gen_flat_list else np.array([]).reshape(0, 2)

    if _6evs_orig_flat.shape[0] > 0:
        _6evs_pdp_detail = check_pdp_match_detailed(
            _6evs_orig_flat, _6evs_gen_flat,
            pdp_variant="fundamental",
        )

        _6evs_d1_pct = _6evs_pdp_detail.get("d1_percentage", 0.0) * 100
        _6evs_d2_pct = _6evs_pdp_detail.get("d2_percentage", 0.0) * 100
        _6evs_d1_mm = _6evs_pdp_detail.get("d1_mismatches", 0)
        _6evs_d2_mm = _6evs_pdp_detail.get("d2_mismatches", 0)
        st.markdown(
            f"**PDP fundamental** | d1: {_6evs_d1_pct:.1f}% ({_6evs_d1_mm} mismatches) | "
            f"d2: {_6evs_d2_pct:.1f}% ({_6evs_d2_mm} mismatches)"
        )

        from matplotlib.colors import ListedColormap as _6evs_LCM
        _6evs_hm_cmap = _6evs_LCM(['#00AA00', '#FFFF00', '#FF0000'])

        # Build point labels: k0, l0, k1, l1, ... (by timestamp, then object)
        _6evs_timestamps_used: list[float] = []
        for _, _, ts in _6evs_plot_data:
            for t in ts:
                if float(t) not in _6evs_timestamps_used:
                    _6evs_timestamps_used.append(float(t))
        _6evs_timestamps_used.sort()

        _6evs_obj_labels_short = []
        for oid in _6evs_sorted_oids:
            _6evs_obj_labels_short.append(OBJECT_LABELS[int(oid) % len(OBJECT_LABELS)])

        # Build label list matching flat order (obj-first: k0 k1 k2 ... l0 l1 l2 ...)
        _6evs_flat_labels: list[str] = []
        for idx_o, oid in enumerate(_6evs_sorted_oids):
            lbl = _6evs_obj_labels_short[idx_o]
            ts_arr = _6evs_vp[oid]
            for li in range(len(ts_arr)):
                t_val = int(ts_arr[li])
                _6evs_flat_labels.append(f"{lbl} t={t_val}")

        # Reorder to interleaved by timestamp: (k_t0, l_t0, k_t1, l_t1, ...)
        _6evs_n_obj = len(_6evs_sorted_oids)
        _6evs_n_ts = len(_6evs_timestamps_used)
        # Build mapping: for each (timestamp_idx, obj_idx) -> flat_idx
        _6evs_reorder_idx: list[int] = []
        _6evs_reorder_labels: list[str] = []
        _6evs_offset = 0
        _6evs_obj_offsets: list[int] = []
        for oid in _6evs_sorted_oids:
            _6evs_obj_offsets.append(_6evs_offset)
            _6evs_offset += _6evs_pp[oid].shape[0]
        for ti, t_val in enumerate(_6evs_timestamps_used):
            for oi, oid in enumerate(_6evs_sorted_oids):
                ts_arr = _6evs_vp[oid]
                for li in range(len(ts_arr)):
                    if abs(float(ts_arr[li]) - t_val) < 0.5:
                        flat_idx = _6evs_obj_offsets[oi] + li
                        _6evs_reorder_idx.append(flat_idx)
                        _6evs_reorder_labels.append(_6evs_flat_labels[flat_idx])
                        break

        def _6evs_reorder_matrix(m: np.ndarray) -> np.ndarray:
            idx = _6evs_reorder_idx
            if len(idx) != m.shape[0]:
                return m
            return m[np.ix_(idx, idx)]

        def _6evs_create_heatmap(matrix: np.ndarray, title: str,
                                  comp_matrix: np.ndarray | None = None) -> Figure:
            n = matrix.shape[0]
            display = _6evs_reorder_matrix(matrix)
            fig_hm, ax_hm = plt.subplots(figsize=(3.5, 3.5))
            ax_hm.imshow(display, cmap=_6evs_hm_cmap, vmin=0, vmax=2, aspect='equal')
            # Highlight differences
            if comp_matrix is not None:
                comp_display = _6evs_reorder_matrix(comp_matrix)
                for i in range(n):
                    for j in range(n):
                        if display[i, j] != comp_display[i, j]:
                            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                  fill=False, edgecolor='black', linewidth=1.5)
                            ax_hm.add_patch(rect)
            if len(_6evs_reorder_labels) == n and n <= 16:
                ax_hm.set_xticks(range(n))
                ax_hm.set_yticks(range(n))
                ax_hm.set_xticklabels(_6evs_reorder_labels, fontsize=5, rotation=90)
                ax_hm.set_yticklabels(_6evs_reorder_labels, fontsize=5)
            else:
                ax_hm.set_xticks([])
                ax_hm.set_yticks([])
            ax_hm.set_title(title, fontsize=9, fontweight='bold')
            fig_hm.tight_layout()
            return fig_hm

        _6evs_orig_d1 = _6evs_pdp_detail.get("original_d1_matrix")
        _6evs_orig_d2 = _6evs_pdp_detail.get("original_d2_matrix")
        _6evs_gen_d1 = _6evs_pdp_detail.get("generated_d1_matrix")
        _6evs_gen_d2 = _6evs_pdp_detail.get("generated_d2_matrix")

        hmc1, hmc2, hmc3, hmc4 = st.columns(4, gap="small")
        with hmc1:
            st.markdown("**Original d1**")
            if _6evs_orig_d1 is not None:
                _fig1 = _6evs_create_heatmap(_6evs_orig_d1, "Original d1 (x)")
                st.pyplot(_fig1)
                plt.close(_fig1)
        with hmc2:
            st.markdown("**Original d2**")
            if _6evs_orig_d2 is not None:
                _fig2 = _6evs_create_heatmap(_6evs_orig_d2, "Original d2 (y)")
                st.pyplot(_fig2)
                plt.close(_fig2)
        with hmc3:
            st.markdown("**Generated d1**")
            if _6evs_gen_d1 is not None:
                _fig3 = _6evs_create_heatmap(_6evs_gen_d1, "Generated d1 (x)",
                                              comp_matrix=_6evs_orig_d1)
                st.pyplot(_fig3)
                plt.close(_fig3)
        with hmc4:
            st.markdown("**Generated d2**")
            if _6evs_gen_d2 is not None:
                _fig4 = _6evs_create_heatmap(_6evs_gen_d2, "Generated d2 (y)",
                                              comp_matrix=_6evs_orig_d2)
                st.pyplot(_fig4)
                plt.close(_fig4)

        st.caption("Legend: 🟩 Green (0) = j > i | 🟨 Yellow (1) = j ≈ i | 🟥 Red (2) = j < i | ▪ Border = differs from original")

    # ---- Data table ----
    with st.expander("📊 Point coordinates", expanded=False):
        _6evs_table_rows = []
        for idx_o, oid in enumerate(_6evs_sorted_oids):
            orig, gen, ts = _6evs_plot_data[idx_o]
            lbl = OBJECT_LABELS[int(oid) % len(OBJECT_LABELS)]
            for li in range(orig.shape[0]):
                _6evs_table_rows.append({
                    "Object": lbl,
                    "t": int(ts[li]),
                    "orig_d1": round(float(orig[li, 0]), 4),
                    "orig_d2": round(float(orig[li, 1]), 4),
                    "gen_d1": round(float(gen[li, 0]), 4),
                    "gen_d2": round(float(gen[li, 1]), 4),
                    "Δd1": round(float(gen[li, 0] - orig[li, 0]), 4),
                    "Δd2": round(float(gen[li, 1] - orig[li, 1]), 4),
                })
        st.dataframe(pd.DataFrame(_6evs_table_rows), use_container_width=True)

    st.markdown("---")

    if st.button("Clear 6-Event Single Results", key="clear_6ev_single_results"):
        st.session_state["_generate_6ev_single_requested"] = False
        st.session_state["_generate_6ev_single_results"] = None
        st.session_state.pop("_6evs_points_plot", None)
        st.session_state.pop("_6evs_vals_plot", None)
        st.session_state.pop("_6evs_browse_idx", None)
        st.session_state.pop("_6evs_continue_from", None)
        st.cache_data.clear()
        st.rerun()

# Display 4-timestamps filtered results
if st.session_state.get("_generate_four_ts_results", None):
    _fts_results = st.session_state["_generate_four_ts_results"]
    _fts_pp = st.session_state.get("_fts_points_plot", all_points_plot)
    _fts_vp = st.session_state.get("_fts_vals_plot", all_vals_plot)
    _fts_sorted_oids = sorted(_fts_pp.keys())

    st.markdown("---")
    _fts_n_per_obj = {oid: _fts_pp[oid].shape[0] for oid in _fts_sorted_oids}
    _fts_n_total = sum(_fts_n_per_obj.values())
    st.markdown(f"### Top 10 — 100 configs × 2500 iter (4 timestamps: 0, 46, 92, 136)")
    st.markdown(f"Exponential | PDP fundamental | **{_fts_n_total} total pts** ({', '.join(f'obj {oid}: {n}' for oid, n in _fts_n_per_obj.items())})")

    _fts_config_metrics: list[dict[str, Any]] = []

    for _fts_rank, (_fts_cnum, _fts_dev, _fts_cfg) in enumerate(_fts_results, 1):
        st.markdown(f"#### Rank {_fts_rank}: Configuration #{_fts_cnum} (PV={_fts_dev:.4f} m²)")

        _fts_iters = _fts_cfg.get("iterations", "N/A")
        _fts_sp = _fts_cfg.get("successful_points", [])
        _fts_gen_map: dict[int, np.ndarray] = {}
        for sp in _fts_sp:
            _fts_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        _fts_max_angle = 0.0
        _fts_max_dist = 0.0
        _fts_plot_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        _fts_gi = 0
        for oid in _fts_sorted_oids:
            n_pts = _fts_pp[oid].shape[0]
            orig = _fts_pp[oid]
            ts = _fts_vp[oid]
            gen = orig.copy()
            for li in range(n_pts):
                gi = _fts_gi + li
                if gi in _fts_gen_map:
                    gen[li] = _fts_gen_map[gi]
            _fts_plot_data.append((orig, gen, ts))
            for i in range(1, n_pts):
                o_dx, o_dy = orig[i, 0] - orig[i-1, 0], orig[i, 1] - orig[i-1, 1]
                g_dx, g_dy = gen[i, 0] - gen[i-1, 0], gen[i, 1] - gen[i-1, 1]
                ad = abs(np.degrees(np.arctan2(g_dy, g_dx) - np.arctan2(o_dy, o_dx)))
                if ad > 180: ad = 360 - ad
                _fts_max_angle = max(_fts_max_angle, ad)
                _fts_max_dist = max(_fts_max_dist, abs(np.linalg.norm([g_dx, g_dy]) - np.linalg.norm([o_dx, o_dy])))
            _fts_gi += n_pts

        _fts_config_metrics.append({
            "config_num": _fts_cnum, "rank": _fts_rank, "perp_variance": _fts_dev,
            "max_angle_deviation": _fts_max_angle, "max_distance_deviation": _fts_max_dist,
        })

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Perp. Variance", f"{_fts_dev:.4f} m²")
        mc2.metric("Max Angle Δ", f"{_fts_max_angle:.1f}°")
        mc3.metric("Max Distance Δ", f"{_fts_max_dist:.2f}m")
        mc4.metric("Iterations", f"{_fts_iters}")

        _fts_all_x: list[float] = []
        _fts_all_y: list[float] = []
        _fts_max_dd = 0.0
        _fts_md_orig = np.array([0.0, 0.0])
        _fts_md_gen = np.array([0.0, 0.0])
        for orig, gen, ts in _fts_plot_data:
            _fts_all_x.extend(orig[:, 0].tolist()); _fts_all_x.extend(gen[:, 0].tolist())
            _fts_all_y.extend(orig[:, 1].tolist()); _fts_all_y.extend(gen[:, 1].tolist())
            for li in range(orig.shape[0]):
                d = float(np.linalg.norm(gen[li] - orig[li]))
                if d > _fts_max_dd:
                    _fts_max_dd = d; _fts_md_orig = orig[li].copy(); _fts_md_gen = gen[li].copy()

        _fts_xlo = min(_fts_all_x) - max((max(_fts_all_x) - min(_fts_all_x)) * 0.05, 1.0)
        _fts_xhi = max(_fts_all_x) + max((max(_fts_all_x) - min(_fts_all_x)) * 0.05, 1.0)
        _fts_ylo = min(_fts_all_y) - max((max(_fts_all_y) - min(_fts_all_y)) * 0.08, 0.5)
        _fts_yhi = max(_fts_all_y) + max((max(_fts_all_y) - min(_fts_all_y)) * 0.08, 0.5)

        _fts_dx = _fts_xhi - _fts_xlo; _fts_dy = _fts_yhi - _fts_ylo
        _fts_tr = 1.3
        if _fts_dx > _fts_dy * _fts_tr:
            pad = (_fts_dx / _fts_tr - _fts_dy) / 2; _fts_ylo -= pad; _fts_yhi += pad; _fts_dy = _fts_yhi - _fts_ylo
        elif _fts_dy * _fts_tr > _fts_dx:
            pad = (_fts_dy * _fts_tr - _fts_dx) / 2; _fts_xlo -= pad; _fts_xhi += pad; _fts_dx = _fts_xhi - _fts_xlo
        _fts_fw = 10.0
        _fts_fh = max(3.0, min(7.5, _fts_fw * (_fts_dy / _fts_dx))) if _fts_dx > 0 else 5.0

        fig_s = Figure(figsize=(_fts_fw, _fts_fh), dpi=150)
        ax_s = fig_s.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_o, oid in enumerate(_fts_sorted_oids):
            orig, gen, ts = _fts_plot_data[idx_o]
            lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
            ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.3, ls='--', label=f"{lbl} orig")
            ax_s.plot(gen[:, 0], gen[:, 1], lw=1.5, color=clr, alpha=1.0, label=f"{lbl} gen")
            ax_s.plot(gen[-1, 0], gen[-1, 1], 'o', color=clr, markersize=6, zorder=5)
        ax_s.annotate(f"max Δ={_fts_max_dd:.2f}m", xy=(_fts_md_gen[0], _fts_md_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_fts_xlo, _fts_xhi); ax_s.set_ylim(_fts_ylo, _fts_yhi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)"); ax_s.set_ylabel("d2 / y-as (m)")
        ax_s.set_title(f"Config #{_fts_cnum}  (PV={_fts_dev:.4f} m²)")
        fig_s.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', dpi=150); buf_s.seek(0)
        st.image(buf_s, use_container_width=True)

        _fts_all_ts_vals: list[float] = []
        for oid in _fts_sorted_oids:
            _fts_all_ts_vals.extend(_fts_vp[oid].tolist())
        _fts_unique_ts = sorted(set(_fts_all_ts_vals))
        _fts_n_frames = len(_fts_unique_ts)

        _fts_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _fts_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _fts_gi2 = 0
        for oid in _fts_sorted_oids:
            n_pts = _fts_pp[oid].shape[0]
            vals = _fts_vp[oid]
            _fts_orig_by_obj[oid] = []; _fts_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_v = float(vals[li])
                oc = _fts_pp[oid][li]
                gc = _fts_gen_map.get(_fts_gi2 + li, oc)
                _fts_orig_by_obj[oid].append((t_v, oc))
                _fts_gen_by_obj[oid].append((t_v, np.array(gc)))
            _fts_gi2 += n_pts

        gif_frames: list[PILImage.Image] = []
        gif_progress = st.progress(0)
        gif_status = st.empty()

        for fi, tc in enumerate(_fts_unique_ts):
            gif_status.text(f"Rendering GIF frame {fi + 1}/{_fts_n_frames} (t ≤ {tc:g})...")
            fig_f = Figure(figsize=(_fts_fw, _fts_fh), dpi=150)
            ax_f = fig_f.add_subplot(111)
            ax_f.set_xlim(_fts_xlo, _fts_xhi)
            ax_f.set_ylim(_fts_ylo, _fts_yhi)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            ax_f.set_title(f"Config #{_fts_cnum} — t ≤ {tc:g}  (PV={_fts_dev:.4f} m²)", fontsize=9)
            for idx_o, oid in enumerate(_fts_sorted_oids):
                lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
                opu = [c for (t, c) in _fts_orig_by_obj[oid] if t <= tc]
                if len(opu) > 1:
                    oa = np.array(opu)
                    ax_f.plot(oa[:, 0], oa[:, 1], lw=1.0, color=clr, alpha=0.35, ls='--', label=f"{lbl} orig")
                gpu = [c for (t, c) in _fts_gen_by_obj[oid] if t <= tc]
                if len(gpu) > 1:
                    ga = np.array(gpu)
                    ax_f.plot(ga[:, 0], ga[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
                    ax_f.plot(ga[-1, 0], ga[-1, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_f.legend(fontsize=7, loc='upper left')
            fig_f.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
            buf_f = io.BytesIO(); fig_f.savefig(buf_f, format='png', dpi=150); buf_f.seek(0)
            gif_frames.append(PILImage.open(buf_f).copy())
            plt.close(fig_f)
            gif_progress.progress((fi + 1) / _fts_n_frames)

        gif_progress.empty(); gif_status.empty()

        if gif_frames:
            gif_buf = io.BytesIO()
            durations = [GIF_FRAME_DURATION_MS] * len(gif_frames)
            if durations:
                durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            gif_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
            gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_fts_cnum}",
                data=gif_buf, file_name=f"config_{_fts_cnum}_four_ts_animation.gif", mime="image/gif",
                key=f"dl_gif_four_ts_{_fts_rank}",
            )

        st.markdown("---")

    if _fts_config_metrics:
        st.markdown("#### Summary")
        _fts_df = pd.DataFrame(_fts_config_metrics)
        st.dataframe(_fts_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={"rank": "Rank", "config_num": "Config #", "perp_variance": "Perp. Variance (m²)",
                     "max_angle_deviation": "Max Angle Δ (°)", "max_distance_deviation": "Max Distance Δ (m)"}
        ), use_container_width=True)

    if st.button("Clear Results & Cache", key="clear_four_ts_results"):
        st.session_state["_generate_four_ts_requested"] = False
        st.session_state["_generate_four_ts_results"] = None
        st.session_state.pop("_fts_points_plot", None)
        st.session_state.pop("_fts_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# Display 2-timestamps filtered results
if st.session_state.get("_generate_two_ts_results", None):
    _tts_results = st.session_state["_generate_two_ts_results"]
    _tts_pp = st.session_state.get("_tts_points_plot", all_points_plot)
    _tts_vp = st.session_state.get("_tts_vals_plot", all_vals_plot)
    _tts_sorted_oids = sorted(_tts_pp.keys())

    st.markdown("---")
    _tts_n_per_obj = {oid: _tts_pp[oid].shape[0] for oid in _tts_sorted_oids}
    _tts_n_total = sum(_tts_n_per_obj.values())
    st.markdown(f"### Top 10 — 100 configs × 2500 iter (2 timestamps: 0, 136)")
    st.markdown(f"Exponential | PDP fundamental | **{_tts_n_total} total pts** ({', '.join(f'obj {oid}: {n}' for oid, n in _tts_n_per_obj.items())})")

    _tts_config_metrics: list[dict[str, Any]] = []

    for _tts_rank, (_tts_cnum, _tts_dev, _tts_cfg) in enumerate(_tts_results, 1):
        st.markdown(f"#### Rank {_tts_rank}: Configuration #{_tts_cnum} (PV={_tts_dev:.4f} m²)")

        _tts_iters = _tts_cfg.get("iterations", "N/A")
        _tts_sp = _tts_cfg.get("successful_points", [])
        _tts_gen_map: dict[int, np.ndarray] = {}
        for sp in _tts_sp:
            _tts_gen_map[int(sp["original_parent_idx"])] = sp["point"]

        _tts_max_angle = 0.0
        _tts_max_dist = 0.0
        _tts_plot_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        _tts_gi = 0
        for oid in _tts_sorted_oids:
            n_pts = _tts_pp[oid].shape[0]
            orig = _tts_pp[oid]
            ts = _tts_vp[oid]
            gen = orig.copy()
            for li in range(n_pts):
                gi = _tts_gi + li
                if gi in _tts_gen_map:
                    gen[li] = _tts_gen_map[gi]
            _tts_plot_data.append((orig, gen, ts))
            for i in range(1, n_pts):
                o_dx, o_dy = orig[i, 0] - orig[i-1, 0], orig[i, 1] - orig[i-1, 1]
                g_dx, g_dy = gen[i, 0] - gen[i-1, 0], gen[i, 1] - gen[i-1, 1]
                ad = abs(np.degrees(np.arctan2(g_dy, g_dx) - np.arctan2(o_dy, o_dx)))
                if ad > 180: ad = 360 - ad
                _tts_max_angle = max(_tts_max_angle, ad)
                _tts_max_dist = max(_tts_max_dist, abs(np.linalg.norm([g_dx, g_dy]) - np.linalg.norm([o_dx, o_dy])))
            _tts_gi += n_pts

        _tts_config_metrics.append({
            "config_num": _tts_cnum, "rank": _tts_rank, "perp_variance": _tts_dev,
            "max_angle_deviation": _tts_max_angle, "max_distance_deviation": _tts_max_dist,
        })

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Perp. Variance", f"{_tts_dev:.4f} m²")
        mc2.metric("Max Angle Δ", f"{_tts_max_angle:.1f}°")
        mc3.metric("Max Distance Δ", f"{_tts_max_dist:.2f}m")
        mc4.metric("Iterations", f"{_tts_iters}")

        _tts_all_x: list[float] = []
        _tts_all_y: list[float] = []
        _tts_max_dd = 0.0
        _tts_md_orig = np.array([0.0, 0.0])
        _tts_md_gen = np.array([0.0, 0.0])
        for orig, gen, ts in _tts_plot_data:
            _tts_all_x.extend(orig[:, 0].tolist()); _tts_all_x.extend(gen[:, 0].tolist())
            _tts_all_y.extend(orig[:, 1].tolist()); _tts_all_y.extend(gen[:, 1].tolist())
            for li in range(orig.shape[0]):
                d = float(np.linalg.norm(gen[li] - orig[li]))
                if d > _tts_max_dd:
                    _tts_max_dd = d; _tts_md_orig = orig[li].copy(); _tts_md_gen = gen[li].copy()

        _tts_xlo = min(_tts_all_x) - max((max(_tts_all_x) - min(_tts_all_x)) * 0.05, 1.0)
        _tts_xhi = max(_tts_all_x) + max((max(_tts_all_x) - min(_tts_all_x)) * 0.05, 1.0)
        _tts_ylo = min(_tts_all_y) - max((max(_tts_all_y) - min(_tts_all_y)) * 0.08, 0.5)
        _tts_yhi = max(_tts_all_y) + max((max(_tts_all_y) - min(_tts_all_y)) * 0.08, 0.5)

        _tts_dx = _tts_xhi - _tts_xlo; _tts_dy = _tts_yhi - _tts_ylo
        _tts_tr = 1.3
        if _tts_dx > _tts_dy * _tts_tr:
            pad = (_tts_dx / _tts_tr - _tts_dy) / 2; _tts_ylo -= pad; _tts_yhi += pad; _tts_dy = _tts_yhi - _tts_ylo
        elif _tts_dy * _tts_tr > _tts_dx:
            pad = (_tts_dy * _tts_tr - _tts_dx) / 2; _tts_xlo -= pad; _tts_xhi += pad; _tts_dx = _tts_xhi - _tts_xlo
        _tts_fw = 10.0
        _tts_fh = max(3.0, min(7.5, _tts_fw * (_tts_dy / _tts_dx))) if _tts_dx > 0 else 5.0

        fig_s = Figure(figsize=(_tts_fw, _tts_fh), dpi=150)
        ax_s = fig_s.add_subplot(111)
        ax_s.set_aspect("equal", adjustable="datalim")
        for idx_o, oid in enumerate(_tts_sorted_oids):
            orig, gen, ts = _tts_plot_data[idx_o]
            lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
            ax_s.plot(orig[:, 0], orig[:, 1], lw=1.0, color=clr, alpha=0.3, ls='--', label=f"{lbl} orig")
            ax_s.plot(gen[:, 0], gen[:, 1], lw=1.5, color=clr, alpha=1.0, label=f"{lbl} gen")
            ax_s.plot(gen[-1, 0], gen[-1, 1], 'o', color=clr, markersize=6, zorder=5)
        ax_s.annotate(f"max Δ={_tts_max_dd:.2f}m", xy=(_tts_md_gen[0], _tts_md_gen[1]),
                      xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax_s.set_xlim(_tts_xlo, _tts_xhi); ax_s.set_ylim(_tts_ylo, _tts_yhi)
        ax_s.legend(fontsize=7, loc='upper left')
        ax_s.set_xlabel("d1 / x-as (m)"); ax_s.set_ylabel("d2 / y-as (m)")
        ax_s.set_title(f"Config #{_tts_cnum}  (PV={_tts_dev:.4f} m²)")
        fig_s.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
        buf_s = io.BytesIO(); fig_s.savefig(buf_s, format='png', dpi=150); buf_s.seek(0)
        st.image(buf_s, use_container_width=True)

        _tts_all_ts_vals: list[float] = []
        for oid in _tts_sorted_oids:
            _tts_all_ts_vals.extend(_tts_vp[oid].tolist())
        _tts_unique_ts = sorted(set(_tts_all_ts_vals))
        _tts_n_frames = len(_tts_unique_ts)

        _tts_orig_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _tts_gen_by_obj: dict[int, list[tuple[float, np.ndarray]]] = {}
        _tts_gi2 = 0
        for oid in _tts_sorted_oids:
            n_pts = _tts_pp[oid].shape[0]
            vals = _tts_vp[oid]
            _tts_orig_by_obj[oid] = []; _tts_gen_by_obj[oid] = []
            for li in range(n_pts):
                t_v = float(vals[li])
                oc = _tts_pp[oid][li]
                gc = _tts_gen_map.get(_tts_gi2 + li, oc)
                _tts_orig_by_obj[oid].append((t_v, oc))
                _tts_gen_by_obj[oid].append((t_v, np.array(gc)))
            _tts_gi2 += n_pts

        gif_frames: list[PILImage.Image] = []
        gif_progress = st.progress(0)
        gif_status = st.empty()

        for fi, tc in enumerate(_tts_unique_ts):
            gif_status.text(f"Rendering GIF frame {fi + 1}/{_tts_n_frames} (t ≤ {tc:g})...")
            fig_f = Figure(figsize=(_tts_fw, _tts_fh), dpi=150)
            ax_f = fig_f.add_subplot(111)
            ax_f.set_xlim(_tts_xlo, _tts_xhi)
            ax_f.set_ylim(_tts_ylo, _tts_yhi)
            ax_f.set_aspect("equal", adjustable="datalim")
            ax_f.set_xlabel("d1 / x-as (m)", fontsize=9)
            ax_f.set_ylabel("d2 / y-as (m)", fontsize=9)
            ax_f.set_title(f"Config #{_tts_cnum} — t ≤ {tc:g}  (PV={_tts_dev:.4f} m²)", fontsize=9)
            for idx_o, oid in enumerate(_tts_sorted_oids):
                lbl = OBJECT_LABELS[oid % len(OBJECT_LABELS)]; clr = f"C{oid}"
                opu = [c for (t, c) in _tts_orig_by_obj[oid] if t <= tc]
                if len(opu) > 1:
                    oa = np.array(opu)
                    ax_f.plot(oa[:, 0], oa[:, 1], lw=1.0, color=clr, alpha=0.35, ls='--', label=f"{lbl} orig")
                gpu = [c for (t, c) in _tts_gen_by_obj[oid] if t <= tc]
                if len(gpu) > 1:
                    ga = np.array(gpu)
                    ax_f.plot(ga[:, 0], ga[:, 1], lw=1.8, color=clr, alpha=1.0, label=f"{lbl} gen")
                    ax_f.plot(ga[-1, 0], ga[-1, 1], 'o', color=clr, markersize=6, zorder=5)
            ax_f.legend(fontsize=7, loc='upper left')
            fig_f.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
            buf_f = io.BytesIO(); fig_f.savefig(buf_f, format='png', dpi=150); buf_f.seek(0)
            gif_frames.append(PILImage.open(buf_f).copy())
            plt.close(fig_f)
            gif_progress.progress((fi + 1) / _tts_n_frames)

        gif_progress.empty(); gif_status.empty()

        if gif_frames:
            gif_buf = io.BytesIO()
            durations = [GIF_FRAME_DURATION_MS] * len(gif_frames)
            if durations:
                durations[-1] = GIF_LAST_FRAME_PAUSE_MS
            gif_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
            gif_buf.seek(0)
            st.download_button(
                label=f"Download GIF — Config #{_tts_cnum}",
                data=gif_buf, file_name=f"config_{_tts_cnum}_two_ts_animation.gif", mime="image/gif",
                key=f"dl_gif_two_ts_{_tts_rank}",
            )

        st.markdown("---")

    if _tts_config_metrics:
        st.markdown("#### Summary")
        _tts_df = pd.DataFrame(_tts_config_metrics)
        st.dataframe(_tts_df[["rank", "config_num", "perp_variance", "max_angle_deviation", "max_distance_deviation"]].rename(
            columns={"rank": "Rank", "config_num": "Config #", "perp_variance": "Perp. Variance (m²)",
                     "max_angle_deviation": "Max Angle Δ (°)", "max_distance_deviation": "Max Distance Δ (m)"}
        ), use_container_width=True)

    if st.button("Clear Results & Cache", key="clear_two_ts_results"):
        st.session_state["_generate_two_ts_requested"] = False
        st.session_state["_generate_two_ts_results"] = None
        st.session_state.pop("_tts_points_plot", None)
        st.session_state.pop("_tts_vals_plot", None)
        st.cache_data.clear()
        st.rerun()

# Display C68 Realistic results
if st.session_state.get("_generate_c68r_results", None):
    _display_top_n_with_gif(
        results=st.session_state["_generate_c68r_results"],
        section_title="Top 10 — Config 68 Realistic (100 configs × 2500 iter)",
        section_description="Config 68 | t=82..160 step 2 | Multiple points same direction | realistic (d1 buf 5m, d2 rough 0.30m) | External pts at lane centers",
        clear_key="clear_c68r_results",
        requested_key="_generate_c68r_requested",
        results_key="_generate_c68r_results",
    )

# Display C68 Fundamental results
if st.session_state.get("_generate_c68f_results", None):
    _display_top_n_with_gif(
        results=st.session_state["_generate_c68f_results"],
        section_title="Top 10 — Config 68 Fundamental (100 configs × 2500 iter)",
        section_description="Config 68 | t=82..160 step 2 | Single point | fundamental | External pts at lane centers",
        clear_key="clear_c68f_results",
        requested_key="_generate_c68f_requested",
        results_key="_generate_c68f_results",
    )

# Display Recursive 6-Event results
if st.session_state.get("_generate_recursive_6event_results", None):
    _r6_results = st.session_state["_generate_recursive_6event_results"]
    _r6_rounds = st.session_state.get("_recursive_6event_rounds", [])
    _r6_pp = st.session_state.get("_recursive_6event_points_plot", all_points_plot)
    _r6_vp = st.session_state.get("_recursive_6event_vals_plot", all_vals_plot)
    _r6_ts = st.session_state.get("_recursive_6event_timestamps", [])
    _r6_oid = st.session_state.get("_recursive_6event_overtaker", None)
    _r6_sorted = sorted(_r6_pp.keys())

    st.markdown("---")
    st.markdown("### 🔄 Recursive 6-Event Generation — Results")
    _r6_ts_nice = ", ".join(f"{t:.0f}" for t in _r6_ts) if _r6_ts else "?"
    st.markdown(f"Overtaking object: **{_r6_oid}** | Events at t = [{_r6_ts_nice}] | **{len(_r6_rounds)} rounds** completed")

    # Show convergence table
    if _r6_rounds:
        _r6_table_data = []
        for _rnd in _r6_rounds:
            _r6_table_data.append({
                "Round": _rnd["round"],
                "Best Variance": f"{_rnd['best_variance']:.6f}",
                "Best Config #": _rnd["best_config"],
                "Valid Configs": _rnd["n_configs"],
                "Early Stopped": _rnd["early_stopped"],
            })
        st.dataframe(_r6_table_data, use_container_width=True)

    # Show top 3 from final round
    for _r6_rank, (_r6_cnum, _r6_dev, _r6_cfg) in enumerate(_r6_results, 1):
        st.markdown(f"#### Rank {_r6_rank}: Config #{_r6_cnum} (PV={_r6_dev:.6f} m²)")
        _r6_sp = _r6_cfg.get("successful_points", [])
        _r6_gm: dict[int, np.ndarray] = {}
        for _sp in _r6_sp:
            _r6_gm[int(_sp["original_parent_idx"])] = _sp["point"]
        # Plot original vs generated for the 6 event points
        fig_r6, ax_r6 = plt.subplots(figsize=(10, 5))
        _r6_gi = 0
        for _oid in _r6_sorted:
            _n = _r6_pp[_oid].shape[0]
            _orig = _r6_pp[_oid]
            _gen = _orig.copy()
            for _li in range(_n):
                _gi = _r6_gi + _li
                if _gi in _r6_gm:
                    _gen[_li] = _r6_gm[_gi]
            _clr = f"C{_oid}"
            _lbl = OBJECT_LABELS[_oid % len(OBJECT_LABELS)]
            ax_r6.plot(_orig[:, 0], _orig[:, 1], '--', color=_clr, alpha=0.4, label=f"{_lbl} orig")
            ax_r6.scatter(_orig[:, 0], _orig[:, 1], color=_clr, alpha=0.4, s=30, zorder=3)
            ax_r6.plot(_gen[:, 0], _gen[:, 1], '-', color=_clr, lw=2, label=f"{_lbl} gen")
            ax_r6.scatter(_gen[:, 0], _gen[:, 1], color=_clr, s=50, zorder=4)
            _r6_gi += _n
        ax_r6.set_xlabel("d1 / x (m)")
        ax_r6.set_ylabel("d2 / y (m)")
        ax_r6.set_ylim(-15, 15)
        ax_r6.set_title(f"Recursive 6-Event — Rank {_r6_rank} (PV={_r6_dev:.6f})")
        ax_r6.legend(fontsize=8)

        st.pyplot(fig_r6)
        plt.close(fig_r6)

        # --- Display d1 and d2 ordering for original vs generated ---
        # Build labelled point list: (label, orig_coord, gen_coord)
        _r6_labelled: list[tuple[str, np.ndarray, np.ndarray]] = []
        _r6_gi2 = 0
        for _oid in _r6_sorted:
            _n = _r6_pp[_oid].shape[0]
            _lbl = OBJECT_LABELS[_oid % len(OBJECT_LABELS)]
            _orig = _r6_pp[_oid]
            _gen = _orig.copy()
            for _li in range(_n):
                _gi = _r6_gi2 + _li
                if _gi in _r6_gm:
                    _gen[_li] = _r6_gm[_gi]
                _r6_labelled.append((f"{_lbl}{_li}", _orig[_li], _gen[_li]))
            _r6_gi2 += _n

        def _build_order_str(points_list: list[tuple[str, float]], dim_name: str) -> str:
            """Build 'dim: a < b < c' string from labelled values."""
            sorted_pts = sorted(points_list, key=lambda x: x[1])
            parts = []
            for i, (lbl, val) in enumerate(sorted_pts):
                parts.append(f"{lbl}({val:.1f})")
                if i < len(sorted_pts) - 1:
                    next_val = sorted_pts[i + 1][1]
                    parts.append(" < " if val < next_val else " = ")
            return f"**{dim_name}**: " + "".join(parts)

        _orig_d1 = [(lbl, float(oc[0])) for lbl, oc, _gc in _r6_labelled]
        _orig_d2 = [(lbl, float(oc[1])) for lbl, oc, _gc in _r6_labelled]
        _gen_d1 = [(lbl, float(gc[0])) for lbl, _oc, gc in _r6_labelled]
        _gen_d2 = [(lbl, float(gc[1])) for lbl, _oc, gc in _r6_labelled]

        st.markdown("**Original ordering:**")
        st.markdown(_build_order_str(_orig_d1, "d1"))
        st.markdown(_build_order_str(_orig_d2, "d2"))
        st.markdown("**Generated ordering:**")
        st.markdown(_build_order_str(_gen_d1, "d1"))
        st.markdown(_build_order_str(_gen_d2, "d2"))

        # Check for mismatches
        _orig_d1_order = [lbl for lbl, _ in sorted(_orig_d1, key=lambda x: x[1])]
        _gen_d1_order = [lbl for lbl, _ in sorted(_gen_d1, key=lambda x: x[1])]
        _orig_d2_order = [lbl for lbl, _ in sorted(_orig_d2, key=lambda x: x[1])]
        _gen_d2_order = [lbl for lbl, _ in sorted(_gen_d2, key=lambda x: x[1])]
        _d1_match = _orig_d1_order == _gen_d1_order
        _d2_match = _orig_d2_order == _gen_d2_order
        if _d1_match and _d2_match:
            st.success("✅ d1 and d2 ordering match")
        else:
            _mismatches = []
            if not _d1_match:
                _mismatches.append("d1")
            if not _d2_match:
                _mismatches.append("d2")
            st.error(f"❌ Ordering mismatch in: {', '.join(_mismatches)}")

    if st.button("🗑 Clear 6-Event Results", key="clear_r6_results"):
        st.session_state["_generate_recursive_6event_requested"] = False
        st.session_state["_generate_recursive_6event_results"] = None
        st.session_state.pop("_recursive_6event_rounds", None)
        st.session_state.pop("_recursive_6event_points_plot", None)
        st.session_state.pop("_recursive_6event_vals_plot", None)
        st.session_state.pop("_recursive_6event_timestamps", None)
        st.session_state.pop("_recursive_6event_overtaker", None)
        st.rerun()

# ============= Drawing (without gridlines) ============

def infer_and_draw_lanes(ax: matplotlib.axes.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    """Draw traffic lanes that follow the trajectory of the active configuration."""
    current_config_raw = st.session_state.get("cfg_c", 0)
    try:
        current_config = int(current_config_raw)
    except (TypeError, ValueError):
        return
    
    logger.debug(f"[INFER_LANES] Called for config {current_config}")

    lane_cfg = LANE_CONFIGURATIONS.get(current_config)
    if not lane_cfg:
        logger.warning(f"[INFER_LANES] No lane config found for {current_config}")
        return
    logger.debug(f"[INFER_LANES] Lane config: {lane_cfg}")

    mode = lane_cfg.get("mode", "data_path")
    if mode == "intersection":
        _draw_intersection_lanes_matplotlib(ax, lane_cfg, xlim, ylim)
        return

    lane_width = float(lane_cfg.get("lane_width", 3.0))  # type: ignore[arg-type]
    lane_count = int(lane_cfg.get("lanes", 3))  # type: ignore[arg-type]
    offset = float(lane_cfg.get("offset", 0.0))  # type: ignore[arg-type]

    # NOTE: Offset is now calculated dynamically inside _build_lane_polylines_from_data()
    # based on actual vehicle positions to center them in lanes
    
    logger.debug(f"[INFER_LANES] About to call _build_lane_polylines_from_data for config {current_config}")
    lane_polylines = _build_lane_polylines_from_data(current_config, lane_width, lane_count, xlim, offset)
    logger.debug(f"[INFER_LANES] Returned from _build_lane_polylines_from_data, result: {lane_polylines is not None}")
    if not lane_polylines:
        logger.debug("[INFER_LANES] No lane polylines returned")
        return

    road_color = "none"
    edge_line_color = "black"
    center_line_color = "black"
    lane_line_width = 0.8

    boundaries = lane_polylines.get("boundaries", [])
    center_lines = lane_polylines.get("center_lines", [])
    is_multi_path = lane_polylines.get("multi_path", False)  # type: ignore[assignment]

    # Draw boundaries (edges) - always solid
    if is_multi_path:
        # Multi-path: draw each boundary as a solid line
        for boundary in boundaries:
            ax.plot(boundary[:, 0], boundary[:, 1], color=edge_line_color, linewidth=lane_line_width, 
                   linestyle='-', alpha=1.0, zorder=1)
    else:
        # Single path: draw road polygon with edges
        if len(boundaries) >= 2:
            left_edge = boundaries[0]
            right_edge = boundaries[-1]
            polygon_arrays: list[np.ndarray] = [left_edge, right_edge[::-1]]
            polygon_points = np.vstack(polygon_arrays)
            ax.fill(polygon_points[:, 0], polygon_points[:, 1], facecolor=road_color, edgecolor='none', zorder=0)  # type: ignore[call-overload]

            for edge in (left_edge, right_edge):
                ax.plot(edge[:, 0], edge[:, 1], color=edge_line_color, linewidth=lane_line_width,  # type: ignore[call-overload]
                       linestyle='-', alpha=1.0, zorder=1)

    for dashed_line in center_lines:
        ax.plot(  # type: ignore[call-overload]
            dashed_line[:, 0],
            dashed_line[:, 1],
            color=center_line_color,
            linewidth=lane_line_width,
            linestyle="--",
            dashes=(10, 10),
            alpha=1.0,
            zorder=1,
        )

    # For curved road configs, draw subtle Frenet coordinate axes
    centerline = lane_polylines.get("centerline")
    if current_config in [15, 17] and centerline is not None:
        # Draw the centerline (basiskromme) as a thin colored line
        ax.plot(centerline[:, 0], centerline[:, 1],  # type: ignore[call-overload]
                color='#666666', linewidth=1.2, linestyle='-', 
                alpha=0.6, zorder=1, label='centerline')
        _draw_frenet_axes(ax, centerline, num_arrows=5)

# _draw_frenet_axes is now imported from pdp_utils.plotting


def setup_square_axes(ax: matplotlib.axes.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    """Configure axes to be square, with simple ticks and labels d1, d2."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)  # type: ignore
        sp.set_color("#222")
    ax.tick_params(axis="both", labelsize=9, width=0.8, color="#222")  # type: ignore
    ax.set_xlabel("d1", fontsize=11, labelpad=8)  # type: ignore
    ax.set_ylabel("d2", fontsize=11, labelpad=8)  # type: ignore
    # Draw inferred lane markings on the background (skip for custom uploads)
    if not _is_custom_upload:
        infer_and_draw_lanes(ax, xlim, ylim)

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
    # Use constrained_layout or fixed padding instead of tight_layout
    # to prevent layout changes based on content
    fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
    return fig

# Color constants and lane drawing helpers now imported from pdp_utils.drawing and pdp_utils.plotting


def create_smooth_animation(
    all_points_plot: dict[int, np.ndarray],
    all_vals_plot: dict[int, np.ndarray],
    latest_generated: dict[tuple[int, int], np.ndarray],
    selected_configs: list[int],
    configs_by_variant: dict[str, list[int]],
    all_configs_list: list[dict[str, Any]],
    external_pts_for_window: list[np.ndarray],
    external_ts_for_window: list[np.ndarray],
    external_points_list: list[tuple[np.ndarray, np.ndarray]],
    n_total_points: int,
    selected_c_int: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    speed_multiplier: float = 1.0,
) -> go.Figure:
    """
    Create a smooth animated trajectory visualization using cubic spline interpolation.
    Objects smoothly traverse through their timestamp coordinates with smooth curves.
    """
    # Create figure
    fig = go.Figure()
    
    # Add lane markings and save them to include in every frame
    if not _is_custom_upload:
        fig = add_lane_markings_to_figure(fig, selected_c_int, xlim, ylim)
    
    # Extract lane marking traces to add to every frame
    lane_marking_traces: list[Any] = [trace for trace in fig.data]
    
    # Configuration to animate - for simplicity, let's animate the first selected config
    # If "Original" is selected, use original points; otherwise use first generated config
    animate_original = "Original" in selected_configs
    
    # Prepare data for each configuration we want to animate
    configs_to_animate: list[dict[str, Any]] = []
    
    if animate_original:
        configs_to_animate.append({
            "name": "Original",
            "data": {obj_id: all_points_plot[obj_id] for obj_id in all_points_plot.keys()},
            "timestamps": all_vals_plot,
            "color_offset": 0
        })
    
    # Add generated configurations
    st.write(f"DEBUG: configs_by_variant keys: {list(configs_by_variant.keys())}")
    st.write(f"DEBUG: selected_configs: {selected_configs}")
    st.write(f"DEBUG: latest_generated has {len(latest_generated)} entries")
    if latest_generated:
        st.write(f"DEBUG: Sample latest_generated keys: {list(latest_generated.keys())[:5]}")
    
    for variant in sorted(configs_by_variant.keys()):
        for config_num in sorted(configs_by_variant[variant]):
            config_label = f"{variant} C{config_num}"
            if config_label not in selected_configs:
                continue
            
            st.write(f"DEBUG: Processing {config_label} (config_num={config_num})")
            
            # Build generated points for this config
            generated_pts_config = {}
            points_found = 0
            searched_keys = []
            for flat_idx in range(n_total_points):
                obj_id, local_idx, _ = get_object_info_for_flat_idx(flat_idx)
                if obj_id == -1:  # Skip external points
                    continue
                if obj_id not in generated_pts_config:
                    generated_pts_config[obj_id] = all_points_plot[obj_id].copy()
                searched_keys.append((config_num, flat_idx))
                if (config_num, flat_idx) in latest_generated:
                    generated_pts_config[obj_id][local_idx] = latest_generated[(config_num, flat_idx)]
                    points_found += 1
            
            st.write(f"DEBUG: Found {points_found}/{n_total_points} generated points for {config_label}")
            st.write(f"DEBUG: Sample searched keys: {searched_keys[:10]}")
            st.write(f"DEBUG: Matching keys in latest_generated: {[k for k in searched_keys[:10] if k in latest_generated]}")
            
            configs_to_animate.append({
                "name": config_label,
                "data": generated_pts_config,
                "timestamps": all_vals_plot,
                "color_offset": len(configs_to_animate)
            })
    
    st.write(f"DEBUG: Total configs_to_animate: {len(configs_to_animate)}")
    for cfg in configs_to_animate:
        st.write(f"  - {cfg['name']}: {len(cfg['data'])} objects")
    
    # Determine time range for animation
    all_timestamps: list[float] = []
    for obj_timestamps in all_vals_plot.values():
        all_timestamps.extend(obj_timestamps)
    
    if not all_timestamps:
        st.warning("No timestamp data available for animation")
        return fig
    
    t_min: float = min(all_timestamps)
    t_max: float = max(all_timestamps)
    
    # Create smooth time samples for animation (more samples = smoother, more fluid animation)
    n_frames = 200  # Increased from 100 for more fluid animation
    t_smooth = np.linspace(t_min, t_max, n_frames)  # type: ignore[misc]
    
    # For each configuration and object, create smooth trajectories
    all_frames_data: list[dict[str, Any]] = []
    
    for config_info in configs_to_animate:
        config_name = config_info["name"]
        config_data = config_info["data"]
        timestamps_data = config_info["timestamps"]
        
        for obj_idx, obj_id in enumerate(sorted(config_data.keys())):
            if obj_id == -1:  # Skip external points
                continue
            
            pts = config_data[obj_id]
            ts = np.array(timestamps_data[obj_id])
            
            if len(pts) < 2:
                # Not enough points for interpolation, skip
                continue
            
            # Create cubic spline interpolation for x and y coordinates
            # Use 'natural' boundary condition for smooth ends
            label = OBJECT_LABELS[int(obj_id) % len(OBJECT_LABELS)]
            try:
                cs_x = CubicSpline(ts, pts[:, 0], bc_type='natural')
                cs_y = CubicSpline(ts, pts[:, 1], bc_type='natural')
                
                # Evaluate splines at smooth time samples
                x_smooth = cs_x(t_smooth)
                y_smooth = cs_y(t_smooth)
                
                color_offset = config_info.get("color_offset", 0)
                color_idx = color_offset * len(config_data) + obj_idx
                # Use Plotly's default color sequence
                plotly_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                color = plotly_colors[color_idx % len(plotly_colors)]
                
                all_frames_data.append({
                    "config_name": config_name,
                    "object_label": label,
                    "color": color,
                    "x_smooth": x_smooth,
                    "y_smooth": y_smooth,
                    "t_smooth": t_smooth,
                })
            except Exception as e:
                st.warning(f"Could not interpolate trajectory for {config_name} - {label}: {e}")
                continue
    
    if not all_frames_data:
        st.warning("No valid trajectory data for animation")
        return fig
    
    # Create animation frames
    frames = []
    for frame_idx in range(n_frames):
        frame_data = []
        
        # Add lane markings to every frame (must come first so they appear behind trajectories)
        frame_data.extend(lane_marking_traces)
        
        for traj_data in all_frames_data:
            config_name = traj_data["config_name"]
            obj_label = traj_data["object_label"]
            color = traj_data["color"]
            x_smooth = traj_data["x_smooth"]
            y_smooth = traj_data["y_smooth"]
            t_smooth = traj_data["t_smooth"]
            
            # Trail: show the path up to current time
            trail_x = x_smooth[:frame_idx+1]
            trail_y = y_smooth[:frame_idx+1]
            
            # Current position
            current_x = x_smooth[frame_idx]
            current_y = y_smooth[frame_idx]
            current_t = t_smooth[frame_idx]
            
            # Add smooth trail with gradient opacity for fading effect
            frame_data.append(
                go.Scatter(
                    x=trail_x,
                    y=trail_y,
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'{config_name} ({obj_label})',
                    showlegend=(frame_idx == 0),
                )
            )
            
            # Add current position marker - larger and more visible
            frame_data.append(
                go.Scatter(
                    x=[current_x],
                    y=[current_y],
                    mode='markers',
                    marker=dict(size=14, color=color, symbol='circle',
                               line=dict(color='white', width=2.5)),
                    name=f'{config_name} ({obj_label}) - current',
                    showlegend=False,
                    hovertemplate=f'<b>{config_name}</b><br>{obj_label}<br>t={current_t:.2f}<br>d1={current_x:.2f}<br>d2={current_y:.2f}<extra></extra>',
                )
            )
        
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    # Add initial traces (first frame)
    for trace in frames[0].data:
        fig.add_trace(trace)
    
    # Add external reference points if present (they don't animate)
    if external_pts_for_window:
        ext_pts_arr = np.array(external_pts_for_window)
        fig.add_trace(  # type: ignore[call-arg]
            go.Scatter(
            x=ext_pts_arr[:, 0],
            y=ext_pts_arr[:, 1],
            mode='markers',
            name='External Reference',
            marker=dict(size=10, symbol='square', color='gray', 
                       line=dict(color='black', width=1.5)),
            showlegend=True,
        ))
    
    # Configure animation
    fig.frames = frames
    
    # Determine axis ranges for animation.
    # Prefer explicit lane-config bounds when available, otherwise use computed defaults.
    lane_cfg = LANE_CONFIGURATIONS.get(selected_c_int, {})
    lane_bounds = lane_cfg.get("bounds") if lane_cfg else None
    x_range: list[float] = [xlim[0], xlim[1]]
    y_range: list[float] = [ylim[0], ylim[1]]

    if isinstance(lane_bounds, dict):
        x_bounds = lane_bounds.get("x")
        y_bounds = lane_bounds.get("y")
        if x_bounds and len(x_bounds) == 2:
            x_range = [float(x_bounds[0]), float(x_bounds[1])]
        if y_bounds and len(y_bounds) == 2:
            y_range = [float(y_bounds[0]), float(y_bounds[1])]

    # If there are no explicit y-bounds, keep the adaptive behavior as fallback.
    if not (isinstance(lane_bounds, dict) and lane_bounds.get("y")):
        all_y_values: list[float] = []
        for traj_data in all_frames_data:
            all_y_values.extend(traj_data["y_smooth"])
        if all_y_values:
            y_min: float = float(min(all_y_values))
            y_max: float = float(max(all_y_values))
            y_margin: float = max((y_max - y_min) * 0.1, 0.25)
            y_range = [y_min - y_margin, y_max + y_margin]

    # Always ensure the right x-bound shows the full trajectory extent.
    all_x_values: list[float] = []
    for traj_data in all_frames_data:
        all_x_values.extend(traj_data["x_smooth"])
    if external_pts_for_window:
        ext_arr = np.array(external_pts_for_window)
        if ext_arr.size > 0:
            all_x_values.extend(ext_arr[:, 0].tolist())
    if all_x_values:
        x_max_data = float(max(all_x_values))
        x_range[1] = max(x_range[1], x_max_data)
    
    fig.update_layout(
        width=1100,
        height=900,
        xaxis=dict(
            range=x_range,
            title="d1",
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            range=y_range,  # Use optimized y-range for better space usage
            title="d2",
            showgrid=True,
            gridcolor='lightgray'
        ),
        title="Smooth Trajectory Animation",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": max(5, int(30 / speed_multiplier)), "redraw": True},
                        "fromcurrent": True,
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                }
            ],
            "x": 0.1,
            "y": 1.15,
            "xanchor": "left",
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "args": [[str(f.name) if hasattr(f, 'name') else str(f_idx)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": f"{float(t_smooth[int(str(f.name) if hasattr(f, 'name') else str(f_idx))]):.1f}",
                    "method": "animate"
                }
                for f_idx, f in enumerate(frames[::max(1, len(frames)//20)])  # Show ~20 slider ticks
            ],
            "x": 0.1,
            "len": 0.85,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top",
            "pad": {"b": 10, "t": 50},
            "currentvalue": {
                "visible": True,
                "prefix": "Time: ",
                "xanchor": "right",
                "font": {"size": 16}
            }
        }]
    )
    
    return fig

# _draw_intersection_lanes_matplotlib is now imported from pdp_utils.plotting

def add_lane_markings_to_figure(fig: go.Figure, c_value: int, xlim: tuple[float, float], ylim: tuple[float, float]) -> go.Figure:
    lane_cfg: dict[str, Any] | None = LANE_CONFIGURATIONS.get(c_value)
    if not lane_cfg:
        return fig

    lane_color = "rgba(0, 0, 0, 0)"
    edge_color = "rgba(0, 0, 0, 1)"
    dashed_color = "rgba(0, 0, 0, 1)"

    mode = lane_cfg.get("mode", "data_path")
    if mode == "intersection":
        return _add_intersection_lanes(fig, lane_cfg, lane_color, edge_color, dashed_color, xlim, ylim)

    lane_width = float(lane_cfg.get("lane_width", 3.0))  # type: ignore[arg-type]
    lane_count = int(lane_cfg.get("lanes", 3))  # type: ignore[arg-type]
    offset = float(lane_cfg.get("offset", 0.0))  # type: ignore[arg-type]

    # Dynamic offset for single-object configurations: force object to rightmost (bottom) lane
    if _df_all is not None:
        config_df = _df_all[_df_all["c"] == c_value]
        if not config_df.empty:
            unique_objects = config_df["o"].unique()
            if len(unique_objects) == 1:
                # Calculate offset to center the road such that y=0 corresponds to the bottom lane center
                offset = (lane_width * (lane_count - 1)) / 2.0

    lane_polylines = _build_lane_polylines_from_data(c_value, lane_width, lane_count, xlim, offset)
    if not lane_polylines:
        return fig

    return _add_lane_polylines_plotly(fig, lane_polylines, lane_color, edge_color, dashed_color)

# _add_intersection_lanes is now imported from pdp_utils.plotting

# annotate_points is now imported from pdp_utils.plotting

def draw_original(ax: matplotlib.axes.Axes) -> None:
    """Draw all object curves in the left panel, including external reference points."""
    # DEBUG: Check what's being drawn
    logger.debug(f"[DRAW_ORIGINAL] all_points_plot.keys()={list(all_points_plot.keys())}")
    for o_id in sorted(all_points_plot.keys()):
        pts = all_points_plot[o_id]
        logger.debug(f"[DRAW_ORIGINAL] o_id={o_id}, shape={pts.shape}, first_pt={pts[0] if pts.shape[0] > 0 else 'EMPTY'}")
    
    # Draw all objects uniformly
    for i, o_id in enumerate(sorted(all_points_plot.keys())):
        pts = all_points_plot[o_id]
        vals = all_vals_plot[o_id]
        if pts.shape[0] > 0:
            color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
            ax.plot(pts[:, 0], pts[:, 1], linewidth=1.2, color=color)  # type: ignore
            annotate_points(ax, pts, vals, label, color)

    # Draw lane markings in the original trajectory view (skip for custom uploads)
    current_config = st.session_state.get("anim_current_config", 1)
    if not _is_custom_upload:
        infer_and_draw_lanes(ax, XLIM, YLIM)
    
    # Draw external reference points (fixed points) with a distinct marker
    if external_pts_for_window:
        ext_pts_arr = np.array(external_pts_for_window)
        ax.scatter(ext_pts_arr[:, 0], ext_pts_arr[:, 1], 
                   s=80, marker='s', color='gray', edgecolors='black', 
                   linewidths=1.5, zorder=5, label='External ref.')  # type: ignore
        # Add labels for external points (use point index from original list)
        n_timestamps = len(selected_ts_window)
        for idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):
            # Calculate which original external point this corresponds to
            ext_point_idx = idx // n_timestamps if n_timestamps > 0 else idx
            ax.annotate(  # type: ignore
                f"ext_{ext_point_idx}",
                xy=(ext_pt[0], ext_pt[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=LABEL_FS - 1,
                color='gray',
                ha="left",
                va="bottom",
            )

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

    # Status text: shows variant, configuration, iteration and step
    current_variant = st.session_state.get("anim_current_variant", "fundamental")
    pdp_variants_list = st.session_state.get("anim_pdp_variants_list", ["fundamental"])
    current_variant_idx = st.session_state.get("anim_current_variant_idx", 0)
    total_variants = len(pdp_variants_list)
    
    if not anim_running and completed_iters > 0:
        if binary_mode:
            step_display = binary_step
        else:
            step_display = st.session_state.get("anim_last_step", 0)
        status_text = f"Variant {current_variant_idx+1}/{total_variants} ({current_variant}) | Config {current_config} | Iteration {completed_iters} | Step {step_display}"
    else:
        if binary_mode:
            step_display = binary_step
        else:
            step_display = search_steps
        status_text = f"Variant {current_variant_idx+1}/{total_variants} ({current_variant}) | Config {current_config} | Iteration {completed_iters + 1} | Step {step_display}"
        st.session_state["anim_last_step"] = step_display

    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=9,  # type: ignore
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    has_animation = st.session_state.get("show_anim_circle", False)
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    gen_pt = st.session_state.get("anim_generated_point", None)
    parent_idx = st.session_state.get("anim_parent_idx", 0)
    in_search = st.session_state.get("anim_in_search", False)

    offsets = [(3, 3), (3, -8), (-8, 3)]

    def make_label(prefix: str, tval: float, gen_marker: str = "") -> str:
        """Helper to build a LaTeX label with italic letter and subscript number."""
        try:
            tnum = float(tval)
        except (TypeError, ValueError):
            tnum = float(np.array(tval, dtype=float))
        lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
        if gen_marker:
            # e.g., k'_0 becomes $\mathit{k}'_0$
            return f"$\\mathit{{{prefix}}}{gen_marker}_{{{lbl}}}$"
        # e.g., k_0 becomes $\mathit{k}_0$
        return f"$\\mathit{{{prefix}}}_{{{lbl}}}$"

    def _get_original_index(sp: SuccessfulPoint) -> int | None:
        """Return original parent index if present, otherwise None."""
        try:
            oi = int(sp["original_parent_idx"])
            return oi
        except (KeyError, TypeError, ValueError):
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
            linewidth=1.2, color=BLUE, alpha=alpha, zorder=1
        )

    # Base l segments
    for i in range(len(l_points_plot) - 1):
        alpha = 0.2 if (i, i+1) in transparent_segments_l else 1.0
        ax.plot(
            [l_points_plot[i, 0], l_points_plot[i+1, 0]],
            [l_points_plot[i, 1], l_points_plot[i+1, 1]],
            linewidth=1.2, color=ORANGE, alpha=alpha, zorder=1
        )

    # Track which original indices already have a generated replacement
    latest_indices: set[int] = set()
    for sp in successful_points:
        oi = _get_original_index(sp)
        if oi is not None:
            latest_indices.add(oi)

    # Draw original k points where there is no generated replacement yet
    for i, ((x, y), tval) in enumerate(zip(k_points_plot, k_vals_plot)):  # type: ignore[misc]
        if i not in latest_indices:
            ax.scatter([x], [y], s=25, zorder=3, color=BLUE, alpha=1.0)  # type: ignore
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
    for i, ((x, y), tval) in enumerate(zip(l_points_plot, l_vals_plot)):  # type: ignore[misc]
        orig_idx = n_k + i
        if orig_idx not in latest_indices:
            ax.scatter([x], [y], s=25, zorder=3, color=ORANGE, alpha=1.0)  # type: ignore
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

    # Draw extra objects (id > 1) - these are just for visualization, not part of PDP
    for i, o_id in enumerate(sorted(all_points_plot.keys())):
        if o_id <= 1:  # Skip k and l, they are handled above with animation logic
            continue
        pts = all_points_plot[o_id]
        vals = all_vals_plot[o_id]
        if pts.shape[0] > 0:
            color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            label_prefix = OBJECT_LABELS[i % len(OBJECT_LABELS)]
            # Draw segments
            for seg_i in range(len(pts) - 1):
                ax.plot(
                    [pts[seg_i, 0], pts[seg_i + 1, 0]],
                    [pts[seg_i, 1], pts[seg_i + 1, 1]],
                    linewidth=1.2, color=color, alpha=0.7, zorder=1
                )
            # Draw points and labels
            for pt_i, ((x, y), tval) in enumerate(zip(pts, vals)):  # type: ignore[misc]
                ax.scatter([x], [y], s=25, zorder=3, color=color, alpha=0.8)  # type: ignore
                off = offsets[pt_i % len(offsets)]
                label = make_label(label_prefix, float(tval))
                ax.annotate(  # type: ignore
                    label,
                    xy=(x, y),
                    xytext=off,
                    textcoords="offset points",
                    fontsize=LABEL_FS,
                    color=color,
                    ha="left" if off[0] >= 0 else "right",
                    va="bottom" if off[1] >= 0 else "top",
                )

    # Draw external reference points (fixed points) - same as in left panel
    if external_pts_for_window:
        ext_pts_arr = np.array(external_pts_for_window)
        ax.scatter(ext_pts_arr[:, 0], ext_pts_arr[:, 1], 
                   s=80, marker='s', color='gray', edgecolors='black', 
                   linewidths=1.5, zorder=5, label='External ref.')  # type: ignore
        # Add labels for external points (use point index from original list)
        n_timestamps = len(selected_ts_window)
        for idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):  # type: ignore[misc]  # type: ignore[misc]
            ext_point_idx = idx // n_timestamps if n_timestamps > 0 else idx
            ax.annotate(  # type: ignore
                f"ext_{ext_point_idx}",
                xy=(ext_pt[0], ext_pt[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=LABEL_FS - 1,
                color='gray',
                ha="left",
                va="bottom",
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
                linewidth=1.2, color=BLUE, alpha=1.0, zorder=4
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
                linewidth=1.2, color=ORANGE, alpha=1.0, zorder=4
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

            # Use unified helper to get object info for this flat index
            obj_id, local_idx, prefix = get_object_info_for_flat_idx(original_parent_idx)
            # Get color based on object position
            sorted_obj_ids = sorted(all_points_plot.keys())
            obj_position = sorted_obj_ids.index(obj_id) if obj_id in sorted_obj_ids else 0
            color = OBJECT_COLORS[obj_position % len(OBJECT_COLORS)]
            tval = get_timestamp_for_flat_idx(original_parent_idx)

            ax.scatter([succ_pt[0]], [succ_pt[1]], s=40, zorder=6, color=color)  # type: ignore
            off = offsets[original_parent_idx % len(offsets)]
            try:
                tval_f = float(tval)  # type: ignore[arg-type]
            except (TypeError, ValueError):
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
    # show_anim_circle controls visibility; draw when animation is active and we have a generated point
    anim_running = st.session_state.get("anim_running", False)
    if (has_animation or anim_running) and gen_pt is not None:
        all_pts = st.session_state.get("anim_all_pts", np.array([]))
        distance = st.session_state.get("anim_distance", 0.0)
        
        # Get multi-point data if available
        selected_indices = st.session_state.get("anim_selected_indices", [])
        generated_points_raw = st.session_state.get("anim_generated_points", {})
        movement_vectors = st.session_state.get("anim_movement_vectors", {})
        
        # Convert parent_idx to int for safe comparison
        parent_idx_int = int(parent_idx) if parent_idx is not None else None
        
        # If no selected_indices, fall back to using current parent_idx
        if not selected_indices and parent_idx_int is not None:
            selected_indices = [parent_idx_int]
        
        # Build the generated_points dict fresh, prioritizing current gen_pt for parent_idx
        # This ensures the visualization always uses the most up-to-date position
        generated_points = {}
        
        # First, add any multi-point data we have (with int keys)
        for k, v in generated_points_raw.items():
            generated_points[int(k)] = v
        
        # Then, ALWAYS update the current parent_idx with the current gen_pt
        # This ensures that even if multi-point data is stale, the active point is correct
        if parent_idx_int is not None and gen_pt is not None:
            generated_points[parent_idx_int] = gen_pt

        # Draw circles and red dots for ALL selected points
        for sel_idx in selected_indices:
            sel_idx_int = int(sel_idx)
            
            # Get parent point position - check successful_points first for updated parent
            # This must match the logic in iteration setup (get_parent_for_idx)
            sel_parent_pt = None
            successful_points_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
            for s in reversed(successful_points_list):
                if int(s.get("original_parent_idx", -1)) == sel_idx_int:
                    sel_parent_pt = s["point"]
                    break
            
            # If not found in successful_points, use original position
            if sel_parent_pt is None:
                if all_pts.size > 0:
                    if sel_idx_int < n_total_points:
                        sel_parent_pt = all_pts[sel_idx_int]
                    else:
                        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                        sidx = sel_idx_int - n_total_points
                        if 0 <= sidx < len(succ_list):
                            sel_parent_pt = succ_list[sidx]["point"]
                        else:
                            sel_parent_pt = np.array([0.0, 0.0])
                else:
                    sel_parent_pt = np.array([0.0, 0.0])
            
            # Get generated point for this index
            sel_gen_pt = generated_points.get(sel_idx_int, None)
            
            # For multi-point mode: ALL circles must have the SAME radius = anim_distance
            # This is the "search radius" for the current iteration step
            circle_radius = float(distance)
            
            # Calculate red dot position: parent + direction × distance
            # Use the movement vector to get the exact direction, place dot at exact distance
            # EXCEPTION: when distance is ~0 (finalized), use stored generated_points directly
            movement_vecs = st.session_state.get("anim_movement_vectors", {})
            mv = movement_vecs.get(sel_idx_int, None)
            
            if circle_radius < 1e-6 and sel_gen_pt is not None:
                # FINALIZED: distance is 0, use stored final position from correct_orders
                # Calculate actual radius as distance from parent to final position
                red_dot_pos = np.array(sel_gen_pt)
                circle_radius = float(np.linalg.norm(red_dot_pos - sel_parent_pt))
                logger.debug(f"[DEBUG ARROW] parent={sel_parent_pt}, red_dot={red_dot_pos}, distance={circle_radius:.4f} (FINALIZED)")
            elif mv is not None and (abs(mv[0]) > 1e-9 or abs(mv[1]) > 1e-9):
                # Normalize the movement vector to get direction
                mv_arr = np.array([float(mv[0]), float(mv[1])])
                mv_mag = float(np.linalg.norm(mv_arr))
                if mv_mag > 1e-9:
                    direction = mv_arr / mv_mag
                else:
                    direction = np.array([1.0, 0.0])
                # Red dot at parent + direction × circle_radius (exact distance, no clipping)
                red_dot_pos = sel_parent_pt + direction * circle_radius
                logger.debug(f"[DEBUG ARROW] parent={sel_parent_pt}, red_dot={red_dot_pos}, distance={circle_radius:.4f}")
            elif sel_gen_pt is not None:
                # Fallback: use the stored generated point position
                red_dot_pos = np.array(sel_gen_pt)
                logger.debug(f"[DEBUG ARROW] parent={sel_parent_pt}, red_dot={red_dot_pos}, distance={circle_radius:.4f} (fallback)")
            else:
                red_dot_pos = None
                logger.debug(f"[DEBUG ARROW] parent={sel_parent_pt}, red_dot=None, distance={circle_radius:.4f}")
            
            # Draw black arrow from parent point (tail) to red dot position (head)
            if red_dot_pos is not None:
                # Draw red dot at arrow head first (lowest z-index: 5)
                ax.scatter([red_dot_pos[0]], [red_dot_pos[1]], s=40, zorder=5, color='red')  # type: ignore
                # Draw arrow on top of red dot (z-index: 6)
                ax.annotate(
                    '',
                    xy=(red_dot_pos[0], red_dot_pos[1]),  # Arrow head
                    xytext=(sel_parent_pt[0], sel_parent_pt[1]),  # Arrow tail
                    arrowprops=dict(
                        arrowstyle='->',
                        color='black',
                        lw=0.8,
                        shrinkA=0,
                        shrinkB=0
                    ),
                    zorder=6
                )
                # Draw small white dot at arrow tail / parent point (highest z-index: 7)
                ax.scatter([sel_parent_pt[0]], [sel_parent_pt[1]], s=6, zorder=7, color='white')  # type: ignore
        
        # ============= Buffer/Rough Visualization =============
        # (Only for the primary generated point for simplicity)
        current_variant = st.session_state.get("anim_current_variant", "fundamental")
        buffer_x_val = st.session_state.get("cfg_buffer_x", 0.0)
        buffer_y_val = st.session_state.get("cfg_buffer_y", 0.0)
        rough_x_val = st.session_state.get("cfg_rough_x", 0.0)
        rough_y_val = st.session_state.get("cfg_rough_y", 0.0)
        
        # Draw buffer points if buffer variant is active
        if current_variant in ["buffer", "bufferrough"] and (buffer_x_val > 0 or buffer_y_val > 0):
            # Draw buffer variants for the current candidate point
            buffer_color = 'purple'
            buffer_alpha = 0.5
            gx, gy = gen_pt[0], gen_pt[1]
            center = np.array([gx, gy])
            
            # Compute the centerline from all_points_plot (the actual plotted points in the window)
            # This gives us the correct road direction for the current view
            try:
                # Collect all points from all objects, sorted by time
                all_window_pts = []
                for o_id in sorted(all_points_plot.keys()):
                    pts = all_points_plot[o_id]
                    ts = all_vals_plot[o_id]
                    for pt, t in zip(pts, ts):
                        all_window_pts.append((float(t), float(pt[0]), float(pt[1])))
                
                # Group by time and compute mean position (centerline)
                from collections import defaultdict
                by_time: defaultdict[float, list[tuple[float, float]]] = defaultdict(list)
                for t, x, y in all_window_pts:
                    by_time[t].append((x, y))
                
                center_samples = []
                for t in sorted(by_time.keys()):
                    pts_at_t: list[tuple[float, float]] = by_time[t]
                    mean_x = np.mean([p[0] for p in pts_at_t])
                    mean_y = np.mean([p[1] for p in pts_at_t])
                    center_samples.append((mean_x, mean_y))
                
                if len(center_samples) >= 2:
                    centerline_window = np.array(center_samples, dtype=float)
                    
                    # Compute tangent from start to end of centerline
                    p_start = centerline_window[0]
                    p_end = centerline_window[-1]
                    vec = p_end - p_start
                    norm = np.linalg.norm(vec)
                    
                    if norm > 1e-6:
                        # Tangent direction (along road)
                        tangent = vec / norm
                        # Normal direction (perpendicular, 90° counter-clockwise)
                        normal = np.array([-tangent[1], tangent[0]])
                        
                        # Buffer points along tangent (d1) and normal (d2) directions
                        buffer_pts_arr = [
                            center - buffer_x_val * tangent,  # -d1
                            center + buffer_x_val * tangent,  # +d1
                            center,                            # center (already drawn as red)
                            center - buffer_y_val * normal,   # -d2
                            center + buffer_y_val * normal,   # +d2
                        ]
                        # Draw buffer points (skip the center one at index 2)
                        for idx, pt in enumerate(buffer_pts_arr):
                            if idx != 2:  # Skip the center point
                                ax.scatter([pt[0]], [pt[1]], s=18, zorder=5, color=buffer_color, alpha=buffer_alpha, marker='x')
                        # Draw rotated buffer cross lines
                        # d1 direction line (tangent)
                        ax.plot([buffer_pts_arr[0][0], buffer_pts_arr[1][0]], [buffer_pts_arr[0][1], buffer_pts_arr[1][1]], 
                                color=buffer_color, alpha=buffer_alpha, linewidth=1, linestyle='--', zorder=4)
                        # d2 direction line (normal)
                        ax.plot([buffer_pts_arr[3][0], buffer_pts_arr[4][0]], [buffer_pts_arr[3][1], buffer_pts_arr[4][1]], 
                                color=buffer_color, alpha=buffer_alpha, linewidth=1, linestyle='--', zorder=4)
                    else:
                        raise ValueError("Centerline too short")
                else:
                    raise ValueError("Not enough centerline points")
            except Exception as e:
                # Fallback: axis-aligned buffer cross (no centerline available)
                logger.warning(f"[RENDER] Rotated buffer cross failed, using axis-aligned fallback: {e}")
                buffer_pts = [
                    (gx - buffer_x_val, gy),  # x - buffer_x
                    (gx + buffer_x_val, gy),  # x + buffer_x
                    (gx, gy),                  # original (already drawn as red)
                    (gx, gy - buffer_y_val),  # y - buffer_y
                    (gx, gy + buffer_y_val),  # y + buffer_y
                ]
                for idx, (bx, by) in enumerate(buffer_pts):
                    if idx != 2:
                        ax.scatter([bx], [by], s=18, zorder=5, color=buffer_color, alpha=buffer_alpha, marker='x')
                ax.plot([gx - buffer_x_val, gx + buffer_x_val], [gy, gy], 
                        color=buffer_color, alpha=buffer_alpha, linewidth=1, linestyle='--', zorder=4)
                ax.plot([gx, gx], [gy - buffer_y_val, gy + buffer_y_val], 
                        color=buffer_color, alpha=buffer_alpha, linewidth=1, linestyle='--', zorder=4)
        
        # Draw rough zone if rough variant is active
        if current_variant in ["rough", "bufferrough"] and (rough_x_val > 0 or rough_y_val > 0):
            # Draw a rectangle around the candidate point showing the roughness tolerance zone
            rough_color = 'green'
            rough_alpha = 0.2
            gx, gy = gen_pt[0], gen_pt[1]
            # Use minimum visible size when one dimension is 0 (thin line like axis thickness)
            min_visible_size = 0.15  # Very thin, similar to axis line thickness
            draw_rough_x = rough_x_val if rough_x_val > 0 else min_visible_size
            draw_rough_y = rough_y_val if rough_y_val > 0 else min_visible_size
            
            # Compute the centerline from all_points_plot (the actual plotted points in the window)
            # This gives us the correct road direction for the current view
            try:
                # Collect all points from all objects, sorted by time
                all_window_pts = []
                for o_id in sorted(all_points_plot.keys()):
                    pts = all_points_plot[o_id]
                    ts = all_vals_plot[o_id]
                    for pt, t in zip(pts, ts):
                        all_window_pts.append((float(t), float(pt[0]), float(pt[1])))
                
                # Group by time and compute mean position (centerline)
                from collections import defaultdict
                by_time: defaultdict[float, list[tuple[float, float]]] = defaultdict(list)
                for t, x, y in all_window_pts:
                    by_time[t].append((x, y))
                
                center_samples = []
                for t in sorted(by_time.keys()):
                    pts_at_t: list[tuple[float, float]] = by_time[t]
                    mean_x = np.mean([p[0] for p in pts_at_t])
                    mean_y = np.mean([p[1] for p in pts_at_t])
                    center_samples.append((mean_x, mean_y))
                
                if len(center_samples) >= 2:
                    centerline_window = np.array(center_samples, dtype=float)
                    
                    # Compute tangent from start to end of centerline
                    p_start = centerline_window[0]
                    p_end = centerline_window[-1]
                    vec = p_end - p_start
                    norm = np.linalg.norm(vec)
                    
                    if norm > 1e-6:
                        # Tangent direction (along road)
                        tangent = vec / norm
                        # Normal direction (perpendicular, 90° counter-clockwise)
                        normal = np.array([-tangent[1], tangent[0]])
                        
                        # Calculate rotated rectangle corners:
                        # Center is at (gx, gy), extend ±rough_x along tangent and ±rough_y along normal
                        center = np.array([gx, gy])
                        corners = [
                            center - draw_rough_x * tangent - draw_rough_y * normal,  # bottom-left
                            center + draw_rough_x * tangent - draw_rough_y * normal,  # bottom-right
                            center + draw_rough_x * tangent + draw_rough_y * normal,  # top-right
                            center - draw_rough_x * tangent + draw_rough_y * normal,  # top-left
                        ]
                        # Draw rotated polygon
                        poly = matplotlib.patches.Polygon(
                            corners,
                            closed=True,
                            edgecolor=rough_color,
                            facecolor=rough_color,
                            alpha=rough_alpha,
                            linewidth=1.5,
                            linestyle=':',
                            zorder=3
                        )
                        ax.add_patch(poly)  # type: ignore
                    else:
                        raise ValueError("Centerline too short")
                else:
                    raise ValueError("Not enough centerline points")
            except Exception as e:
                # Fallback to axis-aligned rectangle if centerline computation fails
                logger.warning(f"[RENDER] Rotated rough zone failed, using axis-aligned fallback: {e}")
                rect = matplotlib.patches.Rectangle(
                    (gx - draw_rough_x, gy - draw_rough_y),
                    2 * draw_rough_x, 2 * draw_rough_y,
                    edgecolor=rough_color, facecolor=rough_color,
                    alpha=rough_alpha, linewidth=1.5, linestyle=':', zorder=3
                )
                ax.add_patch(rect)  # type: ignore

# ============= Helper: choose which config to display on demand ============
def _set_display_config(config_num: int) -> None:
    """
    Set the right-hand plot to show a specific generated configuration.

    This uses st.session_state['anim_all_configs'], finds the matching config_num,
    and updates anim_* state so that the existing drawing + LaTeX code show
    that configuration (without any extra UI changes).
    """
    all_configs: list[dict[str, Any]] = st.session_state.get("anim_all_configs", [])
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
            st.session_state["anim_all_pts"] = all_pts_flat.copy()

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


# ============= Visualization Modes =============
tab_static, tab_animation, tab_slicing = st.tabs(["📊 Static", "▶️ Animation", "🔍 Slicing"])

with tab_static:
    # ============= Layout (two columns) ============
    col1, col2 = st.columns(2, gap="small")

    # Pre-render both figures to ensure identical sizing
    fig_left = render_square_matplotlib_figure(draw_original, XLIM, YLIM)
    fig_right = render_square_matplotlib_figure(draw_generated_empty, XLIM, YLIM)

    # Convert to PNG bytes for consistent display
    _buf_left: IO[bytes] = io.BytesIO()
    fig_left.savefig(_buf_left, format="png", dpi=160)  # type: ignore[call-arg]
    _buf_left.seek(0)

    _buf_right: IO[bytes] = io.BytesIO()
    fig_right.savefig(_buf_right, format="png", dpi=160)  # type: ignore[call-arg]
    _buf_right.seek(0)

    with col1:
        st.markdown("<div class='figure-title'>Original configuration</div>", unsafe_allow_html=True)
        st.latex(make_d1_order_latex())
        st.latex(make_d2_order_latex())
        # Use st.image instead of st.pyplot for consistent sizing
        st.image(_buf_left, width="stretch")

        # Show textual comparison of orderings when animation / generation is active
        if "anim_generated_point" in st.session_state:
            left_d1 = make_d1_order_latex()
            right_d1 = make_d1_order_latex_generated()
            left_order = extract_order_string(left_d1)
            right_order = extract_order_string(right_d1)
            same_d1 = left_order == right_order
            st.caption(f"Left: {left_order}")
            st.caption(f"Right: {right_order}")
            st.markdown(f"**d1 order match: {same_d1}**")

            left_d2 = make_d2_order_latex()
            right_d2 = make_d2_order_latex_generated()
            left_order_d2 = extract_order_string(left_d2)
            right_order_d2 = extract_order_string(right_d2)
            same_d2 = left_order_d2 == right_order_d2
            st.caption(f"Left d2: {left_order_d2}")
            st.caption(f"Right d2: {right_order_d2}")
            st.markdown(f"**d2 order match: {same_d2}**")

        # Download original plot as PNG - reuse the buffer
        _buf_left.seek(0)
        st.download_button(
            label="Save as PNG",
            data=_buf_left.getvalue(),
            file_name="original.png",
            mime="image/png",
            key="dl_left_png",
        )

        # --- 6-Event table for the original configuration ---
        if _6ev_vis_map:
            _ev_rows: list[dict[str, object]] = []
            _ev_labels = {1: "E1 start", 2: "E2 lane-change", 3: "E3 arrived",
                          4: "E4 return", 5: "E5 arrived", 6: "E6 end"}
            _sorted_oids = sorted(all_objects_points.keys())
            for _ev_t, _ev_num in sorted(_6ev_vis_map.items(), key=lambda x: x[1]):
                row: dict[str, object] = {"Event": _ev_labels.get(_ev_num, f"E{_ev_num}"), "t": _ev_t}
                for _i_obj, _oid in enumerate(_sorted_oids):
                    _lbl = OBJECT_LABELS[_i_obj % len(OBJECT_LABELS)]
                    _pts_o, _ts_o = all_objects_points[_oid]
                    _tdiffs = np.abs(_ts_o - _ev_t)
                    _ci = int(np.argmin(_tdiffs))
                    if _tdiffs[_ci] < 1e-3:
                        row[f"{_lbl}_x"] = round(float(_pts_o[_ci, 0]), 2)
                        row[f"{_lbl}_y"] = round(float(_pts_o[_ci, 1]), 2)
                    else:
                        row[f"{_lbl}_x"] = "—"
                        row[f"{_lbl}_y"] = "—"
                _ev_rows.append(row)
            st.markdown("**6-Event coordinates (original)**")
            st.dataframe(pd.DataFrame(_ev_rows), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("<div class='figure-title'>Generated configuration</div>", unsafe_allow_html=True)
        _latex_d1 = make_d1_order_latex_generated()
        _latex_d2 = make_d2_order_latex_generated()
        st.latex(_latex_d1)
        st.latex(_latex_d2)
        # Use st.image instead of st.pyplot for consistent sizing
        st.image(_buf_right, width="stretch")

        if "anim_generated_point" in st.session_state:
            left_d1 = make_d1_order_latex()
            right_d1 = make_d1_order_latex_generated()
            left_order = extract_order_string(left_d1)
            right_order = extract_order_string(right_d1)
            same_d1 = left_order == right_order
            st.caption(f"Left: {left_order}")
            st.caption(f"Right: {right_order}")
            st.markdown(f"**d1 order match: {same_d1}**")

            left_d2 = make_d2_order_latex()
            right_d2 = make_d2_order_latex_generated()
            left_order_d2 = extract_order_string(left_d2)
            right_order_d2 = extract_order_string(right_d2)
            same_d2 = left_order_d2 == right_order_d2
            st.caption(f"Left d2: {left_order_d2}")
            st.caption(f"Right d2: {right_order_d2}")
            st.markdown(f"**d2 order match: {same_d2}**")

        # Download generated plot as PNG + navigation buttons on ONE row
        # Reuse the buffer that was already created above
        _buf_right.seek(0)

        # Determine navigation state
        all_configs_list: list[dict[str, Any]] = st.session_state.get("anim_all_configs", [])
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

        # --- 6-Event table for the generated configuration ---
        if _6ev_vis_map:
            _gen_sp: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
            # Build generated points array (latest gen point per original index)
            _gen_latest: dict[int, np.ndarray] = {}
            for _sp in _gen_sp:
                _orig_idx = int(_sp["original_parent_idx"])
                _gen_latest[_orig_idx] = np.array(_sp["point"])
            _gen_pts_arr = all_pts_flat.copy() if n_total_points > 0 else np.zeros((0, 2))
            for _fi in range(_gen_pts_arr.shape[0]):
                if _fi in _gen_latest:
                    _gen_pts_arr[_fi] = _gen_latest[_fi]

            _ev_rows_g: list[dict[str, object]] = []
            _ev_labels_g = {1: "E1 start", 2: "E2 lane-change", 3: "E3 arrived",
                            4: "E4 return", 5: "E5 arrived", 6: "E6 end"}
            _sorted_oids_g = sorted(all_points_plot.keys())
            # Map flat indices back: for each object, its points occupy a
            # contiguous block in all_pts_flat in the order of sorted object ids.
            _obj_offsets_g: dict[int, int] = {}
            _off = 0
            for _oid_g in _sorted_oids_g:
                _obj_offsets_g[_oid_g] = _off
                _off += all_points_plot[_oid_g].shape[0]

            for _ev_t, _ev_num in sorted(_6ev_vis_map.items(), key=lambda x: x[1]):
                row_g: dict[str, object] = {"Event": _ev_labels_g.get(_ev_num, f"E{_ev_num}"), "t": _ev_t}
                for _i_obj, _oid_g in enumerate(_sorted_oids_g):
                    _lbl = OBJECT_LABELS[_i_obj % len(OBJECT_LABELS)]
                    _vals_g = all_vals_plot[_oid_g]
                    _tdiffs_g = np.abs(_vals_g - _ev_t)
                    _ci_g = int(np.argmin(_tdiffs_g))
                    if _tdiffs_g[_ci_g] < 1e-3:
                        _flat_idx = _obj_offsets_g[_oid_g] + _ci_g
                        if _flat_idx < _gen_pts_arr.shape[0] and _flat_idx in _gen_latest:
                            row_g[f"{_lbl}_x"] = round(float(_gen_pts_arr[_flat_idx, 0]), 2)
                            row_g[f"{_lbl}_y"] = round(float(_gen_pts_arr[_flat_idx, 1]), 2)
                        else:
                            row_g[f"{_lbl}_x"] = "—"
                            row_g[f"{_lbl}_y"] = "—"
                    else:
                        row_g[f"{_lbl}_x"] = "—"
                        row_g[f"{_lbl}_y"] = "—"
                _ev_rows_g.append(row_g)
            if _gen_sp:
                st.markdown("**6-Event coordinates (generated)**")
                st.dataframe(pd.DataFrame(_ev_rows_g), hide_index=True, use_container_width=True)

    # ============= Heat Maps for PDP Inequality Matrices ============
    st.markdown("---")
    st.markdown("### PDP Inequality Matrix Heat Maps")

    # Determine when to update heat maps based on animation settings:
    # a) Auto-advance with wait_interval >= 1000ms: update every step
    # b) Auto-advance with wait_interval < 1000ms: update only at end of iteration
    # c) Manual modes: update only at end of last step/iteration/config

    anim_mode = st.session_state.get("cfg_anim_mode", "Auto-advance")
    wait_interval = int(st.session_state.get("cfg_wait_interval", 2000))
    anim_running = st.session_state.get("anim_running", False)
    search_steps = int(st.session_state.get("anim_search_steps", 0))

    # Determine if we should update heat maps now
    should_update_heatmaps = False

    if not anim_running:
        # Not running animation - always update (final state)
        should_update_heatmaps = True
    elif anim_mode == "Auto-advance":
        if wait_interval >= 1000:
            # a) Auto-advance with slow interval: update every step
            should_update_heatmaps = True
        else:
            # b) Auto-advance with fast interval: only at end of iteration (step 0 after reset, or step 7)
            # We detect end of iteration when search_steps is 0 (just completed) or animation just finished
            should_update_heatmaps = (search_steps == 0)
    else:
        # c) Manual modes: only update at end of step/iteration/config
        # In manual mode, we update after each manual advancement (when not in middle of search)
        should_update_heatmaps = (search_steps == 0)

    pdp_detailed = None

    if n_total_points > 0 and should_update_heatmaps:
        # Get PDP variant parameters from session_state
        pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        pdp_variant = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        rough_x = st.session_state.get("cfg_rough_x", 0.0)
        rough_y = st.session_state.get("cfg_rough_y", 0.0)
        match_threshold = get_match_threshold()

        # Build generated configuration INCLUDING current candidate point
        generated_points: np.ndarray = all_pts_flat.copy()
        successful_points_hm: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])

        # Track the latest generated point for each original index
        latest_generated: dict[int, np.ndarray] = {}
        for sp in successful_points_hm:
            orig_idx = int(sp["original_parent_idx"])
            latest_generated[orig_idx] = sp["point"]

        # Include current candidate point being tested (for live heat map update)
        anim_generated_points = st.session_state.get("anim_generated_points", {})
        gen_pt = st.session_state.get("anim_generated_point", None)

        if anim_generated_points:
            for idx, pt in anim_generated_points.items():
                latest_generated[int(idx)] = np.array(pt)
        elif gen_pt is not None:
            parent_idx = int(st.session_state.get("anim_parent_idx", 0))
            if parent_idx < n_total_points:
                current_original_parent_idx = parent_idx
            else:
                sidx = parent_idx - n_total_points
                if 0 <= sidx < len(successful_points_hm):
                    current_original_parent_idx = int(successful_points_hm[sidx]["original_parent_idx"])
                else:
                    current_original_parent_idx = 0
            latest_generated[current_original_parent_idx] = np.array(gen_pt)

        # Apply all generated points to the configuration
        for flat_idx in range(n_total_points):
            if flat_idx in latest_generated:
                generated_points[flat_idx] = latest_generated[flat_idx]

        # Get threshold settings to pass to detailed check
        mode, pct_threshold, max_mismatch_val = get_threshold_settings()
        max_mismatches_param = max_mismatch_val if mode == "Max mismatches" else None

        # Compute detailed results with the current configuration
        pdp_detailed = check_pdp_match_detailed(
            all_pts_flat,
            generated_points,
            pdp_variant=pdp_variant,
            buffer_x=buffer_x,
            buffer_y=buffer_y,
            rough_x=rough_x,
            rough_y=rough_y,
            match_threshold=match_threshold,
            max_mismatches=max_mismatches_param
        )
        # Store for later use
        st.session_state["pdp_detailed_results"] = pdp_detailed
    elif n_total_points > 0:
        # Use cached results if not updating
        pdp_detailed: dict[str, Any] | None = st.session_state.get("pdp_detailed_results", None)

    if pdp_detailed is not None:
        # Get match percentages
        d1_pct = pdp_detailed.get("d1_percentage", 0.0) * 100
        d2_pct = pdp_detailed.get("d2_percentage", 0.0) * 100
        avg_pct = pdp_detailed.get("avg_percentage", (d1_pct/100 + d2_pct/100) / 2.0) * 100
        d1_match = pdp_detailed.get("d1_match", False)
        d2_match = pdp_detailed.get("d2_match", False)

        # Get threshold settings for display
        mode, pct_threshold, max_mismatches = get_threshold_settings()
        d1_mismatches = pdp_detailed.get("d1_mismatches", 0)
        d2_mismatches = pdp_detailed.get("d2_mismatches", 0)
        total_cells = pdp_detailed.get("total_cells", 0)

        # Re-evaluate match based on new threshold settings
        d1_match, d2_match = check_threshold_match(d1_pct/100, d2_pct/100, total_cells, d1_mismatches, d2_mismatches)
        avg_match = d1_match and d2_match

        # Display match percentages with threshold info
        if mode == "Percentage":
            threshold_display = f"{int(pct_threshold * 100)}%"
            if pct_threshold < 1.0:
                # Show average for relaxed thresholds
                st.markdown(f"**Threshold:** {threshold_display} | **d1:** {d1_pct:.1f}% | **d2:** {d2_pct:.1f}% | **Avg:** {avg_pct:.1f}% {'YES' if avg_match else 'NO'}")
            else:
                # Show individual matches for strict threshold
                st.markdown(f"**Threshold:** {threshold_display} | **d1 Match:** {d1_pct:.1f}% {'YES' if d1_match else 'NO'} | **d2 Match:** {d2_pct:.1f}% {'YES' if d2_match else 'NO'}")
        else:
            # Max mismatches mode
            threshold_display = f"≤{max_mismatches} mismatches"
            st.markdown(f"**Threshold:** {threshold_display} | **d1:** {d1_mismatches} mismatches {'YES' if d1_match else 'NO'} | **d2:** {d2_mismatches} mismatches {'YES' if d2_match else 'NO'}")

        # Create 4 heat map columns: orig_d1, orig_d2 | gen_d1, gen_d2
        hm_col1, hm_col2, hm_col3, hm_col4 = st.columns(4, gap="small")

        # Color map: 0=green (greater precedence), 1=yellow (equal), 2=red (less precedence)
        from matplotlib.colors import ListedColormap
        hm_cmap = ListedColormap(['#00AA00', '#FFFF00', '#FF0000'])  # green, yellow, red

        # Labels should be: k0, l0, k1, l1, k2, l2 (sorted by timestamp first, then by object)
        # But data in all_pts_flat is: k0, k1, k2, l0, l1, l2 (sorted by object first, then by timestamp)
        # We need to reorder the matrix to match the desired label order

        def get_reorder_indices(n: int) -> list[int]:
            """Get indices to reorder from (k0,k1,k2,l0,l1,l2) to (k0,l0,k1,l1,k2,l2)."""
            if n != 6:
                return list(range(n))  # No reordering if not exactly 6 points
            # Original order: k0(0), k1(1), k2(2), l0(3), l1(4), l2(5)
            # Desired order:  k0(0), l0(3), k1(1), l1(4), k2(2), l2(5)
            return [0, 3, 1, 4, 2, 5]

        def reorder_matrix(matrix: np.ndarray) -> np.ndarray:
            """Reorder matrix rows and columns to match desired label order."""
            n = matrix.shape[0]
            if n != 6:
                return matrix
            idx = get_reorder_indices(n)
            # Reorder both rows and columns
            return matrix[np.ix_(idx, idx)]

        def get_point_labels(n: int) -> list[str]:
            """Generate labels: k0, l0, k1, l1, k2, l2 (sorted by timestamp, then object)."""
            if n > 6:
                return []  # Don't show labels if more than 6 points
            labels = []
            for t in range(3):  # 3 timestamps
                for obj in ["k", "l"]:  # 2 objects per timestamp
                    if len(labels) < n:
                        labels.append(f"{obj}{t}")
            return labels

        def create_heatmap_figure(matrix: np.ndarray, title: str, 
                                   comparison_matrix: np.ndarray = None,
                                   highlight_differences: bool = False) -> Figure:
            """Create a heat map figure for an inequality matrix.

            Args:
                matrix: The matrix to display
                title: Title for the heatmap
                comparison_matrix: Original matrix to compare against (for highlighting differences)
                highlight_differences: If True and comparison_matrix provided, highlight differing cells
            """
            fig_hm, ax_hm = plt.subplots(figsize=(3, 3))
            n = matrix.shape[0]

            # Reorder matrix to match label order (k0, l0, k1, l1, k2, l2)
            display_matrix = reorder_matrix(matrix)

            # Create heat map with discrete colors (0, 1, 2 -> green, yellow, red)
            ax_hm.imshow(display_matrix, cmap=hm_cmap, vmin=0, vmax=2, aspect='equal')

            # Highlight differences if requested - subtle style with transparent fill and thin black border
            if highlight_differences and comparison_matrix is not None:
                comparison_display = reorder_matrix(comparison_matrix)
                # Color map for semi-transparent overlays (same colors but with alpha)
                diff_colors = {
                    0: (0.0, 0.67, 0.0, 0.3),   # green with alpha
                    1: (1.0, 1.0, 0.0, 0.3),     # yellow with alpha
                    2: (1.0, 0.0, 0.0, 0.3),     # red with alpha
                }
                # Find cells where values differ
                for i in range(n):
                    for j in range(n):
                        if display_matrix[i, j] != comparison_display[i, j]:
                            # Get the cell's color with transparency
                            cell_val = int(display_matrix[i, j])
                            fill_color = diff_colors.get(cell_val, (0.5, 0.5, 0.5, 0.3))
                            # Draw a rectangle with transparent fill and thin black border
                            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                                fill=True, facecolor=fill_color,
                                                edgecolor='black', linewidth=1.5)
                            ax_hm.add_patch(rect)

            # Add axis labels only if 6 or fewer points
            point_labels = get_point_labels(n)
            if point_labels:
                ax_hm.set_xticks(range(n))
                ax_hm.set_yticks(range(n))
                ax_hm.set_xticklabels(point_labels, fontsize=7)
                ax_hm.set_yticklabels(point_labels, fontsize=7)
            else:
                ax_hm.set_xticks([])
                ax_hm.set_yticks([])

            ax_hm.set_title(title, fontsize=9, fontweight='bold')
            # No axis titles (removed 'Point j' and 'Point i')

            fig_hm.tight_layout()
            return fig_hm

        # Create heat map for each matrix
        orig_d1_matrix = pdp_detailed.get("original_d1_matrix")  # type: ignore[assignment]
        orig_d2_matrix = pdp_detailed.get("original_d2_matrix")  # type: ignore[assignment]
        gen_d1_matrix = pdp_detailed.get("generated_d1_matrix")  # type: ignore[assignment]
        gen_d2_matrix = pdp_detailed.get("generated_d2_matrix")  # type: ignore[assignment]

        # Determine if we should highlight differences (when threshold < 100% or max_mismatches > 0)
        if mode == "Percentage":
            highlight_diffs = pct_threshold < 1.0
        else:
            # In absolute mode, always highlight differences since we're explicitly allowing mismatches
            highlight_diffs = max_mismatches > 0

        with hm_col1:
            st.markdown("**Original d1**")
            if orig_d1_matrix is not None:
                fig_hm1 = create_heatmap_figure(orig_d1_matrix, "Original d1 (x)")
                st.pyplot(fig_hm1)
                plt.close(fig_hm1)

        with hm_col2:
            st.markdown("**Original d2**")
            if orig_d2_matrix is not None:
                fig_hm2 = create_heatmap_figure(orig_d2_matrix, "Original d2 (y)")
                st.pyplot(fig_hm2)
                plt.close(fig_hm2)

        with hm_col3:
            st.markdown("**Generated d1**")
            if gen_d1_matrix is not None:
                fig_hm3 = create_heatmap_figure(gen_d1_matrix, "Generated d1 (x)",
                                               comparison_matrix=orig_d1_matrix,
                                               highlight_differences=highlight_diffs)
                st.pyplot(fig_hm3)
                plt.close(fig_hm3)

        with hm_col4:
            st.markdown("**Generated d2**")
            if gen_d2_matrix is not None:
                fig_hm4 = create_heatmap_figure(gen_d2_matrix, "Generated d2 (y)",
                                               comparison_matrix=orig_d2_matrix,
                                               highlight_differences=highlight_diffs)
                st.pyplot(fig_hm4)
                plt.close(fig_hm4)

        # Legend
        if highlight_diffs:
            st.caption("Legend: Green (0) = j > i | Yellow (1) = j ~ i (equal) | Red (2) = j < i | * Border = differs from original")
        else:
            st.caption("Legend: Green (0) = j > i | Yellow (1) = j ~ i (equal) | Red (2) = j < i")

    else:
        st.info("Heat maps will appear after generating a configuration. Use the animation controls above to generate a configuration.")


with tab_animation:
    # ============= Animation progress (both strategies) ============
    # In manual mode, only process animation when user clicked the appropriate "Next" button
    # In auto mode, always process
    _manual_mode = st.session_state.get("anim_manual_mode", False)
    _manual_step_mode = st.session_state.get("anim_manual_step_mode", False)
    _manual_iteration_mode = st.session_state.get("anim_manual_iteration_mode", False)
    _manual_config_mode = st.session_state.get("anim_manual_config_mode", False)

    # Determine if we should process animation based on mode
    _manual_step_requested = st.session_state.get("anim_manual_step_requested", False)
    _manual_iteration_requested = st.session_state.get("anim_manual_iteration_requested", False)
    _manual_config_requested = st.session_state.get("anim_manual_config_requested", False)

    # For iteration/config modes, we also continue if we're in the middle of completing one
    _iteration_in_progress = st.session_state.get("_iteration_in_progress", False)
    _config_in_progress = st.session_state.get("_config_in_progress", False)

    # Start tracking progress when a request is made
    if _manual_iteration_requested:
        st.session_state["_iteration_in_progress"] = True
        _iteration_in_progress = True
    if _manual_config_requested:
        st.session_state["_config_in_progress"] = True
        _config_in_progress = True

    # INSTANT ITERATION COMPLETION for manual iteration/config modes
    # When user clicks "Next iteration" or "Next config", complete the iteration instantly without animations
    if _manual_iteration_requested or _manual_config_requested:
        # Set flag to skip all wait intervals and complete iteration in single pass
        st.session_state["_skip_wait_intervals"] = True
    else:
        st.session_state["_skip_wait_intervals"] = False

    _should_process_animation = (
        st.session_state.get("anim_running", False) and 
        (not _manual_mode or _manual_step_requested or _manual_iteration_requested or _iteration_in_progress or _manual_config_requested or _config_in_progress)
    )

    if _should_process_animation:
        # Save current state to history before advancing (for "Previous step" functionality)
        # This allows users to go back to previous animation states in manual mode
        if _manual_mode:
            # List of all animation state keys that need to be saved/restored
            # This includes all variables that define the current animation state:
            # - Point positions and parent relationships
            # - Search parameters (distance, angle, steps)
            # - Configuration progress tracking
            # - Circle visualization state
            # - Multi-point animation support variables
            anim_state_keys = [
                "anim_generated_point",
                "anim_parent_idx",
                "anim_successful_points",
                "anim_distance",
                "anim_angle",
                "anim_search_steps",
                "anim_completed_iterations",
                "anim_current_config",
                "anim_in_search",
                "anim_binary_mode",
                "anim_binary_step",
                "anim_ok_point",
                "anim_delta",
                "anim_had_full_match",
                # Linear search state
                "anim_linear_mode",
                "anim_linear_step",
                "anim_linear_current_distance",
                "anim_linear_maxdist",
                "anim_linear_step_size",
                "anim_all_pts",
                "anim_all_ts",
                "diag_rows",
                "binary_iteration_summary",
                # Circle visualization state - needed to restore the search circle position
                "anim_circle_idx",
                "show_anim_circle",
                # Multi-point animation support - needed to restore all selected points and their positions
                "anim_selected_indices",
                "anim_generated_points",
                "anim_movement_vectors",
                # Multi-variant support
                "anim_pdp_variants_list",
                "anim_current_variant_idx",
                "anim_current_variant",
            ]
            # Create a snapshot of current state
            current_state_snapshot = {}
            import copy
            for key in anim_state_keys:
                if key in st.session_state:
                    value = st.session_state[key]
                    # Deep copy numpy arrays, lists, and dicts to avoid reference issues
                    if isinstance(value, np.ndarray):
                        current_state_snapshot[key] = value.copy()
                    elif isinstance(value, (list, dict)):
                        # Deep copy lists (e.g., SuccessfulPoints) and dicts (e.g., generated_points, movement_vectors)
                        current_state_snapshot[key] = copy.deepcopy(value)  # type: ignore[arg-type]
                    else:
                        current_state_snapshot[key] = value

            # Append to history (initialize if not exists)
            if "anim_state_history" not in st.session_state:
                st.session_state["anim_state_history"] = []
            st.session_state["anim_state_history"].append(current_state_snapshot)

            # Limit history size to prevent memory issues (keep last 100 states)
            if len(st.session_state["anim_state_history"]) > 100:
                st.session_state["anim_state_history"] = st.session_state["anim_state_history"][-100:]

        # Clear the manual step flag early so we don't re-process on the next rerun
        if _manual_mode:
            st.session_state["anim_manual_step_requested"] = False

        # Build current generated configuration for PDP comparison
        gen_pt = st.session_state.get("anim_generated_point", None)
        successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])

        # Get parent info for the current candidate point
        parent_idx = int(st.session_state.get("anim_parent_idx", 0))

        # Determine the original parent index for the current candidate
        if parent_idx < n_total_points:
            current_original_parent_idx = parent_idx
        else:
            # Parent is a previously generated point - find its original parent
            sidx = parent_idx - n_total_points
            if 0 <= sidx < len(successful_points):
                current_original_parent_idx = int(successful_points[sidx]["original_parent_idx"])
            else:
                current_original_parent_idx = 0

        # Build generated points array (start with original, then override with generated)
        latest_generated: dict[int, np.ndarray] = {}
        for sp in successful_points:
            orig_idx = int(sp["original_parent_idx"])
            latest_generated[orig_idx] = sp["point"]

        # CRITICAL: Add ALL current candidate points we're testing (multi-point support)!
        # Use anim_generated_points dict which contains all n selected points
        anim_generated_points = st.session_state.get("anim_generated_points", {})
        if anim_generated_points:
            for idx, pt in anim_generated_points.items():
                latest_generated[int(idx)] = np.array(pt)
        elif gen_pt is not None:
            # Fallback for single point (backwards compatibility)
            latest_generated[current_original_parent_idx] = np.array(gen_pt)

        # Construct generated_points array (same order as all_pts_flat)
        generated_points_arr = all_pts_flat.copy()
        for flat_idx in range(n_total_points):
            if flat_idx in latest_generated:
                generated_points_arr[flat_idx] = latest_generated[flat_idx]

        # Use PDP inequality matrix comparison (legacy order strings kept for display)
        left_d1 = make_d1_order_latex()
        left_d2 = make_d2_order_latex()
        right_d1 = make_d1_order_latex_generated()
        right_d2 = make_d2_order_latex_generated()

        # Get PDP variant parameters from session_state
        # Use current variant in animation, or first selected variant otherwise
        pdp_variant = st.session_state.get("anim_current_variant")
        if not pdp_variant:
            pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
            pdp_variant = pdp_variants_list[0] if pdp_variants_list else "fundamental"
        buffer_x = st.session_state.get("cfg_buffer_x", DEFAULT_BUFFER_X)
        buffer_y = st.session_state.get("cfg_buffer_y", DEFAULT_BUFFER_Y)
        rough_x = st.session_state.get("cfg_rough_x", 0.0)
        rough_y = st.session_state.get("cfg_rough_y", 0.0)

        # Use PDP inequality matrix comparison with selected variant
        # Get both threshold parameters
        _thresh, _max_mm = get_threshold_params()
        same_d1, same_d2 = check_pdp_match(
            all_pts_flat,
            generated_points_arr,
            pdp_variant=pdp_variant,
            buffer_x=buffer_x,
            buffer_y=buffer_y,
            rough_x=rough_x,
            rough_y=rough_y,
            match_threshold=_thresh,
            max_mismatches=_max_mm
        )

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
        linear_mode = bool(st.session_state.get("anim_linear_mode", False))
        binary_step = int(st.session_state.get("anim_binary_step", 0))

        current_strategy = st.session_state.get("cfg_strategy", strategy)

        # Wachttijd in seconden voor alle animatie-sleeps
        wait_ms = int(st.session_state.get("cfg_wait_ms", 2000))
        wait_s = wait_ms / 1000.0

        if st.session_state.get("anim_config_complete_wait", False):
            st.session_state["anim_config_complete_wait"] = False
            if st.session_state.get("anim_manual_mode", False):
                # Manual mode: button click already triggered, just rerun
                st.rerun()
            else:
                # Non-blocking wait: sleep in short increments so Streamlit stays responsive
                _elapsed = 0.0
                while _elapsed < wait_s:
                    _step = min(0.1, wait_s - _elapsed)
                    time.sleep(_step)
                    _elapsed += _step
                st.rerun()

        # === Case 1: success (orders match) or distance collapsed to 0 ===
        # For BINARY mode: ONLY complete after 7 steps (distance will be set to 0 after step 7)
        # For EXPONENTIAL mode: complete when orders match or distance <= 0
        binary_complete = binary_mode and distance <= 0.0 and gen_pt is not None
        exponential_complete = not binary_mode and ((same_d1 and same_d2 and gen_pt is not None) or (distance <= 0.0 and gen_pt is not None))

        if binary_complete or exponential_complete:
            # Multi-point support: add ALL n selected points as successful
            anim_generated_points = st.session_state.get("anim_generated_points", {})
            selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])

            # For each selected point, add to successful_points (with damping applied)
            for idx in selected_indices:
                # Get parent point and original parent index
                if idx < n_total_points:
                    parent_point_val = all_pts[idx]
                    original_parent_idx_val = idx
                else:
                    succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                    sidx = int(idx - n_total_points)
                    if 0 <= sidx < len(succ_list):
                        parent_point_val = succ_list[sidx]["point"]
                        original_parent_idx_val = succ_list[sidx]["original_parent_idx"]
                    else:
                        parent_point_val = np.array([0.0, 0.0])
                        original_parent_idx_val = 0

                # Get the final generated point for this index
                final_pt = anim_generated_points.get(idx, gen_pt if idx == parent_idx else np.array([0.0, 0.0]))
                # Apply random damping factor to reduce distance from parent
                damped_pt = apply_damping_factor(parent_point_val, np.array(final_pt, dtype=float))

                sp: SuccessfulPoint = {
                    "point": damped_pt,
                    "parent_idx": idx,
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

            # Flag that an iteration was just completed (for manual iteration mode)
            st.session_state["_iteration_just_completed"] = True

            # <<< hier opnieuw: match evalueren na plaatsing >>>
            update_order_match_flags()

            if completed_iterations + 1 >= max_iterations:
                current_config = int(st.session_state.get("anim_current_config", 1))
                num_configs = int(st.session_state.get("anim_num_configs", default_num_configs))

                all_configs: list[dict[str, Any]] = st.session_state.get("anim_all_configs", [])
                all_configs.append({
                    "config_num": current_config,
                    "points": list(successful_points)
                })
                st.session_state["anim_all_configs"] = all_configs

                for sp in successful_points:
                    sp["config_num"] = current_config  # type: ignore

                # Flag that a configuration was just completed (for manual config mode)
                st.session_state["_config_just_completed"] = True

                if current_config < num_configs:
                    st.session_state["anim_current_config"] = current_config + 1
                    st.session_state["anim_completed_iterations"] = 0
                    st.session_state["anim_search_steps"] = 0
                    st.session_state["anim_running"] = True
                    st.session_state["show_anim_circle"] = True

                    all_pts_reset = all_pts_flat.copy()

                    # Multi-point selection for new configuration
                    selected_indices = select_points_for_iteration()
                    if not selected_indices:
                        movable_indices = get_movable_indices()
                        selected_indices = [int(np.random.choice(movable_indices))] if movable_indices else [0]

                    # Generate movement vectors for all selected points
                    movement_vectors = generate_movement_vectors(selected_indices, maxdist)

                    # Check all points within bounds, retry if needed
                    max_direction_attempts = 10
                    generated_points: dict[int, np.ndarray] = {}
                    for _ in range(max_direction_attempts):
                        all_within_bounds = True
                        generated_points: dict[int, np.ndarray] = {}

                        for idx in selected_indices:
                            dx, dy = movement_vectors.get(idx, (0.0, 0.0))
                            # Get parent position (could be from successful_points)
                            parent_pt = None
                            for s in reversed(successful_points):
                                if int(s.get("original_parent_idx", -1)) == idx:
                                    parent_pt = s["point"]
                                    break
                            if parent_pt is None:
                                parent_pt = all_pts_reset[idx] if idx < len(all_pts_reset) else np.array([0.0, 0.0])

                            new_x = parent_pt[0] + dx
                            new_y = parent_pt[1] + dy

                            if not (COORD_MIN_X <= new_x <= COORD_MAX_X and COORD_MIN_Y <= new_y <= COORD_MAX_Y):
                                all_within_bounds = False

                            # NO CLIPPING - all points must be at exact same distance from parent
                            generated_points[idx] = np.array([new_x, new_y])

                        if all_within_bounds:
                            break
                        # Regenerate movement vectors
                        movement_vectors = generate_movement_vectors(selected_indices, maxdist)

                    # For backwards compatibility, use first point as primary
                    parent_idx_reset = selected_indices[0]
                    parent_pt_reset = all_pts_reset[parent_idx_reset] if parent_idx_reset < len(all_pts_reset) else np.array([0.0, 0.0])
                    new_gen_pt = generated_points.get(parent_idx_reset, parent_pt_reset.copy())
                    angle_local = np.arctan2(new_gen_pt[1] - parent_pt_reset[1], new_gen_pt[0] - parent_pt_reset[0])
                    direction = np.array([np.cos(angle_local), np.sin(angle_local)])

                    st.session_state["anim_parent_idx"] = parent_idx_reset
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_generated_point"] = new_gen_pt
                    st.session_state["anim_distance"] = maxdist
                    st.session_state["anim_all_pts"] = all_pts_reset
                    st.session_state["anim_in_search"] = True
                    st.session_state["anim_config_complete_wait"] = True
                    # CORRECTED Binary search state - PRESERVE binary mode!
                    st.session_state["anim_binary_mode"] = binary_mode  # Keep the same mode
                    st.session_state["anim_binary_step"] = 0  # Init step (will be incremented to 1 on first progress)
                    st.session_state["anim_binary_correct_order"] = parent_pt_reset.copy()  # correct_order = parent (first point)
                    st.session_state["anim_binary_correct_orders"] = {int(idx): all_pts_reset[idx].copy() if idx < len(all_pts_reset) else np.array([0.0, 0.0]) for idx in selected_indices}
                    st.session_state["anim_binary_current_distance"] = maxdist  # Start at maxdist
                    st.session_state["anim_binary_direction"] = direction.copy()  # Direction unit vector (first point)
                    st.session_state["anim_had_full_match"] = False
                    # Sync multi-point data
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    gen_pts_items: list[tuple[int, np.ndarray]] = list(generated_points.items())
                    st.session_state["anim_generated_points"] = {int(k): v for k, v in gen_pts_items}
                    move_vecs_items: list[tuple[int, tuple[float, float]]] = list(movement_vectors.items())
                    st.session_state["anim_movement_vectors"] = {int(k): v for k, v in move_vecs_items}
                else:
                    st.session_state["anim_running"] = False
                    st.session_state["show_anim_circle"] = False
            else:
                # Prepare the next iteration for the same configuration (multi-point support)
                selected_indices = select_points_for_iteration()
                if not selected_indices:
                    movable_indices = get_movable_indices()
                    selected_indices = [int(np.random.choice(movable_indices))] if movable_indices else [0]

                # Generate movement vectors for all selected points
                movement_vectors = generate_movement_vectors(selected_indices, maxdist)

                # Helper to get parent position for an index
                def get_parent_for_idx(idx: int) -> np.ndarray:
                    for s in reversed(successful_points):
                        if int(s.get("original_parent_idx", -1)) == idx:
                            return s["point"]
                    return get_point_for_flat_idx(idx)

                # Check all points within bounds, retry if needed
                max_direction_attempts = 10
                generated_points: dict[int, np.ndarray] = {}
                for _ in range(max_direction_attempts):
                    all_within_bounds = True
                    generated_points: dict[int, np.ndarray] = {}

                    for idx in selected_indices:
                        dx, dy = movement_vectors.get(idx, (0.0, 0.0))
                        parent_pt = get_parent_for_idx(idx)

                        new_x = parent_pt[0] + dx
                        new_y = parent_pt[1] + dy

                        if not (COORD_MIN_X <= new_x <= COORD_MAX_X and COORD_MIN_Y <= new_y <= COORD_MAX_Y):
                            all_within_bounds = False

                        # NO CLIPPING - all points must be at exact same distance from parent
                        generated_points[idx] = np.array([new_x, new_y])

                    if all_within_bounds:
                        break
                    # Regenerate movement vectors
                    movement_vectors = generate_movement_vectors(selected_indices, maxdist)

                # For backwards compatibility, use first point as primary
                parent_idx_new = selected_indices[0]
                parent_pt_new = get_parent_for_idx(parent_idx_new)
                new_gen_pt = generated_points.get(parent_idx_new, parent_pt_new.copy())
                angle_local = np.arctan2(new_gen_pt[1] - parent_pt_new[1], new_gen_pt[0] - parent_pt_new[0])
                direction = np.array([np.cos(angle_local), np.sin(angle_local)])

                st.session_state["anim_parent_idx"] = parent_idx_new
                st.session_state["anim_angle"] = angle_local
                st.session_state["anim_generated_point"] = new_gen_pt
                st.session_state["anim_distance"] = maxdist
                st.session_state["anim_in_search"] = True
                # CORRECTED Binary search state - PRESERVE binary mode!
                st.session_state["anim_binary_mode"] = binary_mode  # Keep the same mode
                st.session_state["anim_binary_step"] = 0  # Init step (will be incremented to 1 on first progress)
                st.session_state["anim_binary_correct_order"] = parent_pt_new.copy()  # correct_order = parent (first point)
                st.session_state["anim_binary_correct_orders"] = {int(idx): get_parent_for_idx(idx).copy() for idx in selected_indices}
                st.session_state["anim_binary_current_distance"] = maxdist  # Start at maxdist
                st.session_state["anim_binary_direction"] = direction.copy()  # Direction unit vector (first point)
                st.session_state["anim_had_full_match"] = False
                # Linear mode state reset for new iteration
                st.session_state["anim_linear_mode"] = linear_mode  # Keep the same mode
                st.session_state["anim_linear_step"] = 0
                st.session_state["anim_linear_current_distance"] = maxdist
                st.session_state["anim_linear_step_size"] = maxdist * 0.1
                # Sync multi-point data
                st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                st.session_state["anim_generated_points"] = {int(k): v for k, v in generated_points.items()}
                st.session_state["anim_movement_vectors"] = {int(k): v for k, v in movement_vectors.items()}
        else:
            # === Case 2: keep searching ===
            # Different behavior for binary vs linear vs exponential strategy

            # DEBUG: Print which strategy branch we're taking
            logger.debug(f"[DEBUG ANIM] binary_mode={binary_mode}, linear_mode={linear_mode}, cfg_strategy={st.session_state.get('cfg_strategy')}, anim_binary_mode={st.session_state.get('anim_binary_mode')}")

            # Check if we should skip wait intervals (manual iteration/config mode)
            _skip_wait_intervals = st.session_state.get("_skip_wait_intervals", False)

            if binary_mode:
                # ============= CORRECTED BINARY SEARCH STRATEGY (7 steps, MULTI-POINT) =============
                # Algorithm:
                # - Init: all n points at distance maxdist, correct_orders = parent coords, current_distance = maxdist
                # - Step 0: halve naar 0.5×maxdist BEFORE testing
                # - Steps 1-7: 
                #   - Test ALL n points for combined PDP order match
                #   - If ALL match: distance += 0.5^(n+1) × maxdist, correct_orders = current positions
                #   - If any no match: distance -= 0.5^(n+1) × maxdist
                # - End: place all n points at their correct_order positions

                binary_step = int(st.session_state.get("anim_binary_step", 0))

                # INSTANT ITERATION: If skip_wait_intervals is set, complete all remaining binary steps at once
                if _skip_wait_intervals and binary_step < 7:
                    # Get all selected indices and movement vectors
                    selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])
                    movement_vectors = st.session_state.get("anim_movement_vectors", {})
                    correct_orders: dict[int, np.ndarray] = st.session_state.get("anim_binary_correct_orders", {})

                    if not correct_orders:
                        for idx in selected_indices:
                            if idx < n_total_points:
                                correct_orders[idx] = all_pts[idx].copy()
                            else:
                                sidx = int(idx - n_total_points)
                                succ_list = st.session_state.get("anim_successful_points", [])
                                if 0 <= sidx < len(succ_list):
                                    correct_orders[idx] = succ_list[sidx]["point"].copy()
                                else:
                                    correct_orders[idx] = np.array([0.0, 0.0])

                    current_distance = float(st.session_state.get("anim_binary_current_distance", maxdist))

                    # Helper function to compute positions at a given distance
                    def _compute_positions_at_distance(dist: float) -> dict[int, np.ndarray]:
                        positions: dict[int, np.ndarray] = {}
                        for idx in selected_indices:
                            parent_pt = None
                            succ_list = st.session_state.get("anim_successful_points", [])
                            for s in reversed(succ_list):
                                if int(s.get("original_parent_idx", -1)) == idx:
                                    parent_pt = s["point"]
                                    break
                            if parent_pt is None:
                                if idx < n_total_points:
                                    parent_pt = all_pts[idx]
                                else:
                                    sidx = int(idx - n_total_points)
                                    if 0 <= sidx < len(succ_list):
                                        parent_pt = succ_list[sidx]["point"]
                                    else:
                                        parent_pt = np.array([0.0, 0.0])

                            orig_vec = movement_vectors.get(idx, (0.0, 0.0))
                            orig_mag = np.sqrt(orig_vec[0]**2 + orig_vec[1]**2)
                            if orig_mag > 1e-9:
                                direction = np.array([orig_vec[0] / orig_mag, orig_vec[1] / orig_mag])
                            else:
                                direction = np.array([1.0, 0.0])

                            new_pt = parent_pt + direction * dist
                            new_pt[0] = np.clip(new_pt[0], COORD_MIN_X, COORD_MAX_X)
                            new_pt[1] = np.clip(new_pt[1], COORD_MIN_Y, COORD_MAX_Y)
                            positions[idx] = new_pt
                        return positions

                    # Helper to check PDP match for a set of positions
                    # Get threshold parameters once outside the helper
                    _thresh_bin, _max_mm_bin = get_threshold_params()
                    def _check_match_for_positions(positions: dict[int, np.ndarray]) -> bool:
                        test_points = all_pts_flat.copy()
                        for idx, pt in positions.items():
                            if idx < len(test_points):
                                test_points[idx] = pt
                        match_d1, match_d2 = check_pdp_match(
                            all_pts_flat, test_points,
                            pdp_variant=pdp_variant, buffer_x=buffer_x, buffer_y=buffer_y,
                            rough_x=rough_x, rough_y=rough_y, match_threshold=_thresh_bin, max_mismatches=_max_mm_bin
                        )
                        return match_d1 and match_d2

                    # Simulate all remaining binary search steps (from current step to 7)
                    logger.debug(f"[DEBUG INSTANT BINARY] Starting instant completion from step {binary_step}")

                    # Step 1: halve to 0.5×maxdist if not done yet
                    if binary_step == 0:
                        current_distance = 0.5 * maxdist
                        binary_step = 1

                    # Steps 2-7: simulate binary search
                    while binary_step < 7:
                        binary_step += 1
                        test_positions = _compute_positions_at_distance(current_distance)
                        matches = _check_match_for_positions(test_positions)

                        delta_term = (0.5 ** binary_step) * maxdist
                        if matches:
                            # Update correct_orders
                            for idx in selected_indices:
                                if idx in test_positions:
                                    correct_orders[int(idx)] = np.array(test_positions[idx])
                            current_distance = current_distance + delta_term
                        else:
                            current_distance = max(current_distance - delta_term, 0.0)

                        logger.debug(f"[DEBUG INSTANT BINARY] Step {binary_step}: match={matches}, distance={current_distance:.4f}")

                    # Finalize: set to step 7+ and trigger completion
                    st.session_state["anim_binary_step"] = 7
                    st.session_state["anim_search_steps"] = 7
                    st.session_state["anim_binary_current_distance"] = current_distance
                    st.session_state["anim_binary_correct_orders"] = {int(k): v.copy() for k, v in correct_orders.items()}

                    # Place final points at correct_order positions
                    final_positions = {int(idx): correct_orders.get(int(idx), all_pts[idx].copy()) for idx in selected_indices}
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_binary_correct_order"] = correct_orders.get(int(first_idx), np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = 0.0  # Trigger success
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = final_positions
                    st.session_state["_skip_wait_intervals"] = False
                    logger.debug(f"[DEBUG INSTANT BINARY] Completed - final distance: {current_distance:.4f}")
                else:
                    # Normal step-by-step processing
                    binary_step += 1
                    st.session_state["anim_binary_step"] = binary_step

                search_steps += 1
                st.session_state["anim_search_steps"] = search_steps

                # Get all selected indices and their current generated positions
                selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])
                anim_generated_points = st.session_state.get("anim_generated_points", {})
                movement_vectors = st.session_state.get("anim_movement_vectors", {})

                # Get correct_orders for all points (multi-point state)
                correct_orders: dict[int, np.ndarray] = st.session_state.get("anim_binary_correct_orders", {})
                logger.debug(f"[DEBUG BINARY LOAD] correct_orders from session_state: {len(correct_orders)} entries, keys={list(correct_orders.keys())}")
                if not correct_orders:
                    # Fallback: initialize from parent positions
                    logger.debug("[DEBUG BINARY LOAD] correct_orders was EMPTY, initializing from parent positions")
                    for idx in selected_indices:
                        if idx < n_total_points:
                            correct_orders[idx] = all_pts[idx].copy()
                        else:
                            sidx = int(idx - n_total_points)
                            succ_list = st.session_state.get("anim_successful_points", [])
                            if 0 <= sidx < len(succ_list):
                                correct_orders[idx] = succ_list[sidx]["point"].copy()
                            else:
                                correct_orders[idx] = np.array([0.0, 0.0])

                # Get current distance (same for all points in synchronized movement)
                current_distance = float(st.session_state.get("anim_binary_current_distance", maxdist))

                # Check if current candidate configuration matches PDP (ALL n points together)
                current_matches = same_d1 and same_d2

                logger.debug(f"[DEBUG BINARY STEP {binary_step}] current_distance={current_distance:.4f}, n_points={len(selected_indices)}, matched={current_matches}")

                # Add diagnostic row
                diag_rows: list[dict[str, Any]] = st.session_state.get("diag_rows", [])
                diag_rows.append({
                    "n": binary_step,
                    "order_match_d1": same_d1,
                    "order_match_d2": same_d2,
                    "current_distance": current_distance,
                    "n_selected_points": len(selected_indices),
                })
                st.session_state["diag_rows"] = diag_rows

                # Helper: compute new positions for all points at given distance
                # MUST check successful_points first for updated parent positions (same logic as iteration setup)
                def compute_new_positions(dist: float) -> dict[int, np.ndarray]:
                    new_positions: dict[int, np.ndarray] = {}
                    for idx in selected_indices:
                        # Get parent position - check successful_points first for updated parent
                        parent_pt = None
                        succ_list = st.session_state.get("anim_successful_points", [])
                        for s in reversed(succ_list):
                            if int(s.get("original_parent_idx", -1)) == idx:
                                parent_pt = s["point"]
                                break

                        # If not found in successful_points, use original position
                        if parent_pt is None:
                            if idx < n_total_points:
                                parent_pt = all_pts[idx]
                            else:
                                sidx = int(idx - n_total_points)
                                if 0 <= sidx < len(succ_list):
                                    parent_pt = succ_list[sidx]["point"]
                                else:
                                    parent_pt = np.array([0.0, 0.0])

                        # Get original movement vector and scale to new distance
                        orig_vec = movement_vectors.get(idx, (0.0, 0.0))
                        orig_mag = np.sqrt(orig_vec[0]**2 + orig_vec[1]**2)
                        if orig_mag > 1e-9:
                            # Unit direction from original vector
                            direction = np.array([orig_vec[0] / orig_mag, orig_vec[1] / orig_mag])
                        else:
                            direction = np.array([1.0, 0.0])  # Fallback direction

                        # New position: parent + direction × dist
                        new_pt = parent_pt + direction * dist
                        new_pt[0] = np.clip(new_pt[0], COORD_MIN_X, COORD_MAX_X)
                        new_pt[1] = np.clip(new_pt[1], COORD_MIN_Y, COORD_MAX_Y)
                        new_positions[idx] = new_pt
                    return new_positions

                if binary_step >= 7:
                    # After 7 steps: finalize at correct_orders for ALL points
                    # If current step matches, update correct_orders first
                    if current_matches:
                        for idx in selected_indices:
                            if idx in anim_generated_points:
                                correct_orders[int(idx)] = np.array(anim_generated_points[idx])
                        st.session_state["anim_binary_correct_orders"] = {int(k): v.copy() for k, v in correct_orders.items()}
                        st.session_state["anim_had_full_match"] = True

                    # DEBUG: print correct_orders contents
                    logger.debug(f"[DEBUG BINARY FINALIZE] correct_orders keys={list(correct_orders.keys())}")
                    for k, v in correct_orders.items():
                        logger.debug(f"[DEBUG BINARY FINALIZE] correct_orders[{k}]={v}")
                    logger.debug(f"[DEBUG BINARY FINALIZE] had_full_match={st.session_state.get('anim_had_full_match', False)}")

                    # Place final points at correct_order positions (use int(idx) for key lookup)
                    final_positions = {int(idx): correct_orders.get(int(idx), all_pts[idx].copy()) for idx in selected_indices}

                    # DEBUG: print final_positions
                    for k, v in final_positions.items():
                        logger.debug(f"[DEBUG BINARY FINALIZE] final_positions[{k}]={v}")

                    # For backwards compatibility, keep single generated_point as first one
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_binary_correct_order"] = correct_orders.get(int(first_idx), np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = 0.0  # Trigger success
                    st.session_state["anim_in_search"] = True
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = final_positions
                    logger.debug(f"[DEBUG BINARY] FINALIZE at correct_orders for {len(selected_indices)} points")

                elif binary_step == 1:
                    # Step 1: Special case - halve distance FIRST before testing
                    # Current points are at maxdist, halve to 0.5×maxdist
                    new_distance = 0.5 * maxdist
                    st.session_state["anim_binary_current_distance"] = new_distance

                    # Compute new positions for ALL points
                    new_positions = compute_new_positions(new_distance)

                    # For backwards compatibility, keep single generated_point as first one
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = new_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = new_distance
                    st.session_state["anim_in_search"] = True
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = {int(k): v for k, v in new_positions.items()}
                    logger.debug(f"[DEBUG BINARY STEP {binary_step}] HALVE! distance {maxdist:.4f} -> {new_distance:.4f} for {len(selected_indices)} points")
                else:
                    # Steps 2-7: Test current position, then apply +/- formula
                    # delta_term = 0.5^(binary_step) × maxdist
                    # (step 2: 0.5², step 3: 0.5³, etc.)
                    delta_term = (0.5 ** binary_step) * maxdist

                    if current_matches:
                        # Match! 
                        # 1. Update correct_orders to current point positions for ALL points
                        for idx in selected_indices:
                            if idx in anim_generated_points:
                                correct_orders[int(idx)] = np.array(anim_generated_points[idx])
                        st.session_state["anim_binary_correct_orders"] = {int(k): v.copy() for k, v in correct_orders.items()}
                        st.session_state["anim_had_full_match"] = True
                        # 2. Add delta_term to distance
                        new_distance = current_distance + delta_term
                        logger.debug(f"[DEBUG BINARY STEP {binary_step}] MATCH! distance {current_distance:.4f} + {delta_term:.4f} = {new_distance:.4f}")
                    else:
                        # No match: subtract delta_term from distance
                        new_distance = current_distance - delta_term
                        logger.debug(f"[DEBUG BINARY STEP {binary_step}] NO MATCH! distance {current_distance:.4f} - {delta_term:.4f} = {new_distance:.4f}")

                    # Ensure distance stays positive
                    new_distance = max(new_distance, 0.0)
                    st.session_state["anim_binary_current_distance"] = new_distance

                    # Compute new positions for ALL points
                    new_positions = compute_new_positions(new_distance)

                    # For backwards compatibility, keep single generated_point as first one
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = new_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_binary_correct_order"] = correct_orders.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = new_distance
                    st.session_state["anim_in_search"] = True
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = {int(k): v for k, v in new_positions.items()}
                    logger.debug(f"[DEBUG BINARY STEP {binary_step}] Next candidates at distance {new_distance:.4f} for {len(selected_indices)} points")

            elif linear_mode:
                # ============= LINEAR SEARCH STRATEGY (MULTI-POINT) =============
                # Algorithm: Decrease distance by 0.1×maxdist per step until ALL n points have order match
                # Stop when: (1) all points match, or (2) distance <= 0

                linear_step = int(st.session_state.get("anim_linear_step", 0))

                # INSTANT ITERATION: If skip_wait_intervals is set, complete all remaining linear steps at once
                if _skip_wait_intervals:
                    # Get all selected indices and movement vectors
                    selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])
                    movement_vectors = st.session_state.get("anim_movement_vectors", {})
                    correct_orders: dict[int, np.ndarray] = st.session_state.get("anim_binary_correct_orders", {})

                    if not correct_orders:
                        for idx in selected_indices:
                            if idx < n_total_points:
                                correct_orders[idx] = all_pts[idx].copy()
                            else:
                                sidx = int(idx - n_total_points)
                                succ_list = st.session_state.get("anim_successful_points", [])
                                if 0 <= sidx < len(succ_list):
                                    correct_orders[idx] = succ_list[sidx]["point"].copy()
                                else:
                                    correct_orders[idx] = np.array([0.0, 0.0])

                    current_distance = float(st.session_state.get("anim_linear_current_distance", maxdist))
                    step_size = float(st.session_state.get("anim_linear_step_size", maxdist * 0.1))

                    # Helper: get parent position for an index
                    def _get_parent_lin(idx: int) -> np.ndarray:
                        succ_list = st.session_state.get("anim_successful_points", [])
                        for s in reversed(succ_list):
                            if int(s.get("original_parent_idx", -1)) == idx:
                                return s["point"]
                        if idx < n_total_points:
                            return all_pts[idx]
                        else:
                            sidx = int(idx - n_total_points)
                            if 0 <= sidx < len(succ_list):
                                return succ_list[sidx]["point"]
                            return np.array([0.0, 0.0])

                    # Helper to compute positions at a given distance
                    def _compute_linear_positions(dist: float) -> dict[int, np.ndarray]:
                        positions: dict[int, np.ndarray] = {}
                        for idx in selected_indices:
                            parent_pt = _get_parent_lin(idx)
                            orig_vec = movement_vectors.get(idx, (0.0, 0.0))
                            orig_mag = np.sqrt(orig_vec[0]**2 + orig_vec[1]**2)
                            if orig_mag > 1e-9:
                                direction = np.array([orig_vec[0] / orig_mag, orig_vec[1] / orig_mag])
                            else:
                                direction = np.array([1.0, 0.0])
                            new_pt = parent_pt + direction * dist
                            new_pt[0] = np.clip(new_pt[0], COORD_MIN_X, COORD_MAX_X)
                            new_pt[1] = np.clip(new_pt[1], COORD_MIN_Y, COORD_MAX_Y)
                            positions[idx] = new_pt
                        return positions

                    # Helper to check PDP match for a set of positions
                    # Get threshold parameters once outside the helper
                    _thresh_lin, _max_mm_lin = get_threshold_params()
                    def _check_linear_match(positions: dict[int, np.ndarray]) -> bool:
                        test_points = all_pts_flat.copy()
                        for idx, pt in positions.items():
                            if idx < len(test_points):
                                test_points[idx] = pt
                        match_d1, match_d2 = check_pdp_match(
                            all_pts_flat, test_points,
                            pdp_variant=pdp_variant, buffer_x=buffer_x, buffer_y=buffer_y,
                            rough_x=rough_x, rough_y=rough_y, match_threshold=_thresh_lin, max_mismatches=_max_mm_lin
                        )
                        return match_d1 and match_d2

                    # Simulate all linear search steps until match or distance <= 0
                    logger.debug(f"[DEBUG INSTANT LINEAR] Starting instant completion from step {linear_step}")
                    max_linear_steps = 100  # Safety limit

                    while linear_step < max_linear_steps and current_distance > 0:
                        linear_step += 1
                        test_positions = _compute_linear_positions(current_distance)
                        matches = _check_linear_match(test_positions)

                        logger.debug(f"[DEBUG INSTANT LINEAR] Step {linear_step}: match={matches}, distance={current_distance:.4f}")

                        if matches:
                            # Update correct_orders and finalize
                            for idx in selected_indices:
                                if idx in test_positions:
                                    correct_orders[int(idx)] = np.array(test_positions[idx])
                            break

                        current_distance = max(current_distance - step_size, 0.0)

                    # Finalize
                    st.session_state["anim_linear_step"] = linear_step
                    st.session_state["anim_search_steps"] = linear_step
                    st.session_state["anim_linear_current_distance"] = current_distance
                    st.session_state["anim_binary_correct_orders"] = {int(k): v.copy() for k, v in correct_orders.items()}

                    # Place final points at correct_order positions
                    final_positions = {int(idx): correct_orders.get(int(idx), all_pts[idx].copy()) for idx in selected_indices}
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_binary_correct_order"] = correct_orders.get(int(first_idx), np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = 0.0  # Trigger success
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = final_positions
                    st.session_state["_skip_wait_intervals"] = False
                    logger.debug(f"[DEBUG INSTANT LINEAR] Completed at step {linear_step}, distance: {current_distance:.4f}")
                else:
                    # Normal step-by-step processing
                    linear_step += 1
                    st.session_state["anim_linear_step"] = linear_step

                search_steps += 1
                st.session_state["anim_search_steps"] = search_steps

                # Get all selected indices and their movement vectors
                selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])
                movement_vectors = st.session_state.get("anim_movement_vectors", {})
                anim_generated_points = st.session_state.get("anim_generated_points", {})

                # Get correct_orders for all points (multi-point state)
                correct_orders: dict[int, np.ndarray] = st.session_state.get("anim_binary_correct_orders", {})
                if not correct_orders:
                    for idx in selected_indices:
                        if idx < n_total_points:
                            correct_orders[idx] = all_pts[idx].copy()
                        else:
                            sidx = int(idx - n_total_points)
                            succ_list = st.session_state.get("anim_successful_points", [])
                            if 0 <= sidx < len(succ_list):
                                correct_orders[idx] = succ_list[sidx]["point"].copy()
                            else:
                                correct_orders[idx] = np.array([0.0, 0.0])

                # Get current distance and step size
                current_distance = float(st.session_state.get("anim_linear_current_distance", maxdist))
                step_size = float(st.session_state.get("anim_linear_step_size", maxdist * 0.1))

                # Check if current candidate configuration matches PDP (ALL n points together)
                current_matches = same_d1 and same_d2

                logger.debug(f"[DEBUG LINEAR STEP {linear_step}] current_distance={current_distance:.4f}, n_points={len(selected_indices)}, matched={current_matches}")

                # Add diagnostic row
                diag_rows = st.session_state.get("diag_rows", [])
                diag_rows.append({
                    "n": linear_step,
                    "order_match_d1": same_d1,
                    "order_match_d2": same_d2,
                    "current_distance": current_distance,
                    "n_selected_points": len(selected_indices),
                })
                st.session_state["diag_rows"] = diag_rows

                # Helper: get parent position for an index
                def get_parent_for_idx_lin(idx: int) -> np.ndarray:
                    succ_list = st.session_state.get("anim_successful_points", [])
                    for s in reversed(succ_list):
                        if int(s.get("original_parent_idx", -1)) == idx:
                            return s["point"]
                    if idx < n_total_points:
                        return all_pts[idx]
                    else:
                        sidx = int(idx - n_total_points)
                        if 0 <= sidx < len(succ_list):
                            return succ_list[sidx]["point"]
                        return np.array([0.0, 0.0])

                # Helper: compute new positions for all points at given distance
                def compute_linear_positions(dist: float) -> dict[int, np.ndarray]:
                    new_positions: dict[int, np.ndarray] = {}
                    for idx in selected_indices:
                        parent_pt = get_parent_for_idx_lin(idx)
                        orig_vec = movement_vectors.get(idx, (0.0, 0.0))
                        orig_mag = np.sqrt(orig_vec[0]**2 + orig_vec[1]**2)
                        if orig_mag > 1e-9:
                            direction = np.array([orig_vec[0] / orig_mag, orig_vec[1] / orig_mag])
                        else:
                            direction = np.array([1.0, 0.0])
                        new_pt = parent_pt + direction * dist
                        new_pt[0] = np.clip(new_pt[0], COORD_MIN_X, COORD_MAX_X)
                        new_pt[1] = np.clip(new_pt[1], COORD_MIN_Y, COORD_MAX_Y)
                        new_positions[idx] = new_pt
                    return new_positions

                if current_matches:
                    # MATCH! Update correct_orders to current positions and FINALIZE
                    for idx in selected_indices:
                        if idx in anim_generated_points:
                            correct_orders[int(idx)] = np.array(anim_generated_points[idx])
                    st.session_state["anim_binary_correct_orders"] = {int(k): v.copy() for k, v in correct_orders.items()}
                    st.session_state["anim_had_full_match"] = True

                    # Place final points at correct_order positions
                    final_positions = {int(idx): correct_orders.get(int(idx), all_pts[idx].copy()) for idx in selected_indices}

                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_binary_correct_order"] = correct_orders.get(int(first_idx), np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = 0.0  # Trigger success
                    st.session_state["anim_in_search"] = True
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = final_positions
                    logger.debug(f"[DEBUG LINEAR] MATCH at step {linear_step}, distance {current_distance:.4f} - FINALIZE")

                else:
                    # NO MATCH: decrease distance by step_size (10% of maxdist)
                    new_distance = current_distance - step_size

                    if new_distance <= 0:
                        # Distance reached 0, finalize at parent positions (no match found)
                        final_positions = {int(idx): get_parent_for_idx_lin(idx).copy() for idx in selected_indices}

                        first_idx = selected_indices[0] if selected_indices else parent_idx
                        st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                        st.session_state["anim_distance"] = 0.0
                        st.session_state["anim_in_search"] = True
                        st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                        st.session_state["anim_generated_points"] = final_positions
                        st.session_state["anim_movement_vectors"] = {}
                        logger.debug(f"[DEBUG LINEAR] Distance reached 0, snapping {len(selected_indices)} points to parents")
                    else:
                        # Update distance and compute new positions
                        st.session_state["anim_linear_current_distance"] = new_distance
                        new_positions = compute_linear_positions(new_distance)

                        first_idx = selected_indices[0] if selected_indices else parent_idx
                        st.session_state["anim_generated_point"] = new_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                        st.session_state["anim_distance"] = new_distance
                        st.session_state["anim_in_search"] = True
                        st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                        st.session_state["anim_generated_points"] = {int(k): v for k, v in new_positions.items()}
                        logger.debug(f"[DEBUG LINEAR STEP {linear_step}] NO MATCH! distance {current_distance:.4f} - {step_size:.4f} = {new_distance:.4f}")

            else:
                # ============= EXPONENTIAL SEARCH STRATEGY (MULTI-POINT) =============
                # Algorithm: Halve distance for ALL n points TOGETHER until ALL n points have order match
                search_steps += 1
                st.session_state["anim_search_steps"] = search_steps

                # Get all selected indices and their movement vectors
                selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])
                movement_vectors = st.session_state.get("anim_movement_vectors", {})
                anim_generated_points = st.session_state.get("anim_generated_points", {})

                # Helper: get parent position for an index
                # MUST check successful_points first for updated parent positions (same logic as iteration setup)
                def get_parent_for_idx_exp(idx: int) -> np.ndarray:
                    # First check if this point was successfully placed in a previous iteration
                    succ_list = st.session_state.get("anim_successful_points", [])
                    for s in reversed(succ_list):
                        if int(s.get("original_parent_idx", -1)) == idx:
                            return s["point"]
                    # If not found, use original position
                    if idx < n_total_points:
                        return all_pts[idx]
                    else:
                        sidx = int(idx - n_total_points)
                        if 0 <= sidx < len(succ_list):
                            return succ_list[sidx]["point"]
                        return np.array([0.0, 0.0])

                # Helper: compute new positions for ALL points at given distance
                def compute_exp_positions(dist: float) -> dict[int, np.ndarray]:
                    new_positions: dict[int, np.ndarray] = {}
                    for idx in selected_indices:
                        parent_pt = get_parent_for_idx_exp(idx)

                        # Get original movement vector and scale to new distance
                        # Direction is preserved from initial generation - NO random tweaks!
                        orig_vec = movement_vectors.get(idx, (0.0, 0.0))
                        orig_mag = np.sqrt(orig_vec[0]**2 + orig_vec[1]**2)
                        if orig_mag > 1e-9:
                            direction = np.array([orig_vec[0] / orig_mag, orig_vec[1] / orig_mag])
                        else:
                            direction = np.array([1.0, 0.0])

                        # New position: parent + direction × dist
                        # NO CLIPPING - all points must be at exact same distance from parent
                        new_pt = parent_pt + direction * dist
                        new_positions[idx] = new_pt
                    return new_positions

                # INSTANT ITERATION: If skip_wait_intervals is set, complete all remaining exponential steps at once
                if _skip_wait_intervals:
                    # Helper to check PDP match for a set of positions
                    # Get threshold parameters once outside the helper
                    _thresh_exp, _max_mm_exp = get_threshold_params()
                    def _check_exp_match(positions: dict[int, np.ndarray]) -> bool:
                        test_points = all_pts_flat.copy()
                        for idx, pt in positions.items():
                            if idx < len(test_points):
                                test_points[idx] = pt
                        match_d1, match_d2 = check_pdp_match(
                            all_pts_flat, test_points,
                            pdp_variant=pdp_variant, buffer_x=buffer_x, buffer_y=buffer_y,
                            rough_x=rough_x, rough_y=rough_y, match_threshold=_thresh_exp, max_mismatches=_max_mm_exp
                        )
                        return match_d1 and match_d2

                    # Simulate all exponential search steps until match or min distance
                    logger.debug(f"[DEBUG INSTANT EXPONENTIAL] Starting instant completion from step {search_steps}")
                    current_dist = float(distance)
                    min_distance = 1e-5
                    final_positions: dict[int, np.ndarray] = {}

                    while search_steps < max_search_steps and current_dist > min_distance:
                        search_steps += 1
                        current_dist = current_dist / 2.0
                        if current_dist < min_distance:
                            current_dist = min_distance * 2.0

                        test_positions = compute_exp_positions(current_dist)
                        matches = _check_exp_match(test_positions)

                        logger.debug(f"[DEBUG INSTANT EXPONENTIAL] Step {search_steps}: match={matches}, distance={current_dist:.4f}")

                        if matches:
                            final_positions = {int(k): v.copy() for k, v in test_positions.items()}
                            break

                    # If no match found, snap to parent positions
                    if not final_positions:
                        for idx in selected_indices:
                            final_positions[int(idx)] = get_parent_for_idx_exp(idx).copy()

                    # Finalize
                    st.session_state["anim_search_steps"] = search_steps
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = 0.0  # Trigger success
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = final_positions
                    st.session_state["_skip_wait_intervals"] = False
                    logger.debug(f"[DEBUG INSTANT EXPONENTIAL] Completed at step {search_steps}, distance: {current_dist:.4f}")
                elif search_steps >= max_search_steps:
                    # If search did not converge, snap ALL points back to parent positions
                    final_positions: dict[int, np.ndarray] = {}
                    for idx in selected_indices:
                        final_positions[int(idx)] = get_parent_for_idx_exp(idx).copy()

                    # For backwards compatibility
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                    st.session_state["anim_distance"] = 0.0
                    st.session_state["anim_in_search"] = True
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = final_positions
                    st.session_state["anim_movement_vectors"] = {}
                    logger.debug(f"[DEBUG EXPONENTIAL] Max steps reached, snapping {len(selected_indices)} points to parents")
                else:
                    # Standard exponential search step: halve distance for ALL n points together
                    new_distance = distance / 2.0
                    min_distance = 1e-5
                    if new_distance < min_distance:
                        new_distance = min_distance * 2.0

                    # Compute new positions for ALL points at the new distance
                    new_positions = compute_exp_positions(new_distance)

                    # For backwards compatibility, use first point
                    first_idx = selected_indices[0] if selected_indices else parent_idx
                    first_pt = new_positions.get(first_idx, np.array([0.0, 0.0]))
                    first_parent = get_parent_for_idx_exp(first_idx)
                    angle_local = np.arctan2(first_pt[1] - first_parent[1], first_pt[0] - first_parent[0])

                    st.session_state["anim_generated_point"] = first_pt.copy()
                    st.session_state["anim_distance"] = new_distance
                    st.session_state["anim_angle"] = angle_local
                    st.session_state["anim_in_search"] = True
                    st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                    st.session_state["anim_generated_points"] = {int(k): v for k, v in new_positions.items()}
                    logger.debug(f"[DEBUG EXPONENTIAL] Halving ALL {len(selected_indices)} points: {distance:.4f} -> {new_distance:.4f}")

        # Determine rerun behavior based on animation mode
        # _manual_step_mode: pause after each step (one search iteration)
        # _manual_iteration_mode: pause only when a point is placed (iteration complete)
        # _manual_config_mode: pause only when a configuration is complete

        # Check if we just completed an iteration (point was placed)
        _iteration_just_completed = st.session_state.get("_iteration_just_completed", False)
        # Check if we just completed a configuration
        _config_just_completed = st.session_state.get("_config_just_completed", False)

        # Clear the completion flags
        st.session_state["_iteration_just_completed"] = False
        st.session_state["_config_just_completed"] = False

        if _manual_step_mode:
            # Manual step-by-step: always pause after each step
            # Clear the step request flag
            st.session_state["anim_manual_step_requested"] = False
            st.rerun()
        elif _manual_iteration_mode:
            if _iteration_just_completed:
                # Iteration complete - pause and wait for user
                st.session_state["anim_manual_iteration_requested"] = False
                st.session_state["_iteration_in_progress"] = False
                st.rerun()
            else:
                # Still in the middle of an iteration - continue automatically
                st.rerun()
        elif _manual_config_mode:
            if _config_just_completed:
                # Configuration complete - pause and wait for user
                st.session_state["anim_manual_config_requested"] = False
                st.session_state["_config_in_progress"] = False
                st.rerun()
            else:
                # Still in the middle of a configuration - continue automatically
                st.rerun()
        else:
            # Auto mode: non-blocking wait in short increments so Streamlit stays responsive
            _elapsed = 0.0
            while _elapsed < wait_s:
                _step = min(0.1, wait_s - _elapsed)
                time.sleep(_step)
                _elapsed += _step
            st.rerun()


with tab_static:
    # ============= CSV Export Section ============
    st.markdown("<hr />", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:1.5rem;'>Generated configuration (CSV)</h3>", unsafe_allow_html=True)

    all_configs_list: list[dict[str, Any]] = st.session_state.get("anim_all_configs", [])
    current_successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    current_config_num = int(st.session_state.get("anim_current_config", 1))

    # Initialize latest_generated and configs_by_variant for use in animation
    latest_generated: dict[tuple[int, int], np.ndarray] = {}
    configs_by_variant: dict[str, list[int]] = {}
    # Build configs_by_variant from all_configs_list
    st.write(f"DEBUG BUILD: all_configs_list has {len(all_configs_list)} entries")
    for i, cfg in enumerate(all_configs_list):
        variant = cfg.get("pdp_variant", "fundamental")
        config_num = cfg.get("config_num", 0)
        if i < 3:  # Show first 3
            st.write(f"  Config {i}: variant={variant}, config_num={config_num}, points={len(cfg.get('points', []))}")
        if variant not in configs_by_variant:
            configs_by_variant[variant] = []
        if config_num not in configs_by_variant[variant]:
            configs_by_variant[variant].append(config_num)

    if all_configs_list or current_successful_points:
        # Collect all generated points, grouped per configuration
        all_points_by_config: dict[int, list[SuccessfulPoint]] = {}

        for config_data in all_configs_list:
            config_num = config_data["config_num"]
            points = config_data["points"]
            all_points_by_config[config_num] = points

        if current_successful_points:
            all_points_by_config[current_config_num] = current_successful_points

        # For each original_index keep the latest generated point (from most recent config)
        latest_generated: dict[tuple[int, int], np.ndarray] = {}
        for config_num, points in all_points_by_config.items():
            st.write(f"DEBUG POPULATE: Config {config_num} has {len(points)} points")
            for sp in points:
                orig_idx = int(sp.get("original_parent_idx", 0))
                latest_generated[(config_num, orig_idx)] = sp["point"]

        all_config_nums = sorted(all_points_by_config.keys())

        csv_rows: list[tuple[int, float, int, float, float]] = []

        # Build rows for each configuration, in the style (c, t, o, x, y)
        for config_num in all_config_nums:
            # Shift configuration id so each configuration has a unique c-value
            c_value = selected_c_int + config_num

            # Iterate over all objects in the flat index order
            for flat_idx in range(n_total_points):
                obj_id, local_idx, _ = get_object_info_for_flat_idx(flat_idx)
                t_val = get_timestamp_for_flat_idx(flat_idx)
                if (config_num, flat_idx) in latest_generated:
                    point = latest_generated[(config_num, flat_idx)]
                else:
                    point = get_point_for_flat_idx(flat_idx)
                csv_rows.append((c_value, float(t_val), int(obj_id), float(point[0]), float(point[1])))

        csv_rows.sort(key=lambda row: (row[0], row[1], row[2]))

        csv_lines = ["c,t,o,x,y"]
        for c, t, o, x, y in csv_rows:
            csv_lines.append(f"{c},{t},{o},{x:.{COORD_CSV_PRECISION}f},{y:.{COORD_CSV_PRECISION}f}")

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

        # ============= Visualization of all configurations (Plotly) ============
        st.markdown("<h3 style='margin-top:1.5rem;'>Visualization of generated configurations</h3>", unsafe_allow_html=True)

        # Build list of all available configurations: "Original" + generated configs
        available_configs = ["Original"]

        # Build config labels for multiselect (format: "variant C{num}")
        for variant in sorted(configs_by_variant.keys()):
            for config_num in sorted(configs_by_variant[variant]):
                available_configs.append(f"{variant} C{config_num}")

        # Configuration selector - default to only "Original"
        st.markdown("**Select configurations to display:**")
        selected_configs = st.multiselect(
            "Show configurations:",
            options=available_configs,
            default=["Original"],
            key="viz_config_filter",
            help="Select which configurations to display in the visualization. "
                 "The 'Original' configuration shows the reference points as entered. "
                 "Generated configurations (e.g., 'fundamental C0') show alternative point arrangements "
                 "that satisfy the same PDP inequality matrix. Select multiple configurations to compare them visually."
        )

        fig = go.Figure()

        # Add lane markings for traffic configurations (skip for custom uploads)
        if not _is_custom_upload:
            fig = add_lane_markings_to_figure(fig, selected_c_int, XLIM, YLIM)

        # 1. Add Original Configuration - loop through ALL objects (only if selected)
        if "Original" in selected_configs:
            for i, obj_id in enumerate(sorted(all_points_plot.keys())):
                pts = all_points_plot[obj_id]
                vals = all_vals_plot[obj_id]
                color = OBJECT_COLORS_PLOTLY[i % len(OBJECT_COLORS_PLOTLY)]
                label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
                # Build hover text for each point
                hover_texts: list[str] = [f"<b>Original</b><br>Object: {label}<br>Point: {label}_{int(t)}<br>t: {int(t)}<br>d1: {pts[j, 0]:.{COORD_DISPLAY_PRECISION}f}<br>d2: {pts[j, 1]:.{COORD_DISPLAY_PRECISION}f}" 
                              for j, t in enumerate(vals)]
                fig.add_trace(  # type: ignore[call-arg]
                    go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode='lines+markers+text',
                    name=f'Original ({label})',
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    text=[f"{label}_{{{int(t)}}}" for t in vals],
                    textposition="top center",
                    legendgroup='Original',
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_texts,
                ))

            # Add external reference points to Original configuration
            if external_pts_for_window:
                ext_pts_arr = np.array(external_pts_for_window)
                hover_texts_ext: list[str] = []
                text_labels_ext: list[str] = []
                n_timestamps = len(selected_ts_window)
                for idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):
                    ext_point_idx = idx % len(external_points_list) if external_points_list else idx
                    hover_texts_ext.append(
                        f"<b>Original</b><br>Type: External Reference<br>Point: ext_{ext_point_idx}<br>"
                        f"d1: {ext_pt[0]:.{COORD_DISPLAY_PRECISION}f}<br>d2: {ext_pt[1]:.{COORD_DISPLAY_PRECISION}f}<br>"
                        f"<i>(Fixed - does not move)</i>"
                    )
                    text_labels_ext.append(f"ext_{ext_point_idx}")

                fig.add_trace(  # type: ignore[call-arg]
                    go.Scatter(
                    x=ext_pts_arr[:, 0],
                    y=ext_pts_arr[:, 1],
                    mode='markers+text',
                    name='Original (external)',
                    marker=dict(size=10, symbol='square', color='gray', line=dict(color='black', width=1.5)),
                    text=text_labels_ext,
                    textposition="top center",
                    legendgroup='Original',
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_texts_ext,
                ))

        # 2. Add Generated Configurations (only those selected)
        for variant in sorted(configs_by_variant.keys()):
            for config_num in sorted(configs_by_variant[variant]):
                config_label = f"{variant} C{config_num}"
                if config_label not in selected_configs:
                    continue  # Skip configurations not selected

                # Build generated points for this config (only movable objects)
                generated_pts_config: dict[int, np.ndarray] = {}
                for flat_idx in range(n_total_points):
                    obj_id, local_idx, _ = get_object_info_for_flat_idx(flat_idx)
                    if obj_id == -1:  # Skip external points
                        continue
                    if obj_id not in generated_pts_config:
                        generated_pts_config[obj_id] = all_points_plot[obj_id].copy()
                    if (config_num, flat_idx) in latest_generated:
                        generated_pts_config[obj_id][local_idx] = latest_generated[(config_num, flat_idx)]

                # Plot each object with its own color
                for i, obj_id in enumerate(sorted(generated_pts_config.keys())):
                    pts = generated_pts_config[obj_id]
                    vals = all_vals_plot[obj_id]  # Get timestamps for point labels
                    color = OBJECT_COLORS_PLOTLY[i % len(OBJECT_COLORS_PLOTLY)]
                    label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
                    # Build hover text for each point showing config info
                    hover_texts = [f"<b>{config_label}</b><br>Variant: {variant}<br>Config: C{config_num}<br>Object: {label}<br>Point: {label}_{int(vals[j])}<br>t: {int(vals[j])}<br>d1: {pts[j, 0]:.{COORD_DISPLAY_PRECISION}f}<br>d2: {pts[j, 1]:.{COORD_DISPLAY_PRECISION}f}" 
                                  for j in range(len(pts))]
                    fig.add_trace(  # type: ignore[call-arg]
                        go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode='lines+markers',
                        name=f'{variant} C{config_num} ({label})',
                        line=dict(color=color, width=1, dash='dash'),
                        marker=dict(size=6, symbol='circle-open'),
                        legendgroup=f'{variant}_C{config_num}',
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_texts,
                    ))

                # Add external reference points to generated config (they remain fixed)
                if external_pts_for_window:
                    ext_pts_arr = np.array(external_pts_for_window)
                    hover_texts_ext = []
                    text_labels_ext = []
                    n_timestamps = len(selected_ts_window)
                    for idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):
                        ext_point_idx = idx % len(external_points_list) if external_points_list else idx
                        hover_texts_ext.append(
                            f"<b>{config_label}</b><br>Type: External Reference<br>Point: ext_{ext_point_idx}<br>"
                            f"d1: {ext_pt[0]:.{COORD_DISPLAY_PRECISION}f}<br>d2: {ext_pt[1]:.{COORD_DISPLAY_PRECISION}f}<br>"
                            f"<i>(Fixed - does not move)</i>"
                        )

                    fig.add_trace(  # type: ignore[call-arg]
                        go.Scatter(
                        x=ext_pts_arr[:, 0],
                        y=ext_pts_arr[:, 1],
                        mode='markers',
                        name=f'{config_label} (external)',
                        marker=dict(size=8, symbol='square-open', color='gray', line=dict(color='black', width=1)),
                        legendgroup=f'{variant}_C{config_num}',
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_texts_ext,
                        showlegend=False,  # Don't show in legend to avoid clutter
                    ))

        fig.update_layout(
            width=800,
            height=800,
            xaxis=dict(
                range=[XLIM[0], XLIM[1]],
                title="d1"
            ),
            yaxis=dict(
                range=[YLIM[0], YLIM[1]],
                title="d2"
            ),
            legend=dict(
                groupclick="toggleitem" # Clicking a legend item toggles the whole group
            ),
            title="Comparison of Selected Configurations"
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.info("Run an animation or use 'Generate configurations' to generate configuration data.")


with tab_animation:
    # ============= Smooth Animation Button (Always Available) ============
    st.markdown("---")
    st.markdown("### 🎬 Smooth Trajectory Animation")
    st.markdown("""
    Animate the selected configurations with smooth, continuous motion:
    - **Cubic spline interpolation** for natural, fluid trajectories
    - **Road boundaries and lane markings** clearly visible (3 meter rijvakken)
    - **Independent y-scale** to maximize space usage
    - Works with both **Original** and **Generated configurations**
    """)

    # Build list of all available configurations for animation
    available_configs_anim = ["Original"]
    for variant in sorted(configs_by_variant.keys()):
        for config_num in sorted(configs_by_variant[variant]):
            available_configs_anim.append(f"{variant} C{config_num}")

    # Configuration selector for animation - default to only "Original"
    st.markdown("**Select configurations to animate:**")
    selected_configs_anim = st.multiselect(
        "Animate configurations:",
        options=available_configs_anim,
        default=["Original"],
        key="anim_config_filter",
        help="Select which configurations to animate. Multiple selections will be shown together in one animation."
    )

    # Animation speed selector
    _speed_options = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
    _speed_label = st.select_slider(
        "Animation speed",
        options=list(_speed_options.keys()),
        value="1x",
        key="anim_speed_selector",
        help="Controls playback speed. Higher = faster animation.",
    )
    _anim_speed = _speed_options[_speed_label]

    if st.button("▶ Play Animation", key="btn_smooth_animation", 
                 help="Opens an interactive animation of all selected configurations with smooth trajectories, road boundaries, and lane markings"):
        st.session_state["show_smooth_animation"] = True
        st.session_state["selected_configs_for_anim"] = selected_configs_anim
        st.session_state["anim_speed_multiplier"] = _anim_speed

    # Display the smooth animation if requested
    if st.session_state.get("show_smooth_animation", False):
        st.markdown("---")
        st.markdown("**Animation Controls:** Use Play/Pause buttons and the slider to control playback.")

        # Get the selected configs from session state
        anim_configs = st.session_state.get("selected_configs_for_anim", ["Original"])

        # Create the animation
        _stored_speed = st.session_state.get("anim_speed_multiplier", 1.0)
        anim_fig = create_smooth_animation(
            all_points_plot=all_points_plot,
            all_vals_plot=all_vals_plot,
            latest_generated=latest_generated,
            selected_configs=anim_configs,
            configs_by_variant=configs_by_variant,
            all_configs_list=all_configs_list,
            external_pts_for_window=external_pts_for_window,
            external_ts_for_window=external_ts_for_window,
            external_points_list=external_points_list,
            n_total_points=n_total_points,
            selected_c_int=selected_c_int,
            xlim=XLIM,
            ylim=YLIM,
            speed_multiplier=_stored_speed,
        )

        # Display the animation
        st.plotly_chart(anim_fig, width="stretch")

        # Add a close button
        if st.button("✖ Close Animation", key="btn_close_animation_v2"):
            st.session_state["show_smooth_animation"] = False
            st.rerun()


with tab_static:
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

    # ============= Diagnostic text box: result per iteration ============
    st.markdown("<h3 style='margin-top:1.5rem;'>Order match per iteration (binary strategy)</h3>", unsafe_allow_html=True)

    iter_log = st.session_state.get("binary_iteration_summary", [])

    if iter_log:
        lines: list[str] = []
        for item in iter_log:
            cnum = item.get("config", 1)
            it = item.get("iteration", 0)
            m1 = item.get("match_d1", False)
            m2 = item.get("match_d2", False)
            lines.append(f"Config {cnum}, iteration {it}: d1 match = {m1}, d2 match = {m2}")
        summary_text = "\n".join(lines)
        st.text_area(
            "Overview of order match after final placement of the point",
            value=summary_text,
            height=160,
            key="binary_iter_overview"
        )
    else:
        st.info("No final points placed with the binary strategy yet.")

    # ============= Angles Between Consecutive Timestamps Graph ============
    st.markdown("<hr />", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:1.5rem;'>Angles Between Consecutive Timestamps</h3>", unsafe_allow_html=True)

    def compute_vector_angle(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute the angle (in degrees) of the vector from p1 to p2.
        Angle is measured from the positive x-axis, counterclockwise.
        Returns angle in degrees [-180, 180].
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))

    def build_angle_series_from_points(points_dict: dict[int, np.ndarray], vals_dict: dict[int, np.ndarray], timestamps: list) -> dict[str, dict[str, float]]:
        """
        Build angle series data from points and timestamps.
        Only calculates angles between consecutive timestamps for the SAME object.
        e.g., k0→k1, k1→k2, l0→l1, l1→l2 (NOT k0→l0 or k0→k2)
        Returns dict of series_name -> {timestamp_label: angle}
        """
        angle_series: dict[str, dict[str, float]] = {}  # type: ignore[misc]

        # Only angles between consecutive timestamps for each object
        for i, o_id in enumerate(sorted(points_dict.keys())):
            pts = points_dict[o_id]
            ts = vals_dict[o_id]
            label = OBJECT_LABELS[i % len(OBJECT_LABELS)]

            for idx in range(len(pts) - 1):
                t_from = ts[idx]
                t_to = ts[idx + 1]
                angle = compute_vector_angle(pts[idx], pts[idx + 1])
                series_name = f"{label}{int(t_from)}→{label}{int(t_to)}"
                ts_label = f"t={int(t_from)}"
                if series_name not in angle_series:
                    angle_series[series_name] = {}
                angle_series[series_name][ts_label] = angle

        return angle_series

    # ============= Angle Comparison Section =============
    st.markdown("<hr />", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:1.5rem;'>📐 Angle Comparison</h3>", unsafe_allow_html=True)

    # Check if we have generated configurations to display
    # Use the same data source as the Plotly comparison chart (anim_all_configs)
    angles_all_configs_list: list[dict[str, Any]] = st.session_state.get("anim_all_configs", [])
    angles_successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    has_generated_data = len(angles_all_configs_list) > 0 or len(angles_successful_points) > 0

    if has_generated_data:
        # Get all timestamps
        all_timestamps = sorted(selected_ts_window)

        # Build angle data for ORIGINAL configuration
        original_angle_series = build_angle_series_from_points(all_points_plot, all_vals_plot, all_timestamps)

        # Build angle data for ALL configurations (1 to max_config)
        generated_angle_series_all: dict[int, dict[str, dict[str, float]]] = {}

        # First, collect all generated points per config
        all_generated_pts_per_config: dict[int, dict[tuple[int, int], np.ndarray]] = {}  # config_num -> {(obj_id, local_idx) -> point}

        # From stored configs
        for config_data in angles_all_configs_list:
            config_num = config_data.get("config_num", 1)
            if config_num not in all_generated_pts_per_config:
                all_generated_pts_per_config[config_num] = {}
            # Note: data is stored as "points" not "successful_points"
            points_list = config_data.get("points", config_data.get("successful_points", []))
            for sp in points_list:
                orig_idx = sp.get("original_parent_idx", sp.get("parent_idx", 0))
                obj_id, local_idx, _ = get_object_info_for_flat_idx(orig_idx)
                if obj_id != -1:
                    all_generated_pts_per_config[config_num][(obj_id, local_idx)] = np.array(sp["point"])

        # From current animation state (if not yet in all_configs)
        current_config_num = st.session_state.get("anim_current_config", 1)
        if current_config_num not in all_generated_pts_per_config:
            all_generated_pts_per_config[current_config_num] = {}
        for sp in angles_successful_points:
            orig_idx = sp.get("original_parent_idx", sp.get("parent_idx", 0))
            obj_id, local_idx, _ = get_object_info_for_flat_idx(orig_idx)
            if obj_id != -1:
                all_generated_pts_per_config[current_config_num][(obj_id, local_idx)] = np.array(sp["point"])

        # Get the maximum config number to ensure we have all configs from 1 to max
        config_nums_list: list[int] = [int(k) for k in all_generated_pts_per_config.keys()]
        max_config_num: int = max(config_nums_list) if config_nums_list else 1

        # Build angle series for each config from 1 to max_config_num
        for config_num in range(1, max_config_num + 1):
            # Build generated points for this config
            gen_points_dict: dict[int, np.ndarray] = {}
            gen_vals_dict: dict[int, np.ndarray] = {}

            # Start with original points
            for o_id in all_points_plot.keys():
                gen_points_dict[o_id] = all_points_plot[o_id].copy()
                gen_vals_dict[o_id] = all_vals_plot[o_id].copy()

            # Apply generated points for this config (if any)
            if config_num in all_generated_pts_per_config:
                for (obj_id, local_idx), gen_pt in all_generated_pts_per_config[config_num].items():
                    if obj_id in gen_points_dict:
                        gen_points_dict[obj_id][local_idx] = gen_pt

            # Calculate angles for this config
            if gen_points_dict:
                generated_angle_series_all[config_num] = build_angle_series_from_points(
                    gen_points_dict, gen_vals_dict, all_timestamps
                )

        # Build the plot: X-axis = Configuration number (1, 2, 3, ...), Y-axis = Angle (0-360°)
        # Each line = one vector pair (e.g., k0→k1), showing its angle across all configurations
        fig_angles = go.Figure()

        colors_plotly = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        # Get all unique vector pairs from the original angle series
        # Each series_name is like "k0→k1" or "k0→l0"
        all_vector_pairs = sorted(original_angle_series.keys())

        # Get all config numbers and sort them (starting from 1)
        # Explicit list comprehension ensures type clarity for Pylance
        all_config_nums: list[int] = sorted([int(k) for k in generated_angle_series_all.keys()])

        # Store stats for each vector pair
        vector_pair_stats: dict[str, dict[str, float]] = {}

        # For each vector pair, collect angles across all configurations
        for idx, vector_pair in enumerate(all_vector_pairs):
            x_vals: list[int] = []  # Config numbers (1, 2, 3, ...)
            y_vals: list[float] = []  # Angles for this vector pair (0-360°)

            # Get angle from each generated configuration
            for config_num in all_config_nums:
                if config_num in generated_angle_series_all:
                    gen_series = generated_angle_series_all[config_num]
                    if vector_pair in gen_series:
                        # Get the first (and should be only) angle value for this vector pair
                        angles_dict = gen_series[vector_pair]
                        if angles_dict:
                            # Take the first angle value (there should be one per vector pair)
                            angle_val = list(angles_dict.values())[0]
                            # Convert from -180/180 to 0/360 range
                            if angle_val < 0:
                                angle_val = angle_val + 360
                            # Convert to 0-360 range first, then take remainder of 90
                            angle_mod90 = angle_val % 90
                            x_vals.append(config_num)
                            y_vals.append(angle_mod90)

            # Calculate stats for this vector pair
            if len(y_vals) >= 1:
                avg_angle: float = float(np.mean(y_vals))
                std_angle: float = float(np.std(y_vals)) if len(y_vals) > 1 else 0.0
                vector_pair_stats[vector_pair] = {
                    "avg": avg_angle,
                    "std": std_angle,
                    "count": len(y_vals)
                }

            # Only add trace if we have data
            if len(x_vals) >= 1:
                # Show k0→k1 by default, hide others (click legend to show)
                is_visible = (vector_pair == "k0→k1")

                fig_angles.add_trace(go.Scatter(
                    name=f"{vector_pair} (avg={avg_angle:.1f}°, σ={std_angle:.1f}°)",
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    line=dict(color=colors_plotly[idx % len(colors_plotly)], width=2),
                    marker=dict(size=8),
                    visible=True if is_visible else "legendonly"  # Hide by default, show in legend
                ))

        # Determine x-axis range
        min_config = min([int(c) for c in all_config_nums]) if all_config_nums else 1  # type: ignore[type-var]
        max_config = max([int(c) for c in all_config_nums]) if all_config_nums else 1  # type: ignore[type-var]

        # Get the search strategy (exponential, linear, binary)
        search_strategy = st.session_state.get("cfg_strategy", "exponential")

        # Get the PDP variants (fundamental, rough, buffer, etc.)
        pdp_variants_list = st.session_state.get("anim_pdp_variants_list", [])
        if not pdp_variants_list:
            pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
        pdp_variants_str = ", ".join(pdp_variants_list) if pdp_variants_list else "fundamental"

        fig_angles.update_layout(
            title=f"Vector Angles (mod 90°) | Strategy: {search_strategy} | PDP Variants: {pdp_variants_str}",
            xaxis_title="Configuration",
            yaxis_title="Angle mod 90° (degrees)",
            height=800,  # Twice as high
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                itemclick="toggle",  # Click to toggle visibility
                itemdoubleclick="toggleothers"  # Double-click to isolate
            ),
            xaxis=dict(
                range=[0, max_config + 1],
                tickmode='linear',
                tick0=0,
                dtick=1 if max_config <= 20 else 2,
                fixedrange=True  # Disable zoom on x-axis
            ),
            yaxis=dict(
                range=[0, 90],
                tickvals=[0, 15, 30, 45, 60, 75, 90],
                gridcolor='lightgray',
                fixedrange=True  # Disable zoom on y-axis
            ),
            dragmode=False  # Disable all drag interactions (pan, zoom)
        )

        # Display without zoom/pan controls
        st.plotly_chart(fig_angles, use_container_width=True, config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False})

        # Display statistics summary
        st.subheader("ðŸ“ˆ Angle Statistics per Vector Pair")
        stats_data = []
        for vp, stats in sorted(vector_pair_stats.items()):
            stats_data.append({
                "Vector Pair": vp,
                "Average (°)": f"{stats['avg']:.2f}",
                "Std Dev (°)": f"{stats['std']:.2f}",
                "Count": int(stats['count'])
            })
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        # Show angle values in expandable table
        with st.expander("ðŸ“Š Angle Values (degrees)"):
            angle_table_data = []
            # Generated angles per config
            for config_num in sorted(generated_angle_series_all.keys()):
                gen_angle_series = generated_angle_series_all[config_num]
                for vector_pair in sorted(gen_angle_series.keys()):
                    angles_dict = gen_angle_series[vector_pair]
                    for ts_label, angle in angles_dict.items():
                        angle_mod90 = angle % 90
                        angle_table_data.append({
                            "Config": config_num,
                            "Vector": vector_pair,
                            "Angle (°)": f"{angle:.2f}",
                            "Mod 90°": f"{angle_mod90:.2f}"
                        })
            if angle_table_data:
                st.dataframe(pd.DataFrame(angle_table_data), use_container_width=True)
    else:
        st.info("Generate configurations to see angle comparisons between original and generated point configurations.")


with tab_slicing:
    st.markdown("### 🔍 Interactive Time Slicing")
    st.markdown("""
    Use the slider below to explore the data at a specific point in time.
    The plot updates in real-time as you move the slider.
    """)

    if _df_all is not None and not _df_all.empty:
        _slice_c = int(st.session_state.get("cfg_c", 0))
        _slice_df_c = _df_all[_df_all["c"] == _slice_c]

        if not _slice_df_c.empty:
            _t_vals = sorted(_slice_df_c["t"].unique())
            _t_min = float(_t_vals[0])
            _t_max = float(_t_vals[-1])

            if len(_t_vals) > 1:
                _selected_t = st.slider(
                    "Timestamp",
                    min_value=_t_min,
                    max_value=_t_max,
                    value=_t_min,
                    step=float(_t_vals[1] - _t_vals[0]) if len(_t_vals) > 1 else 1.0,
                    key="slicing_time_slider",
                )

                # Filter data at selected timestamp
                _slice_df_t = _slice_df_c[_slice_df_c["t"] == _selected_t]

                _fig_slice = go.Figure()

                # Add lane markings
                add_lane_markings_to_figure(_fig_slice, _slice_c, XLIM, YLIM)

                _obj_ids = sorted(_slice_df_t["o"].unique())
                for _i_obj, _oid in enumerate(_obj_ids):
                    _obj_df = _slice_df_t[_slice_df_t["o"] == _oid]
                    _fig_slice.add_trace(go.Scatter(
                        x=_obj_df["x"].values,
                        y=_obj_df["y"].values,
                        mode="markers+text",
                        marker=dict(size=12, color=OBJECT_COLORS_PLOTLY[_i_obj % len(OBJECT_COLORS_PLOTLY)]),
                        text=[OBJECT_LABELS[_i_obj % len(OBJECT_LABELS)]],
                        textposition="top center",
                        name=f"Object {OBJECT_LABELS[_i_obj % len(OBJECT_LABELS)]}",
                    ))

                # Also show trailing trajectory (all timestamps up to selected)
                _slice_df_hist = _slice_df_c[_slice_df_c["t"] <= _selected_t]
                for _i_obj, _oid in enumerate(sorted(_slice_df_hist["o"].unique())):
                    _hist = _slice_df_hist[_slice_df_hist["o"] == _oid].sort_values("t")
                    if len(_hist) > 1:
                        _fig_slice.add_trace(go.Scatter(
                            x=_hist["x"].values,
                            y=_hist["y"].values,
                            mode="lines",
                            line=dict(
                                color=OBJECT_COLORS_PLOTLY[_i_obj % len(OBJECT_COLORS_PLOTLY)],
                                width=2,
                                dash="dot",
                            ),
                            opacity=0.5,
                            showlegend=False,
                            name=f"Trail {OBJECT_LABELS[_i_obj % len(OBJECT_LABELS)]}",
                        ))

                _fig_slice.update_layout(
                    title=f"Configuration {_slice_c} at t = {_selected_t}",
                    xaxis=dict(
                        title="x",
                        range=[XLIM[0] - 2, XLIM[1] + 2],
                    ),
                    yaxis=dict(
                        title="y",
                        range=[YLIM[0] - 2, YLIM[1] + 2],
                    ),
                    height=600,
                    showlegend=True,
                )

                st.plotly_chart(_fig_slice, use_container_width=True)

                # Show data table for selected timestamp
                with st.expander("Data at selected timestamp", expanded=False):
                    st.dataframe(_slice_df_t, use_container_width=True)
            else:
                st.warning("Only one timestamp available \u2014 nothing to slice.")
        else:
            st.info("No data for the selected configuration.")
    else:
        st.info("Load a dataset to use interactive slicing.")

