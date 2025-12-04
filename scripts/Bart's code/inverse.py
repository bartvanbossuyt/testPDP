# -*- coding: utf-8 -*- 
# inverse.py
# Streamlit app with an academic look, two columns with strictly square axes.
# Left: line + points from CSV (c=11, t∈{0,1,2}, o=0). Right: identical axes, initially empty.
# CSV may start with literally: "header: c,t,o,x,y" → that line is skipped.
# Axis labels: d₁ and d₂; point labels: k₀, k₁, k₂ (blue, smaller).
# maxdist = max(||k0-k1||, ||k1-k2||); axes get at least maxdist margin to every border.

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

import plotly.graph_objects as go
import plotly.express as px

# ============= Coordinate Precision Settings =============
# Change these values to adjust coordinate display precision throughout the app
COORD_DISPLAY_PRECISION = 2   # Decimal places for UI display (hover text, status messages)
COORD_CSV_PRECISION = 3       # Decimal places for CSV export (3 digits after decimal point)

# Type definition for successful point data in the search process
class SuccessfulPoint(TypedDict):
    point: np.ndarray              # Coordinates of the generated point
    parent_idx: int                # Index in all_pts (may be a generated point)
    parent_point: np.ndarray       # Actual coordinates of the parent point
    original_parent_idx: int       # Index of the ORIGINAL point (k0, k1, k2, l0, l1, l2)
    iteration: int                 # Iteration number when this point was accepted

# ============= PDP Core Functions (from N_PDP.py) =============
def compute_inequality_matrix(points: np.ndarray, dimension: int, roughness: float = 0.0) -> np.ndarray:
    """
    Compute PDP inequality matrix for a set of points along one dimension.
    
    This follows the exact logic from N_PDP.py:
    - Value 0: point j > point i (beyond roughness)
    - Value 1: |point j - point i| <= roughness (equal within tolerance)
    - Value 2: point j < point i (beyond roughness)
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        dimension: 0 for x, 1 for y
        roughness: tolerance for equality (default 0.0)
    
    Returns:
        (N, N) inequality matrix
    """
    n = len(points)
    inequality_matrix = np.zeros((n, n))
    
    values = points[:, dimension]
    
    for i in range(n):
        for j in range(n):
            diff = values[j] - values[i]
            if abs(diff) <= roughness:
                inequality_matrix[i, j] = 1  # Equal (within roughness)
            elif diff > roughness:
                inequality_matrix[i, j] = 0  # Greater than
            else:
                inequality_matrix[i, j] = 2  # Less than
    
    return inequality_matrix

def compare_inequality_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> bool:
    """
    Compare two inequality matrices for equality.
    
    Returns True if matrices are identical (same PDP pattern).
    """
    return np.array_equal(matrix1, matrix2)

def apply_buffer_transformation(points: np.ndarray, buffer_x: float, buffer_y: float) -> np.ndarray:
    """
    Apply buffer transformation to a set of points.
    
    This creates 5 variants of each point:
    - Original point * 5 + 0: x - buffer_x
    - Original point * 5 + 1: x + buffer_x
    - Original point * 5 + 2: no buffer in x (original x)
    - Original point * 5 + 3: y - buffer_y
    - Original point * 5 + 4: y + buffer_y
    
    The point index is expanded by 5x to accommodate all buffer variants.
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        buffer_x: buffer distance in x-direction
        buffer_y: buffer distance in y-direction
    
    Returns:
        (5*N, 2) array with buffered points
    """
    n = len(points)
    buffered = np.zeros((5 * n, 2))
    
    for i, (x, y) in enumerate(points):
        base_idx = i * 5
        # Variant 0: x - buffer_x
        buffered[base_idx + 0] = [x - buffer_x, y]
        # Variant 1: x + buffer_x
        buffered[base_idx + 1] = [x + buffer_x, y]
        # Variant 2: no buffer in x
        buffered[base_idx + 2] = [x, y]
        # Variant 3: y - buffer_y
        buffered[base_idx + 3] = [x, y - buffer_y]
        # Variant 4: y + buffer_y
        buffered[base_idx + 4] = [x, y + buffer_y]
    
    return buffered

# ============= Page configuration =============
st.set_page_config(
    page_title="pdp inverse",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============= Authentication =============
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Use .get() to safely access the password key, avoiding KeyError if not present
        entered_password = st.session_state.get("password", "")
        if entered_password == "pdp2025":
            st.session_state["password_correct"] = True
            # Safely delete password from state if it exists
            if "password" in st.session_state:
                del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

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
        st.error("😕 Password incorrect")
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

# ---------- Wrapper: Series -> Series ----------
def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, coercing bad values to NaN (Pylance-friendly)."""
    out = pd.to_numeric(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        s, errors="coerce"
    )
    return out

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
            if first_line.startswith("header:"):
                # Skip header line
                from io import StringIO
                _df_all = pd.read_csv(StringIO("\n".join(lines[1:])), header=None, names=names)
            else:
                from io import StringIO
                uploaded_file.seek(0)
                _df_all = pd.read_csv(StringIO(content))
                if not set(names).issubset(_df_all.columns):
                    _df_all = pd.read_csv(StringIO(content), header=None, names=names)
            
            # Clean the dataframe
            for col in names:
                _df_all[col] = pd.to_numeric(_df_all[col], errors="coerce")
            _df_all = _df_all.dropna(subset=names)
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
        num_timestamps = st.number_input(
            "Number of timestamps",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key="rand_num_timestamps",
            help="Number of timestamps (time moments). Each point will have a position at each timestamp."
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
            
            if edited_p_df is not None:
                edited_coords[p] = list(zip(edited_p_df["x"].tolist(), edited_p_df["y"].tolist()))
    
    # Update session state with edited values
    if edited_coords:
        st.session_state["random_all_coords"] = edited_coords
    
    # Build the dataframe from edited coordinates
    all_coords_final = st.session_state.get("random_all_coords", {})
    
    rows = []
    for p, coords in all_coords_final.items():
        for t_idx, (x, y) in enumerate(coords):
            rows.append({"c": 0, "t": t_idx, "o": p, "x": x, "y": y})
    
    _df_all = pd.DataFrame(rows)
    available_configs = [0]
    
    # Save configuration button
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
                # Create CSV content with header
                csv_content = "header: c,t,o,x,y\n"
                for _, row in _df_all.iterrows():
                    csv_content += f"{int(row['c'])},{int(row['t'])},{int(row['o'])},{row['x']:.{COORD_CSV_PRECISION}f},{row['y']:.{COORD_CSV_PRECISION}f}\n"
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(csv_content)
                
                st.success(f"Configuration saved to: {save_path}")
            except Exception as e:
                st.error(f"Error saving file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Also provide download button
    csv_download = "header: c,t,o,x,y\n"
    for _, row in _df_all.iterrows():
        csv_download += f"{int(row['c'])},{int(row['t'])},{int(row['o'])},{row['x']:.{COORD_CSV_PRECISION}f},{row['y']:.{COORD_CSV_PRECISION}f}\n"
    
    st.download_button(
        label="Download Configuration as CSV",
        data=csv_download,
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
    # Calculate bounds from loaded data (use default config if multiple exist)
    # For preset/upload: use first config or config 11 if available
    _default_config = 11 if 11 in available_configs else available_configs[0]
    _df_for_bounds = _df_all[_df_all["c"] == _default_config]

    _data_min_x = float(_df_for_bounds["x"].min())
    _data_max_x = float(_df_for_bounds["x"].max())
    _data_min_y = float(_df_for_bounds["y"].min())
    _data_max_y = float(_df_for_bounds["y"].max())

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

# Use cfg_coord_ widget keys as the source of truth (these are what widgets actually use)
# Fall back to coord_ keys, then to auto-calculated values
_coord_min_x_val = st.session_state.get("cfg_coord_min_x", st.session_state.get("coord_min_x", _auto_min_x))
_coord_max_x_val = st.session_state.get("cfg_coord_max_x", st.session_state.get("coord_max_x", _auto_max_x))
_coord_min_y_val = st.session_state.get("cfg_coord_min_y", st.session_state.get("coord_min_y", _auto_min_y))
_coord_max_y_val = st.session_state.get("cfg_coord_max_y", st.session_state.get("coord_max_y", _auto_max_y))

with bounds_col1:
    coord_min_x = st.number_input(
        "Min X",
        value=_coord_min_x_val,
        step=10.0,
        key="cfg_coord_min_x",
        help="Minimum X coordinate. Generated points cannot have x < this value."
    )
with bounds_col2:
    coord_max_x = st.number_input(
        "Max X",
        value=_coord_max_x_val,
        step=10.0,
        key="cfg_coord_max_x",
        help="Maximum X coordinate. Generated points cannot have x > this value."
    )
with bounds_col3:
    coord_min_y = st.number_input(
        "Min Y",
        value=_coord_min_y_val,
        step=10.0,
        key="cfg_coord_min_y",
        help="Minimum Y coordinate. Generated points cannot have y < this value."
    )
with bounds_col4:
    coord_max_y = st.number_input(
        "Max Y",
        value=_coord_max_y_val,
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

# ============= Auto Detect Coordinate Bounds Button =============
# This button recalculates the coordinate bounds based on the currently selected
# configuration (c) and timestamp window. Useful when switching between configurations
# or changing the number of timestamps, as parent points may fall outside the current bounds.
if data_source != "Create random configuration":
    # Only show Auto Detect button when using preset or uploaded data
    st.markdown('<div class="auto-detect-bounds-wrapper" style="margin-top:0.5rem;">', unsafe_allow_html=True)
    if st.button("🔍 Auto Detect Coordinate Bounds", key="btn_auto_detect_bounds", 
                 help="Recalculate axis bounds based on currently selected configuration (c) and timestamp window. Use this when parent points fall outside the visible area after changing settings."):
        # Get currently selected configuration and timestamps from session state
        _detect_c = int(st.session_state.get("cfg_c", available_configs[0]))
        _detect_k = int(st.session_state.get("cfg_k", 3))  # Number of timestamps
        _detect_start_t = st.session_state.get("cfg_start_t", None)
        
        # Get time values for the selected configuration
        _detect_t_k = sorted(_df_all[(_df_all["c"] == _detect_c) & (_df_all["o"] == 0)]["t"].unique().tolist())
        _detect_t_l = sorted(_df_all[(_df_all["c"] == _detect_c) & (_df_all["o"] == 1)]["t"].unique().tolist())
        _detect_t_common = [t for t in _detect_t_k if t in _detect_t_l]
        
        # Determine the timestamp window
        if _detect_start_t is not None and _detect_start_t in _detect_t_common:
            _detect_start_idx = _detect_t_common.index(_detect_start_t)
        else:
            _detect_start_idx = 0
        _detect_end_idx = min(_detect_start_idx + _detect_k, len(_detect_t_common))
        _detect_ts_window = _detect_t_common[_detect_start_idx:_detect_end_idx]
        
        # Filter dataframe to selected configuration and timestamp window
        _df_filtered = _df_all[
            (_df_all["c"] == _detect_c) & 
            (_df_all["t"].isin(_detect_ts_window))
        ]
        
        if len(_df_filtered) > 0:
            # Calculate bounds from filtered data
            _new_min_x = float(_df_filtered["x"].min())
            _new_max_x = float(_df_filtered["x"].max())
            _new_min_y = float(_df_filtered["y"].min())
            _new_max_y = float(_df_filtered["y"].max())
            
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
            # For small ranges (< 10), round to nearest 1
            # For medium ranges (10-100), round to nearest 5
            # For large ranges (> 100), round to nearest 10
            def smart_round_min(val: float, data_range: float) -> float:
                """Round down to a nice value based on data range."""
                if data_range < 10:
                    return float(np.floor(val))  # Round to nearest 1
                elif data_range < 100:
                    return float(np.floor(val / 5) * 5)  # Round to nearest 5
                else:
                    return float(np.floor(val / 10) * 10)  # Round to nearest 10
            
            def smart_round_max(val: float, data_range: float) -> float:
                """Round up to a nice value based on data range."""
                if data_range < 10:
                    return float(np.ceil(val))  # Round to nearest 1
                elif data_range < 100:
                    return float(np.ceil(val / 5) * 5)  # Round to nearest 5
                else:
                    return float(np.ceil(val / 10) * 10)  # Round to nearest 10
            
            _new_min_x = smart_round_min(_new_min_x, _new_range_x)
            _new_max_x = smart_round_max(_new_max_x, _new_range_x)
            _new_min_y = smart_round_min(_new_min_y, _new_range_y)
            _new_max_y = smart_round_max(_new_max_y, _new_range_y)
            
            # Store pending bounds - these will be applied on next rerun BEFORE widgets are created
            st.session_state["_pending_bounds_update"] = True
            st.session_state["_pending_bounds"] = {
                "min_x": _new_min_x,
                "max_x": _new_max_x,
                "min_y": _new_min_y,
                "max_y": _new_max_y
            }
            
            # Set a special hash to prevent auto-recalculation from overwriting user-triggered detection
            st.session_state["_bounds_data_hash"] = f"auto_detect_{_detect_c}_{_new_min_x:.2f}_{_new_max_x:.2f}_{_new_min_y:.2f}_{_new_max_y:.2f}"
            
            st.success(f"Bounds updated for config {_detect_c}: X=[{_new_min_x:.0f}, {_new_max_x:.0f}], Y=[{_new_min_y:.0f}, {_new_max_y:.0f}]")
            # Rerun to apply the new bounds
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

sc1, sc2, sc3 = st.columns([1,1,2], gap="small")
with sc1:
    # Configuration selector for c (only show if multiple configs available)
    if len(available_configs) > 1:
        selected_c = st.selectbox(
            "Configuration (c)",
            options=available_configs,
            index=available_configs.index(11) if 11 in available_configs else 0,
            key="cfg_c",
            help="Select which configuration to use as the reference. Each configuration has its own set of k and l trajectories."
        )
    else:
        selected_c = available_configs[0]
        st.markdown(f"**Configuration:** {selected_c}")
selected_c_int: int = int(selected_c) if selected_c is not None else int(available_configs[0])

# Time values for k and l in the selected configuration
_t_k = sorted(_df_all[(_df_all["c"] == selected_c_int) & (_df_all["o"] == 0)]["t"].unique().tolist())  # type: ignore
_t_l = sorted(_df_all[(_df_all["c"] == selected_c_int) & (_df_all["o"] == 1)]["t"].unique().tolist())  # type: ignore
_t_common = [t for t in _t_k if t in _t_l]
if not _t_common:
    st.error(f"No overlapping t-values for c={selected_c} between o=0 and o=1.")
    st.stop()

n_timepoints = len(_t_common)
default_window = min(3, n_timepoints)
with sc2:
    # Number of timestamps in the sliding time window (dropdown instead of slider)
    if n_timepoints > 1:
        timestamp_options = list(range(1, n_timepoints + 1))
        default_idx = timestamp_options.index(default_window) if default_window in timestamp_options else 0
        num_timestamps = st.selectbox(
            "Number of timestamps",
            options=timestamp_options,
            index=default_idx,
            key="cfg_k",
            help="Select the number of timestamps to include in the analysis window."
        )
    else:
        st.markdown("**Number of timestamps**")
        st.code(str(n_timepoints))
        num_timestamps = n_timepoints

with sc3:
    # Starting t value of the window (dropdown instead of slider)
    valid_start_count = max(1, n_timepoints - num_timestamps + 1)
    valid_starts = _t_common[:valid_start_count]
    
    if len(valid_starts) > 1:
        start_t = st.selectbox(
            "Starting time (t)",
            options=valid_starts,
            index=0,
            key="cfg_start_t",
            help="Select the starting timestamp for the analysis window."
        )
    else:
        st.markdown("**Starting time (t)**")
        st.code(str(valid_starts[0]))
        start_t = valid_starts[0]

# Strategy, iterations, configurations
st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
sc4, sc5, sc6 = st.columns([1,1,1], gap="small")
with sc4:
    # Choice of search strategy for generating new configurations
    strategy = st.radio(
        "Strategy",
        options=["exponential", "binary"],
        index=0,
        key="cfg_strategy",
        help="Choose the search strategy for configuration generation."
    )
with sc5:
    # Number of iterations per configuration (used by both animate and generate)
    num_iterations = st.number_input(
        "Number of iterations",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
        key="cfg_iterations",
        help="Number of points to generate per configuration. Each iteration replaces one original point with a new generated point that preserves the PDP pattern."
    )
with sc6:
    # How many configurations to generate (used by both animate and generate)
    num_configs = st.number_input(
        "Number of configurations",
        min_value=1,
        max_value=1000,
        value=1,
        step=1,
        key="cfg_num_configs",
        help="How many independent configurations to create when clicking 'Generate configurations'. Each configuration is a complete new set of generated points (all k and l points) that preserves the original PDP pattern. Use this for batch generation without animation."
    )

# Point Selection Mode and Movement Direction
st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
st.markdown("**Point Selection (per iteration)**")

ps_col1, ps_col2 = st.columns([1, 1], gap="small")
with ps_col1:
    point_selection_mode = st.selectbox(
        "Selection mode",
        options=["Single point", "Multiple random points", "Group pattern (Np + Nt)"],
        index=0,
        key="cfg_point_selection_mode",
        help="""How to select points to move in each iteration:
        
• **Single point**: Move 1 random point per iteration (default, current behavior)

• **Multiple random points**: Move N randomly selected points together

• **Group pattern (Np + Nt)**: Move groups of consecutive timestamps. Select N objects (p) and T consecutive timestamps (t) per object. The starting timestamp is random, then T consecutive timestamps are used for each object."""
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
elif point_selection_mode == "Group pattern (Np + Nt)":
    gp_col1, gp_col2 = st.columns([1, 1], gap="small")
    with gp_col1:
        group_num_objects = st.number_input(
            "Objects (p)",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            key="cfg_group_num_objects",
            help="Number of different objects to include in the group. Objects are randomly selected."
        )
    with gp_col2:
        group_num_timestamps = st.number_input(
            "Consecutive timestamps (t)",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            key="cfg_group_num_timestamps",
            help="Number of consecutive timestamps per object. Starting timestamp is random, then t consecutive timestamps are used."
        )

# PDP Variant Selection (Multiple variants)
st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
st.markdown("**PDP Variant Configuration** (select one or more)")

# Multi-select for PDP variants
pdp_variants_selected = st.multiselect(
    "PDP Variants to calculate",
    options=["fundamental", "buffer", "rough", "bufferrough"],
    default=["fundamental"],
    key="cfg_pdp_variants",
    help="""Select PDP variants for configuration generation:

• **fundamental**: Basic PDP with N×N inequality matrix. Two configurations match if ALL pairwise orderings are identical.

• **buffer**: Expands each point to 5 variants (±buffer in x and y directions). Creates 5N×5N matrix. More restrictive - requires all buffer variants to match.

• **rough**: Adds equality tolerance. Points within roughness distance are considered EQUAL (matrix value 1). More permissive - allows small variations.

• **bufferrough**: Combines buffer expansion AND roughness tolerance. 5N×5N matrix with fuzzy equality."""
)

# Show parameter inputs if any variant needs them
needs_buffer = any(v in ["buffer", "bufferrough"] for v in pdp_variants_selected)
needs_rough = any(v in ["rough", "bufferrough"] for v in pdp_variants_selected)

if needs_buffer or needs_rough:
    st.markdown("**Parameters for selected variants:**")
    param_col1, param_col2 = st.columns([1, 1], gap="small")
    
    with param_col1:
        if needs_buffer:
            buffer_x = st.number_input(
                "Buffer X",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
                key="cfg_buffer_x",
                help="Buffer distance in x-direction. Each point is expanded to 5 variants: (x±buffer_x, y) and (x, y±buffer_y) plus the original. The PDP comparison uses these 5N expanded points, creating a 5N×5N inequality matrix. During animation, buffer points are shown as PURPLE X markers connected by dashed lines."
            )
            buffer_y = st.number_input(
                "Buffer Y",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                key="cfg_buffer_y",
                help="Buffer distance in y-direction. Each point is expanded to 5 variants: (x±buffer_x, y) and (x, y±buffer_y) plus the original. The PDP comparison uses these 5N expanded points, creating a 5N×5N inequality matrix. During animation, buffer points are shown as PURPLE X markers connected by dashed lines."
            )
        else:
            buffer_x = 0.0
            buffer_y = 0.0
    
    with param_col2:
        if needs_rough:
            rough_x = st.number_input(
                "Roughness X",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key="cfg_rough_x",
                help="Equality tolerance in x-direction. Two x-coordinates are considered EQUAL if their difference is ≤ roughness_x. This creates 'fuzzy' equality zones in the inequality matrix (value 1 instead of 0 or 2). During animation, the roughness zone is shown as a GREEN semi-transparent rectangle around the candidate point."
            )
            rough_y = st.number_input(
                "Roughness Y",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key="cfg_rough_y",
                help="Equality tolerance in y-direction. Two y-coordinates are considered EQUAL if their difference is ≤ roughness_y. This creates 'fuzzy' equality zones in the inequality matrix (value 1 instead of 0 or 2). During animation, the roughness zone is shown as a GREEN semi-transparent rectangle around the candidate point."
            )
        else:
            rough_x = 0.0
            rough_y = 0.0
else:
    buffer_x = 0.0
    buffer_y = 0.0
    rough_x = 0.0
    rough_y = 0.0

# External (fixed) reference points
st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
st.markdown("**External Reference Points**")

use_external_points = st.checkbox(
    "Use external reference points",
    value=st.session_state.get("use_external_points", False),
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
    
    # Initialize external points if not present (now only x, y - no timestamp)
    if "external_points" not in st.session_state:
        st.session_state["external_points"] = [(0.0, 0.0)]  # Default: one point at origin
    
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

# Animation settings
st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
st.markdown("**Animation Settings**")

anim_mode = st.radio(
    "Animation mode",
    options=["Auto-advance", "Manual step-by-step", "Manual iteration-by-iteration", "Manual config-by-config"],
    index=0,
    horizontal=True,
    key="cfg_anim_mode",
    help=("Choose how the animation advances:\n"
          "• **Auto-advance**: Automatically moves to the next step after a set time interval.\n"
          "• **Manual step-by-step**: Click to advance each search step manually.\n"
          "• **Manual iteration-by-iteration**: Click to complete one full iteration (all search steps until point is placed).\n"
          "• **Manual config-by-config**: Click to complete one full configuration (all iterations).")
)

if anim_mode == "Auto-advance":
    sc_wait, _, _ = st.columns([1, 1, 1], gap="small")
    with sc_wait:
        wait_interval_ms = st.selectbox(
            "Wait interval (ms)",
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
    
    # Determine button labels based on mode
    if is_manual_step_mode:
        prev_label = "◀ Previous step"
        next_label = "▶ Next step"
        prev_help = "Click to go back to the previous animation step."
        next_help = "Click to advance the animation by one step."
        generate_help = "Start generating configurations step-by-step. Click 'Next step' to advance each step manually."
    elif is_manual_iteration_mode:
        prev_label = "◀ Previous iteration"
        next_label = "▶ Next iteration"
        prev_help = "Click to go back to the previous iteration."
        next_help = "Click to complete one full iteration (all search steps until a point is placed)."
        generate_help = "Start generating configurations. Click 'Next iteration' to complete one iteration at a time."
    else:  # is_manual_config_mode
        prev_label = "◀ Previous config"
        next_label = "▶ Next config"
        prev_help = "Click to go back to the previous configuration."
        next_help = "Click to complete one full configuration (all iterations)."
        generate_help = "Start generating configurations. Click 'Next config' to complete one configuration at a time."
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 0.6], gap="small")
    # No "Generate without animation" button in manual mode
    generate_btn = False  # Set to False so the generate_btn handler doesn't trigger
    with col_btn1:
        animate_btn = st.button(
            "Generate", 
            key="btn_animate",
            help=generate_help
        )
    with col_btn2:
        # Show "Previous" button - enabled only when animation is running and there is history
        anim_history = st.session_state.get("anim_state_history", [])
        prev_step_clicked = st.button(
            prev_label, 
            key="btn_prev_step", 
            disabled=not anim_is_running or len(anim_history) == 0,
            help=prev_help + " Only active when animation is running and there is history to go back to."
        )
        if prev_step_clicked and anim_is_running and len(anim_history) > 0:
            # Pop the last state from history and restore it
            previous_state = anim_history.pop()
            st.session_state["anim_state_history"] = anim_history
            # Restore all animation state variables from the previous state
            for key, value in previous_state.items():
                st.session_state[key] = value
            st.rerun()
    with col_btn3:
        # Show "Next" button - enabled only when animation is running
        next_step_clicked = st.button(
            next_label, 
            key="btn_next_step", 
            type="primary", 
            disabled=not anim_is_running,
            help=next_help + " Only active when animation is running."
        )
        if next_step_clicked and anim_is_running:
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
            "⟲ Reset",
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
            "⟲ Reset",
            key="btn_reset_auto",
            disabled=not reset_btn_should_be_enabled,
            help="Halt the animation and reset all graphs to their initial values. Clears all generated points and search state."
        )
        st.markdown('</div>', unsafe_allow_html=True)

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

# ============= Data window (select subset of k and l) ============
# Extract points from _df_all (works for all data sources: preset, uploaded, random)
def extract_points_from_df(df: pd.DataFrame, o_val: int, c_val: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract points and t-values from DataFrame for a given object and configuration."""
    sel = df[(df["c"] == c_val) & (df["o"] == o_val)].sort_values("t").reset_index(drop=True)
    if sel.empty:
        return np.array([]).reshape(0, 2), np.array([])
    pts = sel[["x", "y"]].values.astype(float)
    ts = sel["t"].values.astype(float)
    return pts, ts

# Get all unique object IDs in the selected configuration
all_object_ids = sorted(_df_all[_df_all["c"] == selected_c_int]["o"].unique().tolist())

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
selected_ts_window = _t_common[start_idx:end_idx]
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
    pts_list = []
    ts_list = []
    obj_ids = []
    local_indices = []
    is_fixed = []
    
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
    for ext_idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):
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
    """Get list of flat indices for movable (non-fixed) points only."""
    return [i for i in range(n_total_points) if not is_fixed_point(i)]

def select_points_for_iteration() -> list[int]:
    """
    Select points to move in this iteration based on the point selection mode.
    Returns a list of flat indices of points to move together.
    """
    movable_indices = get_movable_indices()
    if not movable_indices:
        return []
    
    point_selection_mode = st.session_state.get("cfg_point_selection_mode", "Single point")
    
    if point_selection_mode == "Single point":
        # Current default behavior: select one random point
        return [int(np.random.choice(movable_indices))]
    
    elif point_selection_mode == "Multiple random points":
        # Select N random points
        num_points = int(st.session_state.get("cfg_num_random_points", 2))
        num_points = min(num_points, len(movable_indices))  # Can't select more than available
        selected = list(np.random.choice(movable_indices, size=num_points, replace=False))
        return [int(idx) for idx in selected]
    
    elif point_selection_mode == "Group pattern (Np + Nt)":
        # Select N objects, each with T consecutive timestamps
        num_objects = int(st.session_state.get("cfg_group_num_objects", 2))
        num_timestamps = int(st.session_state.get("cfg_group_num_timestamps", 2))
        
        # Group movable indices by object
        indices_by_object: dict[int, list[tuple[int, float]]] = {}  # obj_id -> [(flat_idx, timestamp), ...]
        for flat_idx in movable_indices:
            o_id = all_obj_ids_flat[flat_idx]
            t = all_ts_flat[flat_idx]
            if o_id not in indices_by_object:
                indices_by_object[o_id] = []
            indices_by_object[o_id].append((flat_idx, t))
        
        # Sort by timestamp within each object
        for o_id in indices_by_object:
            indices_by_object[o_id].sort(key=lambda x: x[1])
        
        # Filter to objects with at least num_timestamps points
        valid_objects = [o_id for o_id, pts in indices_by_object.items() if len(pts) >= num_timestamps]
        
        if not valid_objects:
            # Fall back to single point if no valid objects
            return [int(np.random.choice(movable_indices))]
        
        # Select random objects
        num_objects = min(num_objects, len(valid_objects))
        selected_objects = list(np.random.choice(valid_objects, size=num_objects, replace=False))
        
        selected_indices = []
        for o_id in selected_objects:
            pts_list = indices_by_object[o_id]
            # Select random starting position that allows num_timestamps consecutive points
            max_start = len(pts_list) - num_timestamps
            if max_start < 0:
                continue
            start_pos = int(np.random.randint(0, max_start + 1))
            # Add consecutive points
            for i in range(num_timestamps):
                selected_indices.append(pts_list[start_pos + i][0])
        
        return [int(idx) for idx in selected_indices] if selected_indices else [int(np.random.choice(movable_indices))]
    
    # Default fallback
    return [int(np.random.choice(movable_indices))]

def generate_movement_vectors(selected_indices: list[int], base_distance: float) -> dict[int, tuple[float, float]]:
    """
    Generate movement vectors for selected points based on movement direction mode.
    Returns a dict mapping flat_idx -> (delta_x, delta_y)
    """
    if not selected_indices:
        return {}
    
    movement_direction = st.session_state.get("cfg_movement_direction", "Same direction")
    
    if movement_direction == "Same direction":
        # All points move with the same angle
        angle = float(np.random.uniform(0, 2 * np.pi))
        delta_x = base_distance * np.cos(angle)
        delta_y = base_distance * np.sin(angle)
        return {int(idx): (delta_x, delta_y) for idx in selected_indices}
    
    else:  # Random directions
        # Each point gets its own random angle
        vectors = {}
        for idx in selected_indices:
            angle = float(np.random.uniform(0, 2 * np.pi))
            delta_x = base_distance * np.cos(angle)
            delta_y = base_distance * np.sin(angle)
            vectors[int(idx)] = (delta_x, delta_y)
        return vectors

def scale_movement_vectors(vectors: dict[int, tuple[float, float]], scale: float) -> dict[int, tuple[float, float]]:
    """Scale all movement vectors by a factor (e.g., 0.5 to halve distances)."""
    return {int(idx): (dx * scale, dy * scale) for idx, (dx, dy) in vectors.items()}

def apply_movement_vectors(base_points: np.ndarray, vectors: dict[int, tuple[float, float]]) -> dict[int, np.ndarray]:
    """
    Apply movement vectors to base points and return new positions.
    Returns dict mapping flat_idx -> new_position (clipped to bounds)
    """
    new_positions = {}
    for idx, (dx, dy) in vectors.items():
        if 0 <= idx < len(base_points):
            new_x = base_points[idx, 0] + dx
            new_y = base_points[idx, 1] + dy
            # Clip to coordinate bounds
            new_x = np.clip(new_x, COORD_MIN_X, COORD_MAX_X)
            new_y = np.clip(new_y, COORD_MIN_Y, COORD_MAX_Y)
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
        # Calculate pairwise distances between all point groups
        all_pts = np.vstack(all_point_arrays)
        pairwise_dists = []
        for i in range(all_pts.shape[0]):
            for j in range(i + 1, all_pts.shape[0]):
                d = np.hypot(all_pts[i, 0] - all_pts[j, 0], 
                            all_pts[i, 1] - all_pts[j, 1])
                pairwise_dists.append(d)
        maxdist = max(pairwise_dists) if pairwise_dists else 10.0
    else:
        maxdist = 10.0  # Default fallback

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
COORD_MIN_X = float(st.session_state.get("coord_min_x", -50.0))
COORD_MAX_X = float(st.session_state.get("coord_max_x", 150.0))
COORD_MIN_Y = float(st.session_state.get("coord_min_y", -50.0))
COORD_MAX_Y = float(st.session_state.get("coord_max_y", 150.0))

# Compute axis limits for visualization: coordinate bounds + 10% margin
coord_width = COORD_MAX_X - COORD_MIN_X
coord_height = COORD_MAX_Y - COORD_MIN_Y

# Use 10% of each range as margin
margin_x = coord_width * 0.10
margin_y = coord_height * 0.10

# Make it square by using the larger dimension (including margins)
total_width = coord_width + 2 * margin_x
total_height = coord_height + 2 * margin_y
viz_side = max(total_width, total_height)

# Center of coordinate bounds
coord_cx = 0.5 * (COORD_MIN_X + COORD_MAX_X)
coord_cy = 0.5 * (COORD_MIN_Y + COORD_MAX_Y)

# Square limits with percentage-based margin for visualization
XLIM = (coord_cx - viz_side / 2.0, coord_cx + viz_side / 2.0)
YLIM = (coord_cy - viz_side / 2.0, coord_cy + viz_side / 2.0)

# ============= d1/d2 order strings (LaTeX) ============
def _format_t_subscript(tval: float) -> str:
    """Format t-value as an integer subscript if possible, otherwise as a float."""
    try:
        tnum = float(tval)
    except Exception:
        tnum = float(np.array(tval, dtype=float))
    return str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"

def make_d1_order_latex() -> str:
    """Return LaTeX describing the ordering in d1 (x-coordinate) for all objects."""
    entries: list[tuple[float, str]] = []
    for i, o_id in enumerate(sorted(all_points_plot.keys())):
        pts = all_points_plot[o_id]
        ts = all_vals_plot[o_id]
        label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
        for x, t in zip(pts[:, 0].tolist(), ts.tolist()):
            lbl = _format_t_subscript(t)
            entries.append((float(x), rf"{label}_{lbl}"))

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
    """Return LaTeX describing the ordering in d2 (y-coordinate) for all objects."""
    entries: list[tuple[float, str]] = []
    for i, o_id in enumerate(sorted(all_points_plot.keys())):
        pts = all_points_plot[o_id]
        ts = all_vals_plot[o_id]
        label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
        for y, t in zip(pts[:, 1].tolist(), ts.tolist()):
            lbl = _format_t_subscript(t)
            entries.append((float(y), rf"{label}_{lbl}"))

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
    entries.append((float(gen_pt[0]), rf"{parent_label}{parent_primes}_{lbl_parent}"))

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
            entries.append((float(latest_generated[flat_idx][0]), rf"{label}{primes}_{lbl}"))
        else:
            entries.append((float(pt[0]), rf"{label}{primes}_{lbl}"))

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
    entries.append((float(gen_pt[1]), rf"{parent_label}{parent_primes}_{lbl_parent}"))

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
            entries.append((float(latest_generated[flat_idx][1]), rf"{label}{primes}_{lbl}"))
        else:
            entries.append((float(pt[1]), rf"{label}{primes}_{lbl}"))

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
def _strip_primes(text: str) -> str:
    """Remove prime markers and * markers from a LaTeX-like string."""
    text = re.sub(r"\^\{\*\}", "", text)  # legacy, now not used but harmless
    text = re.sub(r"[']+", "", text)
    text = text.replace("*", "")
    return text

def _extract_order_string(latex_str: str) -> str:
    """Strip d_1/d_2 prefixes, prime decorations and braces so only the bare order remains."""
    core = latex_str.replace("d_1:", "").replace("d_2:", "").strip()
    core_no_primes = _strip_primes(core)
    # remove {…} but keep inside, so k_{0} → k_0
    core_no_braces = re.sub(r"\{([^{}]+)\}", r"\1", core_no_primes)
    return core_no_braces

def check_pdp_match(original_points: np.ndarray, generated_points: np.ndarray,
                    pdp_variant: str = "fundamental",
                    buffer_x: float = 25.0,
                    buffer_y: float = 10.0,
                    rough_x: float = 0.0,
                    rough_y: float = 0.0,
                    debug: bool = False) -> tuple[bool, bool]:
    """
    Check if generated configuration matches original using PDP inequality matrices.
    
    This uses the exact PDP logic from N_PDP.py with support for all four variants:
    - fundamental: Basic PDP matching with no tolerance (N×N matrix comparison)
    - buffer: Apply buffer transformation to both configs, compare 5N×5N matrices
    - rough: Use roughness as equality tolerance in N×N matrix comparison
    - bufferrough: Apply buffer transformation AND use roughness tolerance
    
    For buffer variants:
    - Each point is expanded to 5 buffer variants (±buffer_x, ±buffer_y, original)
    - Both original and generated configurations are expanded
    - The resulting 5N×5N inequality matrices are compared
    
    Args:
        original_points: All original points (N, 2) - can be any number of points
        generated_points: All generated points (N, 2) - same count as original
        pdp_variant: PDP variant to use ("fundamental", "buffer", "rough", "bufferrough")
        buffer_x: Buffer distance in x-direction (for buffer variants)
        buffer_y: Buffer distance in y-direction (for buffer variants)
        rough_x: Roughness tolerance in x-direction (for rough variants)
        rough_y: Roughness tolerance in y-direction (for rough variants)
        debug: If True, print debug information
    
    Returns:
        (d1_match, d2_match): Boolean tuple indicating if x and y dimensions match
    """
    # Apply buffer transformation if needed (expands N points to 5N points)
    orig_pts = original_points.copy()
    gen_pts = generated_points.copy()
    
    if pdp_variant in ["buffer", "bufferrough"]:
        n_before = len(orig_pts)
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, buffer_y)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, buffer_y)
        print(f"[DEBUG BUFFER] Applied buffer transform: {n_before} -> {len(orig_pts)} points, buffer=({buffer_x}, {buffer_y})")
    
    # Determine roughness values
    roughness_x = rough_x if pdp_variant in ["rough", "bufferrough"] else 0.0
    roughness_y = rough_y if pdp_variant in ["rough", "bufferrough"] else 0.0
    
    if pdp_variant in ["rough", "bufferrough"]:
        print(f"[DEBUG ROUGH] Using roughness=({roughness_x}, {roughness_y})")
    
    if debug:
        n_orig = len(orig_pts)
        n_gen = len(gen_pts)
        print(f"[DEBUG check_pdp_match] variant={pdp_variant}, points={n_orig}, roughness=({roughness_x}, {roughness_y})")
    
    # Compute inequality matrices for both dimensions
    original_x_matrix = compute_inequality_matrix(orig_pts, 0, roughness_x)
    original_y_matrix = compute_inequality_matrix(orig_pts, 1, roughness_y)
    
    generated_x_matrix = compute_inequality_matrix(gen_pts, 0, roughness_x)
    generated_y_matrix = compute_inequality_matrix(gen_pts, 1, roughness_y)
    
    # Compare matrices
    d1_match = compare_inequality_matrices(original_x_matrix, generated_x_matrix)
    d2_match = compare_inequality_matrices(original_y_matrix, generated_y_matrix)
    
    if pdp_variant in ["buffer", "rough", "bufferrough"]:
        print(f"[DEBUG {pdp_variant.upper()}] Match result: d1={d1_match}, d2={d2_match}")
    
    return d1_match, d2_match

# Legacy wrapper for backward compatibility
def check_pdp_match_legacy(original_k: np.ndarray, original_l: np.ndarray, 
                          generated_k: np.ndarray, generated_l: np.ndarray,
                          pdp_variant: str = "fundamental",
                          buffer_x: float = 25.0,
                          buffer_y: float = 10.0,
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
    """
    # Get all current candidate points (multi-point support)
    anim_generated_points = st.session_state.get("anim_generated_points", {})
    gen_pt = st.session_state.get("anim_generated_point", None)
    
    # Need at least some generated point to check
    if not anim_generated_points and gen_pt is None:
        st.session_state["order_match_d1"] = False
        st.session_state["order_match_d2"] = False
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
    buffer_x = st.session_state.get("cfg_buffer_x", 25.0)
    buffer_y = st.session_state.get("cfg_buffer_y", 10.0)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    # Use PDP inequality matrix comparison with selected variant
    d1_match, d2_match = check_pdp_match(
        all_pts_flat,
        generated_points,
        pdp_variant=pdp_variant,
        buffer_x=buffer_x,
        buffer_y=buffer_y,
        rough_x=rough_x,
        rough_y=rough_y
    )
    
    st.session_state["order_match_d1"] = d1_match
    st.session_state["order_match_d2"] = d2_match

# ============= Helper: Binary search iteration ============
def run_binary_iteration(
    current_points: np.ndarray,
    successful_points: list[SuccessfulPoint],
    pdp_variant: str,
    buffer_x: float,
    buffer_y: float,
    rough_x: float,
    rough_y: float,
    max_binary_steps: int = 7
) -> tuple[list[SuccessfulPoint], bool]:
    """
    Run one iteration of multi-point generation using the 7-step binary search strategy.
    
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
    diag_rows: list = st.session_state.get("diag_rows", [])
    
    # Binary search: 7 steps
    for binary_step in range(max_binary_steps):
        # Compute candidate positions: ok_point + delta for each point
        candidate_positions: dict[int, np.ndarray] = {}
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
            rough_y=rough_y
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
            # Match! Update ok_points to current candidates, keep delta for next step
            had_full_match = True
            for idx in selected_indices:
                ok_points[idx] = candidate_positions[idx].copy()
        else:
            # No match: halve delta (binary search narrowing)
            for idx in selected_indices:
                deltas[idx] = deltas[idx] / 2.0
    
    # Store diagnostics
    st.session_state["diag_rows"] = diag_rows
    st.session_state["anim_had_full_match"] = had_full_match
    
    # Final placement: use the last ok_points
    iteration_num = len([sp for sp in successful_points]) // max(1, len(selected_indices))
    for idx in selected_indices:
        final_pt = ok_points[idx]
        parent_pt = get_parent_position(idx)
        sp: SuccessfulPoint = {
            "point": final_pt,
            "parent_idx": idx,
            "parent_point": parent_pt,
            "original_parent_idx": idx,
            "iteration": iteration_num,
        }
        successful_points.append(sp)
    
    # Record iteration summary
    iter_log: list = st.session_state.get("binary_iteration_summary", [])
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
    buffer_x = st.session_state.get("cfg_buffer_x", 25.0)
    buffer_y = st.session_state.get("cfg_buffer_y", 10.0)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    all_configs: list = []
    current_points = all_pts_flat.copy()
    
    # Reset diagnostics
    st.session_state["diag_rows"] = []
    st.session_state["binary_iteration_summary"] = []
    
    # Process each variant
    for variant_idx, pdp_variant in enumerate(pdp_variants_list):
        st.session_state["anim_current_variant_idx"] = variant_idx
        st.session_state["anim_current_variant"] = pdp_variant
        
        # Generate configurations for this variant
        for config_num in range(1, num_configs + 1):
            st.session_state["anim_current_config"] = config_num
            successful_points: list[SuccessfulPoint] = []
            
            # Reset diagnostics for each configuration
            st.session_state["diag_rows"] = []
            
            # Run iterations for this configuration using binary search
            for iteration in range(num_iterations):
                st.session_state["anim_completed_iterations"] = iteration
                
                successful_points, success = run_binary_iteration(
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
    
    # Store all configurations
    st.session_state["anim_all_configs"] = all_configs
    st.session_state["anim_running"] = False
    st.session_state["anim_completed_iterations"] = num_iterations
    st.session_state["anim_binary_mode"] = True
    
    # Rerun to update the UI
    st.rerun()


# ============= Helper: Multi-point generation iteration ============
def run_multipoint_iteration(
    current_points: np.ndarray,
    successful_points: list[SuccessfulPoint],
    pdp_variant: str,
    buffer_x: float,
    buffer_y: float,
    rough_x: float,
    rough_y: float,
    max_search_steps: int = 7
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
    for search_step in range(max_search_steps):
        # Scale vectors
        scaled_vectors = scale_movement_vectors(movement_vectors, current_scale)
        
        # Get parent points (either from successful_points or original)
        parent_positions = {}
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
        candidate_positions = {}
        for idx, (dx, dy) in scaled_vectors.items():
            parent_pt = parent_positions.get(idx, current_points[idx])
            new_x = np.clip(parent_pt[0] + dx, COORD_MIN_X, COORD_MAX_X)
            new_y = np.clip(parent_pt[1] + dy, COORD_MIN_Y, COORD_MAX_Y)
            candidate_positions[idx] = np.array([new_x, new_y])
        
        # Build config with candidate positions and check PDP
        test_config = build_current_config(candidate_positions)
        
        same_d1, same_d2 = check_pdp_match(
            all_pts_flat,
            test_config,
            pdp_variant=pdp_variant,
            buffer_x=buffer_x,
            buffer_y=buffer_y,
            rough_x=rough_x,
            rough_y=rough_y
        )
        
        if same_d1 and same_d2:
            # Success! Add all candidate points to successful_points
            iteration_num = len([sp for sp in successful_points]) // max(1, len(selected_indices))
            for idx, new_pt in candidate_positions.items():
                parent_pt = parent_positions[idx]
                sp: SuccessfulPoint = {
                    "point": new_pt,
                    "parent_idx": idx,  # Original index used as parent
                    "parent_point": parent_pt,
                    "original_parent_idx": idx,
                    "iteration": iteration_num,
                }
                successful_points.append(sp)
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
    
    # Max search steps reached - snap to parent positions (minimal change)
    # This counts as "success" to ensure each iteration produces a result
    min_scale = 0.001
    scaled_vectors = scale_movement_vectors(movement_vectors, min_scale)
    
    for idx in selected_indices:
        parent_pt = parent_positions.get(idx, current_points[idx])
        dx, dy = scaled_vectors.get(idx, (0, 0))
        new_x = np.clip(parent_pt[0] + dx, COORD_MIN_X, COORD_MAX_X)
        new_y = np.clip(parent_pt[1] + dy, COORD_MIN_Y, COORD_MAX_Y)
        new_pt = np.array([new_x, new_y])
        
        iteration_num = len([sp for sp in successful_points]) // max(1, len(selected_indices))
        sp: SuccessfulPoint = {
            "point": new_pt,
            "parent_idx": idx,
            "parent_point": parent_pt,
            "original_parent_idx": idx,
            "iteration": iteration_num,
        }
        successful_points.append(sp)
    
    return successful_points, True


def generate_exp_multipoint() -> None:
    """
    Multi-point aware version of non-animated exponential generation.
    
    Supports:
    - Single point (default, original behavior)
    - Multiple random points
    - Group pattern (Np + Nt)
    
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
    buffer_x = st.session_state.get("cfg_buffer_x", 25.0)
    buffer_y = st.session_state.get("cfg_buffer_y", 10.0)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    all_configs: list = []
    current_points = all_pts_flat.copy()
    
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
    
    # Store all configurations
    st.session_state["anim_all_configs"] = all_configs
    st.session_state["anim_running"] = False
    st.session_state["anim_completed_iterations"] = num_iterations
    
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
        buffer_x = st.session_state.get("cfg_buffer_x", 25.0)
        buffer_y = st.session_state.get("cfg_buffer_y", 10.0)
        rough_x = st.session_state.get("cfg_rough_x", 0.0)
        rough_y = st.session_state.get("cfg_rough_y", 0.0)
        
        # Use PDP inequality matrix comparison with selected variant
        same_d1, same_d2 = check_pdp_match(
            all_pts_flat,
            generated_points,
            pdp_variant=pdp_variant,
            buffer_x=buffer_x,
            buffer_y=buffer_y,
            rough_x=rough_x,
            rough_y=rough_y
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

        # === Case 1: success (orders match) or distance collapsed to 0 ===
        if (same_d1 and same_d2 and gen_pt is not None) or (distance <= 0.0 and gen_pt is not None):
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
                all_configs: list = st.session_state.get("anim_all_configs", [])
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
                    max_attempts = 20
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
                        max_attempts = 20
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
                max_attempts = 20
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
    
    # Clear animation history when starting a new animation (for "Previous step" functionality)
    st.session_state["anim_state_history"] = []
    
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
        print(f"[DEBUG INIT BINARY] strategy={strategy}, setting anim_binary_mode=True")
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
        
        print(f"[DEBUG BINARY INIT] selected_indices={selected_indices}, maxdist={maxdist}")
        for idx in selected_indices:
            vec = movement_vectors.get(idx, (0.0, 0.0))
            vec_mag = np.sqrt(vec[0]**2 + vec[1]**2)
            print(f"[DEBUG BINARY INIT] idx={idx}, parent={all_pts[idx]}, movement_vec={vec}, magnitude={vec_mag:.4f}")
        
        # Step a-f: Check if all points are within bounds, retry with new directions if not
        max_direction_attempts = 10
        found_valid = False
        for _ in range(max_direction_attempts):
            all_within_bounds = True
            generated_points: dict[int, np.ndarray] = {}
            
            for idx in selected_indices:
                dx, dy = movement_vectors.get(idx, (0.0, 0.0))
                base_pt = all_pts[idx]
                candidate_x = base_pt[0] + dx
                candidate_y = base_pt[1] + dy
                
                # Check if within bounds
                if not (COORD_MIN_X <= candidate_x <= COORD_MAX_X and COORD_MIN_Y <= candidate_y <= COORD_MAX_Y):
                    all_within_bounds = False
                
                # Clip for storage (even if out of bounds, for visualization)
                candidate_x = np.clip(candidate_x, COORD_MIN_X, COORD_MAX_X)
                candidate_y = np.clip(candidate_y, COORD_MIN_Y, COORD_MAX_Y)
                generated_points[idx] = np.array([candidate_x, candidate_y])
            
            if all_within_bounds:
                found_valid = True
                break
            
            # Regenerate movement vectors with new random directions
            movement_vectors = generate_movement_vectors(selected_indices, maxdist)
        
        # DEBUG: Verify distances
        print(f"[DEBUG BINARY INIT] found_valid={found_valid}")
        for idx in selected_indices:
            parent_pt_dbg = all_pts[idx]
            gen_pt_dbg = generated_points.get(idx, parent_pt_dbg)
            actual_dist = np.linalg.norm(gen_pt_dbg - parent_pt_dbg)
            print(f"[DEBUG BINARY INIT] idx={idx}, parent={parent_pt_dbg}, generated={gen_pt_dbg}, actual_distance={actual_dist:.4f}")
        
        if not found_valid:
            # Step f fail: use parent coordinates for all points
            print(f"[DEBUG BINARY] Failed to find valid direction after {max_direction_attempts} attempts, using parents")
            generated_points = {idx: all_pts[idx].copy() for idx in selected_indices}
            current_distance = 0.0
        else:
            current_distance = maxdist
        
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
    
    # Multi-variant support - initialize variant tracking
    pdp_variants_list = st.session_state.get("cfg_pdp_variants", ["fundamental"])
    st.session_state["anim_pdp_variants_list"] = pdp_variants_list
    st.session_state["anim_current_variant_idx"] = 0
    st.session_state["anim_current_variant"] = pdp_variants_list[0] if pdp_variants_list else "fundamental"
    
    # Debug: Print the PDP configuration being used
    print(f"[DEBUG GENERATE] Selected variants: {pdp_variants_list}")
    print(f"[DEBUG GENERATE] Current variant: {st.session_state['anim_current_variant']}")
    print(f"[DEBUG GENERATE] buffer_x: {st.session_state.get('cfg_buffer_x', 'NOT SET')}")
    print(f"[DEBUG GENERATE] buffer_y: {st.session_state.get('cfg_buffer_y', 'NOT SET')}")
    print(f"[DEBUG GENERATE] rough_x: {st.session_state.get('cfg_rough_x', 'NOT SET')}")
    print(f"[DEBUG GENERATE] rough_y: {st.session_state.get('cfg_rough_y', 'NOT SET')}")
    print(f"[DEBUG GENERATE] Strategy selected: '{strategy}'")
    print(f"[DEBUG GENERATE] cfg_strategy from session_state: '{st.session_state.get('cfg_strategy', 'NOT SET')}'")

    if strategy == "exponential":
        # Set up parameters for multi-point generation
        st.session_state["anim_max_iterations"] = int(num_iterations)
        st.session_state["anim_num_configs"] = int(num_configs)
        st.session_state["anim_running"] = True
        st.session_state["show_anim_circle"] = False  # no circle for generate
        
        # Debug: print point selection mode
        point_selection_mode = st.session_state.get("cfg_point_selection_mode", "Single point")
        movement_direction = st.session_state.get("cfg_movement_direction", "Same direction")
        print(f"[DEBUG GENERATE] Point selection mode: {point_selection_mode}")
        print(f"[DEBUG GENERATE] Movement direction: {movement_direction}")

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
        print(f"[DEBUG GENERATE BINARY] Point selection mode: {point_selection_mode}")
        print(f"[DEBUG GENERATE BINARY] Movement direction: {movement_direction}")

        # Run the binary search generator
        generate_binary_multipoint()

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
    # DEBUG: Print figure parameters
    print(f"[DEBUG RENDER] xlim={xlim}, ylim={ylim}, size_inches={size_inches}, dpi={dpi}")
    fig = Figure(figsize=(size_inches, size_inches), dpi=dpi)
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    setup_square_axes(ax, xlim, ylim)
    draw_fn(ax)
    # Use constrained_layout or fixed padding instead of tight_layout
    # to prevent layout changes based on content
    fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
    # DEBUG: Print actual axes limits after drawing
    actual_xlim = ax.get_xlim()
    actual_ylim = ax.get_ylim()
    print(f"[DEBUG RENDER] After draw: actual_xlim={actual_xlim}, actual_ylim={actual_ylim}")
    return fig

BLUE = "C0"
ORANGE = "C1"
LABEL_FS = 9

# Colors for all objects (matplotlib style): blue, orange, green, purple, brown, pink, etc.
OBJECT_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
# Plotly-compatible colors (hex equivalents of matplotlib's default color cycle)
OBJECT_COLORS_PLOTLY = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# Labels for all objects
OBJECT_LABELS = ["k", "l", "m", "n", "p", "q", "r", "s", "u", "v"]

def annotate_points(
    ax: matplotlib.axes.Axes,
    pts: np.ndarray,
    ts: np.ndarray,
    label_prefix: str,
    color: str,
) -> None:
    """Draw points plus labels k_t or l_t with small offsets."""
    offsets = [(3, 3), (3, -8), (-8, 3)]
    for i, ((x, y), tval) in enumerate(zip(pts, ts)):
        ax.scatter([x], [y], s=25, zorder=3, color=color)  # type: ignore
        off = offsets[i % len(offsets)]
        try:
            tnum = float(tval)  # type: ignore[arg-type]
        except Exception:
            tnum = float(np.array(tval, dtype=float))
        lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
        # Use simple text label without LaTeX to avoid parsing issues
        label_text = f"{label_prefix}_{lbl}"
        ax.annotate(  # type: ignore
            label_text,
            xy=(x, y),
            xytext=off,
            textcoords="offset points",
            fontsize=LABEL_FS,
            color=color,
            ha="left" if off[0] >= 0 else "right",
            va="bottom" if off[1] >= 0 else "top",
        )

def draw_original(ax: matplotlib.axes.Axes) -> None:
    """Draw all object curves in the left panel, including external reference points."""
    # Draw all objects uniformly
    for i, o_id in enumerate(sorted(all_points_plot.keys())):
        pts = all_points_plot[o_id]
        vals = all_vals_plot[o_id]
        if pts.shape[0] > 0:
            color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
            ax.plot(pts[:, 0], pts[:, 1], linewidth=1.2, color=color)  # type: ignore
            annotate_points(ax, pts, vals, label, color)
    
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
        """Helper to build a simple label prefix_gen_marker_t."""
        try:
            tnum = float(tval)
        except Exception:
            tnum = float(np.array(tval, dtype=float))
        lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
        if gen_marker:
            return f"{prefix}{gen_marker}_{lbl}"
        return f"{prefix}_{lbl}"

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
    for i, ((x, y), tval) in enumerate(zip(k_points_plot, k_vals_plot)):
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
    for i, ((x, y), tval) in enumerate(zip(l_points_plot, l_vals_plot)):
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
            for pt_i, ((x, y), tval) in enumerate(zip(pts, vals)):
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
        for idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):
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
            
            # Calculate the actual distance from generated point to parent point
            # The circle radius should ALWAYS equal this distance (red dot on circle edge)
            if sel_gen_pt is not None:
                actual_dist = float(np.linalg.norm(np.array(sel_gen_pt) - sel_parent_pt))
            else:
                actual_dist = float(distance)  # Fallback to stored distance if no gen_pt
            
            # DEBUG: Print circle info
            print(f"[DEBUG CIRCLE] parent={sel_parent_pt}, gen_pt={sel_gen_pt}, actual_dist={actual_dist:.4f}")
            
            # Draw circle around parent point with radius = actual distance to red dot
            # This ensures the red dot is ALWAYS on the circle edge
            circle = matplotlib.patches.Circle(
                (sel_parent_pt[0], sel_parent_pt[1]),
                radius=actual_dist,
                edgecolor='red',
                facecolor='none',
                linewidth=1.2,
                zorder=5
            )
            ax.add_patch(circle)  # type: ignore
            
            # Draw red dot at generated point if we have one
            if sel_gen_pt is not None:
                ax.scatter([sel_gen_pt[0]], [sel_gen_pt[1]], s=40, zorder=6, color='red')  # type: ignore
        
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
            # 5 buffer variants: x-buf, x+buf, original, y-buf, y+buf
            buffer_pts = [
                (gx - buffer_x_val, gy),  # x - buffer_x
                (gx + buffer_x_val, gy),  # x + buffer_x
                (gx, gy),                  # original (already drawn as red)
                (gx, gy - buffer_y_val),  # y - buffer_y
                (gx, gy + buffer_y_val),  # y + buffer_y
            ]
            # Draw buffer points (skip the center one at index 2, it's the main point)
            for idx, (bx, by) in enumerate(buffer_pts):
                if idx != 2:  # Skip the center point
                    ax.scatter([bx], [by], s=18, zorder=5, color=buffer_color, alpha=buffer_alpha, marker='x')
            # Draw lines connecting buffer points to show the buffer cross
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
            # Rectangle from (gx - rough_x, gy - rough_y) to (gx + rough_x, gy + rough_y)
            rect = matplotlib.patches.Rectangle(
                (gx - rough_x_val, gy - rough_y_val),
                2 * rough_x_val if rough_x_val > 0 else 0.5,  # width
                2 * rough_y_val if rough_y_val > 0 else 0.5,  # height
                edgecolor=rough_color,
                facecolor=rough_color,
                alpha=rough_alpha,
                linewidth=1.5,
                linestyle=':',
                zorder=3
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

# ============= Layout (two columns) ============
col1, col2 = st.columns(2, gap="small")

# Pre-render both figures to ensure identical sizing
fig_left = render_square_matplotlib_figure(draw_original, XLIM, YLIM)
fig_right = render_square_matplotlib_figure(draw_generated_empty, XLIM, YLIM)

# Convert to PNG bytes for consistent display
_buf_left: IO[bytes] = io.BytesIO()
fig_left.savefig(_buf_left, format="png", dpi=160)
_buf_left.seek(0)

_buf_right: IO[bytes] = io.BytesIO()
fig_right.savefig(_buf_right, format="png", dpi=160)
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

    # Download original plot as PNG - reuse the buffer
    _buf_left.seek(0)
    st.download_button(
        label="Save as PNG",
        data=_buf_left.getvalue(),
        file_name="original.png",
        mime="image/png",
        key="dl_left_png",
    )

with col2:
    st.markdown("<div class='figure-title'>Generated configuration</div>", unsafe_allow_html=True)
    _latex_d1 = make_d1_order_latex_generated()
    _latex_d2 = make_d2_order_latex_generated()
    # DEBUG: Print LaTeX lengths
    print(f"[DEBUG LATEX] d1_len={len(_latex_d1)}, d2_len={len(_latex_d2)}")
    print(f"[DEBUG LATEX] d1={_latex_d1[:100]}...")
    st.latex(_latex_d1)
    st.latex(_latex_d2)
    # DEBUG: Print XLIM, YLIM, and anim_distance to check if they change
    _debug_distance = st.session_state.get("anim_distance", "NOT SET")
    _debug_binary = st.session_state.get("anim_binary_mode", False)
    print(f"[DEBUG DRAW] XLIM={XLIM}, YLIM={YLIM}, anim_distance={_debug_distance}, binary_mode={_debug_binary}")
    # Use st.image instead of st.pyplot for consistent sizing
    st.image(_buf_right, width="stretch")

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
    # Reuse the buffer that was already created above
    _buf_right.seek(0)

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
                    current_state_snapshot[key] = copy.deepcopy(value)
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
    buffer_x = st.session_state.get("cfg_buffer_x", 25.0)
    buffer_y = st.session_state.get("cfg_buffer_y", 10.0)
    rough_x = st.session_state.get("cfg_rough_x", 0.0)
    rough_y = st.session_state.get("cfg_rough_y", 0.0)
    
    # Use PDP inequality matrix comparison with selected variant
    same_d1, same_d2 = check_pdp_match(
        all_pts_flat,
        generated_points_arr,
        pdp_variant=pdp_variant,
        buffer_x=buffer_x,
        buffer_y=buffer_y,
        rough_x=rough_x,
        rough_y=rough_y
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
            time.sleep(wait_s)
            st.rerun()

    def _get_parent_point(all_pts: np.ndarray, parent_idx: int) -> np.ndarray:
        """Return the effective parent point (original or generated) for a given parent_idx."""
        if all_pts.size == 0:
            return np.array([0.0, 0.0])
        if parent_idx < n_total_points:
            return all_pts[parent_idx]
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - n_total_points)
        if 0 <= sidx < len(succ_list):
            return succ_list[sidx]["point"]
        return np.array([0.0, 0.0])

    # === Case 1: success (orders match) or distance collapsed to 0 ===
    # For BINARY mode: ONLY complete after 7 steps (distance will be set to 0 after step 7)
    # For EXPONENTIAL mode: complete when orders match or distance <= 0
    binary_complete = binary_mode and distance <= 0.0 and gen_pt is not None
    exponential_complete = not binary_mode and ((same_d1 and same_d2 and gen_pt is not None) or (distance <= 0.0 and gen_pt is not None))
    
    if binary_complete or exponential_complete:
        # Multi-point support: add ALL n selected points as successful
        anim_generated_points = st.session_state.get("anim_generated_points", {})
        selected_indices = st.session_state.get("anim_selected_indices", [parent_idx])
        
        # For each selected point, add to successful_points
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
            
            sp: SuccessfulPoint = {
                "point": np.array(final_pt, dtype=float),
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

            all_configs: list = st.session_state.get("anim_all_configs", [])
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
                        
                        new_x = np.clip(new_x, COORD_MIN_X, COORD_MAX_X)
                        new_y = np.clip(new_y, COORD_MIN_Y, COORD_MAX_Y)
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
                st.session_state["anim_generated_points"] = {int(k): v for k, v in generated_points.items()}
                st.session_state["anim_movement_vectors"] = {int(k): v for k, v in movement_vectors.items()}
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
                    
                    new_x = np.clip(new_x, COORD_MIN_X, COORD_MAX_X)
                    new_y = np.clip(new_y, COORD_MIN_Y, COORD_MAX_Y)
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
            # Sync multi-point data
            st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
            st.session_state["anim_generated_points"] = {int(k): v for k, v in generated_points.items()}
            st.session_state["anim_movement_vectors"] = {int(k): v for k, v in movement_vectors.items()}
    else:
        # === Case 2: keep searching ===
        # Different behavior for binary vs exponential strategy
        
        # DEBUG: Print which strategy branch we're taking
        print(f"[DEBUG ANIM] binary_mode={binary_mode}, cfg_strategy={st.session_state.get('cfg_strategy')}, anim_binary_mode={st.session_state.get('anim_binary_mode')}")
        
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
            if not correct_orders:
                # Fallback: initialize from parent positions
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
            
            print(f"[DEBUG BINARY STEP {binary_step}] current_distance={current_distance:.4f}, n_points={len(selected_indices)}, matched={current_matches}")
            
            # Add diagnostic row
            diag_rows = st.session_state.get("diag_rows", [])
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
                
                # Place final points at correct_order positions
                final_positions = {int(idx): correct_orders.get(idx, all_pts[idx].copy()) for idx in selected_indices}
                
                # For backwards compatibility, keep single generated_point as first one
                first_idx = selected_indices[0] if selected_indices else parent_idx
                st.session_state["anim_generated_point"] = final_positions.get(first_idx, np.array([0.0, 0.0])).copy()
                st.session_state["anim_binary_correct_order"] = correct_orders.get(first_idx, np.array([0.0, 0.0])).copy()
                st.session_state["anim_distance"] = 0.0  # Trigger success
                st.session_state["anim_in_search"] = True
                st.session_state["anim_selected_indices"] = [int(i) for i in selected_indices]
                st.session_state["anim_generated_points"] = final_positions
                print(f"[DEBUG BINARY] FINALIZE at correct_orders for {len(selected_indices)} points")
                
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
                print(f"[DEBUG BINARY STEP {binary_step}] HALVE! distance {maxdist:.4f} -> {new_distance:.4f} for {len(selected_indices)} points")
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
                    print(f"[DEBUG BINARY STEP {binary_step}] MATCH! distance {current_distance:.4f} + {delta_term:.4f} = {new_distance:.4f}")
                else:
                    # No match: subtract delta_term from distance
                    new_distance = current_distance - delta_term
                    print(f"[DEBUG BINARY STEP {binary_step}] NO MATCH! distance {current_distance:.4f} - {delta_term:.4f} = {new_distance:.4f}")
                
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
                print(f"[DEBUG BINARY STEP {binary_step}] Next candidates at distance {new_distance:.4f} for {len(selected_indices)} points")
        
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
                    orig_vec = movement_vectors.get(idx, (0.0, 0.0))
                    orig_mag = np.sqrt(orig_vec[0]**2 + orig_vec[1]**2)
                    if orig_mag > 1e-9:
                        direction = np.array([orig_vec[0] / orig_mag, orig_vec[1] / orig_mag])
                    else:
                        direction = np.array([1.0, 0.0])
                    
                    # Add small angle tweak per point
                    angle_tweak = float(np.random.uniform(-0.15, 0.15))
                    cos_t, sin_t = np.cos(angle_tweak), np.sin(angle_tweak)
                    direction = np.array([
                        direction[0] * cos_t - direction[1] * sin_t,
                        direction[0] * sin_t + direction[1] * cos_t
                    ])
                    
                    # New position: parent + direction × dist
                    new_pt = parent_pt + direction * dist
                    new_pt[0] = np.clip(new_pt[0], COORD_MIN_X, COORD_MAX_X)
                    new_pt[1] = np.clip(new_pt[1], COORD_MIN_Y, COORD_MAX_Y)
                    new_positions[idx] = new_pt
                return new_positions

            if search_steps >= max_search_steps:
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
                print(f"[DEBUG EXPONENTIAL] Max steps reached, snapping {len(selected_indices)} points to parents")
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
                print(f"[DEBUG EXPONENTIAL] Halving ALL {len(selected_indices)} points: {distance:.4f} -> {new_distance:.4f}")

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
        # Auto mode: sleep and auto-advance
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
    
    # Get unique variants and organize generated configs
    configs_by_variant: dict = {}
    for cfg in all_configs_list:
        variant = cfg.get("pdp_variant", "fundamental")
        config_num = cfg.get("config_num", 0)
        if variant not in configs_by_variant:
            configs_by_variant[variant] = []
        if config_num not in configs_by_variant[variant]:
            configs_by_variant[variant].append(config_num)
    
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

    # 1. Add Original Configuration - loop through ALL objects (only if selected)
    if "Original" in selected_configs:
        for i, obj_id in enumerate(sorted(all_points_plot.keys())):
            pts = all_points_plot[obj_id]
            vals = all_vals_plot[obj_id]
            color = OBJECT_COLORS_PLOTLY[i % len(OBJECT_COLORS_PLOTLY)]
            label = OBJECT_LABELS[i % len(OBJECT_LABELS)]
            # Build hover text for each point
            hover_texts = [f"<b>Original</b><br>Object: {label}<br>Point: {label}_{int(t)}<br>d₁: {pts[j, 0]:.{COORD_DISPLAY_PRECISION}f}<br>d₂: {pts[j, 1]:.{COORD_DISPLAY_PRECISION}f}" 
                          for j, t in enumerate(vals)]
            fig.add_trace(go.Scatter(
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
            hover_texts_ext = []
            text_labels_ext = []
            n_timestamps = len(selected_ts_window)
            for idx, (ext_pt, ext_t) in enumerate(zip(external_pts_for_window, external_ts_for_window)):
                ext_point_idx = idx % len(external_points_list) if external_points_list else idx
                hover_texts_ext.append(
                    f"<b>Original</b><br>Type: External Reference<br>Point: ext_{ext_point_idx}<br>"
                    f"d₁: {ext_pt[0]:.{COORD_DISPLAY_PRECISION}f}<br>d₂: {ext_pt[1]:.{COORD_DISPLAY_PRECISION}f}<br>"
                    f"<i>(Fixed - does not move)</i>"
                )
                text_labels_ext.append(f"ext_{ext_point_idx}")
            
            fig.add_trace(go.Scatter(
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
                hover_texts = [f"<b>{config_label}</b><br>Variant: {variant}<br>Config: C{config_num}<br>Object: {label}<br>Point: {label}_{int(vals[j])}<br>d₁: {pts[j, 0]:.{COORD_DISPLAY_PRECISION}f}<br>d₂: {pts[j, 1]:.{COORD_DISPLAY_PRECISION}f}" 
                              for j in range(len(pts))]
                fig.add_trace(go.Scatter(
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
                        f"d₁: {ext_pt[0]:.{COORD_DISPLAY_PRECISION}f}<br>d₂: {ext_pt[1]:.{COORD_DISPLAY_PRECISION}f}<br>"
                        f"<i>(Fixed - does not move)</i>"
                    )
                
                fig.add_trace(go.Scatter(
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
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
            range=[XLIM[0], XLIM[1]],
            title="d₁"
        ),
        yaxis=dict(
            constrain="domain",
            range=[YLIM[0], YLIM[1]],
            title="d₂"
        ),
        legend=dict(
            groupclick="toggleitem" # Clicking a legend item toggles the whole group
        ),
        title="Comparison of Selected Configurations"
    )
    
    st.plotly_chart(fig, width="stretch")

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
    right_d2 = make_d2_order_latex_generated