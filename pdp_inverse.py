import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
import math
import time

st.set_page_config(layout="wide")

# -----------------------------------------------------------------------------
# PDP inverse — high level explanation
# -----------------------------------------------------------------------------
# This Streamlit app performs an "inverse" Partial Dependence Plot (PDP)-style
# exploration: given a set of time-series point observations for two families
# (called `k` and `l`) the tool attempts to infer or propose candidate
# positions (primes) such that the induced ordering relations (x and y axes)
# match a target configuration.
#
# Key concepts and data shapes used throughout the file:
# - Input CSV rows: columns [c, t, o, x, y]
#   - c: configuration id
#   - t: timepoint index
#   - o: family (0 -> k, 1 -> l)
#   - x, y: 2D coordinates
# - k_points / l_points: numpy arrays shaped (timepoints_n, 2) with floats
# - base_keys: list of keys like 'k|t0', 'l|t1' describing reference points
# - Matrices M1/M2/M3/M4: pairwise comparison matrices across base keys
#   - values are in {-1, 0, 1} representing ordering on x or y axis
# - st.session_state is used to keep interactive state: current primes,
#   counts, checkpoint text placeholders, and simple counters.
#
# Purpose of the algorithms:
# - Build base order matrices for the original points.
# - For candidate/primes, rebuild order matrices and compare to the base to
#   detect mismatches (this is the 'inverse PDP' check: find primes that
#   preserve order relations similar to the original).
# - Expose interactive controls to iterate search strategies (exponential
#   or binary) and optionally snapshot / display particular generated
#   configurations (checkpoints).
#
# Notes for maintainers:
# - Most helper functions return pure data structures (matrices, maps),
#   but a few functions update `st.session_state` (e.g., when committing
#   primes or storing placeholders). Comments near those functions
#   explain side-effects.
# -----------------------------------------------------------------------------

# ---------------- Exact sleep helper (respect user's choice exactly) ---------------- #
def sleep_if_needed():
    """Sleep exactly the number of milliseconds chosen by the user.
    0 ms -> no sleep. No multipliers."""
    delay_ms = st.session_state.get("frame_delay_ms", 0)
    if isinstance(delay_ms, (int, float)) and delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

# ---------------- Initialize session-state keys early ---------------- #
for _k, _v in (
    ("latest_blue_k", {}),
    ("latest_blue_l", {}),
    ("prime_k", {}),
    ("prime_l", {}),
    ("prime_count_k", {}),
    ("prime_count_l", {}),
    ("green_snaps_k", []),
    ("green_snaps_l", []),
    ("repeats_done", 0),

    # buffers for point logic (only used for k0 descendants)
    ("pt_trials_pink", []),     # test point: similarity, iteration not finished (k0)
    ("pt_trials_gray", []),     # test point: no similarity (k0)
    ("pt_final_purple", []),    # final point with similarity (k0)

    # text state
    ("orig_text", ""),
    ("checkpoint_texts", {}),          # {step: html str}
    ("checkpoint_placeholders", {}),   # {step: st.empty()},

    # axes lock
    ("axes_locked", False),
):
    if _k not in st.session_state:
        st.session_state[_k] = _v

# --- Iteration/config settings ---#
MAX_STEPS = 12
CONFIG_STEPS = list(range(1, 100))  # 1..99 (all configs < 100)
SPECIAL_CONFIGS = CONFIG_STEPS

# ---------------- Heatmap helper ---------------- #
def plot_heatmap(
    z, x_labels, y_labels, title,
    width=200, height=200,
    show_values=True, fontsize=14,
    target=None,
    show_axis_labels=True
):
    # invert Y for top-to-bottom display
    y_labels = y_labels[::-1]
    z = z[::-1]

    colorscale = [
        [0.0, "#00FF00"],  # -1
        [0.5, "#FFFF00"],  #  0
        [1.0, "#FF0000"],  #  1
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        showscale=False
    ))

    symbols = {-1: "<", 0: "=", 1: ">"}
    if show_values:
        for i, y_val in enumerate(y_labels):
            for j, x_val in enumerate(x_labels):
                value = z[i][j]
                text_symbol = symbols.get(int(value), str(value))
                fig.add_annotation(
                    x=x_val,
                    y=y_val,
                    text=text_symbol,
                    showarrow=False,
                    font=dict(size=fontsize + 2, color="black")
                )

    xaxis_cfg = dict(
        side="top",
        tickmode="array",
        tickvals=x_labels,
        ticktext=x_labels,
        tickfont=dict(size=fontsize),
        showticklabels=show_axis_labels
    )
    yaxis_cfg = dict(
        tickmode="array",
        tickvals=y_labels,
        ticktext=y_labels,
        tickfont=dict(size=fontsize),
        showticklabels=show_axis_labels
    )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg
    )

    (target or st).plotly_chart(fig, use_container_width=False)

# ---------------- Load data ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("voorbeeld.csv", header=None, sep=None, engine="python")
    if df.shape[1] < 5:
        st.error(f"⚠️ File has {df.shape[1]} columns, expected 5.")
        st.stop()
    df.columns = ["c", "t", "o", "x", "y"]
    return df

df = load_data()
if not all(col in df.columns for col in ["c", "t", "o", "x", "y"]):
    st.error("CSV must contain columns: c, t, o, x, y.")
    st.stop()

# ---------------- Layout ---------------- #
col1, col2 = st.columns([3, 2])

with col1:
    st.title("PDP inverse")

with col2:
    st.subheader("Controls")

with col2:
    configurations = sorted(df["c"].unique())
    try:
        default_con_idx = configurations.index(7)
    except ValueError:
        default_con_idx = 0
    selected_con = st.selectbox("Configuration", configurations, index=default_con_idx)

    all_times = sorted(df[df["c"] == selected_con]["t"].unique())
    try:
        default_t_idx = all_times.index(0)
    except ValueError:
        default_t_idx = 0
    selected_t = st.selectbox("Start time (t)", options=all_times, index=default_t_idx)

    timepoints_n = st.slider("Timepoints in window", min_value=2, max_value=10, value=3, step=1)

    strategy = st.radio(
        "Search strategy",
        options=["Exponential", "Binary"],
        index=0,
        horizontal=True,
        help=(
            "Exponential: choose a fixed angle (ray) from the current base point, "
            "try at starting distance, then repeatedly halve the radius along the same ray. "
            "Stop immediately on the first matrix match; otherwise commit the last correct point.\n\n"
            "Binary: along a fixed ray, start at half the starting distance and run a 1-D binary search by "
            "halving the step size (moving inward/outward based on the previous result) "
            "for a limited number of iterations; commit the last correct point."
        ),
    )

    # Exact delay choice (0 ms = no sleep)
    interval_options = [0, 1, 50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]
    st.selectbox(
        "Interval per step (ms)",
        options=interval_options,
        index=interval_options.index(5000),
        key="frame_delay_ms"
    )

    # Toggle to show/hide test points (pink/gray)
    show_test_points = st.checkbox(
        "Show test points (pink/gray)",
        value=True,
        help="Disable to speed up rendering (purple finals are still shown)"
    )

    iterations_per_config = st.slider("Iterations per configuration", min_value=2, max_value=6, value=3, step=1)

    run1_clicked   = st.button(f"Run 1 config ({iterations_per_config} iterations)")
    run3_clicked   = st.button(f"Run 3 configs ({3*iterations_per_config} iterations)")
    run500_clicked = st.button(f"Run 500 configs ({500*iterations_per_config} iterations)")
    reset_clicked  = st.button("Reset")

# ---------------- Time window (dynamic length) ---------------- #
subset_times = [t for t in all_times if selected_t <= t <= selected_t + (timepoints_n - 1)]
subset = df[(df["c"] == selected_con) & (df["t"].isin(subset_times))].sort_values(["o","t"])

def make_n_timepoints_points(subset_family_df, n):
    pts = subset_family_df.sort_values("t")[["x","y"]].to_numpy(dtype=float)
    out = []
    if len(pts) == 0:
        out = [(0.0, 0.0)] * n
    else:
        for i in range(n):
            if i < len(pts):
                out.append((float(pts[i][0]), float(pts[i][1])))
            else:
                out.append((float(pts[-1][0]), float(pts[-1][1])))
    return np.array(out, dtype=float)

# ---------------- Points per family ---------------- #
k_df = subset[subset["o"] == 0]
l_df = subset[subset["o"] == 1]
k_points = make_n_timepoints_points(k_df, timepoints_n)
l_points = make_n_timepoints_points(l_df, timepoints_n)

labels_points = []
for i in range(timepoints_n):
    labels_points.append((f"k{i}", i, float(k_points[i, 0]), float(k_points[i, 1])))
for i in range(timepoints_n):
    labels_points.append((f"l{i}", i, float(l_points[i, 0]), float(l_points[i, 1])))

# ---------------- Helpers ---------------- #
def parse_index_from_label(label: str) -> int:
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0

def dist_xy(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def cmp(a, b):
    return 1 if a > b else (-1 if a < b else 0)

def clamp_radius(val, maxdist):
    try:
        v = float(val)
    except Exception:
        v = 0.0
    if not np.isfinite(v):
        v = 0.0
    return max(0.0, min(float(maxdist), v))

def build_base_matrices(k_points, l_points, n_tp: int):
    # Build canonical display labels, base keys and a mapping from key -> (x,y)
    # Inputs:
    # - k_points, l_points: arrays with shape (n_tp, 2)
    # - n_tp: number of timepoints (int)
    # Returns:
    # - display_labels: human friendly labels for plotting
    # - base_keys: canonical keys used across matrix computations
    # - key_to_point_base: mapping key->(x,y) for original points
    # - M1, M2: integer matrices (n x n) encoding ordering on x and y axes
    # The matrices contain values in {-1,0,1} using cmp semantics where
    # -1 means less-than, 0 equal, and 1 greater-than.
    display_labels = []
    base_keys = []
    key_to_point_base = {}

    for i in range(n_tp):
        display_labels.append(f"<i>k</i>|<i>t</i><sub>{i}</sub>")
        base_keys.append(f"k|t{i}")
        key_to_point_base[f"k|t{i}"] = (float(k_points[i][0]), float(k_points[i][1]))
        display_labels.append(f"<i>l</i>|<i>t</i><sub>{i}</sub>")
        base_keys.append(f"l|t{i}")
        key_to_point_base[f"l|t{i}"] = (float(l_points[i][0]), float(l_points[i][1]))

    n = len(base_keys)
    M1 = np.zeros((n, n), dtype=int)
    M2 = np.zeros((n, n), dtype=int)
    for i, rk in enumerate(base_keys):
        xi, yi = key_to_point_base[rk]
        for j, ck in enumerate(base_keys):
            xj, yj = key_to_point_base[ck]
            M1[i, j] = cmp(xi, xj)
            M2[i, j] = cmp(yi, yj)
    np.fill_diagonal(M1, 0)
    np.fill_diagonal(M2, 0)
    return display_labels, base_keys, key_to_point_base, M1, M2

def merged_current_points_dict(key_to_point_base):
    # Merge base (original) points with any currently committed prime points
    # stored in session_state. This produces a single mapping used by
    # matrix-building helpers when evaluating candidate/primes.
    merged = dict(key_to_point_base)
    for idx, pt in st.session_state.prime_k.items():
        merged[f"k|t{idx}"] = (float(pt[0]), float(pt[1]))
    for idx, pt in st.session_state.prime_l.items():
        merged[f"l|t{idx}"] = (float(pt[0]), float(pt[1]))
    return merged

def build_prime_matrices(base_keys, key_to_point_current, chosen_label, prime_xy):
    # Construct matrices M3/M4 for the scenario where 'prime_xy' replaces the
    # point at `chosen_label` (e.g. 'k0' or 'l2'). This function is pure and
    # returns new matrices and an updated key->point map.
    key_to_point_prime = dict(key_to_point_current)
    idx_lbl = parse_index_from_label(chosen_label)
    key_to_point_prime[f"{'k' if chosen_label.startswith('k') else 'l'}|t{idx_lbl}"] = (float(prime_xy[0]), float(prime_xy[1]))

    n = len(base_keys)
    M3 = np.zeros((n, n), dtype=int)
    M4 = np.zeros((n, n), dtype=int)
    for i, rk in enumerate(base_keys):
        xi, yi = key_to_point_prime[rk]
        for j, ck in enumerate(base_keys):
            xj, yj = key_to_point_prime[ck]
            M3[i, j] = cmp(xi, xj)
            M4[i, j] = cmp(yi, yj)
    np.fill_diagonal(M3, 0)
    np.fill_diagonal(M4, 0)
    return M3, M4, key_to_point_prime

# prime-label helpers
def prime_marks(n):
    if n <= 0: return ""
    if n == 1: return "′"
    if n == 2: return "″"
    return "‴"

def parse_family_idx(key_str):
    fam = 'k' if key_str.startswith('k') else 'l'
    idx = int(''.join([c for c in key_str if c.isdigit()]))
    return fam, idx

def build_name_map_base(points_dict):
    name_map = {}
    for k in points_dict.keys():
        fam, idx = parse_family_idx(k)
        name_map[k] = f"{fam}{idx}"
    return name_map

def build_name_map_current(points_dict):
    # Build a human-readable name map that includes prime marks (′, ″ ...)
    # based on how many primes have been committed for a particular base
    # point. This queries `st.session_state` for prime counts and thus has a
    # read dependency on session state.
    pk = st.session_state.get("prime_count_k", {})
    pl = st.session_state.get("prime_count_l", {})
    name_map = {}
    for k in points_dict.keys():
        fam, idx = parse_family_idx(k)
        if fam == 'k':
            count = pk.get(idx, 0) if isinstance(pk, dict) else 0
        else:
            count = pl.get(idx, 0) if isinstance(pl, dict) else 0
        count = min(count, 3)
        name_map[k] = f"{fam}{prime_marks(count)}{idx}"
    return name_map

# ordering utilities
def _round9(v): return float(np.round(v, 9))

def order_groups_from_points_dict(points_dict, axis='x'):
    proj = []
    for k, (xv, yv) in points_dict.items():
        val = _round9(xv if axis == 'x' else yv)
        proj.append((k, val))
    proj.sort(key=lambda kv: (kv[1], kv[0]))
    groups = []
    i = 0
    while i < len(proj):
        val = proj[i][1]
        same = [proj[i][0]]
        j = i + 1
        while j < len(proj) and proj[j][1] == val:
            same.append(proj[j][0])
            j += 1
        groups.append(sorted(same))
        i = j
    return groups

def groups_to_string(groups, name_map=None):
    parts = []
    for g in groups:
        items = [(name_map[k] if name_map and k in name_map else k) for k in g]
        parts.append(" = ".join(items))
    return " < ".join(parts) if parts else "(none)"

def group_index_map(groups):
    idx_map = {}
    for gi, g in enumerate(groups):
        for key in g:
            idx_map[key] = gi
    return idx_map

def mismatch_count(base_groups, cur_groups):
    base_idx = group_index_map(base_groups)
    cur_idx  = group_index_map(cur_groups)
    keys = set(base_idx.keys()) & set(cur_idx.keys())
    return sum(1 for k in keys if cur_idx[k] != base_idx[k])

# ---- helpers for exact equality of order (incl. ties) ---- #
def _canonical_groups(points_dict, axis):
    groups = order_groups_from_points_dict(points_dict, axis=axis)
    return [sorted(g) for g in groups]

def _same_order(orig_points, test_points, axis):
    return _canonical_groups(orig_points, axis) == _canonical_groups(test_points, axis)

# ---------------- Compute maxdist ---------------- #
def compute_maxdist(k_points, l_points):
    dists = []
    for i in range(len(k_points)-1):
        dists.append(dist_xy(k_points[i], k_points[i+1]))
    for i in range(len(l_points)-1):
        dists.append(dist_xy(l_points[i], l_points[i+1]))
    return max(dists) if dists else 1.0

maxdist = compute_maxdist(k_points, l_points)

# ---------------- Compute centroid of initial points (kept for info) ---------------- #
def compute_initial_centroid(k_points, l_points):
    pts = []
    if len(k_points) > 0:
        pts.extend([(float(x), float(y)) for x, y in k_points])
    if len(l_points) > 0:
        pts.extend([(float(x), float(y)) for x, y in l_points])
    if not pts:
        return 0.0, 0.0
    arr = np.array(pts, dtype=float)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

centroid_x, centroid_y = compute_initial_centroid(k_points, l_points)

# ---------------- Bounding BOX based on extreme original points +/- maxdist ---------------- #
def compute_original_extents(k_points, l_points):
    pts = []
    if len(k_points) > 0:
        pts.extend([(float(x), float(y)) for x, y in k_points])
    if len(l_points) > 0:
        pts.extend([(float(x), float(y)) for x, y in l_points])
    if not pts:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(pts, dtype=float)
    minx = float(np.min(arr[:, 0]))
    maxx = float(np.max(arr[:, 0]))
    miny = float(np.min(arr[:, 1]))
    maxy = float(np.max(arr[:, 1]))
    return minx, maxx, miny, maxy

minx0, maxx0, miny0, maxy0 = compute_original_extents(k_points, l_points)

# Place box edges at maxdist beyond the extrema (no extra margins)
x_min_new = minx0 - maxdist
x_max_new = maxx0 + maxdist
y_min_new = miny0 - maxdist
y_max_new = maxy0 + maxdist

# Box parameters
box_center = ((x_min_new + x_max_new) / 2.0, (y_min_new + y_max_new) / 2.0)
box_half_x = (x_max_new - x_min_new) / 2.0
box_half_y = (y_max_new - y_min_new) / 2.0

# ---------------- Bounding box and fixed-angle helpers (RECTANGLE) ---------------- #
def is_inside_box_strict(x, y, center, hx, hy, eps=1e-12):
    """Strictly inside the axis-aligned box: not outside and not ON the border."""
    cx, cy = center
    return (cx - hx + eps) < x < (cx + hx - eps) and (cy - hy + eps) < y < (cy + hy - eps)

def ray_box_rmax(origin, theta, center, hx, hy):
    """
    Maximum radius r such that (ox + r cosθ, oy + r sinθ) remains inside/ON
    the axis-aligned rectangle. Returns 0.0 if there is no forward intersection.
    """
    ox, oy = origin
    cx, cy = center
    xmin = cx - hx
    xmax = cx + hx
    ymin = cy - hy
    ymax = cy + hy

    dx = math.cos(theta)
    dy = math.sin(theta)

    t_vals = []

    if abs(dx) > 1e-15:
        t = (xmin - ox) / dx
        y = oy + t * dy
        if t >= 0 and (ymin - 1e-12) <= y <= (ymax + 1e-12):
            t_vals.append(t)
        t = (xmax - ox) / dx
        y = oy + t * dy
        if t >= 0 and (ymin - 1e-12) <= y <= (ymax + 1e-12):
            t_vals.append(t)

    if abs(dy) > 1e-15:
        t = (ymin - oy) / dy
        x = ox + t * dx
        if t >= 0 and (xmin - 1e-12) <= x <= (xmax + 1e-12):
            t_vals.append(t)
        t = (ymax - oy) / dy
        x = ox + t * dx
        if t >= 0 and (xmin - 1e-12) <= x <= (xmax + 1e-12):
            t_vals.append(t)

    if not t_vals:
        return 0.0
    return max(0.0, float(min(t_vals)))

def choose_fixed_theta_for_r_strict_inside_box(origin, r, center, hx, hy, max_attempts=50000, shrink_every=2000):
    """
    Find one angle such that the first point (radius r) lies STRICTLY inside the rectangle.
    Returns (theta, r_eff, rmax_theta). Repeats random angle selection until
    r < rmax - eps. If needed, slightly shrinks r to avoid pathological alignment.
    """
    ox, oy = origin
    if r <= 0:
        theta = random.uniform(0.0, 2.0 * math.pi)
        return theta, 0.0, float("inf")

    r_eff = float(r)
    eps = 1e-10
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        theta = random.uniform(0.0, 2.0 * math.pi)
        rmax = ray_box_rmax((ox, oy), theta, center, hx, hy)
        if rmax > r_eff + eps:
            return theta, r_eff, rmax
        if attempts % shrink_every == 0:
            r_eff *= 0.999  # tiny reduction

    # Fallback: aim at the rectangle center and step just inside
    cx, cy = center
    theta = math.atan2(cy - oy, cx - ox)
    rmax = ray_box_rmax((ox, oy), theta, center, hx, hy)
    r_eff = min(r_eff, max(0.0, rmax - 1e-10))
    return theta, r_eff, rmax

def point_on_fixed_theta_strict(origin, theta, r, rmax, eps=1e-10):
    """Keep the same angle; clamp radius so we remain strictly inside: min(r, rmax - eps)."""
    ox, oy = origin
    r_use = min(max(0.0, float(r)), float(rmax) - eps)
    return (ox + r_use * math.cos(theta), oy + r_use * math.sin(theta))

# ---------------- Reset ---------------- #
reset_clicked_now = False
if reset_clicked:
    st.session_state.latest_blue_k = {}
    st.session_state.latest_blue_l = {}
    st.session_state.prime_k = {}
    st.session_state.prime_l = {}
    st.session_state.prime_count_k = {}
    st.session_state.prime_count_l = {}
    st.session_state.green_snaps_k = []
    st.session_state.green_snaps_l = []
    st.session_state.repeats_done = 0
    st.session_state.pt_trials_pink = []
    st.session_state.pt_trials_gray = []
    st.session_state.pt_final_purple = []
    st.session_state.orig_text = {}
    st.session_state.checkpoint_texts = {}
    st.session_state.checkpoint_placeholders = {}
    st.session_state.axes_locked = False
    reset_clicked_now = True

# ---------------- Base figure (with d1/d2 strings) ---------------- #
def base_fig(x_min_new, x_max_new, y_min_new, y_max_new,
             circle_center=None, circle_radius=None, red_point=None,
             base_d1_text="", cur_d1_text="", base_d2_text="", cur_d2_text="",
             show_tests=True,
             box_center=None, box_half_x=None, box_half_y=None,
             ray_segment=None  # ((x0,y0),(x1,y1)) to visualize the fixed angle
             ):
    fig_local = go.Figure()

    # original k
    if len(k_points) > 0:
        fig_local.add_trace(go.Scatter(
            x=k_points[:, 0], y=k_points[:, 1],
            mode="markers+lines+text", name="k",
            text=[f"<i>k</i><sub>{i}</sub>" for i in range(len(k_points))],
            textposition="top center",
            marker=dict(color="cornflowerblue", size=12),
            line=dict(color="cornflowerblue", dash="dash"),
            textfont=dict(color="cornflowerblue", size=20),
            hoverinfo="none"
        ))
    # original l
    if len(l_points) > 0:
        fig_local.add_trace(go.Scatter(
            x=l_points[:, 0], y=l_points[:, 1],
            mode="markers+lines+text", name="l",
            text=[f"<i>l</i><sub>{i}</sub>" for i in range(len(l_points))],
            textposition="top center",
            marker=dict(color="orange", size=12),
            line=dict(color="orange", dash="dash"),
            textfont=dict(color="orange", size=20),
            hoverinfo="none"
        ))

    # persistent green snapshots
    for (xs, ys) in st.session_state.green_snaps_k:
        if len(xs) >= 2:
            fig_local.add_trace(go.Scatter(
                x=list(xs), y=list(ys), mode="lines",
                line=dict(color="#90EE90", width=2, dash="dot"),
                hoverinfo="none", showlegend=False
            ))
    for (xs, ys) in st.session_state.green_snaps_l:
        if len(xs) >= 2:
            fig_local.add_trace(go.Scatter(
                x=list(xs), y=list(ys), mode="lines",
                line=dict(color="#90EE90", width=2, dash="dot"),
                hoverinfo="none", showlegend=False
            ))

    # hybrid current lines
    def hybrid(points, store):
        if len(points) == 0: return []
        out = []
        for i in range(len(points)):
            out.append(store[i] if i in store else (float(points[i][0]), float(points[i][1])))
        return out

    k_hybrid = hybrid(k_points, st.session_state.prime_k)
    l_hybrid = hybrid(l_points, st.session_state.prime_l)
    if len(k_hybrid) >= 2:
        xs, ys = zip(*k_hybrid)
        fig_local.add_trace(go.Scatter(
            x=list(xs), y=list(ys), mode="lines+markers",
            line=dict(color="#6AA5FF", width=3),
            marker=dict(color="#6AA5FF", size=7),
            name="k′ (hybrid)", hoverinfo="none", showlegend=False
        ))
    if len(l_hybrid) >= 2:
        xs, ys = zip(*l_hybrid)
        fig_local.add_trace(go.Scatter(
            x=list(xs), y=list(ys), mode="lines+markers",
            line=dict(color="#FFB84D", width=3),
            marker=dict(color="#FFB84D", size=7),
            name="l′ (hybrid)", hoverinfo="none", showlegend=False
        ))

    # red circle and trial
    if circle_center is not None and circle_radius is not None:
        cxr, cyr = circle_center
        r = max(0.0, float(circle_radius))
        fig_local.add_shape(
            type="circle", xref="x", yref="y",
            x0=cxr - r, x1=cxr + r, y0=cyr - r, y1=cyr + r,
            line=dict(color="rgba(220,20,60,0.4)", width=3),
            fillcolor="rgba(0,0,0,0)"
        )
    if red_point is not None:
        fig_local.add_trace(go.Scatter(
            x=[red_point[0]], y=[red_point[1]],
            mode="markers", marker=dict(color="red", size=7),
            hoverinfo="none", showlegend=False
        ))

    # fixed angle (debug line)
    if ray_segment is not None:
        (x0, y0), (x1, y1) = ray_segment
        fig_local.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.55)", dash="dot", width=2),
            hoverinfo="none", showlegend=False
        ))

    # latest blue points
    blue_all = list(st.session_state.latest_blue_k.values()) + list(st.session_state.latest_blue_l.values())
    if blue_all:
        fig_local.add_trace(go.Scatter(
            x=[p[0] for p in blue_all], y=[p[1] for p in blue_all],
            mode="markers", marker=dict(color="darkblue", size=9),
            hoverinfo="none", showlegend=False
        ))

    # test points & finals
    if show_tests and st.session_state.pt_trials_pink:
        xp, yp = zip(*st.session_state.pt_trials_pink)
        fig_local.add_trace(go.Scatter(x=list(xp), y=list(yp), mode="markers",
                                       marker=dict(color="#ff69b4", size=11), opacity=0.50,
                                       hoverinfo="none", showlegend=False))
    if show_tests and st.session_state.pt_trials_gray:
        xg, yg = zip(*st.session_state.pt_trials_gray)
        fig_local.add_trace(go.Scatter(x=list(xg), y=list(yg), mode="markers",
                                       marker=dict(color="#808080", size=11), opacity=0.15,
                                       hoverinfo="none", showlegend=False))
    if st.session_state.pt_final_purple:
        xu, yu = zip(*st.session_state.pt_final_purple)
        fig_local.add_trace(go.Scatter(x=list(xu), y=list(yu), mode="markers",
                                       marker=dict(color="#800080", size=11), opacity=0.35,
                                       hoverinfo="none", showlegend=False))

    # dashed box + center mark
    if (box_center is not None) and (box_half_x is not None) and (box_half_y is not None) and np.isfinite(box_half_x) and np.isfinite(box_half_y):
        cx, cy = box_center
        fig_local.add_shape(type="rect", xref="x", yref="y",
                            x0=cx - box_half_x, x1=cx + box_half_x,
                            y0=cy - box_half_y, y1=cy + box_half_y,
                            line=dict(color="black", width=2, dash="dash"),
                            fillcolor="rgba(0,0,0,0)")
        fig_local.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                                       marker=dict(symbol="cross", size=16, color="black",
                                                   line=dict(width=2, color="black")),
                                       hoverinfo="none", showlegend=False))

    # axes labels
    fig_local.add_annotation(xref="paper", yref="paper", x=0.99, y=0.01,
                             text="<i>d</i><sub>1</sub>", showarrow=False,
                             font=dict(size=26, color="black"))
    fig_local.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99,
                             text="<i>d</i><sub>2</sub>", showarrow=False,
                             font=dict(size=26, color="black"))

    # d1/d2 blocks
    if base_d1_text:
        fig_local.add_annotation(xref="paper", yref="paper", x=0.06, y=0.98,
                                 text=f"<b>d1:</b> {base_d1_text}", showarrow=False,
                                 align="left", font=dict(size=18),
                                 borderpad=2, bgcolor="rgba(255,255,255,0.8)")
    if cur_d1_text:
        fig_local.add_annotation(xref="paper", yref="paper", x=0.06, y=0.90,
                                 text=f"<b>d1:</b> {cur_d1_text}", showarrow=False,
                                 align="left", font=dict(size=18),
                                 borderpad=2, bgcolor="rgba(255,255,255,0.8)")
    if base_d2_text:
        fig_local.add_annotation(xref="paper", yref="paper", x=0.06, y=0.82,
                                 text=f"<b>d2:</b> {base_d2_text}", showarrow=False,
                                 align="left", font=dict(size=18),
                                 borderpad=2, bgcolor="rgba(255,255,255,0.8)")
    if cur_d2_text:
        fig_local.add_annotation(xref="paper", yref="paper", x=0.06, y=0.74,
                                 text=f"<b>d2:</b> {cur_d2_text}", showarrow=False,
                                 align="left", font=dict(size=18),
                                 borderpad=2, bgcolor="rgba(255,255,255,0.8)")

    if len(subset_times) > 0:
        time_label = f"t = {subset_times[0]}–{subset_times[-1]}"
        fig_local.add_annotation(xref="paper", yref="paper", x=0.98, y=1.02,
                                 text=f"{time_label}", showarrow=False,
                                 font=dict(size=18, color="dimgray"))

    def first_mult_of_5_leq(val): return np.floor(val / 5.0) * 5.0
    x_tick0 = first_mult_of_5_leq(x_min_new)
    y_tick0 = first_mult_of_5_leq(y_min_new)

    fig_local.update_layout(
        xaxis=dict(tickmode="linear", tick0=x_tick0, dtick=5,
                   showticklabels=True, ticks="outside", ticklen=6,
                   showgrid=False, zeroline=False,
                   showline=True, linecolor="black", linewidth=2,
                   range=[x_min_new, x_max_new],
                   scaleanchor="y", scaleratio=1, tickfont=dict(size=14)),
        yaxis=dict(tickmode="linear", tick0=y_tick0, dtick=5,
                   showticklabels=True, ticks="outside", ticklen=6,
                   showgrid=False, zeroline=False,
                   showline=True, linecolor="black", linewidth=2,
                   range=[y_min_new, y_max_new],
                   tickfont=dict(size=14)),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="white", dragmode=False, showlegend=False
    )
    return fig_local

# ---------------- Helpers: ORDER text + styling ---------------- #
def _format_order_line(points_dict, axis_label):
    axis = 'x' if axis_label == 'd1' else 'y'
    groups = order_groups_from_points_dict(points_dict, axis=axis)
    name_map = build_name_map_base(points_dict)
    parts = []
    for g in groups:
        nice = sorted(name_map[k] for k in g)
        parts.append(" = ".join(nice))
    return f"{axis_label}: " + " < ".join(parts) if parts else f"{axis_label}: (none)"

def _compose_original_text(original_points):
    lines = []
    lines.append(_format_order_line(original_points, 'd1').replace("d1:", "d1 original:"))
    lines.append(_format_order_line(original_points, 'd2').replace("d2:", "d2 original:"))
    return "\n".join(lines)

def _status_html(is_ok: bool) -> str:
    if is_ok:
        return '<span style="color:#1f6bff;"><b>ok</b></span>'
    else:
        return '<span style="color:#c01616;"><b>not correct</b></span>'

def _compose_checkpoint_html(original_points, pts, step):
    ok_d1 = _same_order(original_points, pts, axis='x')
    ok_d2 = _same_order(original_points, pts, axis='y')
    line_d1 = _format_order_line(pts, 'd1').replace("d1:", f"d1 new ({step}th):")
    line_d2 = _format_order_line(pts, 'd2').replace("d2:", f"d2 new ({step}th):")
    html = (
        f"{line_d1}  {_status_html(ok_d1)}<br>"
        f"{line_d2}  {_status_html(ok_d2)}"
    )
    return f"""
    <div style="border:1px solid #ddd;border-radius:6px;padding:8px 10px;background:#fff;">
      {html}
    </div>
    """

# ---------------- Placeholders in COL1 ---------------- #
with col1:
    canvas = st.empty()

    subtitle_row = st.columns(2)
    with subtitle_row[0]:
        st.markdown("**original configuration**")
    with subtitle_row[1]:
        st.markdown("**generated configuration**")

    mats_row = st.columns(4)
    mat_orig_d1 = mats_row[0].empty()
    mat_orig_d2 = mats_row[1].empty()
    mat_gen_d1  = mats_row[2].empty()
    mat_gen_d2  = mats_row[3].empty()

# ---------------- Build original matrices ---------------- #
display_labels, base_keys, key_to_point_base, M1, M2 = build_base_matrices(k_points, l_points, timepoints_n)
show_labels = (timepoints_n < 4)
show_values = (timepoints_n <= 3)

plot_heatmap(M1, display_labels, display_labels, "<i>d</i><sub>1</sub>",
             200, 200, show_values=show_values, fontsize=14, target=mat_orig_d1, show_axis_labels=show_labels)
plot_heatmap(M2, display_labels, display_labels, "<i>d</i><sub>2</sub>",
             200, 200, show_values=show_values, fontsize=14, target=mat_orig_d2, show_axis_labels=show_labels)

plot_heatmap(M1, display_labels, display_labels, "<i>d</i><sub>1</sub>",
             200, 200, show_values=show_values, fontsize=14, target=mat_gen_d1, show_axis_labels=show_labels)
plot_heatmap(M2, display_labels, display_labels, "<i>d</i><sub>2</sub>",
             200, 200, show_values=show_values, fontsize=14, target=mat_gen_d2, show_axis_labels=show_labels)

# ---------------- ORIGINAL text box ---------------- #
with col1:
    st.markdown("### Overview")
    if not st.session_state.orig_text:
        st.session_state.orig_text = _compose_original_text(key_to_point_base)
    st.text_area("Original configuration", value=st.session_state.orig_text, height=110, disabled=True, key="orig_box")

# ---------------- Axis padding updater (locked after originals placed) ---------------- #
def auto_expand_axes_for_point(px, py):
    global x_min_new, x_max_new, y_min_new, y_max_new
    if st.session_state.axes_locked:
        return  # Do NOT rescale after the original points have been placed
    margin = 0.0
    if px < x_min_new + margin: x_min_new = px - margin
    if px > x_max_new - margin: x_max_new = px + margin
    if py < y_min_new + margin: y_min_new = py - margin
    if py > y_max_new - margin: y_max_new = py + margin

# ---------------- Ordering text helpers for figure ---------------- #
def base_order_texts(points_dict):
    base_x_groups = order_groups_from_points_dict(points_dict, axis='x')
    base_y_groups = order_groups_from_points_dict(points_dict, axis='y')
    base_name_map = build_name_map_base(points_dict)
    return (
        groups_to_string(base_x_groups, base_name_map),
        groups_to_string(base_y_groups, base_name_map),
        base_x_groups, base_y_groups
    )

def current_order_texts(points_dict):
    cur_x_groups = order_groups_from_points_dict(points_dict, axis='x')
    cur_y_groups = order_groups_from_points_dict(points_dict, axis='y')
    cur_name_map  = build_name_map_current(points_dict)
    return (
        groups_to_string(cur_x_groups, cur_name_map),
        groups_to_string(cur_y_groups, cur_name_map),
        cur_x_groups, cur_y_groups
    )

# ---------------- One search iteration loop (fixed theta + lastcorrect + early stop) ---------------- #
def run_halving_iteration(chosen_label, x_base_x, x_base_y, maxdist,
                          base_keys, key_to_point_base, M1, M2,
                          canvas, mat_gen_d1, mat_gen_d2,
                          show_labels, strategy,
                          mark_k0_purple=False):
    if maxdist is None or maxdist <= 0:
        return False, None

    fam = 'k' if chosen_label.startswith('k') else 'l'
    idx = parse_index_from_label(chosen_label)
    is_k0 = (fam == 'k' and idx == 0)

    # starting center (point x: previous generation / base)
    if fam == 'k' and idx in st.session_state.prime_k:
        cx, cy = st.session_state.prime_k[idx]
    elif fam == 'l' and idx in st.session_state.prime_l:
        cx, cy = st.session_state.prime_l[idx]
    else:
        cx, cy = float(x_base_x), float(x_base_y)

    bcenter = box_center
    hx, hy = box_half_x, box_half_y

    base_d1_text, base_d2_text, base_x_groups, base_y_groups = base_order_texts(key_to_point_base)

    debug_ray = None  # for visualization of the fixed angle

    def draw_and_compare(trial_xy):
        M3_try, M4_try, _ = build_prime_matrices(
            base_keys, merged_current_points_dict(key_to_point_base),
            chosen_label, trial_xy
        )
        plot_heatmap(M3_try, display_labels, display_labels, "<i>d</i><sub>1</sub>",
                     200, 200, show_values=show_values, fontsize=14,
                     target=mat_gen_d1, show_axis_labels=show_labels)
        plot_heatmap(M4_try, display_labels, display_labels, "<i>d</i><sub>2</sub>",
                     200, 200, show_values=show_values, fontsize=14,
                     target=mat_gen_d2, show_axis_labels=show_labels)

        current_points_for_order = dict(merged_current_points_dict(key_to_point_base))
        chosen_key = f"{'k' if chosen_label.startswith('k') else 'l'}|t{idx}"
        current_points_for_order[chosen_key] = trial_xy
        cur_d1_txt_core, cur_d2_txt_core, cur_x_groups, cur_y_groups = current_order_texts(current_points_for_order)
        mism_x = mismatch_count(base_x_groups, cur_x_groups)
        mism_y = mismatch_count(base_y_groups, cur_y_groups)
        cur_d1_text = f"{cur_d1_txt_core}  {mism_x}"
        cur_d2_text = f"{cur_d2_txt_core}  {mism_y}"

        fig = base_fig(
            x_min_new, x_max_new, y_min_new, y_max_new,
            circle_center=(cx, cy),
            circle_radius=dist_xy((cx, cy), trial_xy),
            red_point=trial_xy,
            base_d1_text=base_d1_text, cur_d1_text=cur_d1_text,
            base_d2_text=base_d2_text, cur_d2_text=cur_d2_text,
            show_tests=show_test_points,
            box_center=bcenter, box_half_x=hx, box_half_y=hy,
            ray_segment=debug_ray
        )
        canvas.plotly_chart(fig, use_container_width=True)
        auto_expand_axes_for_point(trial_xy[0], trial_xy[1])

        matched = (np.array_equal(M1, M3_try) and np.array_equal(M2, M4_try))
        return matched

    def commit(final_xy):
        # Update blue + prime state
        if fam == 'k':
            st.session_state.latest_blue_k[idx] = final_xy
            st.session_state.prime_k[idx] = final_xy
            st.session_state.prime_count_k[idx] = min(st.session_state.prime_count_k.get(idx, 0) + 1, 3)
        else:
            st.session_state.latest_blue_l[idx] = final_xy
            st.session_state.prime_l[idx] = final_xy
            st.session_state.prime_count_l[idx] = min(st.session_state.prime_count_l.get(idx, 0) + 1, 3)

        # k0: register final purple marker when matrix similarity holds
        M3_fin, M4_fin, _ = build_prime_matrices(
            base_keys,
            merged_current_points_dict(key_to_point_base),
            chosen_label,
            final_xy
        )
        if is_k0 and (np.array_equal(M1, M3_fin) and np.array_equal(M2, M4_fin)):
            st.session_state.pt_final_purple.append((final_xy[0], final_xy[1]))

        committed_points = merged_current_points_dict(key_to_point_base)
        c_d1_txt, c_d2_txt, c_x_groups2, c_y_groups2 = current_order_texts(committed_points)
        mism_x2 = mismatch_count(base_x_groups, c_x_groups2)
        mism_y2 = mismatch_count(base_y_groups, c_y_groups2)

        fig_final = base_fig(
            x_min_new, x_max_new, y_min_new, y_max_new,
            circle_center=None, circle_radius=None, red_point=None,
            base_d1_text=base_d1_text, cur_d1_text=f"{c_d1_txt}  {mism_x2}",
            base_d2_text=base_d2_text, cur_d2_text=f"{c_d2_txt}  {mism_y2}",
            show_tests=show_test_points,
            box_center=bcenter, box_half_x=hx, box_half_y=hy,
            ray_segment=debug_ray
        )
        canvas.plotly_chart(fig_final, use_container_width=True)

        plot_heatmap(M3_fin, display_labels, display_labels, "<i>d</i><sub>1</sub>",
                     200, 200, show_values=show_values, fontsize=14, target=mat_gen_d1, show_axis_labels=show_labels)
        plot_heatmap(M4_fin, display_labels, display_labels, "<i>d</i><sub>2</sub>",
                     200, 200, show_values=show_values, fontsize=14, target=mat_gen_d2, show_axis_labels=show_labels)

    def update_checkpoint_text_if_needed():
        iters_done = st.session_state.repeats_done + 1
        if iters_done in SPECIAL_CONFIGS:
            pts_now = dict(merged_current_points_dict(key_to_point_base))
            html = _compose_checkpoint_html(key_to_point_base, pts_now, iters_done)
            st.session_state.checkpoint_texts[iters_done] = html
            ph = st.session_state.checkpoint_placeholders.get(iters_done)
            if ph is not None:
                ph.markdown(html, unsafe_allow_html=True)

    # -------- Strategy implementations -------- #
    if strategy == "Exponential":
        # Find a fixed angle so that the first trial is strictly inside the box
        r = clamp_radius(float(maxdist), maxdist)
        theta, r_eff, rmax_theta = choose_fixed_theta_for_r_strict_inside_box(
            origin=(cx, cy), r=r, center=bcenter, hx=hx, hy=hy
        )

        # Debug segment up to just below rmax
        eps_line = 1e-10
        x1 = cx + (rmax_theta - eps_line) * math.cos(theta)
        y1 = cy + (rmax_theta - eps_line) * math.sin(theta)
        debug_ray = ((cx, cy), (x1, y1))

        # lastcorrect: never commit a wrong trial point
        lastcorrect = (cx, cy)
        found_any_match = False

        # First trial
        trial_xy = point_on_fixed_theta_strict((cx, cy), theta, r_eff, rmax_theta)
        matched = draw_and_compare(trial_xy)
        if is_k0:
            (st.session_state.pt_trials_pink if matched else st.session_state.pt_trials_gray).append((trial_xy[0], trial_xy[1]))
        sleep_if_needed()
        if matched:
            lastcorrect = trial_xy
            found_any_match = True
            commit(lastcorrect)               # STOP immediately on match
            update_checkpoint_text_if_needed()
            sleep_if_needed()
            return True, lastcorrect

        # Further halvings with the same angle
        r_cur = r_eff
        for _ in range(2, MAX_STEPS + 1):
            r_cur *= 0.5
            r_cur = clamp_radius(r_cur, maxdist)
            trial_xy = point_on_fixed_theta_strict((cx, cy), theta, r_cur, rmax_theta)
            matched = draw_and_compare(trial_xy)
            if is_k0:
                (st.session_state.pt_trials_pink if matched else st.session_state.pt_trials_gray).append((trial_xy[0], trial_xy[1]))
            sleep_if_needed()
            if matched:
                lastcorrect = trial_xy
                found_any_match = True
                commit(lastcorrect)           # STOP immediately on match
                update_checkpoint_text_if_needed()
                sleep_if_needed()
                return True, lastcorrect

        # No match found: commit lastcorrect (thus keep previous generation)
        commit(lastcorrect)
        update_checkpoint_text_if_needed()
        sleep_if_needed()
        return found_any_match, lastcorrect

    else:  # Binary
        lastcorrect = (cx, cy)
        found_any_match = False

        halfstep = float(maxdist) / 2.0
        pos = clamp_radius(halfstep, maxdist)

        theta, pos_eff, rmax_theta = choose_fixed_theta_for_r_strict_inside_box(
            origin=(cx, cy), r=pos, center=bcenter, hx=hx, hy=hy
        )

        # Debug segment
        eps_line = 1e-10
        x1 = cx + (rmax_theta - eps_line) * math.cos(theta)
        y1 = cy + (rmax_theta - eps_line) * math.sin(theta)
        debug_ray = ((cx, cy), (x1, y1))

        def safe_point_at_pos(p):
            return point_on_fixed_theta_strict((cx, cy), theta, p, rmax_theta)

        trial_xy = safe_point_at_pos(pos_eff)
        prev_was_match = draw_and_compare(trial_xy)
        if prev_was_match:
            lastcorrect = trial_xy
            found_any_match = True
            if is_k0: st.session_state.pt_trials_pink.append((trial_xy[0], trial_xy[1]))
        else:
            if is_k0: st.session_state.pt_trials_gray.append((trial_xy[0], trial_xy[1]))
        sleep_if_needed()

        pos_cur = pos_eff
        for _ in range(10):
            halfstep *= 0.5
            pos_cur = clamp_radius(pos_cur + (halfstep if prev_was_match else -halfstep), maxdist)
            trial_xy = safe_point_at_pos(pos_cur)
            now_match = draw_and_compare(trial_xy)
            if now_match:
                lastcorrect = trial_xy
                found_any_match = True
                if is_k0: st.session_state.pt_trials_pink.append((trial_xy[0], trial_xy[1]))
            else:
                if is_k0: st.session_state.pt_trials_gray.append((trial_xy[0], trial_xy[1]))
            prev_was_match = now_match
            sleep_if_needed()

        commit(lastcorrect)
        update_checkpoint_text_if_needed()
        sleep_if_needed()
        return found_any_match, lastcorrect

# ---------------- Snapshot helper ---------------- #
def build_hybrid_polyline(base_points, primes_map):
    if len(base_points) == 0:
        return []
    pts = []
    n = len(base_points)
    for i in range(n):
        pts.append(primes_map[i] if i in primes_map else (float(base_points[i][0]), float(base_points[i][1])))
    return pts

def snapshot_current_hybrids():
    k_h = build_hybrid_polyline(k_points, st.session_state.prime_k)
    if len(k_h) >= 2:
        xs_k, ys_k = zip(*k_h)
        st.session_state.green_snaps_k.append((list(xs_k), list(ys_k)))
    l_h = build_hybrid_polyline(l_points, st.session_state.prime_l)
    if len(l_h) >= 2:
        xs_l, ys_l = zip(*l_h)
        st.session_state.green_snaps_l.append((list(xs_l), list(ys_l)))

# ---------------- Config runner ---------------- #
def run_configs(n_configs=1, iters_per_config=3, strategy="Exponential", mark_k0_purple=False):
    if len(labels_points) == 0 or maxdist is None or maxdist <= 0:
        return

    # Axes are already locked after originals (see below).
    # Keep original configuration text up to date.
    captured_original = dict(merged_current_points_dict(key_to_point_base))
    st.session_state.orig_text = _compose_original_text(captured_original)

    for _conf in range(n_configs):
        for _ in range(iters_per_config):
            chosen_label, chosen_idx, xbx, xby = random.choice(labels_points)
            run_halving_iteration(
                chosen_label=chosen_label,
                x_base_x=xbx, x_base_y=xby, maxdist=maxdist,
                base_keys=base_keys,
                key_to_point_base=key_to_point_base,
                M1=M1, M2=M2,
                canvas=canvas,
                mat_gen_d1=mat_gen_d1,
                mat_gen_d2=mat_gen_d2,
                show_labels=show_labels,
                strategy=strategy,
                mark_k0_purple=mark_k0_purple
            )
            st.session_state.repeats_done += 1

        snapshot_current_hybrids()
        fig_snap = base_fig(
            x_min_new, x_max_new, y_min_new, y_max_new,
            circle_center=None, circle_radius=None, red_point=None,
            show_tests=show_test_points,
            box_center=box_center, box_half_x=box_half_x, box_half_y=box_half_y
        )
        canvas.plotly_chart(fig_snap, use_container_width=True)

# ---------------- Initial draw ---------------- #
display_labels, base_keys, key_to_point_base, M1, M2 = build_base_matrices(k_points, l_points, timepoints_n)
base_x_groups0 = order_groups_from_points_dict(key_to_point_base, axis='x')
base_y_groups0 = order_groups_from_points_dict(key_to_point_base, axis='y')
base_name_map0 = build_name_map_base(key_to_point_base)
base_d1_text0 = groups_to_string(base_x_groups0, base_name_map0)
base_d2_text0 = groups_to_string(base_y_groups0, base_name_map0)

cur_points0 = dict(key_to_point_base)
cur_name_map0 = build_name_map_current(cur_points0)
cur_d1_text0 = groups_to_string(base_x_groups0, cur_name_map0) + "  0"
cur_d2_text0 = groups_to_string(base_y_groups0, cur_name_map0) + "  0"

fig0 = base_fig(
    x_min_new, x_max_new, y_min_new, y_max_new,
    circle_center=None, circle_radius=None, red_point=None,
    base_d1_text=base_d1_text0, cur_d1_text=cur_d1_text0,
    base_d2_text=base_d2_text0, cur_d2_text=cur_d2_text0,
    show_tests=show_test_points,
    box_center=box_center, box_half_x=box_half_x, box_half_y=box_half_y
)
canvas.plotly_chart(fig0, use_container_width=True)

# >>> LOCK AXES NOW: once the original points are drawn, do not rescale anymore <<<
st.session_state.axes_locked = True

# ---------------- Configuration text boxes 1..99 (collapsible) ---------------- #
with col1:
    with st.expander("Configuration updates (1 … 99)", expanded=False):
        for step in CONFIG_STEPS:
            if step not in st.session_state.checkpoint_placeholders:
                st.session_state.checkpoint_placeholders[step] = st.empty()
            html = st.session_state.checkpoint_texts.get(step)
            if html is None:
                html = """
                <div style="border:1px dashed #ccc;border-radius:6px;padding:8px 10px;background:#fafafa;color:#666;">
                  (not reached yet)
                </div>
                """
            st.session_state.checkpoint_placeholders[step].markdown(html, unsafe_allow_html=True)

# ---------------- Buttons ---------------- #
if run1_clicked:
    run_configs(n_configs=1, iters_per_config=iterations_per_config, strategy=strategy, mark_k0_purple=False)

if run3_clicked:
    run_configs(n_configs=3, iters_per_config=iterations_per_config, strategy=strategy, mark_k0_purple=False)

if run500_clicked:
    run_configs(n_configs=500, iters_per_config=iterations_per_config, strategy=strategy, mark_k0_purple=True)
