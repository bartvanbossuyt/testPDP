"""
PDP Interactive Learning App
────────────────────────────
An educational Streamlit application that teaches the
Point Descriptor Precedence (PDP) framework step by step.

Run with:
    streamlit run app_pdp_learn.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDP Interactive Learning",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SESSION STATE DEFAULTS ─────────────────────────────────────────────────
_defaults = {
    "s2_descriptor": "X-axis",
    "s3_descriptor": "X-axis",
    "s4_descriptor": "X-axis",
    "s5_descriptor": "X-axis",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── DATA LOADING ───────────────────────────────────────────────────────────
# Architecture note: swap `uploaded_file=None` below for `st.file_uploader(...)`
# to enable CSV upload without changing any downstream logic.

DATA_PATH = Path(__file__).parent / "overtake_event_filtered_timestamps.csv"

HARDCODED = [
    [43, 0, 0, -5.262385883284942,  -3.5254968448771686],
    [43, 0, 1, 45.65635338869608,   -3.5530562770118417],
    [43, 1, 0, 66.89601440518928,   -3.516818421025848],
    [43, 1, 1, 117.81744899912938,  -3.491526098912859],
    [43, 2, 0, 145.13780642466915,   0.008551034100465],
    [43, 2, 1, 190.85319323446487,  -3.453028957317177],
    [43, 3, 0, 470.01018049556,     -0.005633528785371],
    [43, 3, 1, 450.5400127795915,   -3.462785333751645],
    [43, 4, 0, 550.8153596943821,   -3.330869274461852],
    [43, 4, 1, 517.2055479253376,   -3.472193346929517],
    [43, 5, 0, 643.6932491900882,   -3.583586709083216],
    [43, 5, 1, 595.1035921843552,   -3.5615063153962567],
]


def load_data(uploaded_file=None) -> pd.DataFrame:
    """
    Load PDP trajectory data.

    Architecture: pass an UploadedFile object (from st.file_uploader) to use
    an external CSV; leave None to use the hardcoded dataset.
    The CSV must have NO header and exactly 5 columns: conID, tstID, poiID, x, y.
    """
    if uploaded_file is not None:
        return pd.read_csv(
            uploaded_file, header=None, names=["conID", "tstID", "poiID", "x", "y"]
        )
    if DATA_PATH.exists():
        return pd.read_csv(
            DATA_PATH, header=None, names=["conID", "tstID", "poiID", "x", "y"]
        )
    return pd.DataFrame(HARDCODED, columns=["conID", "tstID", "poiID", "x", "y"])


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📐 PDP Learning App")
    st.markdown("---")

    st.subheader("📁 Data Source")
    # ── To enable file upload, uncomment the two lines below ──────────────
    # use_upload = st.checkbox("Load custom CSV", value=False)
    # _uf = st.file_uploader("Upload (no header, 5 cols)", type="csv") if use_upload else None
    _uf = None  # currently locked to built-in dataset
    st.info("Using built-in overtake event\n(6 timesteps, 2 objects)")

    st.markdown("---")
    st.markdown(
        "**Navigate** the tabs above in order — each step builds on the previous one."
    )
    st.markdown("---")
    st.markdown(
        """
**PDP in one sentence:**  
Replace exact coordinates with *who is ahead of whom* along a chosen reference direction.
"""
    )

# ─── LOAD & DERIVE GLOBALS ──────────────────────────────────────────────────
df = load_data(_uf)
N_TST = df["tstID"].nunique()
N_POI = df["poiID"].nunique()
TIMESTEPS = sorted(df["tstID"].unique())
OBJECTS = sorted(df["poiID"].unique())

OBJ_COLORS = ["#E63946", "#2563EB"]  # Obj-0 = crimson, Obj-1 = blue

# ─── PDP MATH HELPERS ───────────────────────────────────────────────────────

def get_scalars(df_sub: pd.DataFrame, descriptor: str, angle_deg: float = 45.0) -> np.ndarray:
    """Project 2D points onto a 1D descriptor axis."""
    if descriptor == "X-axis":
        return df_sub["x"].to_numpy(dtype=float)
    if descriptor == "Y-axis":
        return df_sub["y"].to_numpy(dtype=float)
    # Diagonal: project onto direction (cos θ, sin θ)
    rad = math.radians(angle_deg)
    return df_sub["x"].to_numpy(float) * math.cos(rad) + df_sub["y"].to_numpy(float) * math.sin(rad)


def build_matrix(scalars: np.ndarray, rho: float = 0.0):
    """
    Build the PDP inequality matrix.

    Returns
    -------
    sym  : 2-D array of strings ('<', '=', '>')
    num  : 2-D int array  (0 = '<', 1 = '=', 2 = '>')
    """
    n = len(scalars)
    sym = np.empty((n, n), dtype=object)
    num = np.ones((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            diff = float(scalars[i]) - float(scalars[j])
            if diff > rho:
                sym[i, j], num[i, j] = ">", 2
            elif diff < -rho:
                sym[i, j], num[i, j] = "<", 0
            else:
                sym[i, j], num[i, j] = "=", 1
    return sym, num


# Colorscale: 0=< (green), 1== (gold), 2=> (red)
_CSCALE = [
    [0.00, "#4CAF50"],
    [0.49, "#4CAF50"],
    [0.50, "#FFC107"],
    [0.51, "#FFC107"],
    [0.99, "#FFC107"],
    [1.00, "#F44336"],
]

# ─── REUSABLE MATRIX FIGURE ─────────────────────────────────────────────────

def matrix_fig(sym, labels, title="", height=None, width=None):
    n = len(labels)
    num = np.where(sym == "<", 0, np.where(sym == "=", 1, 2)).astype(float)

    text = sym.tolist()

    cell_px = max(60, min(90, 480 // n))
    h = height or (cell_px * n + 120)
    w = width or (cell_px * n + 120)

    fig = go.Figure(
        go.Heatmap(
            z=num,
            x=labels,
            y=labels,
            colorscale=_CSCALE,
            zmin=0,
            zmax=2,
            text=text,
            texttemplate="%{text}",
            textfont={"size": max(12, cell_px // 4), "color": "white"},
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(side="top", tickangle=-45),
        yaxis=dict(autorange="reversed"),
        height=h,
        width=w,
        margin=dict(l=10, r=10, t=80, b=10),
        template="plotly_white",
    )
    return fig


# ─── TABS ───────────────────────────────────────────────────────────────────
st.title("📐 Understanding PDP: Point Descriptor Precedence")
st.markdown(
    "> **Dataset:** An overtake manoeuvre — 2 vehicles moving forward, "
    "with Object 0 overtaking Object 1 somewhere between t=2 and t=3."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🚀 1 · Core Idea",
        "📏 2 · Descriptors",
        "🔢 3 · Inequality Matrix",
        "⏱ 4 · Temporal PDP",
        "〰 5 · Roughness",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — THE CORE IDEA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Step 1: From Metric Trajectories to Qualitative Precedence")

    col_t, col_v = st.columns([1, 1.5], gap="large")

    with col_t:
        st.markdown(
            """
### What is a Spatiotemporal Trajectory?

A **raw trajectory** records precise **(x, y)** coordinates at every timestep —
full metric information including distances, speeds, and absolute positions.

---

### The PDP Shift: Why Qualitative?

In many domains (traffic analysis, sports, pedestrian flow) the *exact* numbers
matter less than **who is ahead of whom** — the relative *precedence*.

PDP replaces metric coordinates with a single relational question:

> *"Is object A ahead of object B along a chosen reference direction?"*

This makes the representation:
- **Translation-invariant** — absolute position is irrelevant
- **Scale-invariant** — speed and distances don't change the relations
- **Interpretable** — maps directly to domain concepts like "overtake" or "approach"

---

### This Dataset: An Overtake Event

Two vehicles, 6 key timesteps.  
At **t=2**, Object 1 is still ahead in X.  
At **t=3**, Object 0 has crossed over — it overtook Object 1.

> Use the toggle below to switch between raw coordinates and qualitative precedence.
"""
        )

        view = st.radio(
            "View mode:",
            ["📍 Raw Coordinates", "🏁 Qualitative Precedence"],
            key="tab1_view",
            horizontal=True,
        )

    with col_v:
        if "Raw" in view:
            fig_raw = go.Figure()
            for pid in OBJECTS:
                sub = df[df["poiID"] == pid].sort_values("tstID")
                fig_raw.add_trace(
                    go.Scatter(
                        x=sub["x"],
                        y=sub["y"],
                        mode="lines+markers+text",
                        name=f"Object {pid}",
                        line=dict(color=OBJ_COLORS[pid], width=2.5),
                        marker=dict(size=10),
                        text=[f"t={t}" for t in sub["tstID"]],
                        textposition="top center",
                        textfont=dict(size=11),
                    )
                )
            fig_raw.add_vrect(
                x0=440, x1=480,
                fillcolor="gold", opacity=0.15,
                annotation_text="Overtake zone", annotation_position="top left",
            )
            fig_raw.update_layout(
                title="Raw 2D Trajectories",
                xaxis_title="X (forward direction, m)",
                yaxis_title="Y (lateral, m)",
                height=400,
                template="plotly_white",
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_raw, use_container_width=True)

            # X vs time
            fig_xt = go.Figure()
            for pid in OBJECTS:
                sub = df[df["poiID"] == pid].sort_values("tstID")
                fig_xt.add_trace(
                    go.Scatter(
                        x=sub["tstID"], y=sub["x"],
                        mode="lines+markers",
                        name=f"Object {pid}",
                        line=dict(color=OBJ_COLORS[pid], width=2.5),
                        marker=dict(size=9),
                    )
                )
            fig_xt.add_vrect(
                x0=2.5, x1=3.5, fillcolor="gold", opacity=0.2,
                annotation_text="Overtake!", annotation_position="top left",
            )
            fig_xt.update_layout(
                title="X-Position Over Time  (lines crossing = overtake)",
                xaxis_title="Timestep",
                yaxis_title="X coordinate (m)",
                xaxis=dict(tickmode="array", tickvals=TIMESTEPS, ticktext=[f"t={t}" for t in TIMESTEPS]),
                height=280,
                template="plotly_white",
            )
            st.plotly_chart(fig_xt, use_container_width=True)

        else:
            rows = []
            for t in TIMESTEPS:
                x0 = df[(df["tstID"] == t) & (df["poiID"] == 0)]["x"].values[0]
                x1 = df[(df["tstID"] == t) & (df["poiID"] == 1)]["x"].values[0]
                if x0 > x1:
                    rel, leader, color = "Obj 0 > Obj 1", "Object 0", OBJ_COLORS[0]
                elif x0 < x1:
                    rel, leader, color = "Obj 0 < Obj 1", "Object 1", OBJ_COLORS[1]
                else:
                    rel, leader, color = "Obj 0 = Obj 1", "Tied", "gray"
                rows.append(dict(t=t, x0=x0, x1=x1, gap=abs(x0 - x1),
                                 rel=rel, leader=leader, color=color))

            fig_qual = go.Figure()
            for r in rows:
                fig_qual.add_trace(
                    go.Bar(
                        name=r["leader"],
                        x=[f"t={r['t']}"],
                        y=[r["gap"]],
                        marker_color=r["color"],
                        text=f"{r['rel']}<br>gap={r['gap']:.0f}",
                        textposition="outside",
                        showlegend=False,
                    )
                )
            fig_qual.update_layout(
                title="Qualitative Precedence: Who is Ahead in X?",
                xaxis_title="Timestep",
                yaxis_title="Lead gap |x₀ − x₁|",
                height=340,
                template="plotly_white",
            )
            st.plotly_chart(fig_qual, use_container_width=True)

            tbl = pd.DataFrame(
                [{"Timestep": f"t={r['t']}", "x (Obj 0)": f"{r['x0']:.1f}",
                  "Relation": r["rel"], "x (Obj 1)": f"{r['x1']:.1f}",
                  "Leader": r["leader"]} for r in rows]
            )
            st.dataframe(tbl, hide_index=True, use_container_width=True)
            st.success(
                "🏁 **The overtake happens between t=2 and t=3!**  "
                "At t=2, Object 1 is still ahead (higher X). At t=3, Object 0 has crossed past it."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTORS & SCALARISATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Step 2: Descriptors & Scalarisation")

    col_t2, col_v2 = st.columns([1, 1.5], gap="large")

    with col_t2:
        st.markdown(
            r"""
### What is a Descriptor?

A **descriptor** $d : \mathbb{R}^2 \rightarrow \mathbb{R}$ maps each 2D point
to a single **scalar value** — projecting the 2D world onto a 1D measuring stick.

| Descriptor | Formula | Captures |
|---|---|---|
| **X-axis** | $s = x$ | Forward progress |
| **Y-axis** | $s = y$ | Lateral position |
| **Diagonal** | $s = x\cos\theta + y\sin\theta$ | Custom direction |

### How Scalarisation Works

1. Choose a descriptor direction
2. Each point "snaps" perpendicularly onto that direction
3. Its scalar value is its signed distance along the axis

> **Key insight:** changing the descriptor can *reverse* the ordering of two points.
> Choose the descriptor that reflects what matters in your domain.
"""
        )

        desc2 = st.selectbox("🔧 Descriptor:", ["X-axis", "Y-axis", "Diagonal"], key="desc_s2")
        angle2 = 45.0
        if desc2 == "Diagonal":
            angle2 = float(
                st.slider("Angle θ (degrees):", 0, 180, 45, key="angle_s2")
            )
            st.latex(rf"s = x\,\cos({angle2:.0f}°) + y\,\sin({angle2:.0f}°)")
        elif desc2 == "X-axis":
            st.latex(r"s = x")
        else:
            st.latex(r"s = y")

        focus_ts2 = st.select_slider(
            "Focus timestep:",
            options=TIMESTEPS,
            value=TIMESTEPS[2],
            format_func=lambda x: f"t={x}",
            key="focus_ts2",
        )

    with col_v2:
        fig2 = make_subplots(
            rows=2, cols=1,
            row_heights=[0.62, 0.38],
            subplot_titles=("2D Points (highlighted = focus timestep)", "Scalar Number Line"),
            vertical_spacing=0.12,
        )

        # ── Faded full trajectories
        for pid in OBJECTS:
            sub = df[df["poiID"] == pid].sort_values("tstID")
            fig2.add_trace(
                go.Scatter(
                    x=sub["x"], y=sub["y"],
                    mode="lines+markers",
                    name=f"Object {pid}",
                    line=dict(color=OBJ_COLORS[pid], width=1.5, dash="dot"),
                    marker=dict(size=6, opacity=0.35),
                    showlegend=True,
                ), row=1, col=1
            )

        # ── Highlighted timestep
        focus_scalars = []
        for pid in OBJECTS:
            row_pt = df[(df["tstID"] == focus_ts2) & (df["poiID"] == pid)].iloc[0]
            s = float(get_scalars(row_pt.to_frame().T, desc2, angle2)[0])
            focus_scalars.append(s)

            fig2.add_trace(
                go.Scatter(
                    x=[row_pt["x"]], y=[row_pt["y"]],
                    mode="markers+text",
                    name=f"t={focus_ts2} Obj{pid}",
                    marker=dict(size=15, color=OBJ_COLORS[pid],
                                line=dict(color="black", width=2)),
                    text=[f"Obj {pid}\ns={s:.1f}"],
                    textposition="top right",
                    textfont=dict(size=11),
                    showlegend=False,
                ), row=1, col=1
            )

        # ── Number line
        nl_min = min(focus_scalars) - abs(max(focus_scalars) - min(focus_scalars)) * 0.3 - 5
        nl_max = max(focus_scalars) + abs(max(focus_scalars) - min(focus_scalars)) * 0.3 + 5
        fig2.add_shape(
            type="line",
            x0=nl_min, y0=0, x1=nl_max, y1=0,
            line=dict(color="#555", width=2),
            row=2, col=1,
            xref="x2", yref="y2",
        )
        for pid, s in zip(OBJECTS, focus_scalars):
            fig2.add_trace(
                go.Scatter(
                    x=[s], y=[0],
                    mode="markers+text",
                    marker=dict(size=18, color=OBJ_COLORS[pid], symbol="diamond",
                                line=dict(color="black", width=1.5)),
                    text=[f"Obj {pid}  s={s:.1f}"],
                    textposition="top center",
                    showlegend=False,
                ), row=2, col=1
            )

        fig2.update_yaxes(range=[-0.7, 0.7], showticklabels=False, row=2, col=1)
        fig2.update_xaxes(title_text="Scalar value", row=2, col=1)
        fig2.update_layout(height=560, template="plotly_white",
                           legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig2, use_container_width=True)

        # ── Scalar table for all timesteps
        st.markdown("**Scalar values for all timesteps** (current descriptor):")
        s_rows = []
        for t in TIMESTEPS:
            sub_t = df[df["tstID"] == t].sort_values("poiID")
            sv = get_scalars(sub_t, desc2, angle2)
            s_rows.append({
                "Timestep": f"t={t}",
                "s (Object 0)": f"{sv[0]:.3f}",
                "s (Object 1)": f"{sv[1]:.3f}",
                "Relation": ">" if sv[0] > sv[1] else ("<" if sv[0] < sv[1] else "="),
            })
        st.dataframe(pd.DataFrame(s_rows), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INEQUALITY MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Step 3: Building the Inequality Matrix")

    col_t3, col_v3 = st.columns([1, 1.5], gap="large")

    with col_t3:
        st.markdown(
            r"""
### From Scalars to a Total Preorder

Once every point has a scalar value $s_i$, compare **all pairs** $(i, j)$:

$$M[i,j] = \begin{cases}
< & s_i < s_j \\
= & s_i = s_j \\
> & s_i > s_j
\end{cases}$$

This creates a **total preorder matrix** — a complete relational fingerprint
of the configuration at one timestep.

### Reading the Matrix

| Symbol | Colour | Meaning (row vs. column) |
|---|---|---|
| `<` | 🟢 green | row point is **behind** column point |
| `=` | 🟡 gold | row point is **alongside** column point |
| `>` | 🔴 red | row point is **ahead of** column point |

### Properties
- The diagonal is always `=` (a point equals itself)
- The matrix is **antisymmetric**: if M[i,j]=`>` then M[j,i]=`<`

---

### Interactive: Move Object 0

Drag the slider to shift Object 0's position along the descriptor.  
Watch the matrix **flip** when Object 0 crosses Object 1.
"""
        )

        desc3 = st.selectbox("Descriptor:", ["X-axis", "Y-axis", "Diagonal"], key="desc_s3")
        angle3 = 45.0
        if desc3 == "Diagonal":
            angle3 = float(st.slider("Angle θ:", 0, 180, 45, key="angle_s3"))

        ts3 = st.select_slider(
            "Timestep:", options=TIMESTEPS, value=TIMESTEPS[0],
            format_func=lambda x: f"t={x}", key="ts_s3",
        )

        x0_orig = float(df[(df["tstID"] == ts3) & (df["poiID"] == 0)]["x"].values[0])
        x1_val = float(df[(df["tstID"] == ts3) & (df["poiID"] == 1)]["x"].values[0])
        gap_px = abs(x0_orig - x1_val)

        offset = st.slider(
            "Object 0 X offset:",
            min_value=-float(gap_px * 2 + 50),
            max_value=float(gap_px * 2 + 50),
            value=0.0,
            step=1.0,
            key="offset_s3",
            help="Shift Object 0 to simulate passing Object 1. Watch the matrix cells flip!",
        )

    with col_v3:
        df_ts3 = df[df["tstID"] == ts3].copy().sort_values("poiID")
        df_ts3.loc[df_ts3["poiID"] == 0, "x"] = x0_orig + offset

        sv3 = get_scalars(df_ts3, desc3, angle3)
        labels3 = [f"Obj {p}  s={sv3[i]:.1f}" for i, p in enumerate(df_ts3["poiID"].values)]
        sym3, _ = build_matrix(sv3, rho=0.0)

        st.plotly_chart(
            matrix_fig(sym3, labels3, title=f"Inequality Matrix · t={ts3}  (ρ=0)"),
            use_container_width=False,
        )

        # Interpretation sentences
        rel = sym3[0][1]
        st.markdown(
            f"**Object 0** is **`{rel}`** Object 1 along *{desc3}*  "
            f"({sv3[0]:.1f} {rel} {sv3[1]:.1f})"
        )

        x0_new = x0_orig + offset
        if offset != 0 and (np.sign(x0_new - x1_val) != np.sign(x0_orig - x1_val)):
            st.success(
                f"🔄 **Matrix flipped!** Object 0 just crossed Object 1 — "
                f"the ordering reversed."
            )
        elif offset != 0:
            st.info(f"Object 0 moved by {offset:+.0f} units. Cross the gap ({gap_px:.0f}) to flip the matrix.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEMPORAL PDP
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Step 4: Temporal PDP — PDP-S, PDP-D, PDP-G")

    col_t4, col_v4 = st.columns([1, 1.5], gap="large")

    with col_t4:
        st.markdown(
            r"""
### Adding the Temporal Dimension

So far the matrix covers **one timestep**. But motion happens over time.
PDP captures temporal structure through a **sliding window of size $w$**.

### Three Regimes

| $w$ | Name | Description |
|---|---|---|
| $w = 1$ | **PDP-S** (Static) | Snapshot — spatial arrangement at one moment |
| $1 < w < N$ | **PDP-D** (Dynamic) | Sliding window — captures transitions |
| $w = N$ | **PDP-G** (Global) | Full trajectory — complete temporal fingerprint |

### Matrix Growth

With window size $w$ and $n_{\text{obj}}$ objects, the matrix grows to:

$$\text{size} = (w \cdot n_{\text{obj}}) \times (w \cdot n_{\text{obj}})$$

Each row/column is a **(timestep, object)** pair.  
The ordering used here is **time-major**:
$$(t_0,\text{obj}_0),\ (t_0,\text{obj}_1),\ (t_1,\text{obj}_0),\ (t_1,\text{obj}_1),\ \ldots$$
"""
        )

        w4 = st.slider(
            "Window size (w):",
            min_value=1, max_value=N_TST, value=1, step=1,
            key="window_s4",
            help="1 = PDP-S  ·  between = PDP-D  ·  max = PDP-G",
        )

        desc4 = st.selectbox("Descriptor:", ["X-axis", "Y-axis", "Diagonal"], key="desc_s4")
        angle4 = 45.0
        if desc4 == "Diagonal":
            angle4 = float(st.slider("Angle θ:", 0, 180, 45, key="angle_s4"))

        # PDP variant badge
        if w4 == 1:
            pdp_name, pdp_color = "PDP-S (Static)", "#4CAF50"
            pdp_info = "Single snapshot. Captures spatial arrangement at one moment."
        elif w4 == N_TST:
            pdp_name, pdp_color = "PDP-G (Global)", "#F44336"
            pdp_info = "Full trajectory. Every point-time instance compared to every other."
        else:
            pdp_name, pdp_color = f"PDP-D  (w={w4})", "#2196F3"
            pdp_info = f"Sliding window of {w4} timesteps. Captures transitions and dynamics."

        st.markdown(
            f"""<div style="background:{pdp_color}22; border-left:4px solid {pdp_color};
                 padding:10px 14px; border-radius:4px; margin-top:8px;">
            <b style="color:{pdp_color}; font-size:1.05em;">{pdp_name}</b><br>
            <span style="font-size:0.93em;">{pdp_info}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        max_start = N_TST - w4
        start_opts = TIMESTEPS[: max_start + 1]
        if w4 < N_TST:
            start_ts4 = st.select_slider(
                "Window start:", options=start_opts, value=start_opts[0],
                format_func=lambda x: f"t={x}", key="wstart_s4",
            )
        else:
            start_ts4 = TIMESTEPS[0]

    with col_v4:
        start_idx = TIMESTEPS.index(start_ts4)
        win_ts = TIMESTEPS[start_idx: start_idx + w4]

        df_win = df[df["tstID"].isin(win_ts)].sort_values(["tstID", "poiID"])
        sv4 = get_scalars(df_win, desc4, angle4)
        labels4 = [f"t{t}_o{p}" for t in win_ts for p in OBJECTS]
        sym4, _ = build_matrix(sv4, rho=0.0)

        st.markdown(
            f"**Window:** t={win_ts[0]} → t={win_ts[-1]}  ·  "
            f"Matrix size: **{len(labels4)} × {len(labels4)}**"
        )
        st.plotly_chart(
            matrix_fig(sym4, labels4, title=f"{pdp_name} Matrix"),
            use_container_width=False,
        )

        # For PDP-D: show compact text view of all windows
        if 1 < w4 < N_TST:
            st.markdown("---")
            st.markdown("**All windows** at a glance:")
            all_starts = TIMESTEPS[: N_TST - w4 + 1]
            n_cols = min(len(all_starts), 3)
            cols_w = st.columns(n_cols)
            for idx, s in enumerate(all_starts):
                w_ts_i = TIMESTEPS[TIMESTEPS.index(s): TIMESTEPS.index(s) + w4]
                df_wi = df[df["tstID"].isin(w_ts_i)].sort_values(["tstID", "poiID"])
                sv_i = get_scalars(df_wi, desc4, angle4)
                labs_i = [f"t{t}_o{p}" for t in w_ts_i for p in OBJECTS]
                sym_i, _ = build_matrix(sv_i)
                header = " ".join(f"{l:>10s}" for l in labs_i)
                rows_txt = "\n".join(
                    f"{labs_i[r]:>10s}" + "".join(f"{sym_i[r][c]:>10s}" for c in range(len(labs_i)))
                    for r in range(len(labs_i))
                )
                with cols_w[idx % n_cols]:
                    st.markdown(f"**t={w_ts_i[0]}→t={w_ts_i[-1]}**")
                    st.code(header + "\n" + rows_txt, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROUGHNESS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Step 5: Roughness — Tolerating Near-Equality")

    col_t5, col_v5 = st.columns([1, 1.5], gap="large")

    with col_t5:
        st.markdown(
            r"""
### The Problem with Exact Equality

Real-world sensor noise means two objects that are "side by side"
will almost never have *exactly* equal scalars.

Strict PDP would flag them as `<` or `>` even when the difference is 0.001 —
noisy, brittle, and often meaningless.

### Roughness Threshold $\rho_d$

The **roughness extension** adds a per-descriptor tolerance $\rho_d$:

$$M_\rho[i,j] = \begin{cases}
< & s_i < s_j - \rho_d \\
= & |s_i - s_j| \leq \rho_d \\
> & s_i > s_j + \rho_d
\end{cases}$$

When two scalars lie within $\rho_d$ of each other they are treated as **equal**.

### Effect

| $\rho$ | Result |
|---|---|
| $0$ | Exact PDP — maximum discrimination |
| small | Noise-robust — minor fluctuations ignored |
| large | Coarse — objects "travel together" over a wider band |

---

Try increasing $\rho$ past the **actual scalar difference** shown below
and watch the matrix cell snap from `<`/`>` to `=`.
"""
        )

        desc5 = st.selectbox("Descriptor:", ["X-axis", "Y-axis", "Diagonal"], key="desc_s5")
        angle5 = 45.0
        if desc5 == "Diagonal":
            angle5 = float(st.slider("Angle θ:", 0, 180, 45, key="angle_s5"))

        ts5 = st.select_slider(
            "Focus timestep:",
            options=TIMESTEPS,
            value=TIMESTEPS[4],  # t=4: objects are closest in Y
            format_func=lambda x: f"t={x}",
            key="ts_s5",
        )

        sub5 = df[df["tstID"] == ts5].sort_values("poiID")
        sv5_base = get_scalars(sub5, desc5, angle5)
        actual_diff = abs(float(sv5_base[0]) - float(sv5_base[1]))

        st.markdown(
            f"""
> **At t={ts5}:**  
> s(Obj 0) = **{sv5_base[0]:.3f}**  
> s(Obj 1) = **{sv5_base[1]:.3f}**  
> **|difference| = {actual_diff:.3f}**
"""
        )

        rho_max = max(actual_diff * 2.5 + 5.0, 10.0)
        rho = st.slider(
            "Roughness threshold ρ:",
            min_value=0.0,
            max_value=float(rho_max),
            value=0.0,
            step=max(0.1, round(rho_max / 100, 1)),
            key="rho_s5",
            help=f"Actual difference at t={ts5} is {actual_diff:.2f}. Raise ρ past this to force equality.",
        )

        if rho >= actual_diff:
            st.success(f"✅ ρ = {rho:.2f} ≥ {actual_diff:.2f} → now treated as **equal (=)**")
        elif rho > 0:
            st.info(f"ρ = {rho:.2f} < {actual_diff:.2f} → still distinct. Keep increasing.")
        else:
            st.warning(f"ρ = 0 (exact). Difference = {actual_diff:.2f}")

    with col_v5:
        sym5, _ = build_matrix(sv5_base, rho=rho)
        labels5 = [f"Obj {p}\ns={sv5_base[i]:.2f}" for i, p in enumerate(sub5["poiID"].values)]

        st.plotly_chart(
            matrix_fig(sym5, labels5, title=f"Rough PDP (ρ={rho:.2f}) at t={ts5}",
                       height=280, width=320),
            use_container_width=False,
        )

        # ── Full timeline under current ρ
        st.markdown(f"---\n**Full timeline with ρ = {rho:.2f}:**")
        tl_rows = []
        for t in TIMESTEPS:
            sub_t = df[df["tstID"] == t].sort_values("poiID")
            sv_t = get_scalars(sub_t, desc5, angle5)
            sym_t, _ = build_matrix(sv_t, rho=rho)
            diff_t = abs(float(sv_t[0]) - float(sv_t[1]))
            rel_t = sym_t[0][1]
            tl_rows.append({
                "t": t,
                "s(Obj 0)": round(float(sv_t[0]), 2),
                "s(Obj 1)": round(float(sv_t[1]), 2),
                "|diff|": round(diff_t, 2),
                f"Obj0 ? Obj1": rel_t,
                "Equal?": "✅" if rel_t == "=" else "❌",
            })
        st.dataframe(pd.DataFrame(tl_rows), hide_index=True, use_container_width=True)

        # ── Roughness sweep: how many off-diagonal '=' emerge
        st.markdown("**Roughness sweep** — equality count vs. ρ:")
        rho_sweep = np.linspace(0.0, rho_max, 200)
        eq_counts = []
        for rv in rho_sweep:
            eq = 0
            for t in TIMESTEPS:
                sub_t = df[df["tstID"] == t].sort_values("poiID")
                sv_t = get_scalars(sub_t, desc5, angle5)
                _, nm = build_matrix(sv_t, rho=float(rv))
                eq += int(np.sum(nm == 1)) - N_POI  # subtract diagonal
            eq_counts.append(eq)

        fig_sweep = go.Figure()
        fig_sweep.add_trace(
            go.Scatter(
                x=rho_sweep, y=eq_counts,
                mode="lines", fill="tozeroy",
                line=dict(color="#2196F3", width=2.5),
                fillcolor="rgba(33,150,243,0.12)",
            )
        )
        fig_sweep.add_vline(
            x=rho, line_dash="dash", line_color="red",
            annotation_text=f"ρ={rho:.2f}", annotation_position="top right",
        )
        fig_sweep.update_layout(
            xaxis_title="Roughness ρ",
            yaxis_title="Off-diagonal '=' cells (sum over all timesteps)",
            template="plotly_white",
            height=230,
            margin=dict(l=10, r=10, t=20, b=40),
        )
        st.plotly_chart(fig_sweep, use_container_width=True)
