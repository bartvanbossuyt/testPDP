# -*- coding: utf-8 -*-
# inverse.py
# Streamlit-app met academische look, twee kolommen met strikt vierkante assen.
# Links: lijn + punten uit CSV (c=11, t∈{0,1,2}, o=0). Rechts: identieke assen, leeg.
# CSV kan beginnen met letterlijk: "header: c,t,o,x,y" → die regel wordt overgeslagen.
# Aslabels: d₁ en d₂; puntlabels: k₀, k₁, k₂ (blauw, kleiner).
# maxdist = max(||k0-k1||, ||k1-k2||); assen krijgen ≥ maxdist marge tot elke rand.

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

# Type definition for successful point data
class SuccessfulPoint(TypedDict):
    point: np.ndarray
    parent_idx: int  # Index in all_pts (kan gegenereerd punt zijn)
    parent_point: np.ndarray  # Actual coordinates of parent
    original_parent_idx: int  # Index van het ORIGINELE punt (k0, k1, k2, l0, l1, l2)
    iteration: int

# ============= Pagina-instellingen =============
st.set_page_config(
    page_title="pdp inverse",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============= Stijlen (academische look) =============
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

# ============= Hoofdtitel =============
st.markdown("<h1 class='headline'>pdp inverse</h1>", unsafe_allow_html=True)
st.markdown("<hr />", unsafe_allow_html=True)

# ---------- Wrapper: Series -> Series, met gerichte ignore ----------
def to_numeric_series(s: pd.Series) -> pd.Series:
    """Converteer kolom naar numeriek met NaN bij fouten (Pylance-vriendelijk)."""
    out = pd.to_numeric(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        s, errors="coerce"
    )
    return out

# ============= CSV inladen (herkent 'header: c,t,o,x,y') =============
def load_points(csv_name: str = "voorbeeld.csv", o_val: int = 0, c_val: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """
    Leest voorbeeld.csv. Als de eerste regel start met 'header:', wordt die overgeslagen
    en worden kolomnamen ['c','t','o','x','y'] gebruikt. Filter: c=11, o=o_val, t∈{0,1,2}.
    Retourneert:
      - pts: (N,2) numpy array [x,y] gesorteerd op t
      - ts:  (N,) numpy array met t-waarden (gesorteerd)
    """
    csv_path = Path(__file__).with_name(csv_name)
    if not csv_path.exists():
        st.error(f"CSV niet gevonden: {csv_path}")
        st.stop()

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

    # Numeriek forceren met expliciete Series->Series helper
    for col in names:
        df[col] = to_numeric_series(df[col])
    df = df.dropna(subset=names)  # type: ignore
    df: pd.DataFrame = df  # Explicit annotation for type checker

    # Filter enkel op configuratie en object (alle t-waarden worden meegenomen)
    sel = df[(df["c"] == c_val) & (df["o"] == o_val)].sort_values("t").reset_index(drop=True)  # type: ignore
    sel: pd.DataFrame = sel  # Explicit annotation for type checker
    if sel.empty:
        st.error(f"Geen rijen gevonden voor c={c_val}, o={o_val}.")
        st.stop()

    pts = sel[["x", "y"]].to_numpy(dtype=float)  # type: ignore
    ts = sel["t"].to_numpy(dtype=float)  # type: ignore
    return pts, ts

# ============= Settings (configuratie en tijdvenster) =============
# Lees CSV snel in om beschikbare configuraties (c-waarden) te bepalen
def _read_clean_df(csv_name: str) -> pd.DataFrame:
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
    # Numeriek forceren
    for col in names:
        df[col] = to_numeric_series(df[col])
    df = df.dropna(subset=names)  # type: ignore
    df = df.reset_index(drop=True)
    return df  # type: ignore[return-value]

_df_all = _read_clean_df("voorbeeld.csv")
_mask_t = _df_all["t"].isin(_df_all["t"].unique())  # type: ignore[reportUnknownMemberType]
_df_all = _df_all[_mask_t]

available_configs = sorted(_df_all["c"].dropna().unique().astype(int).tolist())  # type: ignore
if not available_configs:
    st.error("Geen configuraties gevonden (kolom 'c' is leeg).")
    st.stop()

# ============= Settings Card: Academic & Sleek =============
st.markdown("""
<div class='settings-card'>
  <h3>settings</h3>
""", unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns([1,1,2], gap="small")
with sc1:
    selected_c = st.selectbox("configuration (c)", options=available_configs, index=available_configs.index(11) if 11 in available_configs else 0, key="cfg_c")
selected_c_int: int = int(selected_c) if selected_c is not None else int(available_configs[0])

# Bepaal unieke, gesorteerde t-waarden die voor zowel o=0 als o=1 bestaan binnen de gekozen configuratie
_t_k = sorted(_df_all[(_df_all["c"] == selected_c_int) & (_df_all["o"] == 0)]["t"].unique().tolist())  # type: ignore
_t_l = sorted(_df_all[(_df_all["c"] == selected_c_int) & (_df_all["o"] == 1)]["t"].unique().tolist())  # type: ignore
_t_common = [t for t in _t_k if t in _t_l]
if not _t_common:
    st.error(f"Geen overlappende t-waarden voor c={selected_c} tussen o=0 en o=1.")
    st.stop()

n_timepoints = len(_t_common)  # n = aantal beschikbare timestamps voor deze config
default_window = min(3, n_timepoints)

# Track timestamp selection changes to reset animation
prev_config = st.session_state.get("prev_config_c", selected_c_int)

# Als configuratie verandert, reset tracking
if selected_c_int != prev_config:
    st.session_state["prev_config_c"] = selected_c_int
    st.session_state["prev_start_t_idx"] = 0
    st.session_state["prev_num_timestamps"] = default_window
    st.session_state["anim_running"] = False
    st.session_state["show_anim_circle"] = False

# Eerst: kies aantal timestamps
prev_num_ts = st.session_state.get("prev_num_timestamps", default_window)

with sc2:
    # Kies eerst hoeveel timestamps je wilt
    num_timestamps = st.slider(
        "number of timestamps",
        min_value=1,
        max_value=n_timepoints,
        value=min(prev_num_ts, n_timepoints),
        step=1,
        key="cfg_k",
        help=f"Select how many timestamps to use (1 to {n_timepoints})"
    )

# Dan: bereken maximum start-positie op basis van gekozen aantal timestamps
# max_start_idx = n - num_timestamps
# Bijvoorbeeld: n=4, num_timestamps=3 → max_start_idx = 4-3 = 1
max_start_idx = n_timepoints - num_timestamps
prev_start_idx = st.session_state.get("prev_start_t_idx", 0)

# Als num_timestamps verandert EN prev_start_idx > max_start_idx, pas start_idx aan
if num_timestamps != prev_num_ts:
    if prev_start_idx > max_start_idx:
        # Automatisch verminderen naar maximum beschikbare
        prev_start_idx = max_start_idx
        st.session_state["prev_start_t_idx"] = prev_start_idx

with sc3:
    # Start-positie mag niet zo hoog zijn dat we buiten bereik vallen
    # min=0, max = n - num_timestamps
    if max_start_idx > 0:
        start_t_idx = st.slider(
            "starting time index",
            min_value=0,
            max_value=max_start_idx,
            value=min(prev_start_idx, max_start_idx),
            step=1,
            key="cfg_start_t_idx",
            help=f"Starting position (0 to {max_start_idx}), will select {num_timestamps} timestamps"
        )
    else:
        # Als num_timestamps == n_timepoints, kan alleen starten op 0
        start_t_idx = 0
        st.caption("Starting at index 0 (only option for this selection)")
    
    start_t = _t_common[start_t_idx]
    end_t = _t_common[start_t_idx + num_timestamps - 1]
    st.caption(f"Range: t = {start_t} to {end_t}")

# Validatie: controleer dat we niet buiten bereik vallen
end_t_idx = start_t_idx + num_timestamps
if end_t_idx > n_timepoints:
    st.error(f"Invalid selection: would request timestamps up to index {end_t_idx-1}, but only {n_timepoints} available!")
    st.stop()

# Als num_timestamps of start_t_idx verandert, reset animatie
if num_timestamps != prev_num_ts or start_t_idx != prev_start_idx:
    st.session_state["anim_running"] = False
    st.session_state["show_anim_circle"] = False
    st.session_state["prev_num_timestamps"] = num_timestamps
    st.session_state["prev_start_t_idx"] = start_t_idx
    st.session_state["prev_start_t"] = start_t

# --- Sleek academic controls ---
st.markdown("<hr style='margin:0.5rem 0 0.7rem 0;' />", unsafe_allow_html=True)
sc4, sc5, sc6 = st.columns([1,1,1], gap="small")
with sc4:
    strategy = st.radio(
        "strategy",
        options=["exponential", "binary"],
        index=0,
        key="cfg_strategy",
        help="Choose the search strategy for configuration generation."
    )
with sc5:
    num_iterations = st.radio(
        "number of iterations",
        options=[3, 4, 5],
        index=0,
        key="cfg_iterations",
        help="How many iterations to run for each configuration."
    )
with sc6:
    num_configs = st.radio(
        "number of configurations",
        options=[1, 3, 10],
        index=0,
        key="cfg_num_configs",
        help="How many configurations to generate."
    )

# --- Action buttons ---
st.markdown("<div style='display:flex;gap:1.2rem;margin-top:0.7rem;'>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1,1,1], gap="small")
with col_btn1:
    animate_btn = st.button("animatie 1 config", key="btn_animate")
with col_btn2:
    animate_5_btn = st.button("animate 5 configs", key="btn_animate_5")
with col_btn3:
    generate_btn = st.button("generate configurations", key="btn_generate")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Gebruik de selectie voor het laden van punten
k_points, k_vals = load_points("voorbeeld.csv", o_val=0, c_val=selected_c_int)  # k-punten (o=0)
l_points, l_vals = load_points("voorbeeld.csv", o_val=1, c_val=selected_c_int)  # l-punten (o=1)

# ============= Selecteer venster op basis van GUI (start_t_idx, num_timestamps) =============
# We gebruiken nu direct de index i.p.v. opnieuw te zoeken
start_idx = int(start_t_idx)
end_idx = start_idx + int(num_timestamps)
selected_ts_window = _t_common[start_idx:end_idx]
selected_ts_set = set(selected_ts_window)

# Filter k- en l-punten op geselecteerde timestamps; volgorde blijft behouden (reeds gesorteerd op t)
mask_k_win = np.isin(k_vals, list(selected_ts_set))
k_points_plot = k_points[mask_k_win]
k_vals_plot = k_vals[mask_k_win]
mask_l_win = np.isin(l_vals, list(selected_ts_set))
l_points_plot = l_points[mask_l_win]
l_vals_plot = l_vals[mask_l_win]

# ============= maxdist berekenen (needed before animation init) =============
def pairwise_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.hypot(d[0], d[1]))

def max_consecutive_dist(pts: np.ndarray) -> float:
    n = pts.shape[0]
    if n < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    return float(np.max(dists))

maxdist: float = max(max_consecutive_dist(k_points_plot), max_consecutive_dist(l_points_plot))

# ============= Assenlimieten met marge ≥ maxdist en vierkante schaal =============
def square_limits_with_margin(
    pts: np.ndarray, margin: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Bouw een venster dat alle punten bevat en waarbij elk punt minstens 'margin'
    van elke rand afligt (in data-eenheden). Daarna maak het venster vierkant
    door de kortste zijde te vergroten tot de langste (marges worden dan alleen groter).
    """
    xmin = float(np.min(pts[:, 0])) - margin
    xmax = float(np.max(pts[:, 0])) + margin
    ymin = float(np.min(pts[:, 1])) - margin
    ymax = float(np.max(pts[:, 1])) + margin

    w = xmax - xmin
    h = ymax - ymin
    side = max(w, h)
    if side <= 0:
        side = 1.0  # minimale zijde om singular axes te vermijden

    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)

    xlim = (cx - side / 2.0, cx + side / 2.0)
    ylim = (cy - side / 2.0, cy + side / 2.0)
    return xlim, ylim

XLIM, YLIM = square_limits_with_margin(
    np.vstack([k_points_plot, l_points_plot]),  # Alle geselecteerde k en l punten samen
    maxdist
)

# ============= Animate button handler =============
if animate_btn or animate_5_btn:
    # Determine how many configs to generate
    num_configs_to_generate = 5 if animate_5_btn else 1
    
    # Kies random punt uit k of l
    all_pts = np.vstack([k_points_plot, l_points_plot])
    all_ts = np.concatenate([k_vals_plot, l_vals_plot])
    n_total = all_pts.shape[0]

    # Kies een random index
    parent_idx = int(np.random.randint(0, n_total))  # type: ignore[arg-type]
    parent_pt = all_pts[parent_idx]

    # Kies random hoek alfa (in radialen)
    alfa = float(np.random.uniform(0, 2 * np.pi))

    # Startafstand: voor binaire strategie beginnen met halve maxdist
    if strategy == "binary":
        distance = maxdist / 2.0
    else:
        distance = maxdist

    # Beginpunt op afstand 'distance' en hoek 'alfa'
    gen_x = parent_pt[0] + distance * np.cos(alfa)
    gen_y = parent_pt[1] + distance * np.sin(alfa)
    generated_point = np.array([gen_x, gen_y])

    # Check if point is within graph bounds; if not, add 180° to angle
    if not (XLIM[0] <= gen_x <= XLIM[1] and YLIM[0] <= gen_y <= YLIM[1]):
        alfa = (alfa + np.pi) % (2 * np.pi)
        gen_x = parent_pt[0] + distance * np.cos(alfa)
        gen_y = parent_pt[1] + distance * np.sin(alfa)
        generated_point = np.array([gen_x, gen_y])

    # Initialiseer animatiestatus (gemeenschappelijk voor beide strategieën)
    st.session_state["show_anim_circle"] = True
    st.session_state["anim_running"] = True
    st.session_state["anim_circle_idx"] = parent_idx
    st.session_state["anim_distance"] = distance
    st.session_state["anim_generated_point"] = generated_point
    st.session_state["anim_parent_idx"] = parent_idx
    st.session_state["anim_all_pts"] = all_pts
    st.session_state["anim_all_ts"] = all_ts
    st.session_state["anim_angle"] = alfa
    st.session_state["anim_strategy"] = strategy  # Store chosen strategy
    st.session_state["anim_iteration"] = 0  # Start bij 0, wordt verhoogd na succesvolle iteratie
    # Stel aantal iteraties in op basis van GUI-keuze (radio)
    gui_iters = int(st.session_state.get("cfg_iterations", 3))
    st.session_state["anim_max_iterations"] = gui_iters
    st.session_state["anim_iterations_per_run"] = gui_iters
    st.session_state["anim_completed_iterations"] = 0  # Aantal voltooide iteraties
    st.session_state["anim_last_update"] = time.time()  # voor 5s wachttijd
    st.session_state["anim_successful_points"] = []  # Succesvol gegenereerde punten met labels
    st.session_state["anim_in_search"] = True  # Bezig met zoeken naar juiste positie
    st.session_state["anim_num_configs"] = num_configs_to_generate  # Number of configs to generate
    st.session_state["anim_current_config"] = 1  # Current config being generated
    st.session_state["anim_all_configs"] = []  # Store all completed configurations
    st.session_state["anim_search_steps"] = 0  # Aantal zoekstappen binnen huidige iteratie
    # CSV accumulator: append 6 rows (t=0,1,2 for k and l) per completed config
    st.session_state["anim_csv_lines"] = []
    
    # Binary strategy specific variables
    if strategy == "binary":
        st.session_state["anim_binary_last_match_point"] = None  # (x, y) van laatste order match
        st.session_state["anim_binary_base_distance"] = distance  # Startafstand (bijv. maxdist)
        st.session_state["anim_binary_base_radius"] = distance    # Startstraal
        st.session_state["anim_binary_step_size"] = distance / 2.0  # Eerste stap (helft van start)
        st.session_state["anim_binary_last_step_match"] = False

# ============= d1/d2-volgorde strings (latex) =============
def _format_t_subscript(tval: float) -> str:
    try:
        tnum = float(tval)
    except Exception:
        tnum = float(np.array(tval, dtype=float))
    return str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"

def make_d1_order_latex() -> str:
    entries: list[tuple[float, str]] = []  # (x, token)
    for x, t in zip(k_points_plot[:, 0].tolist(), k_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(x), rf"k_{{{lbl}}}"))
    for x, t in zip(l_points_plot[:, 0].tolist(), l_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(x), rf"l_{{{lbl}}}"))

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
    entries: list[tuple[float, str]] = []  # (y, token)
    for y, t in zip(k_points_plot[:, 1].tolist(), k_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(y), rf"k_{{{lbl}}}"))
    for y, t in zip(l_points_plot[:, 1].tolist(), l_vals_plot.tolist()):
        lbl = _format_t_subscript(t)
        entries.append((float(y), rf"l_{{{lbl}}}"))

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
    if not st.session_state.get("show_anim_circle", False):
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

    # Bepaal originele parent-index (k of l) ook als parent een gegenereerd punt is
    base_idx = int(parent_idx)
    if parent_idx >= total_original:
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - total_original)
        if 0 <= sidx < len(succ_list):
            base_idx = int(succ_list[sidx]["original_parent_idx"])
    parent_is_k = base_idx < n_k

    # Bepaal voor ALLE originele indices hoeveel succesvolle generaties er zijn (persistent primes)
    in_search = st.session_state.get("anim_in_search", False)
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    generation_counts: dict[int, int] = {}
    for sp in successful_points:
        if "original_parent_idx" in sp:
            oi = int(sp["original_parent_idx"])  # type: ignore[index]
            generation_counts[oi] = generation_counts.get(oi, 0) + 1

    def _prime_str(gen: int) -> str:
        if gen <= 0:
            return ""
        if gen == 1:
            return "'"
        if gen == 2:
            return "''"
        return "^{*}"

    # Huidige parent (zoekpunt of succesvolle dochter) vervangt zijn originele positie
    # Haal altijd de t-waarde van de originele index (ook als parent_idx een generated point is)
    if base_idx < n_k:
        parent_t = k_vals_plot[base_idx]
    else:
        parent_t = l_vals_plot[base_idx - n_k]
    lbl_parent = _format_t_subscript(float(parent_t))
    current_gen_count = generation_counts.get(base_idx, 0)
    # Als we NIET in zoekfase betekent het dat het huidige punt succesvol is en dus één generatie verder gaat
    label_gen_count = current_gen_count + (0 if in_search else 1)
    parent_primes = _prime_str(label_gen_count)
    if parent_is_k:
        entries.append((float(gen_pt[0]), rf"k{parent_primes}_{{{lbl_parent}}}"))
    else:
        entries.append((float(gen_pt[0]), rf"l{parent_primes}_{{{lbl_parent}}}"))

    # Bouw een mapping van originele index naar het LAATSTE gegenereerde punt
    latest_generated: dict[int, np.ndarray] = {}
    for sp in successful_points:
        if "original_parent_idx" in sp:
            orig_idx = int(sp["original_parent_idx"])  # type: ignore[index]
            latest_generated[orig_idx] = sp["point"]

    # Voor alle originele indices (behalve de huidige parent):
    # - Als er een gegenereerde versie bestaat, gebruik die
    # - Anders gebruik het originele punt
    
    # k-punten
    for i, (x, t) in enumerate(zip(k_points_plot[:, 0].tolist(), k_vals_plot.tolist())):
        if i == base_idx:
            continue  # al toegevoegd
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(i, 0)
        primes_i = _prime_str(gen_cnt)
        if i in latest_generated:
            # Gebruik laatste gegenereerde coördinaat
            entries.append((float(latest_generated[i][0]), rf"k{primes_i}_{{{lbl}}}"))
        else:
            entries.append((float(x), rf"k{primes_i}_{{{lbl}}}"))
    
    # l-punten
    for j, (x, t) in enumerate(zip(l_points_plot[:, 0].tolist(), l_vals_plot.tolist())):
        glob_idx = n_k + j
        if glob_idx == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(glob_idx, 0)
        primes_j = _prime_str(gen_cnt)
        if glob_idx in latest_generated:
            entries.append((float(latest_generated[glob_idx][0]), rf"l{primes_j}_{{{lbl}}}"))
        else:
            entries.append((float(x), rf"l{primes_j}_{{{lbl}}}"))
    
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
    if not st.session_state.get("show_anim_circle", False):
        return r"d_2:"
    gen_pt = st.session_state.get("anim_generated_point", None)
    parent_idx = st.session_state.get("anim_parent_idx", 0)
    all_pts = st.session_state.get("anim_all_pts", np.array([]))
    all_ts = st.session_state.get("anim_all_ts", np.array([]))
    if gen_pt is None or all_pts.shape[0] == 0:
        return r"d_2:"
    entries: list[tuple[float, str]] = []
    n_k = k_points_plot.shape[0]
    n_l = l_points_plot.shape[0]
    total_original = n_k + n_l

    # Bepaal originele parent-index (k of l) ook als parent een gegenereerd punt is
    base_idx = int(parent_idx)
    if parent_idx >= total_original:
        succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        sidx = int(parent_idx - total_original)
        if 0 <= sidx < len(succ_list):
            base_idx = int(succ_list[sidx]["original_parent_idx"])
    parent_is_k = base_idx < n_k

    # Persistent primes logica
    in_search = st.session_state.get("anim_in_search", False)
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    generation_counts: dict[int, int] = {}
    for sp in successful_points:
        if "original_parent_idx" in sp:
            oi = int(sp["original_parent_idx"])  # type: ignore[index]
            generation_counts[oi] = generation_counts.get(oi, 0) + 1

    def _prime_str(gen: int) -> str:
        if gen <= 0:
            return ""
        if gen == 1:
            return "'"
        if gen == 2:
            return "''"
        return "^{*}"

    # Haal altijd de t-waarde van de originele index (ook als parent_idx een generated point is)
    if base_idx < n_k:
        parent_t = k_vals_plot[base_idx]
    else:
        parent_t = l_vals_plot[base_idx - n_k]
    lbl_parent = _format_t_subscript(float(parent_t))
    current_gen_count = generation_counts.get(base_idx, 0)
    label_gen_count = current_gen_count + (0 if in_search else 1)
    parent_primes = _prime_str(label_gen_count)
    if parent_is_k:
        entries.append((float(gen_pt[1]), rf"k{parent_primes}_{{{lbl_parent}}}"))
    else:
        entries.append((float(gen_pt[1]), rf"l{parent_primes}_{{{lbl_parent}}}"))

    # Bouw een mapping van originele index naar het LAATSTE gegenereerde punt
    latest_generated: dict[int, np.ndarray] = {}
    for sp in successful_points:
        if "original_parent_idx" in sp:
            orig_idx = int(sp["original_parent_idx"])  # type: ignore[index]
            latest_generated[orig_idx] = sp["point"]

    # Voor alle originele indices (behalve de huidige parent):
    # - Als er een gegenereerde versie bestaat, gebruik die
    # - Anders gebruik het originele punt
    
    # k-punten
    for i, (y, t) in enumerate(zip(k_points_plot[:, 1].tolist(), k_vals_plot.tolist())):
        if i == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(i, 0)
        primes_i = _prime_str(gen_cnt)
        if i in latest_generated:
            entries.append((float(latest_generated[i][1]), rf"k{primes_i}_{{{lbl}}}"))
        else:
            entries.append((float(y), rf"k{primes_i}_{{{lbl}}}"))
    
    # l-punten
    for j, (y, t) in enumerate(zip(l_points_plot[:, 1].tolist(), l_vals_plot.tolist())):
        glob_idx = n_k + j
        if glob_idx == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(glob_idx, 0)
        primes_j = _prime_str(gen_cnt)
        if glob_idx in latest_generated:
            entries.append((float(latest_generated[glob_idx][1]), rf"l{primes_j}_{{{lbl}}}"))
        else:
            entries.append((float(y), rf"l{primes_j}_{{{lbl}}}"))
    
    entries.sort(key=lambda it: it[0])
    tol = 1e-9
    out = [entries[0][1]]
    for i in range(1, len(entries)):
        prev_y = entries[i - 1][0]
        cur_y = entries[i][0]
        connector = " = " if abs(cur_y - prev_y) <= tol else " < "
        out.append(connector + entries[i][1])
    return r"d_2: " + "".join(out)

# ===== Helpers voor volgorde-vergelijking (primes negeren) =====
def _strip_primes(text: str) -> str:
    """Verwijder alle generatie-markers: apostroffen (') en asterisken (^{*})."""
    # Verwijder ^{*} notatie (gebruikt voor generatie 3+)
    text = re.sub(r"\^\{\*\}", "", text)
    # Verwijder alle apostroffen (gebruikt voor generatie 1 en 2)
    text = re.sub(r"[']+", "", text)
    return text

def _extract_order_string(latex_str: str) -> str:
    """Extract the ordering part and strip all primes (apostrophes and asterisks)."""
    # Verwijder prefix "d_1:" / "d_2:" 
    core = latex_str.replace("d_1:", "").replace("d_2:", "").strip()
    # Verwijder alle generatie-markers (primes en asterisken)
    core_no_primes = _strip_primes(core)
    return core_no_primes

# ============= Tekenen (zonder gridlines) =============
def setup_square_axes(ax: matplotlib.axes.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    # geen gridlines
    for sp in ax.spines.values():
        sp: matplotlib.spines.Spine
        sp.set_linewidth(0.9)  # type: ignore
        sp.set_color("#222")
    ax.tick_params(axis="both", labelsize=9, width=0.8, color="#222")  # type: ignore
    # Aslabels met subscripts (unicode): d₁ en d₂ (vermijd mathtext parsing)
    ax.set_xlabel("d₁", fontsize=11, labelpad=8)  # type: ignore
    ax.set_ylabel("d₂", fontsize=11, labelpad=8)  # type: ignore

def render_square_matplotlib_figure(
    draw_fn: Callable[[matplotlib.axes.Axes], None],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    size_inches: float = 5.5,
    dpi: int = 160
) -> Figure:
    fig = Figure(figsize=(size_inches, size_inches), dpi=dpi)
    _ = FigureCanvas(fig)  # backend-canvas
    ax = fig.add_subplot(111)
    setup_square_axes(ax, xlim, ylim)
    draw_fn(ax)
    fig.tight_layout(pad=0.9)
    return fig

# ====== Helpers voor consistent blauw en kleine labels ======
BLUE = "C0"      # Matplotlib default blauw
ORANGE = "C1"    # Matplotlib default oranje
LABEL_FS = 9     # kleiner font voor k_i labels

def annotate_points(
    ax: matplotlib.axes.Axes,
    pts: np.ndarray,
    ts: np.ndarray,
    label_prefix: str,
    color: str,
) -> None:
    """Labels bij de punten met hun echte t-waarde (bv. k4, l4). label_prefix is 'k' of 'l'."""
    offsets = [(3, 3), (3, -8), (-8, 3)]  # kleiner offset = dichter bij punten
    for i, ((x, y), tval) in enumerate(zip(pts, ts)):
        ax.scatter([x], [y], s=40, zorder=3, color=color)  # type: ignore
        off = offsets[i % len(offsets)]
        # Nettere subscript: integer indien mogelijk (converteer robuust naar float)
        lbl: str
        try:
            tnum = float(tval)  # type: ignore[arg-type]
        except Exception:
            # fallback bij exotische types
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
    """Lijn + punten + labels voor k (blauw) en l (oranje)."""
    # k-lijn en punten (blauw)
    ax.plot(k_points_plot[:, 0], k_points_plot[:, 1], linewidth=2.0, color=BLUE)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    annotate_points(ax, k_points_plot, k_vals_plot, "k", BLUE)

    # l-lijn en punten (oranje)
    ax.plot(l_points_plot[:, 0], l_points_plot[:, 1], linewidth=2.0, color=ORANGE)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    annotate_points(ax, l_points_plot, l_vals_plot, "l", ORANGE)

def draw_generated_empty(ax: matplotlib.axes.Axes) -> None:
    """Draw right graph with labeled points and animation elements."""
    n_k = k_points_plot.shape[0]
    n_l_total = l_points_plot.shape[0]
    total_original = n_k + n_l_total
    
    # Toon configuratie en iteratie status op de grafiek zelf
    if st.session_state.get("anim_running", False):
        current_config = st.session_state.get("anim_current_config", 1)
        completed_iters = st.session_state.get("anim_completed_iterations", 0)
        search_steps = st.session_state.get("anim_search_steps", 0)
        status_text = f"Config {current_config} | Iteration {completed_iters + 1} | Step {search_steps}"
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=9,  # type: ignore
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Check if animation is active
    # Toon altijd de laatste generatiepunten als animatie klaar is
    has_animation = st.session_state.get("show_anim_circle", False)
    # Gebruik de meest recente succesvolle punten, ook als animatie niet meer loopt
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    gen_pt = st.session_state.get("anim_generated_point", None) if has_animation else None
    parent_idx = st.session_state.get("anim_parent_idx", 0) if has_animation else 0
    in_search = st.session_state.get("anim_in_search", False) if has_animation else False
    anim_running = st.session_state.get("anim_running", False)
    
    offsets = [(3, 3), (3, -8), (-8, 3)]
    
    # Helper to create label
    def make_label(prefix: str, tval: float, gen_marker: str = "") -> str:
        try:
            tnum = float(tval)
        except Exception:
            tnum = float(np.array(tval, dtype=float))
        lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
        if gen_marker:
            if len(gen_marker) <= 2:  # ' or ''
                return rf"${prefix}{gen_marker}_{{{lbl}}}$"
            else:  # ^{*}
                return rf"${prefix}{gen_marker}_{{{lbl}}}$"
        return rf"${prefix}_{{{lbl}}}$"
    
    # Helper: haal originele index veilig uit een SuccessfulPoint (compatibel met oude sessies)
    def _get_original_index(sp: SuccessfulPoint) -> int | None:
        try:
            if "original_parent_idx" in sp:
                return int(sp["original_parent_idx"])  # type: ignore[index]
            # Fallback: afleiden uit parent_idx indien die direct naar een origineel verwijst
            if "parent_idx" in sp:
                pidx = int(sp["parent_idx"])  # type: ignore[index]
                if 0 <= pidx < total_original:
                    return pidx
        except Exception:
            return None
        return None

    # Bepaal welke lijnen transparant moeten zijn op basis van succesvolle punten
    transparent_segments_k: set[tuple[int, int]] = set()
    transparent_segments_l: set[tuple[int, int]] = set()
    
    for succ_pt_data in successful_points:
        succ_parent_idx = succ_pt_data.get("parent_idx", -1)  # type: ignore[call-arg]
        if succ_parent_idx < n_k:
            # k-punt: segmenten voor en na dit punt worden transparant
            if succ_parent_idx > 0:
                transparent_segments_k.add((succ_parent_idx - 1, succ_parent_idx))
            if succ_parent_idx < n_k - 1:
                transparent_segments_k.add((succ_parent_idx, succ_parent_idx + 1))
        else:
            # l-punt
            local_idx = succ_parent_idx - n_k
            if local_idx > 0:
                transparent_segments_l.add((local_idx - 1, local_idx))
            if local_idx < l_points_plot.shape[0] - 1:
                transparent_segments_l.add((local_idx, local_idx + 1))
    
    # Draw k lines
    for i in range(len(k_points_plot) - 1):
        alpha = 0.2 if (i, i+1) in transparent_segments_k else 1.0
        ax.plot([k_points_plot[i, 0], k_points_plot[i+1, 0]], 
                [k_points_plot[i, 1], k_points_plot[i+1, 1]], 
                linewidth=2.0, color=BLUE, alpha=alpha, zorder=1)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    
    # Bepaal indices die al een succesvol gegenereerd punt hebben (toon geen originele labels meer voor die indices)
    latest_indices: set[int] = set()
    for sp in successful_points:
        oi = _get_original_index(sp)
        if oi is not None:
            latest_indices.add(oi)

    # Draw k points; hide original marker & label entirely if daughters exist for this index
    for i, ((x, y), tval) in enumerate(zip(k_points_plot, k_vals_plot)):
        if i not in latest_indices:
            # Alleen tonen als er GEEN succesvolle gegenereerde punten voor deze index zijn
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
    
    # Draw l lines
    for i in range(len(l_points_plot) - 1):
        alpha = 0.2 if (i, i+1) in transparent_segments_l else 1.0
        ax.plot([l_points_plot[i, 0], l_points_plot[i+1, 0]], 
                [l_points_plot[i, 1], l_points_plot[i+1, 1]], 
                linewidth=2.0, color=ORANGE, alpha=alpha, zorder=1)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    
    # Draw l points; hide original marker & label entirely if daughters exist for this l-index
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
    
    # Overlay: verbind enkel opeenvolgende k-punten en enkel opeenvolgende l-punten,
    # waarbij we voor elk origineel index het meest recente gegenereerde punt gebruiken indien aanwezig.
    # Als animatie volledig klaar is (niet meer draait), teken altijd de overlay lijnen
    if len(successful_points) > 0 or not anim_running:
        latest_by_original: dict[int, np.ndarray] = {}
        for sp in successful_points:
            oi = _get_original_index(sp)
            if oi is not None:
                latest_by_original[oi] = sp["point"]

        # k-overlay
        k_path_pts: list[np.ndarray] = []
        for i in range(n_k):
            pt_k = latest_by_original[i] if i in latest_by_original else k_points_plot[i]
            k_path_pts.append(pt_k)
        for i in range(len(k_path_pts) - 1):
            p0 = k_path_pts[i]
            p1 = k_path_pts[i + 1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=2.2, color=BLUE, alpha=1.0, zorder=4)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        # l-overlay
        n_l = l_points_plot.shape[0]
        l_path_pts: list[np.ndarray] = []
        for j in range(n_l):
            orig_idx = n_k + j
            pt_l = latest_by_original[orig_idx] if orig_idx in latest_by_original else l_points_plot[j]
            l_path_pts.append(pt_l)
        for j in range(len(l_path_pts) - 1):
            q0 = l_path_pts[j]
            q1 = l_path_pts[j + 1]
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], linewidth=2.2, color=ORANGE, alpha=1.0, zorder=4)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    # Teken enkel de LAATSTE succesvolle punten per origineel index met labels
    if len(successful_points) > 0:
        # Bepaal per originele index het laatst gegenereerde punt
        latest_success: dict[int, SuccessfulPoint] = {}
        for sp in successful_points:
            oi = _get_original_index(sp)
            if oi is not None:
                latest_success[oi] = sp

        # Sorteer op originele index voor consistente rendering volgorde
        for original_parent_idx in sorted(latest_success.keys()):
            succ_pt_data = latest_success[original_parent_idx]
            succ_pt = succ_pt_data["point"]
            
            # Tel hoeveel generaties er zijn voor deze originele index
            generation_count = 0
            for sp in successful_points:
                if _get_original_index(sp) == original_parent_idx:
                    generation_count += 1

            # Bepaal generation marker op basis van aantal generaties
            if generation_count == 1:
                gen_marker = "'"
            elif generation_count == 2:
                gen_marker = "''"
            else:
                gen_marker = "^{*}"

            # Bepaal prefix en tval op basis van ORIGINELE parent
            if original_parent_idx < n_k:
                prefix = "k"
                color = BLUE
                tval = k_vals_plot[original_parent_idx]
            else:
                prefix = "l"
                color = ORANGE
                local_idx = original_parent_idx - n_k
                tval = l_vals_plot[local_idx]

            # Teken enkel het laatste punt met label
            ax.scatter([succ_pt[0]], [succ_pt[1]], s=60, zorder=6, color=color)  # type: ignore
            off = offsets[original_parent_idx % len(offsets)]
            # Cast tval expliciet naar float voor type checker
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
    
    # Als we bezig zijn met zoeken (rood punt + cirkel zonder label) - alleen als animatie nog loopt
    if has_animation and in_search and gen_pt is not None and anim_running:
        all_pts = st.session_state.get("anim_all_pts", np.array([]))
        distance = st.session_state.get("anim_distance", 0.0)
        
        if all_pts.shape[0] > 0:
            n_k = k_points_plot.shape[0]
            n_l = l_points_plot.shape[0]
            total_original = n_k + n_l
            if parent_idx < total_original:
                parent_pt = all_pts[parent_idx]
            else:
                # Haal het gegenereerde parent punt uit successful_points
                succ_list: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                sidx = int(parent_idx - total_original)
                if 0 <= sidx < len(succ_list):
                    parent_pt = succ_list[sidx]["point"]
                else:
                    parent_pt = np.array([0.0, 0.0])

            # Teken het huidige gegenereerde rode punt (zonder label)
            ax.scatter([gen_pt[0]], [gen_pt[1]], s=60, zorder=6, color='red')  # type: ignore

            # Teken rode cirkel rond parent punt
            circle = matplotlib.patches.Circle(
                (parent_pt[0], parent_pt[1]),
                radius=distance,
                edgecolor='red',
                facecolor='none',
                linewidth=2.0,
                zorder=5
            )
            ax.add_patch(circle)  # type: ignore

# ============= Layout =============
col1, col2 = st.columns(2, gap="small")

with col1:
    st.markdown("<div class='figure-title'>original configuration</div>", unsafe_allow_html=True)
    # d1-volgorde boven de grafiek
    st.latex(make_d1_order_latex())
    # d2-volgorde boven de grafiek
    st.latex(make_d2_order_latex())
    fig_left = render_square_matplotlib_figure(draw_original, XLIM, YLIM)
    st.pyplot(fig_left, clear_figure=True)
    
    # Vergelijk x-volgorde (d1) wanneer animatie actief is
    if st.session_state.get("show_anim_circle", False):
        left_d1 = make_d1_order_latex()
        right_d1 = make_d1_order_latex_generated()
        left_order = _extract_order_string(left_d1)
        right_order = _extract_order_string(right_d1)
        same_d1 = left_order == right_order
        
        # Debug: toon de volgordes
        st.caption(f"Left: {left_order}")
        st.caption(f"Right: {right_order}")
        st.markdown(f"**d1 order match: {same_d1}**")
        
        # Vergelijk y-volgorde (d2)
        left_d2 = make_d2_order_latex()
        right_d2 = make_d2_order_latex_generated()
        left_order_d2 = _extract_order_string(left_d2)
        right_order_d2 = _extract_order_string(right_d2)
        same_d2 = left_order_d2 == right_order_d2
        
        st.caption(f"Left d2: {left_order_d2}")
        st.caption(f"Right d2: {right_order_d2}")
        st.markdown(f"**d2 order match: {same_d2}**")
    
    # Download button for left figure
    _buf_left: IO[bytes] = io.BytesIO()
    fig_left.savefig(_buf_left, format="png", dpi=160, bbox_inches="tight")  # type: ignore
    st.download_button(
        label="save as png",
        data=_buf_left.getvalue(),
        file_name="original.png",
        mime="image/png",
        key="dl_left_png",
    )

with col2:
    st.markdown("<div class='figure-title'>generated configuration</div>", unsafe_allow_html=True)
    
    # d1-volgorde boven de grafiek (inclusief gegenereerde punt indien animatie actief)
    st.latex(make_d1_order_latex_generated())
    # d2-volgorde boven de grafiek (inclusief gegenereerde punt indien animatie actief)
    st.latex(make_d2_order_latex_generated())
    
    # Render right graph
    fig_right = render_square_matplotlib_figure(draw_generated_empty, XLIM, YLIM)
    
    # Display the figure
    st.pyplot(fig_right, clear_figure=True)
    
    # Vergelijk x-volgorde (d1) wanneer animatie actief is
    if st.session_state.get("show_anim_circle", False):
        left_d1 = make_d1_order_latex()
        right_d1 = make_d1_order_latex_generated()
        left_order = _extract_order_string(left_d1)
        right_order = _extract_order_string(right_d1)
        same_d1 = left_order == right_order
        
        # Debug: toon de volgordes
        st.caption(f"Left: {left_order}")
        st.caption(f"Right: {right_order}")
        st.markdown(f"**d1 order match: {same_d1}**")
        
        # Vergelijk y-volgorde (d2)
        left_d2 = make_d2_order_latex()
        right_d2 = make_d2_order_latex_generated()
        left_order_d2 = _extract_order_string(left_d2)
        right_order_d2 = _extract_order_string(right_d2)
        same_d2 = left_order_d2 == right_order_d2
        
        st.caption(f"Left d2: {left_order_d2}")
        st.caption(f"Right d2: {right_order_d2}")
        st.markdown(f"**d2 order match: {same_d2}**")
    
    # Download button for right figure
    _buf_right: IO[bytes] = io.BytesIO()
    fig_right.savefig(_buf_right, format="png", dpi=160, bbox_inches="tight")  # type: ignore
    st.download_button(
        label="save as png",
        data=_buf_right.getvalue(),
        file_name="generated.png",
        mime="image/png",
        key="dl_right_png",
    )

# ============= Animatie-voortgang: NA het tekenen, halveren om de 5 s tot volgordes gelijk zijn (primes genegeerd) =============
if st.session_state.get("anim_running", False):
    # Bereken volgorde-strings
    left_d1 = make_d1_order_latex()
    left_d2 = make_d2_order_latex()
    right_d1 = make_d1_order_latex_generated()
    right_d2 = make_d2_order_latex_generated()

    # Vergelijk zonder primes
    same_d1 = _extract_order_string(left_d1) == _extract_order_string(right_d1)
    same_d2 = _extract_order_string(left_d2) == _extract_order_string(right_d2)

    completed_iterations = int(st.session_state.get("anim_completed_iterations", 0))
    max_iterations = int(st.session_state.get("anim_max_iterations", 3))
    iterations_per_run = int(st.session_state.get("anim_iterations_per_run", 3))
    search_steps = int(st.session_state.get("anim_search_steps", 0))
    max_search_steps = 7  # Maximum aantal zoekstappen per iteratie

    # Haal animatiestaat (zorg dat imports beschikbaar zijn)
    distance = float(st.session_state.get("anim_distance", maxdist))
    angle = float(st.session_state.get("anim_angle", 0.0))
    gen_pt = st.session_state.get("anim_generated_point", None)
    parent_idx = int(st.session_state.get("anim_parent_idx", 0))
    all_pts = st.session_state.get("anim_all_pts", np.array([]))
    successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
    in_search = bool(st.session_state.get("anim_in_search", True))

    # Check of we wachten na een voltooide configuratie (extra wachttijd)
    if st.session_state.get("anim_config_complete_wait", False):
        # Reset de wait flag
        st.session_state["anim_config_complete_wait"] = False
        # Extra 5 seconden wachttijd en ga dan direct door
        time.sleep(5.0)
        st.rerun()
    
    # Bepaal strategie
    strategy = st.session_state.get("anim_strategy", "exponential")

    # Voor EXPONENTIËLE strategie: originele succesvoorwaarde.
    # Voor BINAIRE strategie: GEEN vroege success bij match; pas succes na 7 stappen (of distance<=0)
    if strategy != "binary" and ((same_d1 and same_d2 and gen_pt is not None) or (distance <= 0.0 and gen_pt is not None)):
        # Registreer succesvolle punt
        # Determine correct parent_point and original_parent_idx
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
        st.session_state["anim_search_steps"] = 0  # Reset search steps
        st.session_state["anim_in_search"] = True  # volgende punt zoeken
        
        # Reset binary-specifieke variabelen na succesvolle iteratie (indien exponentieel triggerde succes is dat irrelevant voor binary)
        if strategy == "binary":
            st.session_state["anim_binary_last_match_point"] = None
            st.session_state["anim_binary_base_distance"] = maxdist / 2.0
            st.session_state["anim_binary_base_radius"] = maxdist / 2.0
            st.session_state["anim_binary_step_size"] = maxdist / 4.0
            st.session_state["anim_binary_last_step_match"] = False

        # Check of we klaar zijn met deze configuratie
        if completed_iterations + 1 >= max_iterations:
            # Configuratie compleet - sla op en start nieuwe config indien nodig
            current_config = int(st.session_state.get("anim_current_config", 1))
            num_configs = int(st.session_state.get("anim_num_configs", 1))
            
            # Sla huidige configuratie op
            # Bewaar configuraties (typeless list to satisfy runtime; ignore static typing complaints)
            all_configs = st.session_state.get("anim_all_configs", [])  # type: ignore[assignment]
            all_configs.append({
                "config_num": current_config,
                "points": list(successful_points)
            })
            st.session_state["anim_all_configs"] = all_configs
            
            # Markeer alle huidige successful_points als "van vorige config"
            for sp in successful_points:
                sp["config_num"] = current_config  # type: ignore[typeddict-item]
            
            # CSV: voeg 6 rijen toe voor deze voltooide configuratie (t=0,1,2 en o in {k=0, l=1})
            try:
                output_c = int(selected_c_int) + int(current_config)  # start bij c_start+1 -> 12,13,...
            except Exception:
                output_c = int(current_config)

            # Bepaal voor alle originele indices de LAATSTE gegenereerde positie (anders origineel punt)
            latest_generated: dict[int, np.ndarray] = {}
            for sp2 in successful_points:
                if "original_parent_idx" in sp2:
                    oi2 = int(sp2["original_parent_idx"])  # type: ignore[index]
                    latest_generated[oi2] = sp2["point"]

            csv_append: list[str] = []
            # k-punten (o=0)
            n_k_loc = k_points_plot.shape[0]
            for i, tval in enumerate(k_vals_plot.tolist()):
                if i >= n_k_loc:
                    break
                if i in latest_generated:
                    px, py = float(latest_generated[i][0]), float(latest_generated[i][1])
                else:
                    px, py = float(k_points_plot[i, 0]), float(k_points_plot[i, 1])
                csv_append.append(f"{output_c},{tval},0,{px:.6f},{py:.6f}")

            # l-punten (o=1)
            n_l_loc = l_points_plot.shape[0]
            for j, tval in enumerate(l_vals_plot.tolist()):
                if j >= n_l_loc:
                    break
                glob_idx = n_k_loc + j
                if glob_idx in latest_generated:
                    px, py = float(latest_generated[glob_idx][0]), float(latest_generated[glob_idx][1])
                else:
                    px, py = float(l_points_plot[j, 0]), float(l_points_plot[j, 1])
                csv_append.append(f"{output_c},{tval},1,{px:.6f},{py:.6f}")

            # Sla op in session_state accumulator
            acc: list[str] = st.session_state.get("anim_csv_lines", [])
            acc.extend(csv_append)
            st.session_state["anim_csv_lines"] = acc
            
            if current_config < num_configs:
                # Start nieuwe configuratie - blijf anim_running op True
                st.session_state["anim_current_config"] = current_config + 1
                st.session_state["anim_completed_iterations"] = 0
                # NIET resetten - behoud alle punten zodat de lijnen correct blijven
                # st.session_state["anim_successful_points"] = []
                st.session_state["anim_search_steps"] = 0
                st.session_state["anim_running"] = True  # Expliciet op True houden
                st.session_state["show_anim_circle"] = True  # Circle blijft zichtbaar
                
                # Kies random originele index
                n_k_reset = k_points_plot.shape[0]
                n_l_reset = l_points_plot.shape[0]
                total_original_reset = n_k_reset + n_l_reset
                all_pts_reset = np.vstack([k_points_plot, l_points_plot])
                all_indices_reset = list(range(total_original_reset))
                if all_indices_reset:
                    chosen_idx_reset = int(np.random.choice(all_indices_reset))
                else:
                    chosen_idx_reset = 0
                
                # Zoek de JONGSTE generatie voor deze index (niet het originele punt!)
                youngest_point_reset = None
                youngest_success_idx_reset = None
                for idx, s in reversed(list(enumerate(successful_points))):
                    oi = s.get("original_parent_idx", None)
                    if oi is not None and int(oi) == chosen_idx_reset:
                        youngest_point_reset = s["point"]
                        youngest_success_idx_reset = idx
                        break
                
                if youngest_point_reset is not None and youngest_success_idx_reset is not None:
                    # Gebruik de jongste generatie
                    parent_pt_reset = youngest_point_reset
                    parent_idx_reset = total_original_reset + youngest_success_idx_reset
                else:
                    # Geen gegenereerde versie, gebruik origineel punt
                    parent_idx_reset = chosen_idx_reset
                    parent_pt_reset = all_pts_reset[parent_idx_reset]
                
                # Reset voor nieuwe config
                distance = maxdist
                angle = float(np.random.uniform(0, 2 * np.pi))
                new_x = parent_pt_reset[0] + distance * np.cos(angle)
                new_y = parent_pt_reset[1] + distance * np.sin(angle)
                new_gen_pt = np.array([new_x, new_y])
                
                if not (XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]):
                    angle = (angle + np.pi) % (2 * np.pi)
                    new_x = parent_pt_reset[0] + distance * np.cos(angle)
                    new_y = parent_pt_reset[1] + distance * np.sin(angle)
                    new_gen_pt = np.array([new_x, new_y])
                
                st.session_state["anim_parent_idx"] = parent_idx_reset
                st.session_state["anim_angle"] = angle
                st.session_state["anim_generated_point"] = new_gen_pt
                st.session_state["anim_distance"] = distance
                st.session_state["anim_all_pts"] = all_pts_reset
                
                # Reset binary-specifieke variabelen voor nieuwe config
                if strategy == "binary":
                    st.session_state["anim_binary_last_match_point"] = None
                    st.session_state["anim_binary_base_distance"] = maxdist / 2.0
                    st.session_state["anim_binary_base_radius"] = maxdist / 2.0
                    st.session_state["anim_binary_step_size"] = maxdist / 4.0
                    st.session_state["anim_binary_last_step_match"] = False
                
                # Zet flag voor extra wachttijd bij volgende rerun
                st.session_state["anim_config_complete_wait"] = True
            else:
                # Alle configuraties compleet
                st.session_state["anim_running"] = False
                st.session_state["show_anim_circle"] = False
        else:
            # Kies nieuwe parent: ALTIJD de jongste generatie van een random originele index
            n_k = k_points_plot.shape[0]
            n_l = l_points_plot.shape[0]
            total_original = n_k + n_l
            # Kies random originele index
            all_indices = list(range(total_original))
            if all_indices:
                chosen_idx = int(np.random.choice(all_indices))
            else:
                chosen_idx = 0

            # Zoek de jongste (laatste generatie) voor deze index
            # Als een generated point bestaat, gebruik die en update parent_idx zodat het naar de juiste plek in successful_points wijst
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
                # parent_idx points to the generated point in successful_points (offset by total_original)
                parent_idx_new = total_original + youngest_success_idx
            else:
                if chosen_idx < n_k:
                    parent_pt_new = k_points_plot[chosen_idx]
                else:
                    parent_pt_new = l_points_plot[chosen_idx - n_k]
                parent_idx_new = chosen_idx
            # Reset afstand naar maxdist voor nieuwe zoekfase
            distance = maxdist
            angle = float(np.random.uniform(0, 2 * np.pi))
            new_x = parent_pt_new[0] + distance * np.cos(angle)
            new_y = parent_pt_new[1] + distance * np.sin(angle)
            new_gen_pt = np.array([new_x, new_y])

            # Als buiten limieten bij start van iteratie: spiegel hoek (R -> R+180°) en plaats opnieuw
            if not (XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]):
                angle = (angle + np.pi) % (2 * np.pi)
                new_x = parent_pt_new[0] + distance * np.cos(angle)
                new_y = parent_pt_new[1] + distance * np.sin(angle)
                new_gen_pt = np.array([new_x, new_y])

            st.session_state["anim_parent_idx"] = parent_idx_new
            st.session_state["anim_angle"] = angle
            st.session_state["anim_generated_point"] = new_gen_pt
            st.session_state["anim_distance"] = distance
            
            # Reset binary-specifieke variabelen voor nieuwe iteratie
            if strategy == "binary":
                st.session_state["anim_binary_last_match_point"] = None
                st.session_state["anim_binary_base_distance"] = maxdist / 2.0
                st.session_state["anim_binary_base_radius"] = maxdist / 2.0
                st.session_state["anim_binary_step_size"] = maxdist / 4.0
                st.session_state["anim_binary_last_step_match"] = False

    else:
        # Strategie-afhankelijke voortgang
        strategy = st.session_state.get("anim_strategy", "exponential")

        if strategy == "binary":
            # Binaire zoek: exact 7 stappen met opslaan van laatste match-positie
            step_size = float(st.session_state.get("anim_binary_step_size", maxdist / 2.0))
            last_match_point = st.session_state.get("anim_binary_last_match_point", None)
            
            # STAP 1: Check huidige order match en sla positie op indien BEIDE matches True
            current_match = bool(same_d1 and same_d2)
            if current_match and gen_pt is not None:
                # Sla huidige (x, y) op als laatste match-positie
                st.session_state["anim_binary_last_match_point"] = np.array(gen_pt, dtype=float)
                last_match_point = np.array(gen_pt, dtype=float)
            
            # STAP 2: Pas afstand aan op basis van match
            search_steps += 1
            st.session_state["anim_search_steps"] = search_steps
            
            if current_match:
                distance = distance + step_size
            else:
                distance = distance - step_size

            # Clamp afstand
            if distance < 0.0:
                distance = 0.0

            # Halveer stap voor volgende ronde
            step_size = step_size / 2.0
            st.session_state["anim_binary_step_size"] = step_size
            st.session_state["anim_distance"] = distance

            # STAP 3: Verplaats punt naar nieuwe positie langs zelfde hoek
            if all_pts.size > 0:
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
                new_x = parent_pt_cur[0] + distance * np.cos(angle)
                new_y = parent_pt_cur[1] + distance * np.sin(angle)
                st.session_state["anim_generated_point"] = np.array([new_x, new_y])

            # STAP 4: Na 7 stappen: gebruik opgeslagen match-positie of parent punt
            if search_steps >= max_search_steps:
                # Bepaal finale positie: gebruik last_match_point als beschikbaar, anders parent punt
                if last_match_point is not None:
                    # Er was minstens 1 keer een volledige match: gebruik die positie
                    final_point = last_match_point.copy()
                else:
                    # Geen enkele volledige match: plaats op parent punt
                    if all_pts.size > 0:
                        n_k_final = k_points_plot.shape[0]
                        n_l_final = l_points_plot.shape[0]
                        total_original_final = n_k_final + n_l_final
                        if parent_idx < total_original_final:
                            final_point = all_pts[parent_idx].copy()
                        else:
                            succ_list_final: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                            sidx_final = int(parent_idx - total_original_final)
                            if 0 <= sidx_final < len(succ_list_final):
                                final_point = succ_list_final[sidx_final]["point"].copy()
                            else:
                                final_point = np.array([0.0, 0.0])
                    else:
                        final_point = np.array([0.0, 0.0])
                
                # Update generated point naar finale positie
                st.session_state["anim_generated_point"] = final_point
                
                # Registreer succesvol punt
                n_k_loc = k_points_plot.shape[0]
                n_l_loc = l_points_plot.shape[0]
                total_original_loc = n_k_loc + n_l_loc
                if all_pts.size > 0 and parent_idx < total_original_loc:
                    parent_point_val = all_pts[parent_idx]
                    original_parent_idx_val = parent_idx
                else:
                    succ_list2: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
                    sidx2 = int(parent_idx - total_original_loc)
                    if 0 <= sidx2 < len(succ_list2):
                        parent_point_val = succ_list2[sidx2]["point"]
                        original_parent_idx_val = succ_list2[sidx2]["original_parent_idx"]
                    else:
                        parent_point_val = np.array([0.0, 0.0])
                        original_parent_idx_val = 0
                
                sp_bin: SuccessfulPoint = {
                    "point": final_point,
                    "parent_idx": parent_idx,
                    "parent_point": parent_point_val,
                    "original_parent_idx": original_parent_idx_val,
                    "iteration": completed_iterations,
                }
                successful_points.append(sp_bin)
                st.session_state["anim_successful_points"] = successful_points
                st.session_state["anim_completed_iterations"] = completed_iterations + 1
                st.session_state["anim_search_steps"] = 0
                st.session_state["anim_in_search"] = True
                # Reset binary vars
                st.session_state["anim_binary_last_match_point"] = None
                st.session_state["anim_binary_base_distance"] = maxdist / 2.0
                st.session_state["anim_binary_base_radius"] = maxdist / 2.0
                st.session_state["anim_binary_step_size"] = maxdist / 4.0
                st.session_state["anim_binary_last_step_match"] = False

                # Check config completion (reuse existing logic triggers by copying essential subset)
                if completed_iterations + 1 >= max_iterations:
                        current_config = int(st.session_state.get("anim_current_config", 1))
                        num_configs = int(st.session_state.get("anim_num_configs", 1))
                        all_configs_local = st.session_state.get("anim_all_configs", [])  # type: ignore[assignment]
                        all_configs_local.append({"config_num": current_config, "points": list(successful_points)})
                        st.session_state["anim_all_configs"] = all_configs_local
                        for spx in successful_points:
                            spx["config_num"] = current_config  # type: ignore[typeddict-item]
                        try:
                            output_c = int(selected_c_int) + int(current_config)
                        except Exception:
                            output_c = int(current_config)
                        latest_generated: dict[int, np.ndarray] = {}
                        for sp2 in successful_points:
                            if "original_parent_idx" in sp2:
                                oi2 = int(sp2["original_parent_idx"])  # type: ignore[index]
                                latest_generated[oi2] = sp2["point"]
                        csv_append: list[str] = []
                        n_k_loc2 = k_points_plot.shape[0]
                        for i, tval in enumerate(k_vals_plot.tolist()):
                            if i >= n_k_loc2:
                                break
                            if i in latest_generated:
                                px, py = float(latest_generated[i][0]), float(latest_generated[i][1])
                            else:
                                px, py = float(k_points_plot[i, 0]), float(k_points_plot[i, 1])
                            csv_append.append(f"{output_c},{tval},0,{px:.6f},{py:.6f}")
                        n_l_loc2 = l_points_plot.shape[0]
                        for j, tval in enumerate(l_vals_plot.tolist()):
                            if j >= n_l_loc2:
                                break
                            glob_idx = n_k_loc2 + j
                            if glob_idx in latest_generated:
                                px, py = float(latest_generated[glob_idx][0]), float(latest_generated[glob_idx][1])
                            else:
                                px, py = float(l_points_plot[j, 0]), float(l_points_plot[j, 1])
                            csv_append.append(f"{output_c},{tval},1,{px:.6f},{py:.6f}")
                        acc_local: list[str] = st.session_state.get("anim_csv_lines", [])
                        acc_local.extend(csv_append)
                        st.session_state["anim_csv_lines"] = acc_local
                        if current_config < num_configs:
                            st.session_state["anim_current_config"] = current_config + 1
                            st.session_state["anim_completed_iterations"] = 0
                            st.session_state["anim_search_steps"] = 0
                            st.session_state["anim_running"] = True
                            st.session_state["show_anim_circle"] = True
                            # Kies nieuwe parent voor start volgende config
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
                            for idx_r, s_r in reversed(list(enumerate(successful_points))):
                                oi_r = s_r.get("original_parent_idx", None)
                                if oi_r is not None and int(oi_r) == chosen_idx_reset:
                                    youngest_point_reset = s_r["point"]
                                    youngest_success_idx_reset = idx_r
                                    break
                            if youngest_point_reset is not None and youngest_success_idx_reset is not None:
                                parent_pt_reset = youngest_point_reset
                                parent_idx_reset = total_original_reset + youngest_success_idx_reset
                            else:
                                parent_idx_reset = chosen_idx_reset
                                parent_pt_reset = all_pts_reset[parent_idx_reset]
                            distance_reset = maxdist / 2.0  # Binaire strategie: start met halve maxdist
                            angle_reset = float(np.random.uniform(0, 2 * np.pi))
                            new_x_reset = parent_pt_reset[0] + distance_reset * np.cos(angle_reset)
                            new_y_reset = parent_pt_reset[1] + distance_reset * np.sin(angle_reset)
                            new_gen_pt_reset = np.array([new_x_reset, new_y_reset])
                            if not (XLIM[0] <= new_x_reset <= XLIM[1] and YLIM[0] <= new_y_reset <= YLIM[1]):
                                angle_reset = (angle_reset + np.pi) % (2 * np.pi)
                                new_x_reset = parent_pt_reset[0] + distance_reset * np.cos(angle_reset)
                                new_y_reset = parent_pt_reset[1] + distance_reset * np.sin(angle_reset)
                                new_gen_pt_reset = np.array([new_x_reset, new_y_reset])
                            st.session_state["anim_parent_idx"] = parent_idx_reset
                            st.session_state["anim_angle"] = angle_reset
                            st.session_state["anim_generated_point"] = new_gen_pt_reset
                            st.session_state["anim_distance"] = distance_reset
                            st.session_state["anim_all_pts"] = all_pts_reset
                            st.session_state["anim_config_complete_wait"] = True
                        else:
                            st.session_state["anim_running"] = False
                            st.session_state["show_anim_circle"] = False
                else:
                    # Iteratie succesvol, maar configuratie nog niet rond: kies een nieuwe parent
                    n_k2 = k_points_plot.shape[0]
                    n_l2 = l_points_plot.shape[0]
                    total_original2 = n_k2 + n_l2
                    all_indices2 = list(range(total_original2))
                    if all_indices2:
                        chosen_idx2 = int(np.random.choice(all_indices2))
                    else:
                        chosen_idx2 = 0
                    youngest_point2 = None
                    youngest_success_idx2 = None
                    for idx2, s2 in reversed(list(enumerate(successful_points))):
                        oi2_chk = s2.get("original_parent_idx", None)
                        if oi2_chk is not None and int(oi2_chk) == chosen_idx2:
                            youngest_point2 = s2["point"]
                            youngest_success_idx2 = idx2
                            break
                    if youngest_point2 is not None and youngest_success_idx2 is not None:
                        parent_pt_new2 = youngest_point2
                        parent_idx_new2 = total_original2 + youngest_success_idx2
                    else:
                        if chosen_idx2 < n_k2:
                            parent_pt_new2 = k_points_plot[chosen_idx2]
                        else:
                            parent_pt_new2 = l_points_plot[chosen_idx2 - n_k2]
                        parent_idx_new2 = chosen_idx2

                    distance2 = maxdist / 2.0  # Binaire strategie: start met halve maxdist
                    angle2 = float(np.random.uniform(0, 2 * np.pi))
                    new_x2 = parent_pt_new2[0] + distance2 * np.cos(angle2)
                    new_y2 = parent_pt_new2[1] + distance2 * np.sin(angle2)
                    new_gen_pt2 = np.array([new_x2, new_y2])
                    if not (XLIM[0] <= new_x2 <= XLIM[1] and YLIM[0] <= new_y2 <= YLIM[1]):
                        angle2 = (angle2 + np.pi) % (2 * np.pi)
                        new_x2 = parent_pt_new2[0] + distance2 * np.cos(angle2)
                        new_y2 = parent_pt_new2[1] + distance2 * np.sin(angle2)
                        new_gen_pt2 = np.array([new_x2, new_y2])

                    st.session_state["anim_parent_idx"] = parent_idx_new2
                    st.session_state["anim_angle"] = angle2
                    st.session_state["anim_generated_point"] = new_gen_pt2
                    st.session_state["anim_distance"] = distance2
                    # Reset binary vars voor nieuwe iteratie
                    st.session_state["anim_binary_last_match_point"] = None
                    st.session_state["anim_binary_base_distance"] = maxdist / 2.0
                    st.session_state["anim_binary_base_radius"] = maxdist / 2.0
                    st.session_state["anim_binary_step_size"] = maxdist / 4.0
                    st.session_state["anim_binary_last_step_match"] = False
                    
                    # Extra pauze voor binaire strategie na lijnen hertekenen
                    time.sleep(5.0)
                    st.rerun()
            # Klaar met deze turn voor binary; skip exponentiële else
        else:
            # EXPONENTIËLE STRATEGIE: handhaaf 7-stappen fallback en halveerafstand
            # Verhoog search_steps en forceer na 7 stappen plaatsing op parent
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
                # Stop verdere halvering deze beurt
                time.sleep(5.0)
                st.rerun()

            # Ga door met halveren zolang we nog niet geforceerd hebben
            # EXPONENTIËLE STRATEGIE: halveer afstand en verplaats punt langs dezelfde hoek
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

                # Halveer afstand, maar stop als te klein
                new_distance = distance / 2.0
                min_distance = 1e-5
                if new_distance < min_distance:
                    # Forceer een kleine random perturbatie zodat we niet vastlopen
                    new_distance = min_distance * 2.0
                    angle = float(np.random.uniform(0, 2 * np.pi))
                # Nieuwe hoek met kleine random jitter om stagnatie te voorkomen
                angle += float(np.random.uniform(-0.25, 0.25))
                angle = angle % (2 * np.pi)
                new_x = parent_pt_cur[0] + new_distance * np.cos(angle)
                new_y = parent_pt_cur[1] + new_distance * np.sin(angle)
                new_gen_pt = np.array([new_x, new_y])

                # Als buiten limieten: spiegel hoek
                if not (XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]):
                    angle = (angle + np.pi) % (2 * np.pi)
                    new_x = parent_pt_cur[0] + new_distance * np.cos(angle)
                    new_y = parent_pt_cur[1] + new_distance * np.sin(angle)
                    new_gen_pt = np.array([new_x, new_y])

                st.session_state["anim_generated_point"] = new_gen_pt
                st.session_state["anim_distance"] = new_distance
                st.session_state["anim_angle"] = angle
                st.session_state["anim_in_search"] = True

    # Auto-advance - ALTIJD rerun (ook bij config complete wait)
    time.sleep(5.0)  # 5000 ms = 5 seconden
    st.rerun()

# ============= CSV Export Section =============
st.markdown("<hr />", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:1.5rem;'>Generated Configurations (CSV)</h3>", unsafe_allow_html=True)

# Build CSV from accumulator that updates on each completed config
csv_acc: list[str] = st.session_state.get("anim_csv_lines", [])
if csv_acc:
    csv_content = "\n".join(["c,t,o,x,y", *csv_acc])
    st.text_area(
        "Copy the generated configurations below:",
        value=csv_content,
        height=220,
        key="csv_export"
    )
    st.download_button(
        label="Download CSV",
        data=csv_content,
        file_name="generated_configs.csv",
        mime="text/csv",
        key="dl_csv"
    )
else:
    st.info("The CSV will update automatically after each completed configuration.")
