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

n_timepoints = len(_t_common)
default_window = min(3, n_timepoints)
with sc2:
    num_timestamps = st.slider("number of timestamps", min_value=1, max_value=n_timepoints, value=default_window, step=1, key="cfg_k")

max_start_pos = max(1, n_timepoints + 1 - num_timestamps)
with sc3:
    # Laat de gebruiker starten op een echte t-waarde i.p.v. een positie-index
    valid_start_count = max(1, n_timepoints - num_timestamps + 1)
    valid_starts = _t_common[:valid_start_count]
    start_t = st.select_slider(
        "starting time (t)",
        options=valid_starts,
        value=valid_starts[0],
        key="cfg_start_t",
    )

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

# ============= Selecteer venster op basis van GUI (start_t, num_timestamps) =============
# Vind index van gekozen start-t in de gemeenschappelijke t-lijst
try:
    start_idx = _t_common.index(start_t)  # type: ignore[arg-type]
except ValueError:
    # Fallback: als om een of andere reden start_t niet in de lijst zit, terug naar 0
    start_idx = 0
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
    if strategy == "binary":
        # Binary strategy:7 iterations, store matches
        if "bin_running" not in st.session_state or not st.session_state["bin_running"]:
            # Initialize binary animation
            all_pts = np.vstack([k_points_plot, l_points_plot])
            all_ts = np.concatenate([k_vals_plot, l_vals_plot])
            n_total = all_pts.shape[0]
            parent_idx = int(np.random.randint(0, n_total))  # type: ignore[arg-type]
            parent_pt = all_pts[parent_idx]
            distance = maxdist
            
            # Random angle with bounds checking
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
            
            # Direct halveren: R = maxdist/2
            distance = distance / 2.0
            gen_x = parent_pt[0] + distance * np.cos(alfa)
            gen_y = parent_pt[1] + distance * np.sin(alfa)
            generated_point = np.array([gen_x, gen_y])
            
            # Initialize state
            st.session_state["bin_running"] = True
            st.session_state["bin_parent_idx"] = parent_idx
            st.session_state["bin_parent_pt"] = parent_pt
            st.session_state["bin_angle"] = alfa
            st.session_state["bin_distance"] = distance  # Start op maxdist
            st.session_state["bin_gen_pt"] = generated_point
            st.session_state["bin_iteration"] = 0  # Stap 0 (n=0)
            st.session_state["bin_matches"] = []
            st.session_state["show_anim_circle"] = True
            st.session_state["anim_in_search"] = True
            st.session_state["anim_generated_point"] = generated_point
            st.session_state["anim_parent_idx"] = parent_idx
            st.session_state["anim_all_pts"] = all_pts
            st.session_state["anim_all_ts"] = all_ts
            st.session_state["anim_distance"] = distance
            st.session_state["anim_successful_points"] = []  # Empty for binary
            st.session_state["anim_angle"] = alfa
            st.rerun()
    elif strategy == "exponential":
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
        # Startafstand
        distance = maxdist

        # Probeer een punt binnen het grafiekveld te vinden
        max_attempts = 20  # Voorkom oneindige loop
        for _ in range(max_attempts):
            alfa = float(np.random.uniform(0, 2 * np.pi))
            gen_x = parent_pt[0] + distance * np.cos(alfa)
            gen_y = parent_pt[1] + distance * np.sin(alfa)
            
            # Check if point is within graph bounds
            if XLIM[0] <= gen_x <= XLIM[1] and YLIM[0] <= gen_y <= YLIM[1]:
                break
            # Als niet binnen veld, probeer opnieuw met nieuwe random hoek
        else:
            # Na max_attempts nog steeds niet binnen veld: clip naar bounds
            gen_x = np.clip(gen_x, XLIM[0], XLIM[1])
            gen_y = np.clip(gen_y, YLIM[0], YLIM[1])
        
        generated_point = np.array([gen_x, gen_y])

        # Initialiseer animatiestatus
        st.session_state["show_anim_circle"] = True
        st.session_state["anim_running"] = True
        st.session_state["anim_circle_idx"] = parent_idx
        st.session_state["anim_distance"] = distance
        st.session_state["anim_generated_point"] = generated_point
        st.session_state["anim_parent_idx"] = parent_idx
        st.session_state["anim_all_pts"] = all_pts
        st.session_state["anim_all_ts"] = all_ts
        st.session_state["anim_angle"] = alfa
        st.session_state["anim_iteration"] = 0  # Start bij 0, wordt verhoogd na succesvolle iteratie
        st.session_state["anim_max_iterations"] = 3  # 3 iteraties per configuratie
        st.session_state["anim_iterations_per_run"] = 3  # Voer 3 iteraties uit per run
        st.session_state["anim_completed_iterations"] = 0  # Aantal voltooide iteraties
        st.session_state["anim_last_update"] = time.time()  # voor 5s wachttijd
        st.session_state["anim_successful_points"] = []  # Succesvol gegenereerde punten met labels
        st.session_state["anim_in_search"] = True  # Bezig met zoeken naar juiste positie
        st.session_state["anim_num_configs"] = num_configs_to_generate  # Number of configs to generate
        st.session_state["anim_current_config"] = 1  # Current config being generated
        st.session_state["anim_all_configs"] = []  # Store all completed configurations
        st.session_state["anim_search_steps"] = 0  # Aantal zoekstappen binnen huidige iteratie

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
        entries.append((float(gen_pt[0]), rf"k_{{{lbl_parent}}}{parent_primes}"))
    else:
        entries.append((float(gen_pt[0]), rf"l_{{{lbl_parent}}}{parent_primes}"))

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
            entries.append((float(latest_generated[i][0]), rf"k_{{{lbl}}}{primes_i}"))
        else:
            entries.append((float(x), rf"k_{{{lbl}}}{primes_i}"))
    
    # l-punten
    for j, (x, t) in enumerate(zip(l_points_plot[:, 0].tolist(), l_vals_plot.tolist())):
        glob_idx = n_k + j
        if glob_idx == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(glob_idx, 0)
        primes_j = _prime_str(gen_cnt)
        if glob_idx in latest_generated:
            entries.append((float(latest_generated[glob_idx][0]), rf"l_{{{lbl}}}{primes_j}"))
        else:
            entries.append((float(x), rf"l_{{{lbl}}}{primes_j}"))
    
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
        entries.append((float(gen_pt[1]), rf"k_{{{lbl_parent}}}{parent_primes}"))
    else:
        entries.append((float(gen_pt[1]), rf"l_{{{lbl_parent}}}{parent_primes}"))

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
            entries.append((float(latest_generated[i][1]), rf"k_{{{lbl}}}{primes_i}"))
        else:
            entries.append((float(y), rf"k_{{{lbl}}}{primes_i}"))
    
    # l-punten
    for j, (y, t) in enumerate(zip(l_points_plot[:, 1].tolist(), l_vals_plot.tolist())):
        glob_idx = n_k + j
        if glob_idx == base_idx:
            continue
        lbl = _format_t_subscript(t)
        gen_cnt = generation_counts.get(glob_idx, 0)
        primes_j = _prime_str(gen_cnt)
        if glob_idx in latest_generated:
            entries.append((float(latest_generated[glob_idx][1]), rf"l_{{{lbl}}}{primes_j}"))
        else:
            entries.append((float(y), rf"l_{{{lbl}}}{primes_j}"))
    
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
            rf"${label_prefix}_{{{lbl}}}$",
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
    
    # Binary strategy: show status info on graph
    if st.session_state.get("bin_running", False):
        iteration = int(st.session_state.get("bin_iteration", 0))
        status_text = f"Config 1 | Iteration 1 | Step {iteration + 1}"
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
            # LaTeX syntax: subscript VOOR superscript: k_{0}' of k_{0}^{*}
            return rf"{prefix}_{{{lbl}}}{gen_marker}"
        return rf"{prefix}_{{{lbl}}}"
    
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
    
    # Als we bezig zijn met zoeken (rood punt + cirkel zonder label)
    # Voor binary: check bin_running, voor exponential: check anim_running
    bin_running = st.session_state.get("bin_running", False)
    if has_animation and in_search and gen_pt is not None and (anim_running or bin_running):
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

# ============= Binary strategy animation logic =============
if st.session_state.get("bin_running", False):
    iteration = int(st.session_state.get("bin_iteration", 0))
    parent_pt = st.session_state.get("bin_parent_pt", np.array([0.0, 0.0]))
    parent_idx = int(st.session_state.get("bin_parent_idx", 0))
    angle = float(st.session_state.get("bin_angle", 0.0))
    distance = float(st.session_state.get("bin_distance", maxdist))
    matches: list = st.session_state.get("bin_matches", [])
    
    # Check order
    left_d1 = make_d1_order_latex()
    left_d2 = make_d2_order_latex()
    right_d1 = make_d1_order_latex_generated()
    right_d2 = make_d2_order_latex_generated()
    same_d1 = _extract_order_string(left_d1) == _extract_order_string(right_d1)
    same_d2 = _extract_order_string(left_d2) == _extract_order_string(right_d2)
    
    # Scenario 1: BEIDE order matches - sla ALLE punten op
    gen_pt = st.session_state.get("bin_gen_pt")
    if same_d1 and same_d2 and gen_pt is not None:
        # Sla alle punten op: k_points, l_points, en het gegenereerde punt
        all_points_snapshot = {
            "k_points": k_points_plot.copy(),
            "l_points": l_points_plot.copy(), 
            "generated_point": np.array(gen_pt, dtype=float),
            "parent_idx": parent_idx,
            "iteration": iteration
        }
        matches.append(all_points_snapshot)
        st.session_state["bin_matches"] = matches
    
    # Check of we klaar zijn (NA het uitvoeren van de huidige stap)
    if iteration >= 6:  # iteration 0-6 = 7 stappen
        # Done - verhoog iteration voor correcte telling
        iteration += 1
        st.session_state["bin_iteration"] = iteration
        st.session_state["bin_running"] = False
        st.session_state["show_anim_circle"] = False
        if len(matches) > 0:
            final_snapshot = matches[-1]
            final_pt = final_snapshot["generated_point"]
        else:
            final_pt = parent_pt
        st.success(f"Binary strategie klaar! {len(matches)} matches in {iteration} stappen. Finaal: [{final_pt[0]:.2f}, {final_pt[1]:.2f}]")
    else:
        # Ga door naar volgende stap
        n = iteration  # Huidige stap nummer (0-6)
        iteration += 1
        st.session_state["bin_iteration"] = iteration
        
        # Bereken delta: (maxdist/2) * (0.5^n)
        # Stap 0: delta = (maxdist/2) * 1 = maxdist/2 ... WACHT NEE
        # Stap 0: delta = maxdist * (0.5^2) = maxdist/4 = 0.25 (bij maxdist=1.0)
        # Stap 1: delta = maxdist * (0.5^3) = maxdist/8 = 0.125
        delta = maxdist * (0.5 ** (n + 2))
        
        # Bij BEIDE matches: TEL OP
        # Bij GEEN of PARTIËLE match: TREK AF
        if same_d1 and same_d2:
            # BEIDE matches: R = R + delta
            new_distance = distance + delta
        else:
            # GEEN of PARTIËLE match: R = R - delta
            new_distance = distance - delta
        
        # Herbereken punt met nieuwe afstand
        gen_x = parent_pt[0] + new_distance * np.cos(angle)
        gen_y = parent_pt[1] + new_distance * np.sin(angle)
        new_gen_pt = np.array([gen_x, gen_y])
        
        st.session_state["bin_distance"] = new_distance
        st.session_state["bin_gen_pt"] = new_gen_pt
        st.session_state["anim_generated_point"] = new_gen_pt
        st.session_state["anim_distance"] = new_distance
        
        time.sleep(2.0)
        st.rerun()

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
        # Extra 2 seconden wachttijd en ga dan direct door
        time.sleep(2.0)
        st.rerun()
    
    # Normale animatie logica
    # Als volgordes matchen OF distance is 0: beschouw als succesvolle generatie
    if (same_d1 and same_d2 and gen_pt is not None) or (distance <= 0.0 and gen_pt is not None):
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

        # Check of we klaar zijn met deze configuratie
        if completed_iterations + 1 >= max_iterations:
            # Configuratie compleet - sla op en start nieuwe config indien nodig
            current_config = int(st.session_state.get("anim_current_config", 1))
            num_configs = int(st.session_state.get("anim_num_configs", 1))
            
            # Sla huidige configuratie op
            all_configs: list = st.session_state.get("anim_all_configs", [])
            all_configs.append({
                "config_num": current_config,
                "points": list(successful_points)
            })
            st.session_state["anim_all_configs"] = all_configs
            
            # Markeer alle huidige successful_points als "van vorige config"
            for sp in successful_points:
                sp["config_num"] = current_config  # type: ignore[typeddict-item]
            
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
                
                # Probeer een punt binnen het grafiekveld te vinden
                max_attempts = 20  # Voorkom oneindige loop
                for _ in range(max_attempts):
                    angle = float(np.random.uniform(0, 2 * np.pi))
                    new_x = parent_pt_reset[0] + distance * np.cos(angle)
                    new_y = parent_pt_reset[1] + distance * np.sin(angle)
                    
                    # Check if point is within graph bounds
                    if XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]:
                        break
                    # Als niet binnen veld, probeer opnieuw met nieuwe random hoek
                else:
                    # Na max_attempts nog steeds niet binnen veld: clip naar bounds
                    new_x = np.clip(new_x, XLIM[0], XLIM[1])
                    new_y = np.clip(new_y, YLIM[0], YLIM[1])
                
                new_gen_pt = np.array([new_x, new_y])
                
                st.session_state["anim_parent_idx"] = parent_idx_reset
                st.session_state["anim_angle"] = angle
                st.session_state["anim_generated_point"] = new_gen_pt
                st.session_state["anim_distance"] = distance
                st.session_state["anim_all_pts"] = all_pts_reset
                
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
            
            # Probeer een punt binnen het grafiekveld te vinden
            max_attempts = 20  # Voorkom oneindige loop
            for _ in range(max_attempts):
                angle = float(np.random.uniform(0, 2 * np.pi))
                new_x = parent_pt_new[0] + distance * np.cos(angle)
                new_y = parent_pt_new[1] + distance * np.sin(angle)
                
                # Check if point is within graph bounds
                if XLIM[0] <= new_x <= XLIM[1] and YLIM[0] <= new_y <= YLIM[1]:
                    break
                # Als niet binnen veld, probeer opnieuw met nieuwe random hoek
            else:
                # Na max_attempts nog steeds niet binnen veld: clip naar bounds
                new_x = np.clip(new_x, XLIM[0], XLIM[1])
                new_y = np.clip(new_y, YLIM[0], YLIM[1])
            
            new_gen_pt = np.array([new_x, new_y])

            st.session_state["anim_parent_idx"] = parent_idx_new
            st.session_state["anim_angle"] = angle
            st.session_state["anim_generated_point"] = new_gen_pt
            st.session_state["anim_distance"] = distance

    else:
        # Geen match: verhoog search_steps
        search_steps += 1
        st.session_state["anim_search_steps"] = search_steps
        
        # Na 7 stappen: plaats punt exact op parent positie (straal = 0)
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
                
                # Plaats punt exact op parent positie
                st.session_state["anim_generated_point"] = parent_pt_cur.copy()
                st.session_state["anim_distance"] = 0.0
                st.session_state["anim_in_search"] = True
        else:
            # Halveer afstand en verplaats punt langs dezelfde hoek
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
    time.sleep(2.0)  # 2000 ms = 2 seconden
    st.rerun()

# ============= CSV Export Section =============
st.markdown("<hr />", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:1.5rem;'>Generated Configuration (CSV)</h3>", unsafe_allow_html=True)

# Build CSV from ALL configurations (completed + current)
all_configs_list: list = st.session_state.get("anim_all_configs", [])
current_successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
current_config_num = int(st.session_state.get("anim_current_config", 1))

if all_configs_list or current_successful_points:
    # Collect all points from all configurations
    all_points_by_config: dict[int, list[SuccessfulPoint]] = {}
    
    # Add completed configurations
    for config_data in all_configs_list:
        config_num = config_data["config_num"]
        points = config_data["points"]
        all_points_by_config[config_num] = points
    
    # Add current configuration if it has points
    if current_successful_points:
        all_points_by_config[current_config_num] = current_successful_points
    
    # Build mapping: (config_num, original_idx) -> generated point
    # Only keep the LATEST point for each (config, original_idx) combination
    latest_generated: dict[tuple[int, int], np.ndarray] = {}
    for config_num, points in all_points_by_config.items():
        for sp in points:
            orig_idx = sp.get("original_parent_idx", 0)
            latest_generated[(config_num, orig_idx)] = sp["point"]
    
    # Get all unique config_nums
    all_config_nums = sorted(all_points_by_config.keys())
    
    # Create CSV rows for ALL points (k and l) at ALL timestamps for EACH config
    csv_rows: list[tuple[int, float, int, float, float]] = []
    
    n_k = k_points_plot.shape[0]
    n_l = l_points_plot.shape[0]
    
    for config_num in all_config_nums:
        # Calculate c value: selected_c + config_num
        c_value = selected_c_int + config_num
        
        # Add k-points (o=0) for all timestamps
        for i in range(n_k):
            t_val = float(k_vals_plot[i])
            if (config_num, i) in latest_generated:
                # Use generated point for this config
                point = latest_generated[(config_num, i)]
            else:
                # Use original point
                point = k_points_plot[i]
            csv_rows.append((c_value, t_val, 0, float(point[0]), float(point[1])))
        
        # Add l-points (o=1) for all timestamps
        for j in range(n_l):
            t_val = float(l_vals_plot[j])
            orig_idx = n_k + j
            if (config_num, orig_idx) in latest_generated:
                # Use generated point for this config
                point = latest_generated[(config_num, orig_idx)]
            else:
                # Use original point
                point = l_points_plot[j]
            csv_rows.append((c_value, t_val, 1, float(point[0]), float(point[1])))
    
    # Sort by c, then t, then o
    csv_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    
    # Build CSV string
    csv_lines = ["c,t,o,x,y"]
    for c, t, o, x, y in csv_rows:
        csv_lines.append(f"{c},{t},{o},{x:.6f},{y:.6f}")
    
    csv_content = "\n".join(csv_lines)
    
    # Display in text area
    st.text_area(
        "Copy the generated configuration below:",
        value=csv_content,
        height=200,
        key="csv_export"
    )
    
    # Download button
    st.download_button(
        label="Download as CSV",
        data=csv_content,
        file_name=f"generated_config_c{selected_c_int}.csv",
        mime="text/csv",
        key="dl_csv"
    )
else:
    st.info("Run an animation to generate configuration data.")
