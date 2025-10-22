import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# Page configuration
st.set_page_config(layout="wide", page_title="N_C Visualization Dashboard")
st.title("🔍 N_C Visualization Dashboard")

# --- Sidebar: path settings ---
st.sidebar.header("Path settings")

base_folder = st.sidebar.text_input(
    "Base folder (contains all result files):",
    value=r"C:\Users\jmverdoo\OneDrive - UGent\2022-... Werk Ugent\01 Projecten\OPEN\Micro-analysis Traffic\NEW_PDP_results\2POI_ROUGH"
)

dataset_path = st.sidebar.text_input(
    "Dataset file (CSV or Parquet):",
    value=os.path.join(base_folder, "N_C_Dataset.csv")
)


# Sidebar: selection for distance matrix and navigation
st.sidebar.header("Settings")
matrix_option = st.sidebar.selectbox(
    "Choose which distance matrix to investigate:",
    [
        "Buffer + Rough",
        "Buffer only",
        "Rough only",
        "Fundamental"
    ]
)
view = st.sidebar.radio("Choose a view:", [
    "Heatmap", "Clustering", "MDS & t-SNE", "Per Configuration", "Inequality Matrices", "TopK Results"
])

# Load files (now with parameter for the chosen distance matrix)
@st.cache_data
def load_data(chosen_matrix: str, dataset_path: str, base_folder: str):
    df_data = pd.read_csv(dataset_path, header=None)
    df_data.columns = ["con", "tst", "poi", "poi_x", "poi_y"]

    # Matrix kiezen
    if chosen_matrix == "Buffer + Rough":
        matrix_filename = "N_C_PDPg_bufferrough_DistanceMatrix.csv"
    elif chosen_matrix == "Buffer only":
        matrix_filename = "N_C_PDPg_buffer_DistanceMatrix.csv"
    elif chosen_matrix == "Rough only":
        matrix_filename = "N_C_PDPg_rough_DistanceMatrix.csv"
    else:
        matrix_filename = "N_C_PDPg_fundamental_DistanceMatrix.csv"

    full_matrix_path = os.path.join(base_folder, matrix_filename)
    df_dist_raw = pd.read_csv(full_matrix_path, header=None)

    df_dist = pd.DataFrame(
        df_dist_raw.values,
        index=[str(i) for i in range(df_dist_raw.shape[0])],
        columns=[str(i) for i in range(df_dist_raw.shape[1])]
    )
    return df_data, df_dist

def style_highway(fig):
    # Achtergrond grijs maken
    fig.update_layout(
        plot_bgcolor="lightgray",
        xaxis=dict(range=[0, 600]),
        yaxis=dict(range=[-5, 5])
    )

    # Witte lijnen voor de rijstrookmarkeringen
    for y_lane in [-1.75, 1.75]:
        fig.add_shape(
            type="line",
            x0=0, x1=600,
            y0=y_lane, y1=y_lane,
            line=dict(color="white", width=2, dash="dash")
        )

    return fig

# Load data with the chosen matrix
df_data, df_dist = load_data(matrix_option, dataset_path, base_folder)

# --- View 1A: Heatmap ---
if view == "Heatmap":
    st.header("🌡️ Distance Matrix Heatmap")
    st.write(f"**Selected distance matrix:** `{matrix_option}`")

    show_values = st.checkbox("Toon celwaarden", value=False)
    zmin = st.slider("Z-min (kleine afstanden)", 0, 100, 0)
    zmax = st.slider("Z-max (grote afstanden)", 0, 100, 100)

    # Optioneel: laat gebruiker hoogte en breedte aanpassen
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider("Breedte figuur (px)", 600, 1800, 1200, step=100)
    with col2:
        height = st.slider("Hoogte figuur (px)", 400, 1200, 800, step=100)

    # Bouw kwargs dynamisch
    hm_kwargs = dict(
        img=df_dist.values.astype(float),
        labels=dict(x="config", y="config", color="distance"),
        x=df_dist.columns.tolist(),
        y=df_dist.index.tolist(),
        aspect="equal",
        color_continuous_scale="OrRd",
        zmin=zmin,
        zmax=zmax,
        origin="lower",
    )
    if show_values:
        hm_kwargs["text_auto"] = ".2f"

    fig_hm = px.imshow(**hm_kwargs)

    # Grotere layout
    fig_hm.update_layout(
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=70, b=70)
    )

    st.plotly_chart(fig_hm, use_container_width=False)


# --- View 1B: Clustering (Dendrogram + Inspectie) ---
elif view == "Clustering":
    st.header("🌳 Hierarchical Clustering")
    st.write(f"**Selected distance matrix:** `{matrix_option}`")

    # Condense matrix once
    condensed = squareform(df_dist.values.astype(float))

    linkage_method = st.selectbox(
        "Linkage method:",
        ["ward", "average", "single", "complete"],
        index=0,
        help="Ward werkt het best met (quasi-)Euclidische afstanden."
    )

    # Cache de linkage per methode voor snelheid
    @st.cache_data
    def _compute_linkage(condensed_arr, method):
        from scipy.cluster.hierarchy import linkage
        return linkage(condensed_arr, method=method)

    linkage_matrix = _compute_linkage(condensed, linkage_method)

    # Dendrogram
    fig_dendro = ff.create_dendrogram(
        df_dist.values.astype(float),
        orientation='bottom',
        labels=df_dist.index.tolist(),
        linkagefun=lambda _: linkage_matrix
    )
    fig_dendro.update_layout(width=1000, height=800, margin=dict(t=50, b=150))
    st.plotly_chart(fig_dendro, use_container_width=True)

    # Cluster inspection
    from scipy.cluster.hierarchy import fcluster

    st.subheader("🔎 Cluster Inspection")
    max_clusters = st.slider("Aantal clusters:", 2, 20, 2)
    cluster_labels = fcluster(linkage_matrix, t=max_clusters, criterion='maxclust')

    cluster_dict = {}
    for label, config in zip(cluster_labels, df_dist.index):
        cluster_dict.setdefault(label, []).append(config)

    selected_cluster = st.selectbox(
        "Kies cluster:",
        sorted(cluster_dict.keys())
    )
    members = cluster_dict[selected_cluster]
    st.write(f"Cluster {selected_cluster} bevat {len(members)} configuraties: {', '.join(members)}")

    # Representatieve config = laagste som van afstanden binnen submatrix
    submatrix = df_dist.loc[members, members].astype(float)
    rep_config = submatrix.sum(axis=1).idxmin()
    st.success(f"Representative configuration: {rep_config}")

    # Toon animatie van representatieve configuratie
    df_rep = df_data[df_data["con"] == int(rep_config)].copy().sort_values("tst")
    df_rep["poi"] = df_rep["poi"].astype(str)

    x_min, x_max = df_rep["poi_x"].min(), df_rep["poi_x"].max()
    y_min, y_max = df_rep["poi_y"].min(), df_rep["poi_y"].max()
    x_span = max(x_max - x_min, 1e-9)
    y_span = y_max - y_min
    base_width = 400
    height = max(int(base_width * (y_span / x_span)) + 50, 400)

    fig_rep_anim = px.scatter(
        df_rep,
        x="poi_x", y="poi_y",
        animation_frame="tst",
        animation_group="poi",
        color="poi",
        color_discrete_map={"0": "blue", "1": "red"},
        labels={"poi_x": "X", "poi_y": "Y", "poi": "POI"},
        title=f"Representatieve beweging — config {rep_config}",
        range_x=[x_min - 1, x_max + 1],
        range_y=[y_min - 1, y_max + 1]
    )
    fig_rep_anim = style_highway(fig_rep_anim)
    fig_rep_anim.update_layout(width=base_width, height=height)
    st.plotly_chart(fig_rep_anim, use_container_width=False)

# --- View 2: Dimensionality Reduction: MDS & t-SNE ---
elif view == "MDS & t-SNE":
    st.header("📉 Dimensionality Reduction: MDS & t-SNE")
    st.write(f"**Selected distance matrix:** `{matrix_option}`")

    dim_red_option = st.selectbox("Choose method:", ["MDS", "t-SNE"], index=0)
    n_components = st.slider("Number of dimensions for projection:", 2, 3, 2)

    # Determine how many samples we have = number of configurations
    n_samples = df_dist.shape[0]

    if dim_red_option == "MDS":
        st.info("MDS based on the distance matrix")
        mds = MDS(
            n_components=n_components,
            dissimilarity='precomputed',
            metric=True,
            random_state=1,
            normalized_stress='auto'
        )
        coords = mds.fit_transform(df_dist.values)

    else:
        st.info("t-SNE based on the distance matrix")

        # Keep perplexity < n_samples. Maximize at n_samples-1.
        max_perp = max(1, n_samples - 1)
        perplexity = st.slider(
            "Perplexity (must < #samples):",
            min_value=1,
            max_value=max_perp,
            value=min(30, max_perp)
        )

        tsne = TSNE(
            n_components=n_components,
            metric='precomputed',
            init='random',
            random_state=0,
            perplexity=perplexity
        )
        coords = tsne.fit_transform(df_dist.values)

    coords_df = pd.DataFrame(coords, columns=[f"Dim{i+1}" for i in range(n_components)])
    coords_df["config"] = df_dist.index

    if n_components == 2:
        fig = px.scatter(
            coords_df,
            x="Dim1",
            y="Dim2",
            text="config",
            title=f"{dim_red_option} (2D)"
        )
    else:
        fig = px.scatter_3d(
            coords_df,
            x="Dim1",
            y="Dim2",
            z="Dim3",
            text="config",
            title=f"{dim_red_option} (3D)"
        )

    st.plotly_chart(fig, use_container_width=True)

    # === Vergelijking tussen configuraties ===
    st.markdown("### 🔍 Vergelijk twee configuraties")
    
    # Selectievakken voor twee configuraties
    col1, col2 = st.columns(2)
    with col1:
        config_1 = st.selectbox(
            "Selecteer eerste configuratie:",
            options=sorted(df_data["con"].unique()),
            key="mds_config_1"
        )
    with col2:
        remaining_configs = [c for c in sorted(df_data["con"].unique()) if c != config_1]
        config_2 = st.selectbox(
            "Selecteer tweede configuratie:",
            options=remaining_configs,
            key="mds_config_2"
        )

    # Toon de bewegingen van beide configuraties
    cols = st.columns(2)

    def plot_con_animation(df_con, title_suffix=""):
        df_con_sorted = df_con.sort_values("tst").copy()
        df_con_sorted["poi"] = df_con_sorted["poi"].astype(str)
        fig_anim = px.scatter(
            df_con_sorted,
            x="poi_x", y="poi_y",
            animation_frame="tst",
            animation_group="poi",
            color="poi",
            color_discrete_map={"0": "blue", "1": "red"},
            title=title_suffix,
            range_x=[df_con_sorted["poi_x"].min() - 1, df_con_sorted["poi_x"].max() + 1],
            range_y=[df_con_sorted["poi_y"].min() - 1, df_con_sorted["poi_y"].max() + 1]
        )
        fig_anim.update_layout(width=500, height=450)
        return style_highway(fig_anim)

    with cols[0]:
        st.write(f"**Configuratie {config_1}**")
        df_con_1 = df_data[df_data["con"] == config_1]
        st.plotly_chart(plot_con_animation(df_con_1, f"con {config_1}"), use_container_width=True)

    with cols[1]:
        st.write(f"**Configuratie {config_2}**")
        df_con_2 = df_data[df_data["con"] == config_2]
        st.plotly_chart(plot_con_animation(df_con_2, f"con {config_2}"), use_container_width=True)

    # Toon de afstand tussen deze configuraties
    distance = df_dist.loc[str(config_1), str(config_2)]
    st.info(f"Afstand tussen configuratie {config_1} en {config_2}: {distance:.2f}")


# --- View 3: Visualization per Configuration ---
elif view == "Per Configuration":
    st.header("🧭 Visualization per Configuration")
    st.write(f"**Selected distance matrix:** `{matrix_option}`")

    selected_con = st.selectbox("Select configuration (con):", sorted(df_data["con"].unique()))
    df_con = df_data[df_data["con"] == selected_con].copy()

    # Static plot with lines between tst
    st.subheader("Static XY plot with lines")
    df_con_sorted = df_con.sort_values("tst")
    fig_static = px.line(
        df_con_sorted,
        x="poi_x",
        y="poi_y",
        color="poi",
        title=f"XY line plot of tst/poi for configuration {selected_con}",
        labels={"poi_x": "X", "poi_y": "Y"}
    )
    fig_static = style_highway(fig_static)
    st.plotly_chart(fig_static, use_container_width=True)

    # Animation over time
    st.subheader("Animation over time (per tst)")
    fig_anim = px.scatter(
        df_con,
        x="poi_x",
        y="poi_y",
        animation_frame="tst",
        color="tst",
        title=f"Animation over time for configuration {selected_con}",
        range_x=[df_con["poi_x"].min() - 1, df_con["poi_x"].max() + 1],
        range_y=[df_con["poi_y"].min() - 1, df_con["poi_y"].max() + 1]
    )
    fig_anim = style_highway(fig_anim)
    st.plotly_chart(fig_anim, use_container_width=True)


# --- View 4: Inequality Matrices (INTERACTIEF + vergelijking) ---
elif view == "Inequality Matrices":
    st.header("⚖️ Inequality Matrices — window-based & interactief")
    st.write(f"**Selected distance matrix:** `{matrix_option}`")

    # === 1) Selectie van configuratie en descriptor ===
    selected_con = st.selectbox("Select configuration (con):", sorted(df_data["con"].unique()))
    descriptor = st.radio("Choose descriptor:", ["d0 (X)", "d1 (Y)"], horizontal=True, index=0)
    dim = 0 if descriptor.startswith("d0") else 1
    value_col = "poi_x" if dim == 0 else "poi_y"

    df_con = df_data[df_data["con"] == selected_con].copy()
    if df_con.empty:
        st.warning("Geen data voor deze configuratie.")
        st.stop()

    # === 2) Window-instellingen ===
    poi_count = int(df_con["poi"].nunique())
    max_tst_total = int(df_con["tst"].max())  # laatste index-waarde (0-based)
    total_timestamps = max_tst_total + 1  # +1 omdat we vanaf 0 tellen

    colA, colB, colC, colD = st.columns(4)
    with colA:
        window_length = st.number_input(
            "Window length (#timestamps)", min_value=1, max_value=total_timestamps, value=total_timestamps-1, step=1,
            help=f"Aantal opeenvolgende timestamps per window (max {total_timestamps}, want timestamps lopen van 0 t/m {max_tst_total})."
        )
    with colB:
        max_start = max(0, max_tst_total - (window_length - 1))
        start_tst = st.slider("Start timestamp (tst)", 0, max_start, 0)
    with colC:
        default_rough = 0.0
        rough = st.number_input("rough tolerance", value=default_rough, step=0.1,
                                help="Waarden |Δ| ≤ rough → gelijk (1); Δ > rough → 0; Δ < -rough → 2.")
    with colD:
        show_numbers = st.checkbox("Annotaties (celwaarden)", value=True)

    # Optie: POIs sorteren op (eerste) waarde of gewoon op poi-id
    colE, colF = st.columns(2)
    with colE:
        sort_by_value = st.checkbox("Sorteer POIs op waarde (eerste timestamp in window)", value=False)
    with colF:
        palette_mode = st.selectbox("Kleurschema", ["groen/geel/rood (1/==/2)", "blauw/grijs/rood (-1/0/+1)"], index=0)

    # === 3) Subset & ordening binnen window ===
    window_ts = list(range(start_tst, start_tst + window_length))
    df_w = df_con[df_con["tst"].isin(window_ts)].copy()
    if df_w.empty or df_w["tst"].nunique() < window_length:
        st.error("Window valt (gedeeltelijk) buiten bereik of data ontbreekt voor sommige timestamps.")
        st.stop()

    counts = df_w.groupby(["poi", "tst"]).size().reset_index(name="n")
    if (counts["n"] != 1).any() or counts.shape[0] != poi_count * window_length:
        st.warning("Onvolledige window: niet elke (poi, tst) combinatie is aanwezig. Matrix kan gaten bevatten.")

    # Volgorde = eerst poi oplopend, dan tst oplopend
    if sort_by_value:
        first_t = window_ts[0]
        df_first = df_w[df_w["tst"] == first_t][["poi", value_col]].sort_values(value_col)
        poi_order = df_first["poi"].tolist()
    else:
        poi_order = sorted(df_w["poi"].unique().tolist())  # Remove [::-1] to have poi0 first

    df_w["poi"] = df_w["poi"].astype(int)
    df_w = df_w.set_index(["poi", "tst"]).sort_index().reset_index()
    df_w["poi"] = pd.Categorical(df_w["poi"], categories=poi_order, ordered=True)
    df_w = df_w.sort_values(["poi", "tst"]).reset_index(drop=True)

    # === 4) Maak de inequality-matrix ===
    import numpy as np
    n = int(poi_count * window_length)
    values = df_w[value_col].to_numpy()
    diffs = values[np.newaxis, :] - values[:, np.newaxis]
    A = np.empty((n, n), dtype=int)
    A[(np.abs(diffs) <= rough)] = 1
    A[(diffs > rough)] = 0
    A[(diffs < -rough)] = 2

    ticks = []
    for poi in poi_order:
        for w_idx, t in enumerate(window_ts):
            ticks.append(f"c{selected_con}_t{t}_d{'x' if dim==0 else 'y'}_p{int(poi)}_w{w_idx}")

    import plotly.express as px

    if palette_mode.startswith("groen"):
        colorscale = [[0.0, "#16a34a"], [0.5, "#facc15"], [1.0, "#dc2626"]]
        zmin, zmax = 0, 2
        legend_caption = "0: > rough, 1: |Δ|≤rough, 2: < -rough"
    else:
        colorscale = [[0.0, "#1d4ed8"], [0.5, "#e5e7eb"], [1.0, "#dc2626"]]
        zmin, zmax = 0, 2
        legend_caption = "0: > rough, 1: |Δ|≤rough, 2: < -rough"


    # --- FIGUUR: numerieke assen + eigen ticklabels (zodat middenlijnen werken) ---
    idx = list(range(len(ticks)))
    fig = px.imshow(
        A,
        x=idx, y=idx,                    # numerieke assen i.p.v. strings
        origin="lower",
        aspect="equal",
        zmin=zmin, zmax=zmax,
        color_continuous_scale=colorscale,
        labels={"x": "j", "y": "i", "color": "klasse"},
        title=(f"Inequality matrix — con {selected_con}, window=({start_tst}..{start_tst+window_length-1}), "
            f"{'X' if dim==0 else 'Y'}; rough={rough}"),
    )

    # Toon je string-ticks, maar enkel elke N-de om overlap te vermijden
    max_labels = 25  # max aantal labels dat we willen tonen
    step = max(1, len(ticks) // max_labels)  # toon 1 op elke 'step'

    visible_tick_idx = list(range(0, len(ticks), step))
    visible_tick_labels = [ticks[i] for i in visible_tick_idx]

    fig.update_xaxes(
        tickmode="array",
        tickvals=visible_tick_idx,
        ticktext=visible_tick_labels,
        tickangle=45
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=visible_tick_idx,
        ticktext=visible_tick_labels
    )


    fig.update_layout(width=1300, height=950, margin=dict(l=50, r=50, t=70, b=70))

    # Hover
    fig.update_traces(hovertemplate="i=%{y}<br>j=%{x}<br>waarde=%{z}<extra></extra>")

    # Optionele annotaties (let op: x=j, y=i zijn nu numeriek)
    if show_numbers and n <= 200:
        for i in range(n):
            for j in range(n):
                fig.add_annotation(x=j, y=i, text=str(int(A[i, j])),
                                showarrow=False, font=dict(size=12))

    # === Middenlijnen (snel, maar zichtbaar) ===
    show_midlines = st.checkbox("Toon middenlijnen (kwadranten verdeling)", value=True)
    if show_midlines:
        mid = len(ticks) / 2 - 0.5               # midden tussen cellen
        fig.add_shape(
            type="line",
            x0=mid, x1=mid, y0=-0.5, y1=len(ticks)-0.5,
            xref="x", yref="y",
            line=dict(color="black", width=2),
            layer="above"
        )
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(ticks)-0.5, y0=mid, y1=mid,
            xref="x", yref="y",
            line=dict(color="black", width=2),
            layer="above"
        )

    st.plotly_chart(fig, use_container_width=False)
    st.caption(legend_caption)

   
    # === 6) Vergelijking met een andere configuratie (zoals TopK) ===
    st.markdown("### 🔍 Vergelijk met een andere configuratie")
    compare_con = st.selectbox(
        "Kies een configuratie om te vergelijken:",
        [c for c in sorted(df_data["con"].unique()) if c != selected_con],
    )
    cols = st.columns(2)

    def plot_con_animation(df_con, title_suffix=""):
        df_con_sorted = df_con.sort_values("tst").copy()
        df_con_sorted["poi"] = df_con_sorted["poi"].astype(str)
        fig_anim = px.scatter(
            df_con_sorted,
            x="poi_x", y="poi_y",
            animation_frame="tst", animation_group="poi",
            color="poi", color_discrete_map={"0": "blue", "1": "red"},
            title=title_suffix,
            range_x=[df_con_sorted["poi_x"].min() - 1, df_con_sorted["poi_x"].max() + 1],
            range_y=[df_con_sorted["poi_y"].min() - 1, df_con_sorted["poi_y"].max() + 1],
        )
        fig_anim.update_layout(width=500, height=450)
        return fig_anim

    with cols[0]:
        st.write(f"**Referentie: con {selected_con}**")
        df_con_ref = df_data[df_data["con"] == int(selected_con)]
        st.plotly_chart(plot_con_animation(df_con_ref, f"con {selected_con}"), use_container_width=True)

    with cols[1]:
        st.write(f"**Vergelijk: con {compare_con}**")
        df_con_cmp = df_data[df_data["con"] == int(compare_con)]
        st.plotly_chart(plot_con_animation(df_con_cmp, f"con {compare_con}"), use_container_width=True)



# --- View 5: TopK Results (INTERACTIEF) ---
elif view == "TopK Results":
    st.header("🏆 TopK Results — interactief")
    st.write(f"**Selected distance matrix:** `{matrix_option}`")

    # 1) Kies referentie-configuratie
    all_cons = sorted(df_data["con"].unique().tolist())
    selected_con = st.selectbox("Referentie-configuratie (con):", all_cons, index=0)

    # 2) Haal de juiste rij uit de afstandsmatrix
    # Opgelet: df_dist heeft string-indexen/kolommen ("0","1",...)
    row = df_dist.loc[str(selected_con)].astype(float)

    # 3) Instellingen
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        exclude_self = st.checkbox("Exclude self-distance (con → con)", value=True)
    with col_b:
        sort_order = st.radio("Sorteer", ["Kleinste eerst", "Grootste eerst"], index=0, horizontal=True)
    with col_c:
        max_k = len(row) - (1 if exclude_self else 0)
        K = st.slider("K", min_value=1, max_value=max_k, value=min(max_k, max_k), step=1)

    # 4) Filteren/sorteren
    series = row.copy()
    if exclude_self and str(selected_con) in series.index:
        series = series.drop(labels=str(selected_con))

    ascending = (sort_order == "Kleinste eerst")
    series_sorted = series.sort_values(ascending=ascending)
    topk = series_sorted.head(K).reset_index()
    topk.columns = ["config", "distance"]
    topk["config"] = topk["config"].astype(int)

    # 5) Plotly-bar + tabel
    st.subheader(f"Top {K} voor con {selected_con}")
    
    # Sorteer op afstand (distance)
    topk = topk.sort_values("distance", ascending=ascending)
    
    import plotly.graph_objects as go
    
    # Maak arrays voor x en y, waarbij we de volgorde behouden
    x_data = topk["config"].astype(str).tolist()
    y_data = topk["distance"].tolist()
    
    fig_topk = go.Figure(data=[
        go.Bar(
            x=x_data,
            y=y_data
        )
    ])
    
    # Update de layout met specifieke x-as configuratie
    fig_topk.update_layout(
        title=f"Top {K} {'dichtste' if ascending else 'verst'} configuraties t.o.v. {selected_con}",
        xaxis=dict(
            title="config",
            type='category',  # Forceer categorische as
            categoryorder='array',  # Gebruik een specifieke volgorde
            categoryarray=x_data  # Gebruik dezelfde volgorde als de data
        ),
        yaxis_title="distance"
    )
    st.plotly_chart(fig_topk, use_container_width=True)
    st.dataframe(topk, use_container_width=True)

    st.caption(
        "Tip: De afstanden komen rechtstreeks uit de geselecteerde PDP-matrix. "
        "Kies hierboven een andere matrix om het effect te zien."
    )

    # 6) Snel vergelijken van trajecten (optioneel)
    st.markdown("### 🔍 Vergelijk traject met één van de TopK-configuraties")
    compare_con = st.selectbox(
        "Kies een configuratie uit de TopK:",
        options=topk["config"].tolist(),
        index=0,
    )
    cols = st.columns(2)

    # Helper om één con als animatie te tonen
    def plot_con_animation(df_con, title_suffix=""):
        df_con_sorted = df_con.sort_values("tst").copy()
        df_con_sorted["poi"] = df_con_sorted["poi"].astype(str)
        x_min, x_max = df_con_sorted["poi_x"].min(), df_con_sorted["poi_x"].max()
        y_min, y_max = df_con_sorted["poi_y"].min(), df_con_sorted["poi_y"].max()

        fig_anim = px.scatter(
            df_con_sorted,
            x="poi_x",
            y="poi_y",
            animation_frame="tst",
            animation_group="poi",
            color="poi",
            color_discrete_map={"0": "blue", "1": "red"},
            title=title_suffix,
            range_x=[x_min - 1, x_max + 1],
            range_y=[y_min - 1, y_max + 1],
        )
        return style_highway(fig_anim)

    with cols[0]:
        st.write(f"**Referentie: con {selected_con}**")
        df_con_ref = df_data[df_data["con"] == int(selected_con)]
        fig_ref = plot_con_animation(df_con_ref, f"con {selected_con}")
        st.plotly_chart(fig_ref, use_container_width=True)

    with cols[1]:
        st.write(f"**Vergelijk: con {compare_con}**")
        df_con_cmp = df_data[df_data["con"] == int(compare_con)]
        fig_cmp = plot_con_animation(df_con_cmp, f"con {compare_con}")
        st.plotly_chart(fig_cmp, use_container_width=True)
