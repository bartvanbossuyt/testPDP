"""
Full analysis, visualization, and PDP conversion for Data_IMEC_04.

Produces:
  1. Console summary of both scenes (10_10, 10_55) across all snapshot files
  2. Static visualizations (PNG) in IMEC_visualizations/
  3. Interactive Leaflet map (HTML) with ground truth overlay
  4. PDP-format CSVs (sparse + interpolated) per scene, ready for N_PDP.py
"""

import os
import json
import math
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR / "Data_IMEC_04"
VIZ_DIR = SCRIPT_DIR / "IMEC_visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

ORIGIN_LAT = 51.045101
ORIGIN_LON = 3.713908
HEADING_DEG = 49.4
METERS_PER_DEG_LAT = 111320
METERS_PER_DEG_LON = 111320 * np.cos(np.radians(ORIGIN_LAT))

SCENES = ["09_56", "10_10_new", "10_27", "10_43", "10_55_new"]

CLASS_COLORS_GT = {
    "Pedestrian": "#FF3B30",
    "Cyclist": "#10B981",
    "Vehicle": "#2563EB",
    "Car": "#2563EB",
    "Van": "#7C3AED",
}


def local_to_gps(x, y, heading_deg=HEADING_DEG):
    h = np.radians(heading_deg)
    north = x * np.cos(h) - y * np.sin(h)
    east = x * np.sin(h) + y * np.cos(h)
    lat = ORIGIN_LAT + north / METERS_PER_DEG_LAT
    lon = ORIGIN_LON + east / METERS_PER_DEG_LON
    return float(lat), float(lon)


# ===================================================================
# 1. Load helpers
# ===================================================================
def load_scene_files(scene: str) -> dict[int, pd.DataFrame]:
    """Return {snapshot_ts: DataFrame} for a scene."""
    # Try multiple possible folder structures
    candidates = [
        DATA_DIR / scene / "track_dataframes",
        DATA_DIR / scene / scene / "track_dataframes",
        DATA_DIR / scene / scene / "dataframes",
    ]
    folder = None
    for c in candidates:
        if c.exists():
            folder = c
            break
    if folder is None:
        return {}
    out = {}
    for fp in sorted(folder.glob("tracks_df_*.csv")):
        ts = int(fp.stem.split("_")[-1])
        df = pd.read_csv(fp)
        # Drop the unnamed index column if present
        if "" in df.columns:
            df = df.drop(columns=[""])
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        out[ts] = df
    return out


def load_ground_truth() -> pd.DataFrame | None:
    gt_path = SCRIPT_DIR / "IMEC_GroundTruth_PDP_full.csv"
    if gt_path.exists():
        return pd.read_csv(gt_path)
    return None


# ===================================================================
# 2. Console analysis
# ===================================================================
def print_scene_summary(scene: str, snapshots: dict[int, pd.DataFrame]):
    print(f"\n{'='*70}")
    print(f"SCENE: {scene}")
    print(f"{'='*70}")

    snap_keys = sorted(snapshots.keys())
    total_rows = sum(len(df) for df in snapshots.values())
    all_ids = set()
    for df in snapshots.values():
        all_ids.update(df["ID"].unique())

    print(f"  Snapshot files : {len(snap_keys)}")
    print(f"  Snapshot range : {snap_keys[0]} → {snap_keys[-1]}  (step {snap_keys[1]-snap_keys[0] if len(snap_keys)>1 else 'N/A'})")
    print(f"  Total rows     : {total_rows:,}")
    print(f"  Unique IDs     : {len(all_ids)}")

    # Use the largest snapshot as representative
    biggest_key = max(snap_keys, key=lambda k: len(snapshots[k]))
    df = snapshots[biggest_key]
    print(f"\n  Representative snapshot: tracks_df_{biggest_key}.csv ({len(df):,} rows)")
    print(f"    Tracks       : {df['ID'].nunique()}")
    ts_range = df["timestamp"]
    print(f"    Timestamp    : {ts_range.min():.0f} → {ts_range.max():.0f}")

    # x/y ranges
    print(f"    x range      : [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"    y range      : [{df['y'].min():.2f}, {df['y'].max():.2f}]")

    # Velocity stats
    speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    print(f"    Speed (m/ts) : median={speed.median():.3f}, p95={speed.quantile(0.95):.3f}, max={speed.max():.3f}")

    # LiDAR coverage
    lidar_ok = df["X_LiDAR"].notna().sum()
    print(f"    LiDAR coords : {lidar_ok}/{len(df)} rows ({100*lidar_ok/len(df):.1f}%)")

    # BBox coverage
    bbox_ok = df["L"].notna().sum()
    print(f"    BBox coords  : {bbox_ok}/{len(df)} rows ({100*bbox_ok/len(df):.1f}%)")

    # Track length distribution
    track_lens = df.groupby("ID").size()
    print(f"    Track lengths: min={track_lens.min()}, median={track_lens.median():.0f}, max={track_lens.max()}")

    # Missing values overview
    missing = df.isna().sum()
    cols_with_na = missing[missing > 0]
    if len(cols_with_na):
        print(f"    Columns with NaN:")
        for c, cnt in cols_with_na.items():
            print(f"      {c:15s}: {cnt:>6d}  ({100*cnt/len(df):5.1f}%)")


# ===================================================================
# 3. Static visualizations
# ===================================================================
def plot_birdseye(scene: str, df: pd.DataFrame, output: Path):
    """Bird's-eye trajectory plot coloured by track ID."""
    fig, ax = plt.subplots(figsize=(13, 10))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(20, df["ID"].nunique())))

    for idx, (tid, grp) in enumerate(df.groupby("ID")):
        grp = grp.sort_values("timestamp")
        c = cmap[idx % len(cmap)]
        ax.plot(grp["x"], grp["y"], "-", color=c, linewidth=1.5, alpha=0.7)
        ax.scatter(grp["x"].iloc[0], grp["y"].iloc[0], color=c, marker="o",
                   s=60, edgecolors="black", linewidth=0.7, zorder=5)

    ax.scatter(0, 0, color="red", marker="^", s=200, zorder=10, label="Sensor")
    ax.set_xlabel("X (m) — forward")
    ax.set_ylabel("Y (m) — left")
    ax.set_title(f"Data_IMEC_04 / {scene} — bird's-eye trajectories\n"
                 f"{df['ID'].nunique()} tracks, {len(df):,} detections")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output.name}")


def plot_time_coloured(scene: str, df: pd.DataFrame, output: Path):
    """Trajectories with colour = timestamp progression."""
    fig, ax = plt.subplots(figsize=(13, 10))
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()

    for tid, grp in df.groupby("ID"):
        grp = grp.sort_values("timestamp")
        pts = np.column_stack([grp["x"], grp["y"]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        times = (grp["timestamp"].values[:-1] - t_min) / max(t_max - t_min, 1)
        lc = LineCollection(segs, cmap="viridis", linewidth=2, alpha=0.8)
        lc.set_array(times)
        ax.add_collection(lc)

    ax.scatter(0, 0, color="red", marker="^", s=200, zorder=10)
    ax.autoscale()
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(t_min, t_max))
    plt.colorbar(sm, ax=ax, label="Timestamp")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Data_IMEC_04 / {scene} — colour = time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output.name}")


def plot_track_stats(scene: str, df: pd.DataFrame, output: Path):
    """4-panel stats dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1 — track length histogram
    ax = axes[0, 0]
    tl = df.groupby("ID").size()
    ax.hist(tl, bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel("Detections per track")
    ax.set_ylabel("# tracks")
    ax.set_title("Track length distribution")

    # 2 — active objects per timestamp
    ax = axes[0, 1]
    active = df.groupby("timestamp")["ID"].nunique()
    ax.plot(active.index, active.values, "b-", linewidth=1.2)
    ax.fill_between(active.index, active.values, alpha=0.25)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Active objects")
    ax.set_title("Concurrent objects over time")

    # 3 — speed histogram
    ax = axes[1, 0]
    speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    ax.hist(speed, bins=50, color="coral", edgecolor="black")
    ax.set_xlabel("Speed (m/ts)")
    ax.set_ylabel("Count")
    ax.set_title("Speed distribution")

    # 4 — distance from sensor
    ax = axes[1, 1]
    dist = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    ax.hist(dist, bins=50, color="mediumseagreen", edgecolor="black")
    ax.set_xlabel("Distance from sensor (m)")
    ax.set_ylabel("Count")
    ax.set_title("Distance from sensor")

    fig.suptitle(f"Data_IMEC_04 / {scene} — statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output.name}")


def plot_snapshot_growth(scene: str, snapshots: dict[int, pd.DataFrame], output: Path):
    """Bar chart: rows and unique IDs per snapshot."""
    keys = sorted(snapshots.keys())
    rows_counts = [len(snapshots[k]) for k in keys]
    id_counts = [snapshots[k]["ID"].nunique() for k in keys]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(keys))
    w = 0.35
    ax1.bar(x - w / 2, rows_counts, w, label="Rows", color="steelblue")
    ax1.set_ylabel("Rows")
    ax1.set_xlabel("Snapshot file (tracks_df_XXX)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in keys], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, id_counts, w, label="Unique IDs", color="tomato", alpha=0.8)
    ax2.set_ylabel("Unique IDs")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"Data_IMEC_04 / {scene} — snapshot growth")
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output.name}")


# ===================================================================
# 4. Interactive HTML map
# ===================================================================
def create_interactive_map(
    scene_data: dict[str, pd.DataFrame],
    gt_df: pd.DataFrame | None,
    output: Path,
):
    """Leaflet map with toggleable scene layers and ground truth."""
    import folium

    m = folium.Map(location=[ORIGIN_LAT, ORIGIN_LON], zoom_start=18,
                   tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)

    # Sensor marker
    folium.Marker(
        [ORIGIN_LAT, ORIGIN_LON], popup="LiDAR Sensor", tooltip="Sensor",
        icon=folium.Icon(color="red", icon="record"),
    ).add_to(m)

    # Ground truth layer
    if gt_df is not None:
        gt_layer = folium.FeatureGroup(name="Ground truth", show=True)
        for tid, grp in gt_df.groupby("track_id"):
            grp = grp.sort_values("timestamp")
            cls = str(grp["class"].iloc[0])
            color = CLASS_COLORS_GT.get(cls, "gray")
            coords = [local_to_gps(r.x, r.y) for r in grp.itertuples()]
            if len(coords) < 2:
                continue
            folium.PolyLine(
                coords, color=color, weight=5, opacity=0.9,
                tooltip=f"GT {cls} #{tid}",
            ).add_to(gt_layer)
            folium.CircleMarker(coords[0], radius=6, color="black",
                                fill=True, fill_color=color, fill_opacity=1,
                                tooltip=f"GT start #{tid}").add_to(gt_layer)
            folium.CircleMarker(coords[-1], radius=6, color=color,
                                fill=True, fill_color="white", fill_opacity=1,
                                tooltip=f"GT end #{tid}").add_to(gt_layer)
        gt_layer.add_to(m)

    # Scene layers
    palette = {
        "09_56": "#EF4444",
        "10_10_new": "#F59E0B",
        "10_27": "#10B981",
        "10_43": "#8B5CF6",
        "10_55_new": "#06B6D4",
    }
    for scene, df in scene_data.items():
        layer = folium.FeatureGroup(name=f"IMEC_04 {scene}", show=True)
        color = palette.get(scene, "gray")
        for tid, grp in df.groupby("ID"):
            grp = grp.sort_values("timestamp")
            coords = [local_to_gps(r.x, r.y) for r in grp.itertuples()]
            if len(coords) < 2:
                continue
            folium.PolyLine(
                coords, color=color, weight=2, opacity=0.6,
                tooltip=f"{scene} ID {int(tid)}",
            ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(output))
    print(f"  Saved interactive map: {output.name}")


# ===================================================================
# 5. PDP conversion
# ===================================================================
def convert_to_pdp_sparse(df: pd.DataFrame, con_id: int = 0) -> pd.DataFrame:
    """Sparse PDP: only rows where data exists."""
    ts_map = {ts: i for i, ts in enumerate(sorted(df["timestamp"].unique()))}
    id_map = {tid: i for i, tid in enumerate(sorted(df["ID"].unique()))}
    pdp = pd.DataFrame({
        "conID": con_id,
        "tstID": df["timestamp"].map(ts_map),
        "poiID": df["ID"].map(id_map),
        "x": df["x"].values,
        "y": df["y"].values,
    })
    return pdp.sort_values(["conID", "tstID", "poiID"]).reset_index(drop=True)


def convert_to_pdp_interpolated(df: pd.DataFrame, con_id: int = 0) -> pd.DataFrame:
    """Dense PDP: interpolate gaps so every object appears at every timestep."""
    unique_ts = sorted(df["timestamp"].unique())
    unique_ids = sorted(df["ID"].unique())
    ts_map = {ts: i for i, ts in enumerate(unique_ts)}
    id_map = {tid: i for i, tid in enumerate(unique_ids)}

    rows = []
    for tid in unique_ids:
        track = df[df["ID"] == tid].set_index("timestamp")[["x", "y"]]
        track = track.reindex(unique_ts).interpolate("linear").ffill().bfill()
        for ts in unique_ts:
            rows.append((con_id, ts_map[ts], id_map[tid],
                         float(track.loc[ts, "x"]), float(track.loc[ts, "y"])))
    pdp = pd.DataFrame(rows, columns=["conID", "tstID", "poiID", "x", "y"])
    return pdp.sort_values(["conID", "tstID", "poiID"]).reset_index(drop=True)


def save_pdp(pdp: pd.DataFrame, path: Path, label: str):
    pdp.to_csv(path, index=False, header=False)
    print(f"    {label}: {path.name}  "
          f"({pdp['tstID'].nunique()} timesteps × {pdp['poiID'].nunique()} objects "
          f"= {len(pdp):,} rows)")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("  DATA_IMEC_04 — FULL ANALYSIS, VISUALISATION & PDP CONVERSION")
    print("=" * 70)

    gt_df = load_ground_truth()
    if gt_df is not None:
        print(f"\nGround truth loaded: {gt_df['track_id'].nunique()} tracks, "
              f"{len(gt_df):,} rows")

    # Per-scene work
    map_data: dict[str, pd.DataFrame] = {}

    for scene in SCENES:
        snapshots = load_scene_files(scene)
        if not snapshots:
            print(f"\n⚠  No data found for scene {scene}, skipping.")
            continue

        # --- Console summary ---
        print_scene_summary(scene, snapshots)

        # Use the largest snapshot for visualizations and PDP
        biggest_key = max(snapshots, key=lambda k: len(snapshots[k]))
        df_rep = snapshots[biggest_key]
        map_data[scene] = df_rep

        # --- Visualizations ---
        print(f"\n  Creating visualizations for {scene} (snapshot {biggest_key})...")
        plot_birdseye(scene, df_rep,
                      VIZ_DIR / f"imec04_{scene}_birdseye.png")
        plot_time_coloured(scene, df_rep,
                           VIZ_DIR / f"imec04_{scene}_time_colour.png")
        plot_track_stats(scene, df_rep,
                         VIZ_DIR / f"imec04_{scene}_stats.png")
        plot_snapshot_growth(scene, snapshots,
                            VIZ_DIR / f"imec04_{scene}_snapshot_growth.png")

        # --- PDP conversion ---
        print(f"\n  Converting {scene} (snapshot {biggest_key}) to PDP format...")
        pdp_sparse = convert_to_pdp_sparse(df_rep)
        save_pdp(pdp_sparse,
                 SCRIPT_DIR / f"IMEC_04_{scene}_PDP_sparse.csv",
                 "sparse")

        pdp_interp = convert_to_pdp_interpolated(df_rep)
        save_pdp(pdp_interp,
                 SCRIPT_DIR / f"IMEC_04_{scene}_PDP_interpolated.csv",
                 "interpolated")

    # --- Interactive map ---
    if map_data:
        print(f"\n  Building interactive map...")
        create_interactive_map(
            map_data, gt_df,
            VIZ_DIR / "imec04_full_map.html",
        )

    # --- Combined PDP (all scenes as separate configs) ---
    if len(map_data) >= 2:
        scene_labels = ", ".join(f"{s} = conID {i}" for i, s in enumerate(s for s in SCENES if s in map_data))
        print(f"\n  Creating combined PDP ({scene_labels})...")
        parts = []
        for i, scene in enumerate(s for s in SCENES if s in map_data):
            parts.append(convert_to_pdp_sparse(map_data[scene], con_id=i))
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values(["conID", "tstID", "poiID"]).reset_index(drop=True)
        save_pdp(combined,
                 SCRIPT_DIR / "IMEC_04_combined_PDP.csv",
                 "combined-sparse")

        parts_i = []
        for i, scene in enumerate(s for s in SCENES if s in map_data):
            parts_i.append(convert_to_pdp_interpolated(map_data[scene], con_id=i))
        combined_i = pd.concat(parts_i, ignore_index=True)
        combined_i = combined_i.sort_values(["conID", "tstID", "poiID"]).reset_index(drop=True)
        save_pdp(combined_i,
                 SCRIPT_DIR / "IMEC_04_combined_PDP_interpolated.csv",
                 "combined-interpolated")

    print(f"\n{'='*70}")
    print("  DONE — all outputs in:")
    print(f"    Visualisations : {VIZ_DIR}")
    print(f"    PDP CSVs       : {SCRIPT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
