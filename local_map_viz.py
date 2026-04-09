# -*- coding: utf-8 -*-
"""
Visualize IMEC Ground Truth on a map background in LOCAL coordinates.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import os

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False

# Sensor GPS position and initial estimated heading
ORIGIN_LAT = 51.045101
ORIGIN_LON = 3.713908
HEADING_DEG = 49.4 

# Conversion factors
METERS_PER_DEG_LAT = 111320
METERS_PER_DEG_LON = 111320 * np.cos(np.radians(ORIGIN_LAT))

# Class styling
CLASS_COLORS = {
    "Pedestrian": "#FF4444", "Cyclist": "#00CC88", "Vehicle": "#4488FF"
}
CLASS_MARKERS = {
    "Pedestrian": "o", "Cyclist": "^", "Vehicle": "s"
}

def gps_to_web_mercator(lon, lat):
    """Converts GPS lon/lat to Web Mercator x/y."""
    r_major = 6378137.0
    x = r_major * np.radians(lon)
    y = r_major * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y

def plot_gt_on_rotated_map(df, output_path):
    """
    Plots trajectories in local LiDAR coordinates (x, y) with a rotated
    map background underneath.
    """
    if not HAS_CTX:
        print("Contextily is required for map background.")
        return

    fig, ax = plt.subplots(figsize=(16, 14))

    # --- Plot Trajectories in Local Coordinates (LiDAR frame) ---
    for track_id in sorted(df["track_id"].unique()):
        track = df[df["track_id"] == track_id].sort_values("timestamp")
        obj_class = track["class"].iloc[0]
        color = CLASS_COLORS.get(obj_class, "gray")
        marker = CLASS_MARKERS.get(obj_class, "o")

        ax.plot(track["x"], track["y"], "-", color=color, linewidth=3, alpha=0.8)
        ax.scatter(track["x"].iloc[0], track["y"].iloc[0], color=color, marker=marker,
                   s=180, edgecolors="black", linewidth=1.5, zorder=5)

    ax.scatter(0, 0, color="red", marker="*", s=500, edgecolors="white", zorder=10)

    # --- Prepare Map Transformation ---
    # 1. Determine bounding box in local coordinates
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # 2. Convert local coordinate corners to GPS to fetch map tile
    corners_local = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    
    heading_rad = np.radians(HEADING_DEG)
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    
    # Rotation matrix to convert from (North, East) to (x, y)
    # x = North*cos + East*sin
    # y = -North*sin + East*cos
    # So, North = x*cos - y*sin
    #     East = x*sin + y*cos
    north_offsets = corners_local[:, 0] * cos_h - corners_local[:, 1] * sin_h
    east_offsets = corners_local[:, 0] * sin_h + corners_local[:, 1] * cos_h
    
    corner_lats = ORIGIN_LAT + (north_offsets / METERS_PER_DEG_LAT)
    corner_lons = ORIGIN_LON + (east_offsets / METERS_PER_DEG_LON)
    
    # 3. Fetch map tile in Web Mercator projection
    try:
        img, ext = ctx.bounds2img(
            min(corner_lons), min(corner_lats),
            max(corner_lons), max(corner_lats),
            ll=True,  # Input is lon/lat
            source=ctx.providers.OpenStreetMap.Mapnik
        )
    except Exception as e:
        print(f"Could not download map tile: {e}")
        return

    # 4. Define the transformation from Web Mercator to local coordinates
    # Origin in Web Mercator
    origin_mx, origin_my = gps_to_web_mercator(ORIGIN_LON, ORIGIN_LAT)
    
    # Rotation matrix from (x, y) local to (East, North) meter offsets
    # East = x*sin + y*cos
    # North = x*cos - y*sin
    rot_matrix_inv = np.array([[sin_h, cos_h], [cos_h, -sin_h]])

    # Define the transform pipeline
    transform = (
        mtransforms.Affine2D().translate(-origin_mx, -origin_my)
        .rotate_deg(-HEADING_DEG)
        .scale(METERS_PER_DEG_LON / (2 * np.pi * 6378137.0) * (2 * np.pi), 1)  # Mercator scaling fix
        .inverted()
        .rotate_deg(HEADING_DEG)
        + ax.transData
    )
    
    ax.imshow(img, extent=ext, transform=transform, zorder=0, alpha=0.9)
    
    # --- Final Touches ---
    ax.set_xlabel("X (meters) - Forward", fontsize=12)
    ax.set_ylabel("Y (meters) - Left", fontsize=12)
    ax.set_title(f"IMEC Ground Truth - Local Coordinates on Rotated Map\n"
                 f"Heading: {HEADING_DEG}° | Location: Ghent, Belgium",
                 fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved local map to: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "IMEC_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    gt_file = os.path.join(script_dir, "IMEC_GroundTruth_PDP_full.csv")
    df_gt = pd.read_csv(gt_file)
    plot_gt_on_rotated_map(df_gt, os.path.join(output_dir, "local_coords_on_map.png"))

