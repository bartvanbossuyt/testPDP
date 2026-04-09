# -*- coding: utf-8 -*-
"""
Visualize IMEC Ground Truth on a map background in LOCAL coordinates.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("Contextily not installed, map background will not be available.")

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
        print("Contextily is required for map background. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 14))

    # --- Plot Trajectories in Local Coordinates (LiDAR frame: x-forward, y-left) ---
    for track_id in sorted(df["track_id"].unique()):
        track = df[df["track_id"] == track_id].sort_values("timestamp")
        obj_class = track["class"].iloc[0]
        color = CLASS_COLORS.get(obj_class, "gray")
        marker = CLASS_MARKERS.get(obj_class, "o")

        # Plot y vs x to align "forward" with the vertical axis
        ax.plot(track["y"], track["x"], "-", color=color, linewidth=3, alpha=0.8, zorder=5)
        ax.scatter(track["y"].iloc[0], track["x"].iloc[0], color=color, marker=marker,
                   s=180, edgecolors="black", linewidth=1.5, zorder=6)

    # Mark the sensor at (0,0) in the local y,x plot
    ax.scatter(0, 0, color="red", marker="*", s=500, edgecolors="white", zorder=10)
    ax.annotate('SENSOR', (0, 0), xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                zorder=10)

    # --- Prepare and Add Rotated Map Background ---
    # Set plot limits based on data
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Define corners in local data coordinates (y,x)
    corners_local = np.array([
        [xmin, ymin], [xmax, ymin],
        [xmax, ymax], [xmin, ymax]
    ])

    # Transform corners from local (y,x) to GPS (lon,lat)
    # Local y is 'left', local x is 'forward'
    x_coords = corners_local[:, 1]
    y_coords = corners_local[:, 0]
    
    heading_rad = np.radians(HEADING_DEG)
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    
    # Transformation from local (x,y) to (North, East) offsets
    north_offsets = x_coords * cos_h - y_coords * sin_h
    east_offsets = x_coords * sin_h + y_coords * cos_h

    corner_lats = ORIGIN_LAT + (north_offsets / METERS_PER_DEG_LAT)
    corner_lons = ORIGIN_LON + (east_offsets / METERS_PER_DEG_LON)
    
    # Fetch map tile using the GPS bounding box
    try:
        image, extent = ctx.bounds2img(
            min(corner_lons), min(corner_lats),
            max(corner_lons), max(corner_lats),
            ll=True,
            source=ctx.providers.OpenStreetMap.Mapnik
        )
    except Exception as e:
        print(f"Could not fetch map tile: {e}")
        # Plot without map if download fails
        image, extent = None, None

    # Add the map as the background image
    if image is not None:
        # We need to rotate the image itself, not the plot
        from scipy.ndimage import rotate
        rotated_image = rotate(image, HEADING_DEG, reshape=False)
        ax.imshow(rotated_image, extent=(xmin, xmax, ymin, ymax), zorder=0, alpha=0.8)

    # --- Final Touches ---
    ax.set_xlabel("Y (meters) - Left of Sensor", fontsize=12)
    ax.set_ylabel("X (meters) - Forward from Sensor", fontsize=12)
    ax.set_title(f"IMEC Ground Truth - Local Coordinates on Map\n"
                 f"Heading: {HEADING_DEG}° | Location: Ghent, Belgium",
                 fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", "box")

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, label=l)
        for l, c in CLASS_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved local map visualization to: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "IMEC_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    gt_file = os.path.join(script_dir, "IMEC_GroundTruth_PDP_full.csv")
    
    try:
        df_gt = pd.read_csv(gt_file)
        plot_gt_on_rotated_map(df_gt, os.path.join(output_dir, "local_coords_on_map.png"))
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_file}")
        print("Please ensure the file exists and the path is correct.")

