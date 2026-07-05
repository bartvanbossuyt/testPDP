"""
Visualize IMEC Ground Truth with real map background.
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
    print("contextily not installed")

# Sensor GPS position
ORIGIN_LAT = 51.045101
ORIGIN_LON = 3.713908
HEADING_DEG = 19.4  # 4.4° base + 15° adjustment

# Conversion factors
METERS_PER_DEG_LAT = 111320
METERS_PER_DEG_LON = 111320 * np.cos(np.radians(ORIGIN_LAT))

# Class styling
CLASS_COLORS = {
    'Pedestrian': '#FF4444',
    'Cyclist': '#00CC88', 
    'Vehicle': '#4488FF',
}
CLASS_MARKERS = {
    'Pedestrian': 'o',
    'Cyclist': '^',
    'Vehicle': 's',
}


def local_to_gps(x, y, heading_deg=HEADING_DEG):
    """Convert local LiDAR coords to GPS."""
    heading_rad = np.radians(heading_deg)
    north_offset = x * np.cos(heading_rad) - y * np.sin(heading_rad)
    east_offset = x * np.sin(heading_rad) + y * np.cos(heading_rad)
    lat = ORIGIN_LAT + (north_offset / METERS_PER_DEG_LAT)
    lon = ORIGIN_LON + (east_offset / METERS_PER_DEG_LON)
    return lat, lon


def plot_on_map(df, output_path, map_style='osm'):
    """
    Plot ground truth trajectories on a real map.
    
    map_style options:
    - 'osm': OpenStreetMap
    - 'satellite': Esri satellite imagery
    - 'topo': OpenTopoMap
    """
    # Convert all points to GPS
    lats, lons = [], []
    for _, row in df.iterrows():
        lat, lon = local_to_gps(row['x'], row['y'])
        lats.append(lat)
        lons.append(lon)
    
    df = df.copy()
    df['lat'] = lats
    df['lon'] = lons
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Plot trajectories
    for track_id in sorted(df['track_id'].unique()):
        track = df[df['track_id'] == track_id].sort_values('timestamp')
        obj_class = track['class'].iloc[0]
        color = CLASS_COLORS.get(obj_class, 'gray')
        marker = CLASS_MARKERS.get(obj_class, 'o')
        
        # Trajectory line
        ax.plot(track['lon'], track['lat'], '-', color=color, 
                linewidth=3, alpha=0.9, zorder=3)
        
        # Start marker (filled)
        ax.scatter(track['lon'].iloc[0], track['lat'].iloc[0],
                   color=color, marker=marker, s=180, 
                   edgecolors='black', linewidth=2, zorder=5)
        
        # End marker (white fill)
        ax.scatter(track['lon'].iloc[-1], track['lat'].iloc[-1],
                   color='white', marker=marker, s=180,
                   edgecolors=color, linewidth=3, zorder=5)
        
        # Label at midpoint
        mid = len(track) // 2
        ax.annotate(f'{obj_class[:3]}:{track_id}',
                    (track['lon'].iloc[mid], track['lat'].iloc[mid]),
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9),
                    zorder=6)
    
    # Sensor position
    ax.scatter(ORIGIN_LON, ORIGIN_LAT, color='red', marker='*', s=500,
               edgecolors='white', linewidth=2, zorder=10)
    ax.annotate('SENSOR', (ORIGIN_LON, ORIGIN_LAT), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                zorder=10)
    
    # Add map background
    if HAS_CTX:
        try:
            if map_style == 'satellite':
                source = ctx.providers.Esri.WorldImagery
            elif map_style == 'topo':
                source = ctx.providers.OpenTopoMap
            else:
                source = ctx.providers.OpenStreetMap.Mapnik
            
            ctx.add_basemap(ax, crs='EPSG:4326', source=source, zoom=19)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS['Pedestrian'],
               markersize=14, markeredgecolor='black', label='Pedestrian'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CLASS_COLORS['Cyclist'],
               markersize=14, markeredgecolor='black', label='Cyclist'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLASS_COLORS['Vehicle'],
               markersize=14, markeredgecolor='black', label='Vehicle'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=18, markeredgecolor='white', label='LiDAR Sensor'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
              framealpha=0.95, edgecolor='black')
    
    # Labels
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_aspect(1 / np.cos(np.radians(ORIGIN_LAT)))
    ax.set_title(f'IMEC Ground Truth on Map\n'
                 f'Location: {ORIGIN_LAT:.6f}°N, {ORIGIN_LON:.6f}°E (Ghent, Belgium)\n'
                 f'10 Tracks: 6 Cyclists, 3 Pedestrians, 1 Vehicle',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison_local_vs_map(df, output_path):
    """Side-by-side: local coordinates vs GPS on map."""
    
    # Convert to GPS
    df = df.copy()
    lats, lons = [], []
    for _, row in df.iterrows():
        lat, lon = local_to_gps(row['x'], row['y'])
        lats.append(lat)
        lons.append(lon)
    df['lat'] = lats
    df['lon'] = lons
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # LEFT: Local coordinates
    ax = axes[0]
    for track_id in sorted(df['track_id'].unique()):
        track = df[df['track_id'] == track_id].sort_values('timestamp')
        obj_class = track['class'].iloc[0]
        color = CLASS_COLORS.get(obj_class, 'gray')
        marker = CLASS_MARKERS.get(obj_class, 'o')
        
        ax.plot(track['x'], track['y'], '-', color=color, linewidth=2.5, alpha=0.8)
        ax.scatter(track['x'].iloc[0], track['y'].iloc[0], color=color, 
                   marker=marker, s=120, edgecolors='black', zorder=5)
    
    ax.scatter(0, 0, color='red', marker='*', s=300, zorder=10)
    ax.set_xlabel('X (meters) - Forward', fontsize=11)
    ax.set_ylabel('Y (meters) - Left', fontsize=11)
    ax.set_title('Local LiDAR Coordinates', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # RIGHT: GPS on map
    ax = axes[1]
    for track_id in sorted(df['track_id'].unique()):
        track = df[df['track_id'] == track_id].sort_values('timestamp')
        obj_class = track['class'].iloc[0]
        color = CLASS_COLORS.get(obj_class, 'gray')
        marker = CLASS_MARKERS.get(obj_class, 'o')
        
        ax.plot(track['lon'], track['lat'], '-', color=color, linewidth=2.5, alpha=0.8)
        ax.scatter(track['lon'].iloc[0], track['lat'].iloc[0], color=color,
                   marker=marker, s=120, edgecolors='black', zorder=5)
    
    ax.scatter(ORIGIN_LON, ORIGIN_LAT, color='red', marker='*', s=300, zorder=10)
    
    if HAS_CTX:
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=19)
        except Exception as e:
            print(f"Map error: {e}")
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_aspect(1 / np.cos(np.radians(ORIGIN_LAT)))
    ax.set_title(f'GPS Coordinates on Map\n({ORIGIN_LAT:.4f}°N, {ORIGIN_LON:.4f}°E)', 
                 fontsize=13, fontweight='bold')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS['Pedestrian'],
               markersize=10, label='Pedestrian'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CLASS_COLORS['Cyclist'],
               markersize=10, label='Cyclist'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLASS_COLORS['Vehicle'],
               markersize=10, label='Vehicle'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.suptitle('IMEC Ground Truth: Local vs GPS', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "IMEC_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth
    gt_file = os.path.join(script_dir, "IMEC_GroundTruth_PDP_full.csv")
    df = pd.read_csv(gt_file)
    
    print("=" * 60)
    print("IMEC GROUND TRUTH - MAP VISUALIZATION")
    print(f"Sensor: {ORIGIN_LAT}°N, {ORIGIN_LON}°E")
    print(f"Heading: {HEADING_DEG}° (adjust if needed)")
    print("=" * 60)
    print(f"Tracks: {df['track_id'].nunique()}")
    print(f"Classes: {df['class'].value_counts().to_dict()}")
    
    # 1. OpenStreetMap background
    print("\n1. Creating OSM map visualization...")
    plot_on_map(df, os.path.join(output_dir, "gt_on_map_osm.png"), map_style='osm')
    
    # 2. Satellite background
    print("\n2. Creating satellite map visualization...")
    plot_on_map(df, os.path.join(output_dir, "gt_on_map_satellite.png"), map_style='satellite')
    
    # 3. Comparison view
    print("\n3. Creating comparison view...")
    plot_comparison_local_vs_map(df, os.path.join(output_dir, "gt_local_vs_map.png"))
    
    print("\n" + "=" * 60)
    print(f"Maps saved to: {output_dir}")
    print("\nIf trajectories don't align with roads, adjust HEADING_DEG")
    print("Try: 0, 90, 180, 270, or intermediate values")
    print("=" * 60)
