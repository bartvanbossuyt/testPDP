"""
Visualize IMEC tracking data with real-world map underlay.

Uses GPS coordinates as reference point and overlays trajectories on OpenStreetMap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Try to import mapping libraries
try:
    import folium
    from folium.plugins import AntPath
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("folium not installed. Run: pip install folium")

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("contextily not installed. Run: pip install contextily")

# Camera/LiDAR position (provided by user)
ORIGIN_LAT = 51.045101
ORIGIN_LON = 3.713908

# LiDAR orientation: X points forward, Y points left
# We need to determine the heading - assuming camera points roughly North for now
# You can adjust HEADING_DEG to rotate the trajectories
HEADING_DEG = 0  # 0 = North, 90 = East, 180 = South, 270 = West

# Conversion factors (approximate at this latitude)
# 1 degree latitude ≈ 111,320 meters
# 1 degree longitude ≈ 111,320 * cos(lat) meters
METERS_PER_DEG_LAT = 111320
METERS_PER_DEG_LON = 111320 * np.cos(np.radians(ORIGIN_LAT))


def local_to_gps(x, y, origin_lat, origin_lon, heading_deg=0):
    """
    Convert local LiDAR coordinates (x=forward, y=left) to GPS coordinates.
    
    Parameters:
    -----------
    x, y : float
        Local coordinates in meters
    origin_lat, origin_lon : float
        GPS coordinates of sensor origin
    heading_deg : float
        Heading of the sensor's X-axis in degrees (0=North, 90=East)
    """
    # Rotate coordinates based on heading
    heading_rad = np.radians(heading_deg)
    
    # In LiDAR frame: x=forward, y=left
    # Convert to North/East offsets
    # If heading=0 (pointing North): forward=North, left=West
    north_offset = x * np.cos(heading_rad) - y * np.sin(heading_rad)
    east_offset = x * np.sin(heading_rad) + y * np.cos(heading_rad)
    
    # Convert meters to degrees
    lat = origin_lat + (north_offset / METERS_PER_DEG_LAT)
    lon = origin_lon + (east_offset / METERS_PER_DEG_LON)
    
    return lat, lon


def create_folium_map(df, output_path, heading_deg=HEADING_DEG):
    """Create interactive Folium map with trajectories."""
    if not HAS_FOLIUM:
        print("Folium not available, skipping interactive map")
        return
    
    # Create base map centered on origin
    m = folium.Map(
        location=[ORIGIN_LAT, ORIGIN_LON],
        zoom_start=18,
        tiles='OpenStreetMap'
    )
    
    # Add satellite layer option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Mark sensor position
    folium.Marker(
        [ORIGIN_LAT, ORIGIN_LON],
        popup='LiDAR Sensor',
        icon=folium.Icon(color='red', icon='video-camera', prefix='fa'),
        tooltip='Sensor Position'
    ).add_to(m)
    
    # Color palette
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
              'darkpurple', 'pink', 'lightblue', 'lightgreen']
    
    # Plot each track
    for idx, track_id in enumerate(df['ID'].unique()):
        track = df[df['ID'] == track_id].sort_values('timestamp')
        color = colors[idx % len(colors)]
        
        # Convert to GPS coordinates
        coords = []
        for _, row in track.iterrows():
            lat, lon = local_to_gps(row['x'], row['y'], ORIGIN_LAT, ORIGIN_LON, heading_deg)
            coords.append([lat, lon])
        
        if len(coords) > 1:
            # Add trajectory line with animation
            AntPath(
                coords,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f'Track {track_id}'
            ).add_to(m)
            
            # Mark start point
            folium.CircleMarker(
                coords[0],
                radius=8,
                color=color,
                fill=True,
                popup=f'Track {track_id} - Start',
                tooltip=f'ID:{track_id} Start'
            ).add_to(m)
            
            # Mark end point
            folium.CircleMarker(
                coords[-1],
                radius=8,
                color=color,
                fill=True,
                fill_color='white',
                popup=f'Track {track_id} - End',
                tooltip=f'ID:{track_id} End'
            ).add_to(m)
    
    # Save map
    m.save(output_path)
    print(f"Saved interactive map to: {output_path}")


def create_static_map_contextily(df, output_path, heading_deg=HEADING_DEG):
    """Create static map with contextily background."""
    if not HAS_CONTEXTILY:
        print("Contextily not available, creating simple plot instead")
        create_simple_gps_plot(df, output_path, heading_deg)
        return
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Convert all points to GPS
    lats, lons, ids = [], [], []
    for _, row in df.iterrows():
        lat, lon = local_to_gps(row['x'], row['y'], ORIGIN_LAT, ORIGIN_LON, heading_deg)
        lats.append(lat)
        lons.append(lon)
        ids.append(row['ID'])
    
    df_gps = df.copy()
    df_gps['lat'] = lats
    df_gps['lon'] = lons
    
    # Plot trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for idx, track_id in enumerate(df_gps['ID'].unique()):
        track = df_gps[df_gps['ID'] == track_id].sort_values('timestamp')
        color = colors[idx % len(colors)]
        
        ax.plot(track['lon'], track['lat'], '-', color=color, linewidth=3, alpha=0.8)
        ax.scatter(track['lon'].iloc[0], track['lat'].iloc[0], 
                   color=color, s=100, marker='o', edgecolors='black', zorder=5)
        ax.scatter(track['lon'].iloc[-1], track['lat'].iloc[-1], 
                   color=color, s=100, marker='s', edgecolors='black', zorder=5)
    
    # Mark sensor
    ax.scatter(ORIGIN_LON, ORIGIN_LAT, color='red', s=300, marker='^', 
               edgecolors='black', linewidth=2, zorder=10)
    ax.annotate('SENSOR', (ORIGIN_LON, ORIGIN_LAT), xytext=(5, 5), 
               textcoords='offset points', fontweight='bold', color='red')
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Could not add basemap: {e}")
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect(1 / np.cos(np.radians(ORIGIN_LAT)))
    ax.set_title(f'IMEC Trajectories on Map\nSensor: ({ORIGIN_LAT:.6f}, {ORIGIN_LON:.6f})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static map to: {output_path}")


def create_simple_gps_plot(df, output_path, heading_deg=HEADING_DEG):
    """Create simple GPS plot without map background."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for idx, track_id in enumerate(df['ID'].unique()):
        track = df[df['ID'] == track_id].sort_values('timestamp')
        color = colors[idx % len(colors)]
        
        # Convert to GPS
        lats, lons = [], []
        for _, row in track.iterrows():
            lat, lon = local_to_gps(row['x'], row['y'], ORIGIN_LAT, ORIGIN_LON, heading_deg)
            lats.append(lat)
            lons.append(lon)
        
        ax.plot(lons, lats, '-', color=color, linewidth=2, alpha=0.8, label=f'ID:{track_id}')
        ax.scatter(lons[0], lats[0], color=color, s=100, marker='o', edgecolors='black', zorder=5)
        ax.scatter(lons[-1], lats[-1], color=color, s=100, marker='s', edgecolors='black', zorder=5)
    
    # Sensor position
    ax.scatter(ORIGIN_LON, ORIGIN_LAT, color='red', s=300, marker='^', 
               edgecolors='black', linewidth=2, zorder=10, label='Sensor')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'IMEC Trajectories (GPS)\nOrigin: {ORIGIN_LAT:.6f}°N, {ORIGIN_LON:.6f}°E', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Correct aspect ratio for GPS (instead of 'equal')
    ax.set_aspect(1 / np.cos(np.radians(ORIGIN_LAT)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved GPS plot to: {output_path}")


def create_dual_view(df, output_path, heading_deg=HEADING_DEG):
    """Create side-by-side view: local coords + GPS coords."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Left: Local coordinates
    ax = axes[0]
    for idx, track_id in enumerate(df['ID'].unique()):
        track = df[df['ID'] == track_id].sort_values('timestamp')
        color = colors[idx % len(colors)]
        ax.plot(track['x'], track['y'], '-', color=color, linewidth=2, alpha=0.8)
        ax.scatter(track['x'].iloc[0], track['y'].iloc[0], 
                   color=color, s=80, marker='o', edgecolors='black')
    
    ax.scatter(0, 0, color='red', s=200, marker='^', zorder=10)
    ax.set_xlabel('X (meters) - Forward')
    ax.set_ylabel('Y (meters) - Left')
    ax.set_title('Local LiDAR Coordinates', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Right: GPS coordinates
    ax = axes[1]
    for idx, track_id in enumerate(df['ID'].unique()):
        track = df[df['ID'] == track_id].sort_values('timestamp')
        color = colors[idx % len(colors)]
        
        lats, lons = [], []
        for _, row in track.iterrows():
            lat, lon = local_to_gps(row['x'], row['y'], ORIGIN_LAT, ORIGIN_LON, heading_deg)
            lats.append(lat)
            lons.append(lon)
        
        ax.plot(lons, lats, '-', color=color, linewidth=2, alpha=0.8, label=f'ID:{track_id}')
    
    ax.scatter(ORIGIN_LON, ORIGIN_LAT, color='red', s=200, marker='^', zorder=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect(1 / np.cos(np.radians(ORIGIN_LAT)))
    ax.set_title(f'GPS Coordinates (Heading: {heading_deg}°)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    plt.suptitle(f'IMEC Tracking Data\nSensor: {ORIGIN_LAT:.6f}°N, {ORIGIN_LON:.6f}°E', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved dual view to: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_file = os.path.join(
        script_dir, "Data_IMEC", "tracked0001", "0001", "track_dataframes", "tracks_df_1.csv"
    )
    output_dir = os.path.join(script_dir, "IMEC_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("IMEC Data Map Visualization")
    print(f"Sensor Position: {ORIGIN_LAT}°N, {ORIGIN_LON}°E")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(track_file)
    print(f"Loaded {len(df)} records, {df['ID'].nunique()} tracks")
    
    # Try different headings to find best match
    # You can adjust HEADING_DEG at the top of this file
    print(f"\nUsing heading: {HEADING_DEG}° (0=North, 90=East)")
    print("Tip: Adjust HEADING_DEG if trajectories don't align with roads\n")
    
    # 1. Interactive Folium map (HTML)
    print("1. Creating interactive HTML map...")
    create_folium_map(df, os.path.join(output_dir, "map_interactive.html"))
    
    # 2. Static map with contextily
    print("\n2. Creating static map...")
    create_static_map_contextily(df, os.path.join(output_dir, "map_static.png"))
    
    # 3. Simple GPS plot
    print("\n3. Creating GPS plot...")
    create_simple_gps_plot(df, os.path.join(output_dir, "map_gps_simple.png"))
    
    # 4. Dual view
    print("\n4. Creating dual view...")
    create_dual_view(df, os.path.join(output_dir, "map_dual_view.png"))
    
    print("\n" + "=" * 60)
    print(f"All maps saved to: {output_dir}")
    print("\nOpen map_interactive.html in a browser for the best experience!")
    print("=" * 60)
