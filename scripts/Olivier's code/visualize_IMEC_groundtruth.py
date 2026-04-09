"""
Visualize IMEC Ground Truth data with map overlay.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

try:
    import folium
    from folium.plugins import AntPath
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# Sensor position
ORIGIN_LAT = 51.045101
ORIGIN_LON = 3.713908
HEADING_DEG = 19.4  # 4.4° base + 15° adjustment

# Conversion
METERS_PER_DEG_LAT = 111320
METERS_PER_DEG_LON = 111320 * np.cos(np.radians(ORIGIN_LAT))

# Class colors
CLASS_COLORS = {
    'Pedestrian': '#FF6B6B',  # Red
    'Cyclist': '#4ECDC4',     # Teal
    'Vehicle': '#45B7D1',     # Blue
}

CLASS_MARKERS = {
    'Pedestrian': 'o',
    'Cyclist': '^',
    'Vehicle': 's',
}


def local_to_gps(x, y, heading_deg=0):
    heading_rad = np.radians(heading_deg)
    north_offset = x * np.cos(heading_rad) - y * np.sin(heading_rad)
    east_offset = x * np.sin(heading_rad) + y * np.cos(heading_rad)
    lat = ORIGIN_LAT + (north_offset / METERS_PER_DEG_LAT)
    lon = ORIGIN_LON + (east_offset / METERS_PER_DEG_LON)
    return lat, lon


def plot_ground_truth_birdseye(df, output_path):
    """Bird's eye view of ground truth trajectories."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot each track
    for track_id in sorted(df['track_id'].unique()):
        track = df[df['track_id'] == track_id].sort_values('timestamp')
        obj_class = track['class'].iloc[0]
        color = CLASS_COLORS.get(obj_class, 'gray')
        marker = CLASS_MARKERS.get(obj_class, 'o')
        
        # Trajectory line
        ax.plot(track['x'], track['y'], '-', color=color, linewidth=2.5, alpha=0.8)
        
        # Start point
        ax.scatter(track['x'].iloc[0], track['y'].iloc[0], 
                   color=color, marker=marker, s=150, edgecolors='black', 
                   linewidth=2, zorder=5)
        
        # End point  
        ax.scatter(track['x'].iloc[-1], track['y'].iloc[-1], 
                   color=color, marker=marker, s=150, edgecolors='white', 
                   linewidth=2, zorder=5)
        
        # Label
        mid = len(track) // 2
        ax.annotate(f'{obj_class[:3]}:{track_id}', 
                    (track['x'].iloc[mid], track['y'].iloc[mid]),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Sensor
    ax.scatter(0, 0, color='red', marker='*', s=400, zorder=10, label='LiDAR Sensor')
    ax.annotate('SENSOR', (0, 0.5), fontsize=10, fontweight='bold', color='red')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS['Pedestrian'], 
               markersize=12, label='Pedestrian'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CLASS_COLORS['Cyclist'], 
               markersize=12, label='Cyclist'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLASS_COLORS['Vehicle'], 
               markersize=12, label='Vehicle'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=15, label='Sensor'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    ax.set_xlabel('X (meters) - Forward', fontsize=12)
    ax.set_ylabel('Y (meters) - Left', fontsize=12)
    ax.set_title('IMEC Ground Truth - Bird\'s Eye View\n10 Tracks: 6 Cyclists, 3 Pedestrians, 1 Vehicle', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ground_truth_animated_frames(df, output_dir):
    """Create animation frames."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamps = sorted(df['timestamp'].unique())
    x_min, x_max = df['x'].min() - 3, df['x'].max() + 3
    y_min, y_max = df['y'].min() - 3, df['y'].max() + 3
    
    for i, ts in enumerate(timestamps):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot trajectories up to now
        for track_id in df['track_id'].unique():
            track = df[(df['track_id'] == track_id) & (df['timestamp'] <= ts)].sort_values('timestamp')
            if len(track) > 0:
                obj_class = track['class'].iloc[0]
                color = CLASS_COLORS.get(obj_class, 'gray')
                ax.plot(track['x'], track['y'], '-', color=color, linewidth=2, alpha=0.5)
        
        # Current positions
        current = df[df['timestamp'] == ts]
        for _, row in current.iterrows():
            color = CLASS_COLORS.get(row['class'], 'gray')
            marker = CLASS_MARKERS.get(row['class'], 'o')
            ax.scatter(row['x'], row['y'], color=color, s=200, marker=marker,
                       edgecolors='black', linewidth=2, zorder=5)
            ax.annotate(f"{row['class'][:3]}:{int(row['track_id'])}", 
                        (row['x'], row['y']), xytext=(3, 3), 
                        textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Sensor
        ax.scatter(0, 0, color='red', marker='*', s=300, zorder=10)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'IMEC Ground Truth - Frame {i+1}/{len(timestamps)}\nTimestamp: {ts}', 
                     fontsize=12, fontweight='bold')
        
        plt.savefig(os.path.join(output_dir, f'gt_frame_{i:04d}.png'), dpi=100, bbox_inches='tight')
        plt.close()
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(timestamps)}")
    
    print(f"Saved {len(timestamps)} frames to: {output_dir}")


def create_folium_map_gt(df, output_path, heading_deg=HEADING_DEG):
    """Interactive map with ground truth."""
    if not HAS_FOLIUM:
        print("Folium not available")
        return
    
    m = folium.Map(location=[ORIGIN_LAT, ORIGIN_LON], zoom_start=18, tiles='OpenStreetMap')
    
    # Satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite', overlay=False
    ).add_to(m)
    folium.LayerControl().add_to(m)
    
    # Sensor marker
    folium.Marker(
        [ORIGIN_LAT, ORIGIN_LON],
        popup='LiDAR Sensor',
        icon=folium.Icon(color='red', icon='video-camera', prefix='fa'),
    ).add_to(m)
    
    # Color map
    folium_colors = {'Pedestrian': 'red', 'Cyclist': 'green', 'Vehicle': 'blue'}
    
    for track_id in sorted(df['track_id'].unique()):
        track = df[df['track_id'] == track_id].sort_values('timestamp')
        obj_class = track['class'].iloc[0]
        color = folium_colors.get(obj_class, 'gray')
        
        coords = []
        for _, row in track.iterrows():
            lat, lon = local_to_gps(row['x'], row['y'], heading_deg)
            coords.append([lat, lon])
        
        if len(coords) > 1:
            AntPath(coords, color=color, weight=4, opacity=0.8,
                    tooltip=f'{obj_class} - Track {track_id}').add_to(m)
            
            # Start/end markers
            folium.CircleMarker(coords[0], radius=8, color=color, fill=True,
                                popup=f'{obj_class} {track_id} - Start').add_to(m)
            folium.CircleMarker(coords[-1], radius=8, color=color, fill=True,
                                fill_color='white', popup=f'{obj_class} {track_id} - End').add_to(m)
    
    m.save(output_path)
    print(f"Saved: {output_path}")


def plot_class_analysis(df, output_path):
    """Analyze by class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Trajectories by class
    ax = axes[0, 0]
    for obj_class in df['class'].unique():
        class_df = df[df['class'] == obj_class]
        color = CLASS_COLORS.get(obj_class, 'gray')
        for track_id in class_df['track_id'].unique():
            track = class_df[class_df['track_id'] == track_id].sort_values('timestamp')
            ax.plot(track['x'], track['y'], '-', color=color, linewidth=2, alpha=0.7)
    ax.scatter(0, 0, color='red', marker='*', s=200, zorder=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectories by Class')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Class distribution
    ax = axes[0, 1]
    class_counts = df.groupby('class')['track_id'].nunique()
    colors = [CLASS_COLORS.get(c, 'gray') for c in class_counts.index]
    ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Tracks per Class')
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    # 3. Detection count over time
    ax = axes[1, 0]
    detections_per_ts = df.groupby('timestamp').size()
    ax.plot(detections_per_ts.index, detections_per_ts.values, 'b-', linewidth=2)
    ax.fill_between(detections_per_ts.index, detections_per_ts.values, alpha=0.3)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Number of Objects')
    ax.set_title('Objects Visible per Frame')
    ax.grid(True, alpha=0.3)
    
    # 4. Distance from sensor
    ax = axes[1, 1]
    df['distance'] = np.sqrt(df['x']**2 + df['y']**2)
    for obj_class in df['class'].unique():
        class_df = df[df['class'] == obj_class]
        color = CLASS_COLORS.get(obj_class, 'gray')
        ax.hist(class_df['distance'], bins=20, alpha=0.5, color=color, 
                label=obj_class, edgecolor='black')
    ax.set_xlabel('Distance from Sensor (m)')
    ax.set_ylabel('Count')
    ax.set_title('Object Distance Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_gif(frames_dir, output_path, duration=150):
    """Create GIF from frames."""
    from PIL import Image
    frames = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    for f in frame_files:
        img = Image.open(os.path.join(frames_dir, f))
        frames.append(img.copy())
        img.close()
    
    if frames:
        frames[0].save(output_path, save_all=True, append_images=frames[1:],
                       duration=duration, loop=0)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "IMEC_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth
    gt_file = os.path.join(script_dir, "IMEC_GroundTruth_PDP_full.csv")
    df = pd.read_csv(gt_file)
    
    print("=" * 60)
    print("IMEC GROUND TRUTH VISUALIZATION")
    print("=" * 60)
    print(f"Total observations: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Classes: {df['class'].value_counts().to_dict()}")
    print(f"Timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # 1. Bird's eye view
    print("\n1. Creating bird's eye view...")
    plot_ground_truth_birdseye(df, os.path.join(output_dir, "gt_birdseye.png"))
    
    # 2. Class analysis
    print("\n2. Creating class analysis...")
    plot_class_analysis(df, os.path.join(output_dir, "gt_class_analysis.png"))
    
    # 3. Interactive map
    print("\n3. Creating interactive map...")
    create_folium_map_gt(df, os.path.join(output_dir, "gt_map_interactive.html"))
    
    # 4. Animation frames
    print("\n4. Creating animation frames...")
    anim_dir = os.path.join(output_dir, "gt_animation_frames")
    plot_ground_truth_animated_frames(df, anim_dir)
    
    # 5. Create GIF
    print("\n5. Creating animated GIF...")
    create_gif(anim_dir, os.path.join(output_dir, "gt_animated.gif"))
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)
