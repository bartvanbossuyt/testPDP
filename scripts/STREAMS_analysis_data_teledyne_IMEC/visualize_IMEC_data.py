"""
Visualize IMEC tracking data.

Creates:
1. Bird's-eye view of trajectories  
2. Animated GIF of trajectories
3. Overlay on camera images (using calibration)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import json
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Color palette for tracks
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def load_imec_data(base_path):
    """Load tracking data and calibration."""
    track_file = os.path.join(base_path, "track_dataframes", "tracks_df_1.csv")
    calib_file = os.path.join(base_path, "calib.json")
    
    df = pd.read_csv(track_file)
    
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    
    return df, calib


def plot_trajectories_birdseye(df, output_path, title="IMEC Tracking Data - Bird's Eye View"):
    """
    Create bird's-eye view of all trajectories.
    X = forward (LiDAR pointing direction), Y = left
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    unique_ids = df['ID'].unique()
    
    for idx, track_id in enumerate(unique_ids):
        track = df[df['ID'] == track_id].sort_values('timestamp')
        color = COLORS[idx % len(COLORS)]
        
        # Plot trajectory line
        ax.plot(track['x'], track['y'], '-', color=color, linewidth=2, alpha=0.7)
        
        # Mark start and end
        ax.scatter(track['x'].iloc[0], track['y'].iloc[0], 
                   color=color, marker='o', s=100, edgecolors='black', zorder=5)
        ax.scatter(track['x'].iloc[-1], track['y'].iloc[-1], 
                   color=color, marker='s', s=100, edgecolors='black', zorder=5)
        
        # Label with ID
        mid_idx = len(track) // 2
        ax.annotate(f'ID:{track_id}', 
                    (track['x'].iloc[mid_idx], track['y'].iloc[mid_idx]),
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add sensor position
    ax.scatter(0, 0, color='red', marker='^', s=200, label='LiDAR Sensor', zorder=10)
    ax.annotate('SENSOR', (0, 0), xytext=(1, 1), fontsize=10, fontweight='bold', color='red')
    
    # Format axes
    ax.set_xlabel('X (meters) - Forward direction', fontsize=12)
    ax.set_ylabel('Y (meters) - Left direction', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='gray', alpha=0.7, label='Trajectories'),
        plt.scatter([], [], color='gray', marker='o', s=100, edgecolors='black', label='Start'),
        plt.scatter([], [], color='gray', marker='s', s=100, edgecolors='black', label='End'),
        plt.scatter([], [], color='red', marker='^', s=200, label='LiDAR Sensor')
    ]
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bird's-eye view to: {output_path}")


def plot_trajectories_colored_by_time(df, output_path):
    """
    Plot trajectories with color gradient showing time progression.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    unique_ids = df['ID'].unique()
    
    # Normalize timestamps for colormap
    t_min, t_max = df['timestamp'].min(), df['timestamp'].max()
    
    for track_id in unique_ids:
        track = df[df['ID'] == track_id].sort_values('timestamp')
        
        # Create line segments colored by time
        points = np.array([track['x'], track['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Normalize times for this track
        times = (track['timestamp'].values[:-1] - t_min) / (t_max - t_min)
        
        lc = LineCollection(segments, cmap='viridis', alpha=0.8, linewidth=3)
        lc.set_array(times)
        ax.add_collection(lc)
    
    # Add sensor
    ax.scatter(0, 0, color='red', marker='^', s=200, zorder=10)
    
    # Auto-scale
    ax.autoscale()
    ax.set_aspect('equal')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(t_min, t_max))
    cbar = plt.colorbar(sm, ax=ax, label='Timestamp')
    
    ax.set_xlabel('X (meters) - Forward', fontsize=12)
    ax.set_ylabel('Y (meters) - Left', fontsize=12)
    ax.set_title('IMEC Trajectories - Color = Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved time-colored trajectories to: {output_path}")


def plot_single_frame_overlay(df, calib, frame_path, timestamp, output_path):
    """
    Overlay track positions on a camera frame using calibration.
    """
    # Load image
    img = Image.open(frame_path)
    img_array = np.array(img)
    
    # Get calibration matrices
    K = np.array(calib['intrinsics']['intrinsic_matrix'])
    E = np.array(calib['extrinsics'])
    
    # Get positions at this timestamp
    frame_data = df[df['timestamp'] == timestamp]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img_array)
    
    for _, row in frame_data.iterrows():
        # 3D point in LiDAR frame
        if pd.notna(row.get('X_LiDAR')):
            pt_3d = np.array([row['X_LiDAR'], row['Y_LiDAR'], row['Z_LiDAR'], 1.0])
        else:
            pt_3d = np.array([row['x'], row['y'], 0, 1.0])  # Use tracker estimate at z=0
        
        # Transform to camera frame
        pt_cam = E @ pt_3d
        
        # Project to image
        if pt_cam[2] > 0:  # Only if in front of camera
            pt_2d = K @ pt_cam[:3]
            u = pt_2d[0] / pt_2d[2]
            v = pt_2d[1] / pt_2d[2]
            
            # Check if in image bounds
            if 0 <= u < img_array.shape[1] and 0 <= v < img_array.shape[0]:
                color = COLORS[int(row['ID']) % len(COLORS)]
                ax.scatter(u, v, color=color, s=200, marker='o', edgecolors='white', linewidth=2)
                ax.annotate(f"ID:{int(row['ID'])}", (u, v), xytext=(5, 5), 
                           textcoords='offset points', color='white', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_title(f'Frame {int(timestamp)} - Track Overlay', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved frame overlay to: {output_path}")


def create_animation_frames(df, output_dir, calib=None, frames_dir=None):
    """
    Create frame-by-frame visualization for animation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamps = sorted(df['timestamp'].unique())
    
    # Get bounds for consistent axes
    x_min, x_max = df['x'].min() - 2, df['x'].max() + 2
    y_min, y_max = df['y'].min() - 2, df['y'].max() + 2
    
    for i, ts in enumerate(timestamps):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all trajectories up to current time (faded)
        for track_id in df['ID'].unique():
            track = df[(df['ID'] == track_id) & (df['timestamp'] <= ts)].sort_values('timestamp')
            if len(track) > 0:
                color = COLORS[int(track_id) % len(COLORS)]
                ax.plot(track['x'], track['y'], '-', color=color, linewidth=2, alpha=0.5)
        
        # Plot current positions
        current = df[df['timestamp'] == ts]
        for _, row in current.iterrows():
            color = COLORS[int(row['ID']) % len(COLORS)]
            ax.scatter(row['x'], row['y'], color=color, s=200, 
                      marker='o', edgecolors='black', linewidth=2, zorder=5)
            ax.annotate(f"ID:{int(row['ID'])}", (row['x'], row['y']), 
                       xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        # Sensor
        ax.scatter(0, 0, color='red', marker='^', s=200, zorder=10)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'IMEC Tracking - Timestamp: {ts:.0f}', fontsize=14, fontweight='bold')
        
        frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{len(timestamps)} frames...")
    
    print(f"Saved {len(timestamps)} animation frames to: {output_dir}")


def create_gif(frames_dir, output_path, duration=200):
    """Create animated GIF from frames."""
    frames = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    for f in frame_files:
        img = Image.open(os.path.join(frames_dir, f))
        frames.append(img.copy())
        img.close()
    
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved animated GIF to: {output_path}")


def plot_velocity_analysis(df, output_path):
    """Plot velocity analysis of tracks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Speed over time per track
    ax = axes[0, 0]
    for track_id in df['ID'].unique():
        track = df[df['ID'] == track_id].sort_values('timestamp')
        speed = np.sqrt(track['vx']**2 + track['vy']**2)
        color = COLORS[int(track_id) % len(COLORS)]
        ax.plot(track['timestamp'], speed, '-', color=color, alpha=0.7, label=f'ID:{track_id}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Histogram of speeds
    ax = axes[0, 1]
    speeds = np.sqrt(df['vx']**2 + df['vy']**2)
    ax.hist(speeds, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Count')
    ax.set_title('Speed Distribution')
    ax.axvline(speeds.mean(), color='red', linestyle='--', label=f'Mean: {speeds.mean():.2f}')
    ax.legend()
    
    # Track lengths
    ax = axes[1, 0]
    track_lengths = df.groupby('ID').size()
    ax.bar(track_lengths.index, track_lengths.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Track ID')
    ax.set_ylabel('Number of detections')
    ax.set_title('Track Lengths')
    
    # Object positions heatmap
    ax = axes[1, 1]
    h = ax.hist2d(df['x'], df['y'], bins=30, cmap='hot')
    plt.colorbar(h[3], ax=ax, label='Density')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Position Heatmap')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved velocity analysis to: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "Data_IMEC", "tracked0001", "0001")
    output_dir = os.path.join(script_dir, "IMEC_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("IMEC Data Visualization")
    print("=" * 60)
    
    # Load data
    df, calib = load_imec_data(base_path)
    print(f"Loaded {len(df)} tracking records, {df['ID'].nunique()} unique tracks")
    
    # 1. Bird's-eye view
    print("\n1. Creating bird's-eye view...")
    plot_trajectories_birdseye(df, os.path.join(output_dir, "trajectories_birdseye.png"))
    
    # 2. Time-colored trajectories
    print("\n2. Creating time-colored trajectories...")
    plot_trajectories_colored_by_time(df, os.path.join(output_dir, "trajectories_time_colored.png"))
    
    # 3. Velocity analysis
    print("\n3. Creating velocity analysis...")
    plot_velocity_analysis(df, os.path.join(output_dir, "velocity_analysis.png"))
    
    # 4. Camera overlay for first frame
    print("\n4. Creating camera frame overlay...")
    frames_dir = os.path.join(base_path, "Frames_2")
    first_ts = df['timestamp'].min()
    first_frame = os.path.join(frames_dir, f"0{int(first_ts)}.png")
    if os.path.exists(first_frame):
        plot_single_frame_overlay(df, calib, first_frame, first_ts, 
                                  os.path.join(output_dir, "frame_overlay.png"))
    else:
        print(f"  Frame not found: {first_frame}")
    
    # 5. Animation frames
    print("\n5. Creating animation frames...")
    anim_dir = os.path.join(output_dir, "animation_frames")
    create_animation_frames(df, anim_dir)
    
    # 6. Create GIF
    print("\n6. Creating animated GIF...")
    create_gif(anim_dir, os.path.join(output_dir, "trajectories_animated.gif"), duration=150)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)
