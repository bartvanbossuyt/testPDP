#!/usr/bin/env python3
"""
Configuration Visualizer for data_prep_lvlX_clean.py outputs
Visualizes multi-track configurations with playback and interaction analysis.
"""

import sys
import argparse
import re
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.image import imread
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QSlider, QLabel, QCheckBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
import os


class RecordingMetaLoader:
    """Load recording metadata and parameters."""
    
    @staticmethod
    def extract_recording_number(filepath: str) -> Optional[str]:
        """Extract recording number from file path (e.g., '13' from path containing '/13/')."""
        # First, try to find folder number in path (e.g., "/13/" or "\13\")
        # This is more reliable than filename which may contain other numbers like "C100_"
        path_parts = Path(filepath).parts
        for part in path_parts:
            if part.isdigit():
                return part
        
        # Fallback: try to find pattern like "13_tracks.csv" or "13_background.png" in filename
        filename = os.path.basename(filepath)
        match = re.search(r'(\d+)_', filename)
        if match:
            return match.group(1)
        
        return None
    
    @staticmethod
    def load_recording_meta(meta_path: str) -> Dict:
        """Load recording metadata CSV and extract orthoPxToMeter."""
        try:
            meta_df = pd.read_csv(meta_path)
            if "orthoPxToMeter" in meta_df.columns:
                ortho_px = float(meta_df["orthoPxToMeter"].iloc[0])
                return {"orthoPxToMeter": ortho_px}
        except Exception as e:
            print(f"Warning: Could not load metadata from {meta_path}: {e}")
        
        return {"orthoPxToMeter": None}
    
    @staticmethod
    def get_scale_down_factor() -> float:
        """Get scale down factor for inD dataset (all recordings use same factor)."""
        # From drone-dataset-tools visualizer_params.json
        # inD: 12, rounD: 10, exiD: 6, uniD: 2
        return 12.0  # Default for inD dataset
    
    @staticmethod
    def build_paths(recording_num: str, base_dir: str = None) -> Dict[str, str]:
        """Build file paths for a recording number."""
        if base_dir is None:
            # Try to infer base directory structure
            base_dir = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\inD"
        
        rec_padded = f"{int(recording_num):02d}"  # e.g., "13" -> "13", "2" -> "02"
        
        return {
            "background": os.path.join(base_dir, "inD-dataset-v1.1", "data", f"{rec_padded}_background.png"),
            "meta": os.path.join(base_dir, "inD-dataset-v1.1", "data", f"{rec_padded}_recordingMeta.csv"),
        }


class ConfigDataLoader:
    """Load and parse configuration CSV files."""
    
    @staticmethod
    def load_config_file(filepath: str) -> pd.DataFrame:
        """
        Load configuration CSV file.
        Expected format: config_index, frame, trackId, x, y[, class]
        """
        # Try loading with class column
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=["config_index", "frame", "trackId", "x", "y", "class"],
                dtype={
                    "config_index": int,
                    "frame": int,
                    "trackId": int,
                    "x": float,
                    "y": float
                }
            )
        except:
            # Fallback: no class column
            df = pd.read_csv(
                filepath,
                header=None,
                names=["config_index", "frame", "trackId", "x", "y"],
                dtype={
                    "config_index": int,
                    "frame": int,
                    "trackId": int,
                    "x": float,
                    "y": float
                }
            )
            df["class"] = "unknown"
        
        return df
    
    @staticmethod
    def get_config_info(df: pd.DataFrame) -> Dict:
        """Extract configuration metadata."""
        return {
            "num_configs": df["config_index"].nunique(),
            "config_ids": sorted(df["config_index"].unique()),
            "tracks_per_config": df.groupby("config_index")["trackId"].nunique().to_dict(),
            "frames_per_config": df.groupby("config_index")["frame"].nunique().to_dict(),
        }


class ConfigVisualizer(QMainWindow):
    """Interactive visualizer for track configurations."""
    
    def __init__(self, config_df: pd.DataFrame, config_info: Dict, background_image_path: Optional[str] = None, 
                 ortho_px_to_meter: Optional[float] = None, scale_down_factor: float = 1.0,
                 x_utm_origin: float = 0.0, y_utm_origin: float = 0.0):
        super().__init__()
        self.config_df = config_df
        self.config_info = config_info
        self.background_image_path = background_image_path
        self.background_image = None
        self.ortho_px_to_meter = ortho_px_to_meter
        self.scale_down_factor = scale_down_factor
        self.x_utm_origin = x_utm_origin
        self.y_utm_origin = y_utm_origin
        
        # Load background image if provided
        if background_image_path and os.path.exists(background_image_path):
            print(f"Loading background image: {background_image_path}")
            self.background_image = imread(background_image_path)
            (self.image_height, self.image_width) = self.background_image.shape[:2]
            print(f"Background image size: {self.image_width}x{self.image_height}")
        else:
            if background_image_path:
                print(f"Warning: Background image not found: {background_image_path}")
            print("Using plain grid background")
        
        # Current state
        self.current_config_idx = 0
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1  # frames to skip
        
        # Visualization settings
        self.show_trajectory = False
        self.show_future_trajectory = False
        self.show_track_ids = True
        self.show_distance = False
        
        # Colors for different trackIds
        self.track_colors = {
            0: '#3498db',  # Blue
            1: '#e74c3c',  # Red
            2: '#2ecc71',  # Green
            3: '#f39c12',  # Orange
            4: '#9b59b6',  # Purple
            5: '#1abc9c',  # Turquoise
        }
        
        self.init_ui()
        self.load_config(0)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Configuration Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Configuration selector
        self.config_label = QLabel("Config: 0")
        control_layout.addWidget(self.config_label)
        
        self.prev_config_btn = QPushButton("◀◀ Prev Config")
        self.prev_config_btn.clicked.connect(self.prev_config)
        control_layout.addWidget(self.prev_config_btn)
        
        self.next_config_btn = QPushButton("Next Config ▶▶")
        self.next_config_btn.clicked.connect(self.next_config)
        control_layout.addWidget(self.next_config_btn)
        
        # Playback controls
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)
        
        self.prev_frame_btn = QPushButton("◀ Frame")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        control_layout.addWidget(self.prev_frame_btn)
        
        self.next_frame_btn = QPushButton("Frame ▶")
        self.next_frame_btn.clicked.connect(self.next_frame)
        control_layout.addWidget(self.next_frame_btn)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        control_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("Frame: 0/0")
        control_layout.addWidget(self.frame_label)
        
        layout.addLayout(control_layout)
        
        # Display options
        options_layout = QHBoxLayout()
        
        self.trajectory_cb = QCheckBox("Show Trajectory")
        self.trajectory_cb.stateChanged.connect(self.toggle_trajectory)
        options_layout.addWidget(self.trajectory_cb)
        
        self.future_trajectory_cb = QCheckBox("Show Future")
        self.future_trajectory_cb.stateChanged.connect(self.toggle_future_trajectory)
        options_layout.addWidget(self.future_trajectory_cb)
        
        self.track_id_cb = QCheckBox("Show Track IDs")
        self.track_id_cb.setChecked(True)
        self.track_id_cb.stateChanged.connect(self.toggle_track_ids)
        options_layout.addWidget(self.track_id_cb)
        
        self.distance_cb = QCheckBox("Show Distance")
        self.distance_cb.stateChanged.connect(self.toggle_distance)
        options_layout.addWidget(self.distance_cb)
        
        # Info label
        self.info_label = QLabel("")
        options_layout.addWidget(self.info_label)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_frame)
    
    def load_config(self, config_idx: int):
        """Load a specific configuration."""
        self.current_config_idx = config_idx
        
        # Get data for this configuration
        self.current_data = self.config_df[
            self.config_df["config_index"] == config_idx
        ].copy()
        
        if self.current_data.empty:
            return
        
        # Get frames and tracks
        self.frames = sorted(self.current_data["frame"].unique())
        self.tracks = sorted(self.current_data["trackId"].unique())
        
        # Reset to first frame
        self.current_frame = 0
        self.frame_slider.setMaximum(len(self.frames) - 1)
        self.frame_slider.setValue(0)
        
        # Update labels
        self.config_label.setText(f"Config: {config_idx} ({len(self.frames)} frames, {len(self.tracks)} tracks)")
        self.update_frame_label()
        
        # Draw initial frame
        self.draw_frame()
    
    def draw_frame(self):
        """Draw the current frame."""
        self.ax.clear()
        
        if self.current_frame >= len(self.frames):
            return
        
        frame_num = self.frames[self.current_frame]
        frame_data = self.current_data[self.current_data["frame"] == frame_num]
        
        # Draw background image if available
        if self.background_image is not None:
            self.ax.imshow(self.background_image, zorder=0, alpha=0.7)
            # Set axis limits to image dimensions
            self.ax.set_xlim(0, self.image_width)
            self.ax.set_ylim(self.image_height, 0)  # Invert Y-axis for image coordinates
        
        # Draw trajectories if enabled
        if self.show_trajectory or self.show_future_trajectory:
            for track_id in self.tracks:
                track_data = self.current_data[self.current_data["trackId"] == track_id].copy()
                color = self.track_colors.get(track_id, '#95a5a6')
                
                # Convert coordinates if needed
                if self.background_image is not None and self.ortho_px_to_meter is not None:
                    track_data["x_vis"] = ((track_data["x"] - self.x_utm_origin) / self.ortho_px_to_meter) / self.scale_down_factor
                    track_data["y_vis"] = (-(track_data["y"] - self.y_utm_origin) / self.ortho_px_to_meter) / self.scale_down_factor
                else:
                    track_data["x_vis"] = track_data["x"]
                    track_data["y_vis"] = track_data["y"]
                
                if self.show_trajectory:
                    # Past trajectory
                    past = track_data[track_data["frame"] <= frame_num]
                    self.ax.plot(past["x_vis"], past["y_vis"], 
                               color=color, alpha=0.3, linewidth=1, linestyle='--', zorder=10)
                
                if self.show_future_trajectory:
                    # Future trajectory
                    future = track_data[track_data["frame"] >= frame_num]
                    self.ax.plot(future["x_vis"], future["y_vis"], 
                               color=color, alpha=0.2, linewidth=1, linestyle=':', zorder=10)
        
        # Draw current positions
        for _, row in frame_data.iterrows():
            track_id = row["trackId"]
            x, y = row["x"], row["y"]
            
            # Convert coordinates if using background image and conversion factor provided
            if self.background_image is not None and self.ortho_px_to_meter is not None:
                # Convert meters to pixels, then scale down for downscaled background image
                x_vis = ((x - self.x_utm_origin) / self.ortho_px_to_meter) / self.scale_down_factor
                y_vis = (-(y - self.y_utm_origin) / self.ortho_px_to_meter) / self.scale_down_factor
                print(f"Track {track_id}: Meters ({x:.2f}, {y:.2f}) -> Pixel ({x_vis:.2f}, {y_vis:.2f}) [Image: {self.image_width}x{self.image_height}]")
            else:
                x_vis, y_vis = x, y
            
            color = self.track_colors.get(track_id, '#95a5a6')
            
            # Draw point with higher zorder to appear on top
            self.ax.plot(x_vis, y_vis, 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=2, zorder=30)
            
            # Annotate with track ID - always shown with high zorder
            if self.show_track_ids:
                # Adjust text offset based on whether we have a background image
                # For pixel coordinates (with image), use smaller offset; for meters, use larger
                y_offset = -2.5 if self.background_image is not None else 0.5
                self.ax.text(x_vis, y_vis + y_offset, f"ID{track_id}", 
                           fontsize=8, ha='center', va='center', color='black',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.2),
                           zorder=22)
        
        # Draw distance lines if enabled
        if self.show_distance and len(frame_data) == 2:
            positions = frame_data[["x", "y"]].values
            if len(positions) == 2:
                # Draw line
                self.ax.plot([positions[0][0], positions[1][0]], 
                           [positions[0][1], positions[1][1]], 
                           'k--', linewidth=1, alpha=0.5)
                
                # Calculate and display distance
                distance = np.sqrt((positions[1][0] - positions[0][0])**2 + 
                                 (positions[1][1] - positions[0][1])**2)
                mid_x = (positions[0][0] + positions[1][0]) / 2
                mid_y = (positions[0][1] + positions[1][1]) / 2
                self.ax.text(mid_x, mid_y, f"{distance:.2f}m", 
                           fontsize=9, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Set axis properties
        if self.background_image is not None:
            # With background image, use pixel coordinates
            self.ax.set_xlabel("X Position (pixels)", fontsize=10)
            self.ax.set_ylabel("Y Position (pixels)", fontsize=10)
        else:
            # Without background, use meter coordinates
            self.ax.set_xlabel("X Position (m)", fontsize=10)
            self.ax.set_ylabel("Y Position (m)", fontsize=10)
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal', adjustable='box')
        
        self.ax.set_title(f"Config {self.current_config_idx} - Frame {frame_num}", fontsize=12, fontweight='bold')
        
        # Update info
        if len(frame_data) == 2:
            positions = frame_data[["x", "y"]].values
            distance = np.sqrt((positions[1][0] - positions[0][0])**2 + 
                             (positions[1][1] - positions[0][1])**2)
            self.info_label.setText(f"Distance: {distance:.2f}m")
        else:
            self.info_label.setText("")
        
        self.canvas.draw()
    
    def update_frame_label(self):
        """Update frame counter label."""
        if self.frames:
            self.frame_label.setText(f"Frame: {self.current_frame + 1}/{len(self.frames)}")
    
    def toggle_play(self):
        """Toggle playback."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("⏸ Pause")
            self.timer.start(100)  # Update every 100ms
        else:
            self.play_btn.setText("▶ Play")
            self.timer.stop()
    
    def advance_frame(self):
        """Advance to next frame during playback."""
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += self.playback_speed
            if self.current_frame >= len(self.frames):
                self.current_frame = len(self.frames) - 1
            self.frame_slider.setValue(self.current_frame)
            self.draw_frame()
            self.update_frame_label()
        else:
            # Loop back to start
            self.current_frame = 0
            self.frame_slider.setValue(0)
            self.draw_frame()
            self.update_frame_label()
    
    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)
            self.draw_frame()
            self.update_frame_label()
    
    def next_frame(self):
        """Go to next frame."""
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.draw_frame()
            self.update_frame_label()
    
    def prev_config(self):
        """Go to previous configuration."""
        if self.is_playing:
            self.toggle_play()
        
        configs = self.config_info["config_ids"]
        current_idx_in_list = configs.index(self.current_config_idx)
        if current_idx_in_list > 0:
            self.load_config(configs[current_idx_in_list - 1])
    
    def next_config(self):
        """Go to next configuration."""
        if self.is_playing:
            self.toggle_play()
        
        configs = self.config_info["config_ids"]
        current_idx_in_list = configs.index(self.current_config_idx)
        if current_idx_in_list < len(configs) - 1:
            self.load_config(configs[current_idx_in_list + 1])
    
    def slider_changed(self, value):
        """Handle slider movement."""
        self.current_frame = value
        self.draw_frame()
        self.update_frame_label()
    
    def toggle_trajectory(self, state):
        """Toggle trajectory display."""
        self.show_trajectory = (state == Qt.Checked)
        self.draw_frame()
    
    def toggle_future_trajectory(self, state):
        """Toggle future trajectory display."""
        self.show_future_trajectory = (state == Qt.Checked)
        self.draw_frame()
    
    def toggle_track_ids(self, state):
        """Toggle track ID labels."""
        self.show_track_ids = (state == Qt.Checked)
        self.draw_frame()
    
    def toggle_distance(self, state):
        """Toggle distance display."""
        self.show_distance = (state == Qt.Checked)
        self.draw_frame()


def main():
    # ============================================================================
    # CONFIGURATION: Set your file paths here
    # ============================================================================
    
    # Path to your configuration CSV file (output from data_prep_lvlX_clean.py)
    CONFIG_FILE = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\inD\Dataset_for_testing_new_pdp\18\maxdist4_minframes3_minmov3\C20_CB_NC_inD_F100.csv"
    
    # Base directory for inD dataset (optional - will be inferred if not set)
    BASE_DIR = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\inD"
    
    # ============================================================================
    
    # Auto-detect recording number from file path
    recording_num = RecordingMetaLoader.extract_recording_number(CONFIG_FILE)
    if recording_num is None:
        print("Warning: Could not detect recording number from file path")
        print(f"File: {CONFIG_FILE}")
        recording_num = input("Please enter recording number (e.g., 13): ").strip()
    
    print(f"\n{'='*70}")
    print(f"Auto-detected recording number: {recording_num}")
    print(f"{'='*70}\n")
    
    # Build paths for this recording
    paths = RecordingMetaLoader.build_paths(recording_num, BASE_DIR)
    BACKGROUND_IMAGE = paths["background"]
    META_PATH = paths["meta"]
    
    # Load recording metadata
    print(f"Loading metadata from: {META_PATH}")
    meta_info = RecordingMetaLoader.load_recording_meta(META_PATH)
    ORTHO_PX_TO_METER = meta_info.get("orthoPxToMeter")
    
    # Get scale down factor
    SCALE_DOWN_FACTOR = RecordingMetaLoader.get_scale_down_factor()
    
    # Coordinates are relative to recording origin
    X_UTM_ORIGIN = 0.0
    Y_UTM_ORIGIN = 0.0
    
    print(f"Background image: {BACKGROUND_IMAGE}")
    print(f"  Exists: {os.path.exists(BACKGROUND_IMAGE)}")
    print(f"Coordinate conversion:")
    print(f"  orthoPxToMeter: {ORTHO_PX_TO_METER}")
    print(f"  scale_down_factor: {SCALE_DOWN_FACTOR}")
    print()
    
    # Load data
    print(f"Loading configuration file: {CONFIG_FILE}")
    config_df = ConfigDataLoader.load_config_file(CONFIG_FILE)
    config_info = ConfigDataLoader.get_config_info(config_df)
    
    print(f"Loaded {config_info['num_configs']} configurations")
    print(f"Tracks per config: {config_info['tracks_per_config']}")
    print(f"Frames per config: {config_info['frames_per_config']}")
    print()
    
    # Create Qt application
    app = QApplication(sys.argv)
    visualizer = ConfigVisualizer(config_df, config_info, BACKGROUND_IMAGE, ORTHO_PX_TO_METER, SCALE_DOWN_FACTOR, X_UTM_ORIGIN, Y_UTM_ORIGIN)
    visualizer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
