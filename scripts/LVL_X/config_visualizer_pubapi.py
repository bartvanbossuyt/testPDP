#!/usr/bin/env python3
"""
Configuration Visualizer for data_prep_pubapi.py outputs
Visualizes multi-track configurations with playback on OpenStreetMap background.
"""

import sys
import os
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QSlider, QLabel, QCheckBox,
                              QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer

# For map tiles
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Warning: contextily not installed. Run: pip install contextily")

# For coordinate transformation (GPS to Web Mercator)
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("Warning: pyproj not installed. Run: pip install pyproj")


class ConfigDataLoader:
    """Load and parse configuration CSV files from pubapi data prep."""
    
    @staticmethod
    def load_config_file(filepath: str) -> pd.DataFrame:
        """
        Load configuration CSV file.
        Expected format: config_index, sample, trackId, x (longitude), y (latitude), class_id, timestamp
        """
        # Try loading with class and timestamp columns (7 columns)
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=["config_index", "sample", "trackId", "x", "y", "class_id", "timestamp"],
                dtype={
                    "config_index": int,
                    "sample": int,
                    "trackId": int,
                    "x": float,
                    "y": float
                }
            )
        except:
            # Try with 6 columns (class but no timestamp)
            try:
                df = pd.read_csv(
                    filepath,
                    header=None,
                    names=["config_index", "sample", "trackId", "x", "y", "class_id"],
                    dtype={
                        "config_index": int,
                        "sample": int,
                        "trackId": int,
                        "x": float,
                        "y": float
                    }
                )
                df["timestamp"] = ""
            except:
                # Fallback: 5 columns (no class, no timestamp)
                df = pd.read_csv(
                    filepath,
                    header=None,
                    names=["config_index", "sample", "trackId", "x", "y"],
                    dtype={
                        "config_index": int,
                        "sample": int,
                        "trackId": int,
                        "x": float,
                        "y": float
                    }
                )
                df["class_id"] = -1
                df["timestamp"] = ""
        
        return df
    
    @staticmethod
    def get_config_info(df: pd.DataFrame) -> Dict:
        """Extract configuration metadata."""
        return {
            "num_configs": df["config_index"].nunique(),
            "config_ids": sorted(df["config_index"].unique()),
            "tracks_per_config": df.groupby("config_index")["trackId"].nunique().to_dict(),
            "samples_per_config": df.groupby("config_index")["sample"].nunique().to_dict(),
        }


class CoordinateTransformer:
    """Handle world coordinates to GPS and Web Mercator transformations."""
    
    # Camera position (known reference point)
    CAMERA_LAT = 50.812976
    CAMERA_LON = 3.235785
    
    # Meters per degree at this latitude
    # 1 degree latitude ≈ 111,320 meters (constant)
    # 1 degree longitude ≈ 111,320 * cos(lat) meters
    METERS_PER_DEG_LAT = 111320.0
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.radians(CAMERA_LAT))  # ~70,400m at 50.8°N
    
    def __init__(self):
        if HAS_PYPROJ:
            # WGS84 (GPS) to Web Mercator (EPSG:3857)
            self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            self.inverse_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        else:
            self.transformer = None
            self.inverse_transformer = None
    
    def world_to_gps(self, world_x: float, world_y: float) -> tuple:
        """
        Convert world coordinates (meters relative to camera) to GPS (lon, lat).
        
        World coordinate system (from camera's perspective):
        - X: positive = right of camera
        - Y: positive = forward from camera (away from camera)
        
        Note: This assumes the camera is facing roughly north. Adjust if different.
        """
        # Convert meters to degrees
        delta_lon = world_x / self.METERS_PER_DEG_LON
        delta_lat = world_y / self.METERS_PER_DEG_LAT
        
        # Add to camera position
        lon = self.CAMERA_LON + delta_lon
        lat = self.CAMERA_LAT + delta_lat
        
        return lon, lat
    
    def world_to_gps_array(self, world_x: np.ndarray, world_y: np.ndarray) -> tuple:
        """Convert arrays of world coordinates to GPS."""
        delta_lon = world_x / self.METERS_PER_DEG_LON
        delta_lat = world_y / self.METERS_PER_DEG_LAT
        
        lons = self.CAMERA_LON + delta_lon
        lats = self.CAMERA_LAT + delta_lat
        
        return lons, lats
    
    def gps_to_mercator(self, lon: float, lat: float) -> tuple:
        """Convert GPS (lon, lat) to Web Mercator (x, y)."""
        if self.transformer:
            return self.transformer.transform(lon, lat)
        else:
            # Approximate conversion (good enough for small areas)
            x = lon * 20037508.34 / 180
            y = np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180)
            y = y * 20037508.34 / 180
            return x, y
    
    def gps_to_mercator_array(self, lons: np.ndarray, lats: np.ndarray) -> tuple:
        """Convert arrays of GPS coordinates to Web Mercator."""
        if self.transformer:
            return self.transformer.transform(lons, lats)
        else:
            xs = lons * 20037508.34 / 180
            ys = np.log(np.tan((90 + lats) * np.pi / 360)) / (np.pi / 180)
            ys = ys * 20037508.34 / 180
            return xs, ys
    
    def world_to_mercator_array(self, world_x: np.ndarray, world_y: np.ndarray) -> tuple:
        """Convert world coordinates directly to Web Mercator."""
        lons, lats = self.world_to_gps_array(world_x, world_y)
        return self.gps_to_mercator_array(lons, lats)


class ConfigVisualizerPubapi(QMainWindow):
    """Interactive visualizer for pubapi track configurations with OSM background."""
    
    # Class ID names from pubapi
    CLASS_ID_NAMES = {
        0: "unknown",
        1: "person",
        2: "bicycle",
        5: "car",
        7: "truck",
        12: "motorcycle",
        14: "bus",
        24: "trailer",
    }
    
    def __init__(self, config_df: pd.DataFrame, config_info: Dict):
        super().__init__()
        self.config_df = config_df
        self.config_info = config_info
        self.coord_transformer = CoordinateTransformer()
        
        # Pre-compute mercator coordinates for all data
        self._precompute_mercator_coords()
        
        # Current state
        self.current_config_idx = 0
        self.current_sample = 0
        self.is_playing = False
        self.playback_speed = 1
        
        # Visualization settings
        self.show_trajectory = False
        self.show_future_trajectory = False
        self.show_track_ids = True
        self.show_distance = False
        self.show_class = True
        
        # Map settings
        self.map_buffer = 0.0002  # Buffer around track bounds in degrees (~20m)
        self.cached_bounds = None
        self.cached_config_idx = None
        self.use_fixed_view = True  # Use same map view for all configs
        
        # Pre-compute global bounds for all data
        self._compute_global_bounds()
        
        # Colors for different trackIds
        self.track_colors = {
            0: '#3498db',  # Blue
            1: '#e74c3c',  # Red
            2: '#2ecc71',  # Green
            3: '#f39c12',  # Orange
            4: '#9b59b6',  # Purple
            5: '#1abc9c',  # Turquoise
        }
        
        # Class-based colors (alternative)
        self.class_colors = {
            0: '#95a5a6',  # unknown - gray
            1: '#e74c3c',  # person - red
            2: '#2ecc71',  # bicycle - green
            5: '#3498db',  # car - blue
            7: '#f39c12',  # truck - orange
            12: '#9b59b6', # motorcycle - purple
            14: '#1abc9c', # bus - turquoise
            24: '#e67e22', # trailer - dark orange
        }
        
        self.init_ui()
        self.load_config(self.config_info["config_ids"][0] if self.config_info["config_ids"] else 0)
    
    def _precompute_mercator_coords(self):
        """Pre-compute Web Mercator coordinates from world coordinates."""
        # x and y in the CSV are now world coordinates (meters relative to camera)
        world_x = self.config_df["x"].values
        world_y = self.config_df["y"].values
        
        # Convert world coords to GPS, then to mercator
        lons, lats = self.coord_transformer.world_to_gps_array(world_x, world_y)
        mercator_x, mercator_y = self.coord_transformer.gps_to_mercator_array(lons, lats)
        
        # Store GPS and mercator coordinates
        self.config_df["gps_lon"] = lons
        self.config_df["gps_lat"] = lats
        self.config_df["mercator_x"] = mercator_x
        self.config_df["mercator_y"] = mercator_y
    
    def _compute_global_bounds(self):
        """Compute global bounds covering all data for fixed map view."""
        # Use the computed GPS coordinates
        min_lon = self.config_df["gps_lon"].min()
        max_lon = self.config_df["gps_lon"].max()
        min_lat = self.config_df["gps_lat"].min()
        max_lat = self.config_df["gps_lat"].max()
        
        # Add buffer
        buffer_deg = self.map_buffer
        self.global_bounds_lon = (min_lon - buffer_deg, max_lon + buffer_deg)
        self.global_bounds_lat = (min_lat - buffer_deg, max_lat + buffer_deg)
        
        # Convert to mercator
        self.global_bounds_mercator_x = self.coord_transformer.gps_to_mercator_array(
            np.array(self.global_bounds_lon), np.array([self.global_bounds_lat[0], self.global_bounds_lat[0]])
        )[0]
        self.global_bounds_mercator_y = self.coord_transformer.gps_to_mercator_array(
            np.array([self.global_bounds_lon[0], self.global_bounds_lon[0]]), np.array(self.global_bounds_lat)
        )[1]
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Pubapi Configuration Visualizer (OpenStreetMap)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(12, 9))
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
        
        # Config jump
        control_layout.addWidget(QLabel("  Go to:"))
        self.config_spinbox = QSpinBox()
        self.config_spinbox.setMinimum(0)
        self.config_spinbox.setMaximum(max(self.config_info["config_ids"]) if self.config_info["config_ids"] else 0)
        self.config_spinbox.valueChanged.connect(self.jump_to_config)
        control_layout.addWidget(self.config_spinbox)
        
        # Playback controls
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)
        
        self.prev_frame_btn = QPushButton("◀ Sample")
        self.prev_frame_btn.clicked.connect(self.prev_sample)
        control_layout.addWidget(self.prev_frame_btn)
        
        self.next_frame_btn = QPushButton("Sample ▶")
        self.next_frame_btn.clicked.connect(self.next_sample)
        control_layout.addWidget(self.next_frame_btn)
        
        # Sample slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        control_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("Sample: 0/0")
        control_layout.addWidget(self.frame_label)
        
        layout.addLayout(control_layout)
        
        # Display options row 1
        options_layout = QHBoxLayout()
        
        self.trajectory_cb = QCheckBox("Show Past Trajectory")
        self.trajectory_cb.stateChanged.connect(self.toggle_trajectory)
        options_layout.addWidget(self.trajectory_cb)
        
        self.future_trajectory_cb = QCheckBox("Show Future")
        self.future_trajectory_cb.stateChanged.connect(self.toggle_future_trajectory)
        options_layout.addWidget(self.future_trajectory_cb)
        
        self.track_id_cb = QCheckBox("Show Track IDs")
        self.track_id_cb.setChecked(True)
        self.track_id_cb.stateChanged.connect(self.toggle_track_ids)
        options_layout.addWidget(self.track_id_cb)
        
        self.class_cb = QCheckBox("Show Class")
        self.class_cb.setChecked(True)
        self.class_cb.stateChanged.connect(self.toggle_class)
        options_layout.addWidget(self.class_cb)
        
        self.distance_cb = QCheckBox("Show Distance")
        self.distance_cb.stateChanged.connect(self.toggle_distance)
        options_layout.addWidget(self.distance_cb)
        
        self.fixed_view_cb = QCheckBox("Fixed Map View")
        self.fixed_view_cb.setChecked(True)
        self.fixed_view_cb.stateChanged.connect(self.toggle_fixed_view)
        options_layout.addWidget(self.fixed_view_cb)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Display options row 2 - Map settings
        map_layout = QHBoxLayout()
        
        map_layout.addWidget(QLabel("Map Style:"))
        self.map_style_combo = QComboBox()
        self.map_style_combo.addItems([
            "OpenStreetMap",
            "CartoDB Positron",
            "CartoDB Dark",
            "CartoDB Voyager",
            "ESRI WorldStreetMap",
        ])
        self.map_style_combo.currentIndexChanged.connect(self.change_map_style)
        map_layout.addWidget(self.map_style_combo)
        
        map_layout.addWidget(QLabel("  Buffer:"))
        self.buffer_spinbox = QSpinBox()
        self.buffer_spinbox.setMinimum(10)
        self.buffer_spinbox.setMaximum(500)
        self.buffer_spinbox.setValue(20)
        self.buffer_spinbox.setSuffix(" m")
        self.buffer_spinbox.valueChanged.connect(self.change_buffer)
        map_layout.addWidget(self.buffer_spinbox)
        
        # Info label
        self.info_label = QLabel("")
        map_layout.addWidget(self.info_label)
        
        map_layout.addStretch()
        layout.addLayout(map_layout)
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_sample)
    
    def get_map_provider(self):
        """Get the current map tile provider."""
        if not HAS_CONTEXTILY:
            return None
        
        style = self.map_style_combo.currentText()
        providers = {
            "OpenStreetMap": ctx.providers.OpenStreetMap.Mapnik,
            "CartoDB Positron": ctx.providers.CartoDB.Positron,
            "CartoDB Dark": ctx.providers.CartoDB.DarkMatter,
            "CartoDB Voyager": ctx.providers.CartoDB.Voyager,
            "ESRI WorldStreetMap": ctx.providers.Esri.WorldStreetMap,
        }
        return providers.get(style, ctx.providers.OpenStreetMap.Mapnik)
    
    def load_config(self, config_idx: int):
        """Load a specific configuration."""
        if config_idx not in self.config_info["config_ids"]:
            return
        
        self.current_config_idx = config_idx
        
        # Get data for this configuration
        self.current_data = self.config_df[
            self.config_df["config_index"] == config_idx
        ].copy()
        
        if self.current_data.empty:
            return
        
        # Get samples and tracks
        self.samples = sorted(self.current_data["sample"].unique())
        self.tracks = sorted(self.current_data["trackId"].unique())
        
        # Get class info
        class_info = []
        for tid in self.tracks:
            track_data = self.current_data[self.current_data["trackId"] == tid]
            if not track_data.empty:
                cid = int(track_data["class_id"].iloc[0])
                class_name = self.CLASS_ID_NAMES.get(cid, f"class_{cid}")
                class_info.append(f"T{tid}:{class_name}")
        
        # Reset to first sample
        self.current_sample = 0
        self.frame_slider.setMaximum(len(self.samples) - 1)
        self.frame_slider.setValue(0)
        
        # Update labels
        self.config_label.setText(
            f"Config: {config_idx} ({len(self.samples)} samples, {len(self.tracks)} tracks) [{', '.join(class_info)}]"
        )
        self.update_frame_label()
        
        # Clear cached bounds to force map reload
        self.cached_config_idx = None
        
        # Draw initial frame
        self.draw_frame()
    
    def draw_frame(self):
        """Draw the current sample."""
        self.ax.clear()
        
        if self.current_sample >= len(self.samples):
            return
        
        sample_num = self.samples[self.current_sample]
        sample_data = self.current_data[self.current_data["sample"] == sample_num]
        
        # Use global bounds (fixed view) or per-config bounds
        if self.use_fixed_view:
            # Use pre-computed global bounds for consistent map view
            bounds_x = self.global_bounds_mercator_x
            bounds_y = self.global_bounds_mercator_y
        else:
            # Calculate bounds for this configuration (with buffer)
            if self.cached_config_idx != self.current_config_idx:
                min_lon = self.current_data["gps_lon"].min()
                max_lon = self.current_data["gps_lon"].max()
                min_lat = self.current_data["gps_lat"].min()
                max_lat = self.current_data["gps_lat"].max()
                
                # Convert buffer from meters to approximate degrees
                buffer_deg = self.map_buffer
                
                self.bounds_lon = (min_lon - buffer_deg, max_lon + buffer_deg)
                self.bounds_lat = (min_lat - buffer_deg, max_lat + buffer_deg)
                
                # Convert bounds to mercator
                self.bounds_mercator_x = self.coord_transformer.gps_to_mercator_array(
                    np.array(self.bounds_lon), np.array([self.bounds_lat[0], self.bounds_lat[0]])
                )[0]
                self.bounds_mercator_y = self.coord_transformer.gps_to_mercator_array(
                    np.array([self.bounds_lon[0], self.bounds_lon[0]]), np.array(self.bounds_lat)
                )[1]
                
                self.cached_config_idx = self.current_config_idx
            
            bounds_x = self.bounds_mercator_x
            bounds_y = self.bounds_mercator_y
        
        # Set axis limits in mercator coordinates
        self.ax.set_xlim(bounds_x[0], bounds_x[1])
        self.ax.set_ylim(bounds_y[0], bounds_y[1])
        
        # Add basemap FIRST (so it's behind everything)
        if HAS_CONTEXTILY:
            try:
                # CRS 3857 is Web Mercator (the projection used by contextily)
                # reset_extent=False keeps our axis limits unchanged
                ctx.add_basemap(self.ax, crs="EPSG:3857", source=self.get_map_provider(), 
                               zoom='auto', zorder=0, reset_extent=False)
            except Exception as e:
                print(f"Warning: Could not load map tiles: {e}")
                import traceback
                traceback.print_exc()
                self.ax.set_facecolor('#e8e8e8')
                self.ax.grid(True, alpha=0.3)
        else:
            self.ax.set_facecolor('#e8e8e8')
            self.ax.grid(True, alpha=0.3)
        
        # Re-apply axis limits after basemap (in case it changed them)
        self.ax.set_xlim(bounds_x[0], bounds_x[1])
        self.ax.set_ylim(bounds_y[0], bounds_y[1])
        
        # Set equal aspect ratio so map is not distorted (top-down view)
        self.ax.set_aspect('equal', adjustable='datalim')
        
        # Draw trajectories if enabled
        if self.show_trajectory or self.show_future_trajectory:
            for track_id in self.tracks:
                track_data = self.current_data[self.current_data["trackId"] == track_id].copy()
                color = self.track_colors.get(track_id, '#95a5a6')
                
                if self.show_trajectory:
                    # Past trajectory
                    past = track_data[track_data["sample"] <= sample_num]
                    if len(past) > 1:
                        self.ax.plot(past["mercator_x"], past["mercator_y"], 
                                   color=color, alpha=0.6, linewidth=2, linestyle='-', zorder=10)
                
                if self.show_future_trajectory:
                    # Future trajectory
                    future = track_data[track_data["sample"] >= sample_num]
                    if len(future) > 1:
                        self.ax.plot(future["mercator_x"], future["mercator_y"], 
                                   color=color, alpha=0.3, linewidth=1.5, linestyle=':', zorder=10)
        
        # Draw current positions
        for _, row in sample_data.iterrows():
            track_id = int(row["trackId"])
            class_id = int(row["class_id"]) if pd.notna(row["class_id"]) else -1
            x_merc = row["mercator_x"]
            y_merc = row["mercator_y"]
            
            # Use class-based color if showing class, otherwise track-based
            if self.show_class:
                color = self.class_colors.get(class_id, '#95a5a6')
            else:
                color = self.track_colors.get(track_id, '#95a5a6')
            
            # Draw point
            self.ax.plot(x_merc, y_merc, 'o', color=color, markersize=14, 
                        markeredgecolor='white', markeredgewidth=2, zorder=30)
            
            # Build label
            label_parts = []
            if self.show_track_ids:
                label_parts.append(f"T{track_id}")
            if self.show_class:
                class_name = self.CLASS_ID_NAMES.get(class_id, f"?")
                label_parts.append(class_name)
            
            if label_parts:
                label = "\n".join(label_parts)
                # Calculate offset in mercator units (roughly 5 meters)
                offset = (bounds_x[1] - bounds_x[0]) * 0.02
                self.ax.annotate(label, (x_merc, y_merc), 
                               xytext=(x_merc + offset, y_merc + offset),
                               fontsize=8, ha='left', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, 
                                        edgecolor='white', linewidth=1),
                               color='white', fontweight='bold',
                               zorder=35)
        
        # Draw distance lines if enabled
        if self.show_distance and len(sample_data) >= 2:
            positions = sample_data[["x", "y", "mercator_x", "mercator_y"]].values
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    # Draw line in mercator
                    self.ax.plot([positions[i][2], positions[j][2]], 
                               [positions[i][3], positions[j][3]], 
                               'w-', linewidth=2, alpha=0.8, zorder=25)
                    self.ax.plot([positions[i][2], positions[j][2]], 
                               [positions[i][3], positions[j][3]], 
                               'k--', linewidth=1, alpha=0.8, zorder=26)
                    
                    # Calculate distance directly from world coordinates (already in meters)
                    distance = self._euclidean_distance(
                        positions[i][0], positions[i][1],
                        positions[j][0], positions[j][1]
                    )
                    
                    mid_x = (positions[i][2] + positions[j][2]) / 2
                    mid_y = (positions[i][3] + positions[j][3]) / 2
                    self.ax.text(mid_x, mid_y, f"{distance:.1f}m", 
                               fontsize=10, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                        alpha=0.9, edgecolor='black'),
                               zorder=40)
        
        # Get timestamp for current sample (use first row's timestamp)
        timestamp_str = ""
        if "timestamp" in sample_data.columns and not sample_data.empty:
            ts = sample_data["timestamp"].iloc[0]
            if pd.notna(ts) and str(ts).strip():
                timestamp_str = f" - Time: {ts}"
        
        # Set title and labels
        self.ax.set_title(f"Config {self.current_config_idx} - Sample {sample_num}{timestamp_str}", 
                         fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Longitude (Web Mercator)")
        self.ax.set_ylabel("Latitude (Web Mercator)")
        
        # Update distance info
        if len(sample_data) == 2:
            pos = sample_data[["x", "y"]].values
            # Use euclidean distance since world coords are in meters
            distance = self._euclidean_distance(pos[0][0], pos[0][1], pos[1][0], pos[1][1])
            self.info_label.setText(f"Distance: {distance:.2f}m")
        else:
            self.info_label.setText("")
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points (for world coordinates in meters)."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _haversine_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate distance between two GPS points in meters using Haversine formula."""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def update_frame_label(self):
        """Update sample counter label."""
        if self.samples:
            self.frame_label.setText(f"Sample: {self.current_sample + 1}/{len(self.samples)}")
    
    def toggle_play(self):
        """Toggle playback."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("⏸ Pause")
            self.timer.start(150)  # Update every 150ms
        else:
            self.play_btn.setText("▶ Play")
            self.timer.stop()
    
    def advance_sample(self):
        """Advance to next sample during playback."""
        if self.current_sample < len(self.samples) - 1:
            self.current_sample += self.playback_speed
            if self.current_sample >= len(self.samples):
                self.current_sample = len(self.samples) - 1
            self.frame_slider.setValue(self.current_sample)
            self.draw_frame()
            self.update_frame_label()
        else:
            # Loop back to start
            self.current_sample = 0
            self.frame_slider.setValue(0)
            self.draw_frame()
            self.update_frame_label()
    
    def prev_sample(self):
        """Go to previous sample."""
        if self.current_sample > 0:
            self.current_sample -= 1
            self.frame_slider.setValue(self.current_sample)
            self.draw_frame()
            self.update_frame_label()
    
    def next_sample(self):
        """Go to next sample."""
        if self.current_sample < len(self.samples) - 1:
            self.current_sample += 1
            self.frame_slider.setValue(self.current_sample)
            self.draw_frame()
            self.update_frame_label()
    
    def prev_config(self):
        """Go to previous configuration."""
        if self.is_playing:
            self.toggle_play()
        
        configs = self.config_info["config_ids"]
        current_idx_in_list = configs.index(self.current_config_idx)
        if current_idx_in_list > 0:
            new_config = configs[current_idx_in_list - 1]
            self.config_spinbox.setValue(new_config)
            self.load_config(new_config)
    
    def next_config(self):
        """Go to next configuration."""
        if self.is_playing:
            self.toggle_play()
        
        configs = self.config_info["config_ids"]
        current_idx_in_list = configs.index(self.current_config_idx)
        if current_idx_in_list < len(configs) - 1:
            new_config = configs[current_idx_in_list + 1]
            self.config_spinbox.setValue(new_config)
            self.load_config(new_config)
    
    def jump_to_config(self, config_idx: int):
        """Jump to a specific configuration."""
        if config_idx in self.config_info["config_ids"]:
            if self.is_playing:
                self.toggle_play()
            self.load_config(config_idx)
    
    def slider_changed(self, value):
        """Handle slider movement."""
        self.current_sample = value
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
    
    def toggle_class(self, state):
        """Toggle class display."""
        self.show_class = (state == Qt.Checked)
        self.draw_frame()
    
    def toggle_distance(self, state):
        """Toggle distance display."""
        self.show_distance = (state == Qt.Checked)
        self.draw_frame()
    
    def toggle_fixed_view(self, state):
        """Toggle fixed map view (same view for all configurations)."""
        self.use_fixed_view = (state == Qt.Checked)
        self.cached_config_idx = None  # Force bounds recalculation
        self.draw_frame()
    
    def change_map_style(self):
        """Change map tile provider."""
        self.cached_config_idx = None  # Force redraw
        self.draw_frame()
    
    def change_buffer(self, value):
        """Change map buffer size."""
        # Convert meters to approximate degrees (at ~50° latitude)
        self.map_buffer = value / 111000  # 111km per degree
        self._compute_global_bounds()  # Recompute global bounds with new buffer
        self.cached_config_idx = None  # Force bounds recalculation
        self.draw_frame()


def main():
    # ============================================================================
    # CONFIGURATION: Set your file paths here
    # ============================================================================
    
    # Path to your configuration CSV file (output from data_prep_pubapi.py)
    # This should be the file with class info (ending in _CL_)
    CONFIG_FILE = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\Test_data_Teledyne\Output\20260126_153511_F30_minmov1.0\C149_2OBJ_CL_pubapi_F30.csv"
    
    # ============================================================================
    
    # Check if file exists
    if not os.path.exists(CONFIG_FILE):
        print(f"Configuration file not found: {CONFIG_FILE}")
        print("\nPlease update CONFIG_FILE path in the script to point to your pubapi output.")
        print("Example: C:\\path\\to\\C5_CB_CL_pubapi_minF50.csv")
        
        # Try to find recent output files
        output_dir = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\Test_data_Teledyne\Output"
        if os.path.exists(output_dir):
            print(f"\nSearching for CSV files in: {output_dir}")
            csv_files = []
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    if f.endswith("_CL_pubapi_minF50.csv") or "_CL_" in f:
                        csv_files.append(os.path.join(root, f))
            
            if csv_files:
                print("\nFound configuration files:")
                for i, f in enumerate(csv_files[:10]):
                    print(f"  {i+1}. {f}")
                
                choice = input("\nEnter number to load (or press Enter to exit): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
                    CONFIG_FILE = csv_files[int(choice) - 1]
                else:
                    return
            else:
                print("No configuration files found. Run data_prep_pubapi.py first.")
                return
        else:
            return
    
    # Check dependencies
    if not HAS_CONTEXTILY:
        print("\n⚠️  Warning: contextily not installed. Map background will be plain.")
        print("   Install with: pip install contextily")
    
    if not HAS_PYPROJ:
        print("\n⚠️  Warning: pyproj not installed. Using approximate coordinate conversion.")
        print("   Install with: pip install pyproj")
    
    print(f"\n{'='*70}")
    print(f"Loading configuration file:")
    print(f"  {CONFIG_FILE}")
    print(f"{'='*70}\n")
    
    # Load data
    config_df = ConfigDataLoader.load_config_file(CONFIG_FILE)
    config_info = ConfigDataLoader.get_config_info(config_df)
    
    print(f"Loaded {config_info['num_configs']} configurations")
    print(f"Config IDs: {config_info['config_ids'][:10]}{'...' if len(config_info['config_ids']) > 10 else ''}")
    print(f"Samples per config: {list(config_info['samples_per_config'].values())[:5]}...")
    
    # Print coordinate bounds
    print(f"\nCoordinate bounds:")
    print(f"  Longitude: {config_df['x'].min():.6f} to {config_df['x'].max():.6f}")
    print(f"  Latitude:  {config_df['y'].min():.6f} to {config_df['y'].max():.6f}")
    print()
    
    # Create Qt application
    app = QApplication(sys.argv)
    visualizer = ConfigVisualizerPubapi(config_df, config_info)
    visualizer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
