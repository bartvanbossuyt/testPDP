"""
Visualize object detections from multiple scenarios on a single interactive map.

This script reads data from specified scenarios, converts local 'world' coordinates
to real-world GPS coordinates based on the camera's known location, and plots
everything on an interactive map using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --- Configuration ---

# GPS locations for each camera, provided by the user.
# The script maps CSV filenames to these locations.
CAMERA_LOCATIONS = {
    # Main pole cameras
    "Pole_Front": {"lat": 50.811406, "lon": 3.231778, "heading": 0},
    "Pole_left": {"lat": 50.811406, "lon": 3.231778, "heading": 0},
    "Pole_right": {"lat": 50.811406, "lon": 3.231778, "heading": 0},
    # Individual cameras
    "10_167_222_123": {"lat": 50.811596, "lon": 3.231945, "heading": 0},
    "10_167_222_195": {"lat": 50.811706, "lon": 3.232191, "heading": 0},
    "10_167_222_226": {"lat": 50.811598, "lon": 3.231934, "heading": 0},
    "10_167_222_228": {"lat": 50.811607, "lon": 3.231969, "heading": 0},
}

# Scenarios to process
SCENARIOS_TO_PROCESS = ["V2_Scenarios/Scenario_1", "V2_Scenarios/Scenario_5"]

# Output file name
OUTPUT_FILE = "scenario_1_and_5_map.html"

# Class names for mapping class_id to a human-readable name
CLASS_NAMES = {
    0: "Person", 1: "Bicycle", 2: "Motorcycle", 5: "Car", 7: "Van",
    10: "SmallTruck", 12: "LargeTruck", 14: "Bus", 25: "Scooter"
}

# --- End Configuration ---

def local_to_gps(x, y, origin_lat, origin_lon, heading_deg=0):
    """
    Convert local coordinates (x=forward, y=left) to GPS coordinates.
    Adapted from visualize_IMEC_map.py.
    """
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(origin_lat))

    heading_rad = np.radians(heading_deg)
    
    # Convert to North/East offsets
    north_offset = x * np.cos(heading_rad) - y * np.sin(heading_rad)
    east_offset = x * np.sin(heading_rad) + y * np.cos(heading_rad)
    
    # Convert meters to degrees
    lat = origin_lat + (north_offset / meters_per_deg_lat)
    lon = origin_lon + (east_offset / meters_per_deg_lon)
    
    return lat, lon

def process_data():
    """
    Load data from CSVs, convert coordinates, and return a combined DataFrame.
    """
    base_path = Path(__file__).parent / 'trackdata_tools' / 'TrackData_csv'
    all_dfs = []

    print("Starting data processing...")
    for scenario_path in SCENARIOS_TO_PROCESS:
        folder = base_path / scenario_path
        print(f"Processing folder: {folder}")
        
        for csv_file in folder.glob('*.csv'):
            filename_stem = csv_file.stem.lower()
            camera_key = None
            for key in CAMERA_LOCATIONS:
                if key.lower() in filename_stem:
                    camera_key = key
                    break

            if camera_key not in CAMERA_LOCATIONS:
                print(f"  - Warning: No camera location found for {csv_file.name}. Skipping.")
                continue

            origin = CAMERA_LOCATIONS[camera_key]
            print(f"  - Processing file: {csv_file.name} (Camera: {camera_key})")
            
            df = pd.read_csv(csv_file)
            
            # Apply coordinate conversion
            gps_coords = df.apply(
                lambda row: local_to_gps(row['world_x'], row['world_y'], origin['lat'], origin['lon'], origin['heading']),
                axis=1
            )
            
            df[['gps_lat', 'gps_lon']] = pd.DataFrame(gps_coords.tolist(), index=df.index)
            df['camera_origin'] = camera_key
            all_dfs.append(df)

    if not all_dfs:
        print("No data processed. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['class_name'] = combined_df['class_id'].map(CLASS_NAMES).fillna('Unknown')
    print(f"Total data points loaded: {len(combined_df)}")

    # Create a DataFrame for camera markers
    camera_df = pd.DataFrame.from_dict(CAMERA_LOCATIONS, orient='index').reset_index()
    camera_df.rename(columns={'index': 'name'}, inplace=True)
    
    return combined_df, camera_df

def create_map(df, camera_df):
    """
    Create an interactive Plotly map and save it as an HTML file.
    """
    if df.empty:
        print("Cannot create map because no data was loaded.")
        return

    print("Creating interactive map...")
    
    # Center the map on the average coordinate
    map_center = {"lat": df['gps_lat'].mean(), "lon": df['gps_lon'].mean()}

    fig = px.scatter_map(
        df,
        lat="gps_lat",
        lon="gps_lon",
        color="class_name",
        hover_name="id",
        hover_data={"speed": True, "camera_origin": True, "object_time": True},
        map_style="open-street-map",
        zoom=18,
        center=map_center,
        title="Object Detections from Scenarios 1 & 5",
        height=900
    )

    # Add camera markers
    fig.add_trace(px.scatter_map(
        camera_df,
        lat="lat",
        lon="lon",
        hover_name="name",
        map_style="open-street-map"
    ).data[0])

    # Customize camera markers
    fig.data[-1].marker.symbol = 'star'
    fig.data[-1].marker.size = 15
    fig.data[-1].marker.color = 'red'
    fig.data[-1].name = 'Cameras'

    # Save to HTML
    output_path = Path(__file__).parent / OUTPUT_FILE
    fig.write_html(output_path)
    print(f"Successfully saved map to: {output_path}")

if __name__ == "__main__":
    data, cameras = process_data()
    create_map(data, cameras)
