"""Quick map regeneration with adjusted heading."""
import pandas as pd
import numpy as np
import os
import folium
from folium.plugins import AntPath

ORIGIN_LAT = 51.045101
ORIGIN_LON = 3.713908
HEADING_DEG = 49.4  # 34.4° + 15° more

METERS_PER_DEG_LAT = 111320
METERS_PER_DEG_LON = 111320 * np.cos(np.radians(ORIGIN_LAT))

def local_to_gps(x, y):
    heading_rad = np.radians(HEADING_DEG)
    north_offset = x * np.cos(heading_rad) - y * np.sin(heading_rad)
    east_offset = x * np.sin(heading_rad) + y * np.cos(heading_rad)
    lat = ORIGIN_LAT + (north_offset / METERS_PER_DEG_LAT)
    lon = ORIGIN_LON + (east_offset / METERS_PER_DEG_LON)
    return lat, lon

script_dir = os.path.dirname(os.path.abspath(__file__))
gt_file = os.path.join(script_dir, "IMEC_GroundTruth_PDP_full.csv")
df = pd.read_csv(gt_file)

m = folium.Map(location=[ORIGIN_LAT, ORIGIN_LON], zoom_start=18, tiles='OpenStreetMap')
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satellite', overlay=False
).add_to(m)
folium.LayerControl().add_to(m)

folium.Marker(
    [ORIGIN_LAT, ORIGIN_LON], 
    popup='LiDAR Sensor',
    icon=folium.Icon(color='red', icon='video-camera', prefix='fa')
).add_to(m)

colors = {'Pedestrian': 'red', 'Cyclist': 'green', 'Vehicle': 'blue'}

for track_id in sorted(df['track_id'].unique()):
    track = df[df['track_id'] == track_id].sort_values('timestamp')
    obj_class = track['class'].iloc[0]
    color = colors.get(obj_class, 'gray')
    
    coords = []
    for _, row in track.iterrows():
        lat, lon = local_to_gps(row['x'], row['y'])
        coords.append([lat, lon])
    
    if len(coords) > 1:
        AntPath(coords, color=color, weight=4, opacity=0.8,
                tooltip=f'{obj_class} - Track {track_id}').add_to(m)
        folium.CircleMarker(coords[0], radius=8, color=color, fill=True,
                            popup=f'{obj_class} {track_id} - Start').add_to(m)
        folium.CircleMarker(coords[-1], radius=8, color=color, fill=True,
                            fill_color='white', popup=f'{obj_class} {track_id} - End').add_to(m)

output = os.path.join(script_dir, "IMEC_visualizations", "gt_map_interactive.html")
m.save(output)
print(f"Saved with heading {HEADING_DEG}° to: {output}")
