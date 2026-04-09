"""
Create interactive map with heading rotation control and enhanced zoom.
"""
import pandas as pd
import numpy as np
import os
import json

ORIGIN_LAT = 51.045101
ORIGIN_LON = 3.713908

script_dir = os.path.dirname(os.path.abspath(__file__))
gt_file = os.path.join(script_dir, "IMEC_GroundTruth_PDP_full.csv")
df = pd.read_csv(gt_file)

# Prepare track data as JSON for JavaScript
tracks_data = []
for track_id in sorted(df['track_id'].unique()):
    track = df[df['track_id'] == track_id].sort_values('timestamp')
    obj_class = track['class'].iloc[0]
    points = [[float(row['x']), float(row['y'])] for _, row in track.iterrows()]
    tracks_data.append({
        'id': int(track_id),
        'class': obj_class,
        'points': points
    })

tracks_json = json.dumps(tracks_data)

html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>IMEC Ground Truth - Interactive Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ position: absolute; top: 60px; bottom: 0; width: 100%; }}
        #controls {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: #333;
            color: white;
            display: flex;
            align-items: center;
            padding: 0 20px;
            z-index: 1000;
            gap: 20px;
        }}
        #controls label {{ font-weight: bold; }}
        #headingSlider {{ width: 200px; }}
        #headingValue {{ 
            background: #555; 
            padding: 5px 15px; 
            border-radius: 5px;
            min-width: 60px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }}
        .legend {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid #333;
        }}
        #info {{
            background: #444;
            padding: 5px 15px;
            border-radius: 5px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="controls">
        <label>Heading:</label>
        <input type="range" id="headingSlider" min="0" max="360" value="49" step="1">
        <span id="headingValue">49°</span>
        <label>Zoom:</label>
        <button onclick="map.zoomIn()">+</button>
        <button onclick="map.zoomOut()">-</button>
        <span id="zoomLevel">18</span>
        <div id="info">
            Sensor: {ORIGIN_LAT:.6f}°N, {ORIGIN_LON:.6f}°E | 
            Tracks: 10 | Drag slider to rotate
        </div>
    </div>
    <div id="map"></div>

    <script>
        // Constants
        const ORIGIN_LAT = {ORIGIN_LAT};
        const ORIGIN_LON = {ORIGIN_LON};
        const METERS_PER_DEG_LAT = 111320;
        const METERS_PER_DEG_LON = 111320 * Math.cos(ORIGIN_LAT * Math.PI / 180);
        
        // Track data
        const tracksData = {tracks_json};
        
        // Class colors
        const classColors = {{
            'Pedestrian': '#FF4444',
            'Cyclist': '#00CC88',
            'Vehicle': '#4488FF'
        }};
        
        // Initialize map with higher max zoom
        const map = L.map('map', {{
            center: [ORIGIN_LAT, ORIGIN_LON],
            zoom: 18,
            maxZoom: 22
        }});
        
        // Add tile layers
        const osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 22,
            maxNativeZoom: 19,
            attribution: '© OpenStreetMap'
        }});
        
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 22,
            maxNativeZoom: 19,
            attribution: '© Esri'
        }});
        
        osmLayer.addTo(map);
        
        L.control.layers({{
            'OpenStreetMap': osmLayer,
            'Satellite': satelliteLayer
        }}).addTo(map);
        
        // Update zoom display
        map.on('zoomend', function() {{
            document.getElementById('zoomLevel').textContent = map.getZoom();
        }});
        
        // Legend
        const legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <div class="legend-item"><div class="legend-color" style="background: #FF4444"></div>Pedestrian</div>
                <div class="legend-item"><div class="legend-color" style="background: #00CC88"></div>Cyclist</div>
                <div class="legend-item"><div class="legend-color" style="background: #4488FF"></div>Vehicle</div>
                <div class="legend-item"><div class="legend-color" style="background: red"></div>Sensor</div>
            `;
            return div;
        }};
        legend.addTo(map);
        
        // Sensor marker
        const sensorMarker = L.circleMarker([ORIGIN_LAT, ORIGIN_LON], {{
            radius: 10,
            color: 'white',
            fillColor: 'red',
            fillOpacity: 1,
            weight: 3
        }}).addTo(map).bindPopup('LiDAR Sensor');
        
        // Track layers
        let trackLayers = [];
        
        function localToGPS(x, y, headingDeg) {{
            const headingRad = headingDeg * Math.PI / 180;
            const northOffset = x * Math.cos(headingRad) - y * Math.sin(headingRad);
            const eastOffset = x * Math.sin(headingRad) + y * Math.cos(headingRad);
            const lat = ORIGIN_LAT + (northOffset / METERS_PER_DEG_LAT);
            const lon = ORIGIN_LON + (eastOffset / METERS_PER_DEG_LON);
            return [lat, lon];
        }}
        
        function updateTracks(headingDeg) {{
            // Remove existing track layers
            trackLayers.forEach(layer => map.removeLayer(layer));
            trackLayers = [];
            
            // Draw new tracks
            tracksData.forEach(track => {{
                const color = classColors[track.class] || 'gray';
                const coords = track.points.map(p => localToGPS(p[0], p[1], headingDeg));
                
                // Trajectory line
                const polyline = L.polyline(coords, {{
                    color: color,
                    weight: 4,
                    opacity: 0.8
                }}).addTo(map);
                polyline.bindTooltip(`${{track.class}} - Track ${{track.id}}`);
                trackLayers.push(polyline);
                
                // Start marker
                const startMarker = L.circleMarker(coords[0], {{
                    radius: 8,
                    color: 'black',
                    fillColor: color,
                    fillOpacity: 1,
                    weight: 2
                }}).addTo(map).bindPopup(`${{track.class}} ${{track.id}} - Start`);
                trackLayers.push(startMarker);
                
                // End marker
                const endMarker = L.circleMarker(coords[coords.length - 1], {{
                    radius: 8,
                    color: color,
                    fillColor: 'white',
                    fillOpacity: 1,
                    weight: 3
                }}).addTo(map).bindPopup(`${{track.class}} ${{track.id}} - End`);
                trackLayers.push(endMarker);
            }});
        }}
        
        // Initial render
        updateTracks(49);
        
        // Slider control
        const slider = document.getElementById('headingSlider');
        const valueDisplay = document.getElementById('headingValue');
        
        slider.addEventListener('input', function() {{
            const heading = parseInt(this.value);
            valueDisplay.textContent = heading + '°';
            updateTracks(heading);
        }});
        
        // Keyboard controls
        document.addEventListener('keydown', function(e) {{
            let heading = parseInt(slider.value);
            if (e.key === 'ArrowLeft') {{
                heading = (heading - 1 + 360) % 360;
            }} else if (e.key === 'ArrowRight') {{
                heading = (heading + 1) % 360;
            }} else if (e.key === 'ArrowUp') {{
                map.zoomIn();
            }} else if (e.key === 'ArrowDown') {{
                map.zoomOut();
            }}
            slider.value = heading;
            valueDisplay.textContent = heading + '°';
            updateTracks(heading);
        }});
    </script>
</body>
</html>
'''

output_path = os.path.join(script_dir, "IMEC_visualizations", "gt_map_rotatable.html")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Saved interactive rotatable map to: {output_path}")
print("Controls:")
print("  - Slider: rotate heading 0-360°")
print("  - Arrow keys: Left/Right to rotate, Up/Down to zoom")
print("  - Layer control: toggle OSM/Satellite")
