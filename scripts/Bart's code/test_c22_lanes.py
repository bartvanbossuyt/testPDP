"""Test script to debug lane drawing for configuration c=22"""
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('voorbeeld.csv')

# Filter for c=22
c22 = df[df['c'] == 22]

print("=" * 60)
print("DATA FOR C=22")
print("=" * 60)
print(c22)
print()

# Check unique objects
objects = c22['o'].unique()
print(f"Number of objects: {len(objects)}")
print(f"Object IDs: {objects}")
print()

# Check speeds and directions for each object
for obj_id in objects:
    obj_data = c22[c22['o'] == obj_id].sort_values('t')
    if len(obj_data) >= 2:
        # Calculate speed
        first = obj_data.iloc[0]
        last = obj_data.iloc[-1]
        dx = last['x'] - first['x']
        dy = last['y'] - first['y']
        dt = last['t'] - first['t']
        
        dist = np.sqrt(dx**2 + dy**2)
        speed_m_per_frame = dist / dt if dt > 0 else 0
        speed_kmh = speed_m_per_frame * 3.6  # Assuming 1 frame = 1 second
        
        # Calculate direction
        direction = np.array([dx, dy])
        direction_norm = direction / (np.linalg.norm(direction) + 1e-10)
        
        print(f"Object {obj_id}:")
        print(f"  Speed: {speed_kmh:.1f} km/h")
        print(f"  Direction: {direction_norm}")
        print(f"  Start: ({first['x']:.2f}, {first['y']:.2f})")
        print(f"  End: ({last['x']:.2f}, {last['y']:.2f})")
        print()

# Check if vehicles are traveling in same direction
if len(objects) == 2:
    obj0_data = c22[c22['o'] == 0].sort_values('t')
    obj1_data = c22[c22['o'] == 1].sort_values('t')
    
    if len(obj0_data) >= 2 and len(obj1_data) >= 2:
        # Direction for object 0
        dx0 = obj0_data.iloc[-1]['x'] - obj0_data.iloc[0]['x']
        dy0 = obj0_data.iloc[-1]['y'] - obj0_data.iloc[0]['y']
        dir0 = np.array([dx0, dy0])
        dir0_norm = dir0 / (np.linalg.norm(dir0) + 1e-10)
        
        # Direction for object 1
        dx1 = obj1_data.iloc[-1]['x'] - obj1_data.iloc[0]['x']
        dy1 = obj1_data.iloc[-1]['y'] - obj1_data.iloc[0]['y']
        dir1 = np.array([dx1, dy1])
        dir1_norm = dir1 / (np.linalg.norm(dir1) + 1e-10)
        
        # Calculate angle between directions
        dot_product = np.dot(dir0_norm, dir1_norm)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        
        print(f"Angle between vehicles: {angle_deg:.1f}°")
        print(f"Same direction (< 45°): {angle_deg < 45.0}")
        print()

# Calculate lane count based on max speed
speeds = []
for obj_id in objects:
    obj_data = c22[c22['o'] == obj_id].sort_values('t')
    if len(obj_data) >= 2:
        first = obj_data.iloc[0]
        last = obj_data.iloc[-1]
        dx = last['x'] - first['x']
        dy = last['y'] - first['y']
        dt = last['t'] - first['t']
        dist = np.sqrt(dx**2 + dy**2)
        speed_kmh = (dist / dt if dt > 0 else 0) * 3.6
        speeds.append(speed_kmh)

max_speed = max(speeds) if speeds else 0.0
lane_count = 3 if max_speed > 100.0 else 2

print(f"Max speed: {max_speed:.1f} km/h")
print(f"Lane count: {lane_count}")
print()

# Calculate expected number of dashed lines
interior_count = max(0, lane_count - 1)
print(f"Expected dashed lines per vehicle (interior_count): {interior_count}")

# If different directions, it would create lines for each vehicle
if angle_deg >= 45.0:
    total_lines = interior_count * len(objects)
    print(f"ISSUE: Different directions detected!")
    print(f"Total dashed lines (INCORRECT): {total_lines}")
    print(f"Should be: {interior_count}")
else:
    print(f"Same direction - dashed lines: {interior_count}")
