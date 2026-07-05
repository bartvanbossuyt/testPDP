import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Motorcycle",
    3: "MotorcyclePlus",
    4: "VRU",
    5: "Car",
    6: "SmallVehicle",
    7: "Van",
    8: "LargeVehicle",
    9: "Vehicle",
    10: "SmallTruck",
    11: "MiddleTruck",
    12: "LargeTruck",
    13: "Truck",
    14: "Bus",
    15: "DoubleBus",
    16: "CarTrailer",
    17: "Box",
    18: "Cone",
    19: "ObjectOfInterest",
    20: "CarAndTrailer",
    21: "VanAndTrailer",
    22: "TruckTrailer",
    23: "TruckHead",
    24: "TruckAndTrailer",
    25: "Scooter",
    26: "MiddleTruckSmall",
    27: "MiddleTruckLarge"
}

# Find all CSV files in TrackData_csv folder
trackdata_csv_folder = Path(__file__).parent / 'TrackData_csv'
csv_files = list(trackdata_csv_folder.rglob('*.csv'))

print(f"Found {len(csv_files)} CSV files in TrackData_csv folder")
print("=" * 80)

# Load all data into one dataframe with scenario info
all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Extract scenario from path
    parts = csv_file.relative_to(trackdata_csv_folder).parts
    df['scenario'] = parts[0]
    df['file_name'] = csv_file.stem
    df['source_file'] = str(csv_file.relative_to(trackdata_csv_folder))
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

print(f"\nOVERALL DATA SUMMARY")
print("-" * 40)
print(f"Total data points (rows): {len(combined_df):,}")
print(f"Total unique track IDs: {combined_df['id'].nunique()}")
print(f"Columns: {list(combined_df.columns)}")

# ============================================================================
# OBJECT CLASS ANALYSIS
# ============================================================================
print(f"\n\nOBJECT CLASS DISTRIBUTION")
print("=" * 80)

# Get unique tracks with their class (one row per track)
unique_tracks = combined_df.groupby(['scenario', 'file_name', 'id']).agg({
    'class_id': 'first',
    'world_x': ['min', 'max', 'mean'],
    'world_y': ['min', 'max', 'mean'],
    'speed': 'mean',
    'object_time': ['min', 'max']
}).reset_index()

# Flatten column names
unique_tracks.columns = ['scenario', 'file_name', 'id', 'class_id', 
                         'world_x_min', 'world_x_max', 'world_x_mean',
                         'world_y_min', 'world_y_max', 'world_y_mean',
                         'avg_speed', 'time_start', 'time_end']

unique_tracks['class_name'] = unique_tracks['class_id'].map(CLASS_NAMES).fillna('unknown')
unique_tracks['duration'] = unique_tracks['time_end'] - unique_tracks['time_start']

# Overall class counts
print("\nTotal unique objects by class (across all scenarios):")
class_counts = unique_tracks['class_name'].value_counts()
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

# Per scenario breakdown
print("\nObjects per scenario:")
scenario_class = unique_tracks.groupby(['scenario', 'class_name']).size().unstack(fill_value=0)
print(scenario_class.to_string())

# ============================================================================
# SPATIAL ANALYSIS
# ============================================================================
print(f"\n\nSPATIAL ANALYSIS")
print("=" * 80)

print("\nWorld coordinate ranges (all data):")
print(f"  world_x: {combined_df['world_x'].min():.2f} to {combined_df['world_x'].max():.2f}")
print(f"  world_y: {combined_df['world_y'].min():.2f} to {combined_df['world_y'].max():.2f}")
print(f"  world_z: {combined_df['world_z'].min():.2f} to {combined_df['world_z'].max():.2f}")

print("\nCoordinate ranges per scenario:")
for scenario in sorted(combined_df['scenario'].unique()):
    scenario_data = combined_df[combined_df['scenario'] == scenario]
    print(f"\n  {scenario}:")
    print(f"    world_x: {scenario_data['world_x'].min():.2f} to {scenario_data['world_x'].max():.2f}")
    print(f"    world_y: {scenario_data['world_y'].min():.2f} to {scenario_data['world_y'].max():.2f}")

# Check if GPS data is available
gps_valid = combined_df[(combined_df['gps_latitude'] != 0) | (combined_df['gps_longitude'] != 0)]
print(f"\nGPS data available: {len(gps_valid)} rows with non-zero GPS coordinates")

# ============================================================================
# ARE OBJECTS IN THE SAME SPACE?
# ============================================================================
print(f"\n\nSPATIAL OVERLAP ANALYSIS")
print("=" * 80)

# Calculate bounding boxes per scenario
print("\nBounding box per scenario (world coordinates):")
bounding_boxes = {}
for scenario in sorted(combined_df['scenario'].unique()):
    scenario_data = combined_df[combined_df['scenario'] == scenario]
    bbox = {
        'x_min': scenario_data['world_x'].min(),
        'x_max': scenario_data['world_x'].max(),
        'y_min': scenario_data['world_y'].min(),
        'y_max': scenario_data['world_y'].max()
    }
    bounding_boxes[scenario] = bbox
    area = (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])
    print(f"  {scenario}: X[{bbox['x_min']:.1f}, {bbox['x_max']:.1f}], Y[{bbox['y_min']:.1f}, {bbox['y_max']:.1f}] (area: {area:.1f} m²)")

# Check overlap between scenarios
print("\nChecking if scenarios share the same space:")
scenarios = sorted(bounding_boxes.keys())
for i, s1 in enumerate(scenarios):
    for s2 in scenarios[i+1:]:
        b1, b2 = bounding_boxes[s1], bounding_boxes[s2]
        # Check for overlap
        x_overlap = max(0, min(b1['x_max'], b2['x_max']) - max(b1['x_min'], b2['x_min']))
        y_overlap = max(0, min(b1['y_max'], b2['y_max']) - max(b1['y_min'], b2['y_min']))
        overlap_area = x_overlap * y_overlap
        if overlap_area > 0:
            print(f"  {s1} & {s2}: OVERLAP (area: {overlap_area:.1f} m²)")
        else:
            print(f"  {s1} & {s2}: NO OVERLAP")

# ============================================================================
# SPEED & MOVEMENT ANALYSIS
# ============================================================================
print(f"\n\nSPEED & MOVEMENT ANALYSIS")
print("=" * 80)

print("\nAverage speed by class:")
speed_by_class = unique_tracks.groupby('class_name')['avg_speed'].agg(['mean', 'min', 'max', 'std'])
for class_name, row in speed_by_class.iterrows():
    print(f"  {class_name}: avg={row['mean']:.2f} m/s, range=[{row['min']:.2f}, {row['max']:.2f}]")

print("\nTrack duration statistics by class:")
duration_by_class = unique_tracks.groupby('class_name')['duration'].agg(['mean', 'min', 'max'])
for class_name, row in duration_by_class.iterrows():
    print(f"  {class_name}: avg={row['mean']:.1f}s, range=[{row['min']:.1f}s, {row['max']:.1f}s]")

# ============================================================================
# PER-FILE BREAKDOWN
# ============================================================================
print(f"\n\nDETAILED FILE BREAKDOWN")
print("=" * 80)

file_summary = unique_tracks.groupby(['scenario', 'file_name']).agg({
    'id': 'count',
    'class_name': lambda x: dict(x.value_counts())
}).reset_index()
file_summary.columns = ['scenario', 'file_name', 'total_objects', 'class_breakdown']

for _, row in file_summary.iterrows():
    print(f"\n{row['scenario']}/{row['file_name']}:")
    print(f"  Total objects: {row['total_objects']}")
    print(f"  Classes: {row['class_breakdown']}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print(f"\n\nGENERATING VISUALIZATIONS...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scatter plot of all object positions colored by scenario
ax1 = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
for scenario, color in zip(scenarios, colors):
    scenario_tracks = unique_tracks[unique_tracks['scenario'] == scenario]
    ax1.scatter(scenario_tracks['world_x_mean'], scenario_tracks['world_y_mean'], 
                c=[color], label=scenario, alpha=0.7, s=50)
ax1.set_xlabel('World X (m)')
ax1.set_ylabel('World Y (m)')
ax1.set_title('Object Positions by Scenario')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Bar chart of object classes
ax2 = axes[0, 1]
class_counts.plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_xlabel('Object Class')
ax2.set_ylabel('Count')
ax2.set_title('Object Class Distribution')
ax2.tick_params(axis='x', rotation=45)

# 3. Scatter plot colored by class
ax3 = axes[1, 0]
class_colors = {'unknown': 'gray', 'person': 'red', 'bicycle': 'green', 
                'car': 'blue', 'truck': 'orange', 'motorcycle': 'purple', 
                'bus': 'brown', 'trailer': 'pink'}
for class_name in unique_tracks['class_name'].unique():
    class_tracks = unique_tracks[unique_tracks['class_name'] == class_name]
    ax3.scatter(class_tracks['world_x_mean'], class_tracks['world_y_mean'], 
                c=class_colors.get(class_name, 'gray'), label=class_name, alpha=0.7, s=50)
ax3.set_xlabel('World X (m)')
ax3.set_ylabel('World Y (m)')
ax3.set_title('Object Positions by Class')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Objects per scenario
ax4 = axes[1, 1]
scenario_counts = unique_tracks.groupby('scenario').size()
scenario_counts.plot(kind='bar', ax=ax4, color='coral')
ax4.set_xlabel('Scenario')
ax4.set_ylabel('Number of Objects')
ax4.set_title('Objects per Scenario')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'trackdata_analysis.png', dpi=150)
print("Saved visualization to: trackdata_analysis.png")

plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
