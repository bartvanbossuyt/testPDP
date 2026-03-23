import pandas as pd
from pathlib import Path

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Motorcycle",
    5: "Car",
    7: "Van",
    10: "SmallTruck",
    12: "LargeTruck",
    14: "Bus",
    20: "CarAndTrailer",
    21: "VanAndTrailer",
    24: "TruckAndTrailer",
    25: "Scooter",
    26: "Unknown"
}

def check_class_changes(csv_file):
    """Check for class_id changes in a single CSV file."""
    df = pd.read_csv(csv_file)
    
    # Check for each ID if class_id changes
    class_changes = df.groupby('id')['class_id'].nunique()
    changing_ids = class_changes[class_changes > 1]
    
    return df['id'].nunique(), changing_ids, df

# Find all CSV files in TrackData_csv folder
trackdata_csv_folder = Path(r'C:\Users\oliverme\OneDrive - UGent\Documents\pythonProject1\TrackData_csv')
csv_files = list(trackdata_csv_folder.rglob('*.csv'))

print(f"Found {len(csv_files)} CSV files in TrackData_csv folder\n")
print("=" * 80)

total_tracks = 0
total_changing = 0

for csv_file in sorted(csv_files):
    relative_path = csv_file.relative_to(trackdata_csv_folder)
    print(f"\n📁 {relative_path}")
    print("-" * 40)
    
    unique_ids, changing_ids, df = check_class_changes(csv_file)
    total_tracks += unique_ids
    total_changing += len(changing_ids)
    
    print(f"Total unique track IDs: {unique_ids}")
    print(f"Track IDs with changing class_id: {len(changing_ids)}")
    
    if len(changing_ids) > 0:
        print("IDs that change classification:")
        for track_id in changing_ids.index:
            track_data = df[df['id'] == track_id]
            classes = track_data['class_id'].unique()
            counts = track_data.groupby('class_id').size()
            class_names = [CLASS_NAMES.get(c, f"class_{c}") for c in classes]
            print(f"  ID {track_id}: {class_names} -> counts: {dict(counts)}")
    else:
        print("✅ All IDs have consistent class_id")

print("\n" + "=" * 80)
print(f"\n📊 SUMMARY:")
print(f"Total files processed: {len(csv_files)}")
print(f"Total unique track IDs across all files: {total_tracks}")
print(f"Total track IDs with changing class_id: {total_changing}")
