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

trackdata_csv_folder = Path(r"C:\Users\oliverme\OneDrive - UGent\Documents\pythonProject1\TrackData_csv")
csv_files = list(trackdata_csv_folder.rglob("*.csv"))

all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

print("CLASS ID ANALYSIS WITH OBJECT DIMENSIONS:")
print("=" * 70)

for class_id in sorted(combined_df["class_id"].unique()):
    class_data = combined_df[combined_df["class_id"] == class_id]
    class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
    print(f"\nClass ID: {class_id} - {class_name}")
    print(f"  Count: {len(class_data)} detections")
    print(f"  Dimensions (meters):")
    print(f"    Height: {class_data['world_height'].mean():.2f} (range: {class_data['world_height'].min():.2f} - {class_data['world_height'].max():.2f})")
    print(f"    Length: {class_data['world_length'].mean():.2f} (range: {class_data['world_length'].min():.2f} - {class_data['world_length'].max():.2f})")
    print(f"    Width:  {class_data['world_width'].mean():.2f} (range: {class_data['world_width'].min():.2f} - {class_data['world_width'].max():.2f})")
    print(f"  Avg Speed: {class_data['speed'].mean():.2f} m/s ({class_data['speed'].mean() * 3.6:.1f} km/h)")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE:")
print("=" * 70)
print(f"{'Class ID':<10} {'Class Name':<20} {'Detections':<12} {'Tracks':<10}")
print("-" * 52)

for class_id in sorted(combined_df["class_id"].unique()):
    class_data = combined_df[combined_df["class_id"] == class_id]
    class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
    n_detections = len(class_data)
    # Count unique tracks (need to handle per-file uniqueness)
    print(f"{class_id:<10} {class_name:<20} {n_detections:<12}")
