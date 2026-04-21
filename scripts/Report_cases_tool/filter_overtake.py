"""
Extract specific timestamps from overtake_event_extended_plus20_both_sides.csv and renumber them.
"""
import pandas as pd
from pathlib import Path

# Load data
csv_path = Path(__file__).parent / "overtake_event_extended_plus20_both_sides.csv"
df = pd.read_csv(csv_path, header=None, names=["conID", "tstID", "poiID", "x", "y"])

# Define timestamp mapping
timestamp_mapping = {
    0: 0,
    34: 1,
    67: 2,
    183: 3,
    213: 4,
    249: 5
}

# Filter to only these timestamps
target_timestamps = list(timestamp_mapping.keys())
df_filtered = df[df['tstID'].isin(target_timestamps)].copy()

# Renumber the timestamps
df_filtered['tstID'] = df_filtered['tstID'].map(timestamp_mapping)

# Sort by conID, new tstID, and poiID
df_filtered = df_filtered.sort_values(['conID', 'tstID', 'poiID']).reset_index(drop=True)

# Save without header
output_path = Path(__file__).parent / "overtake_event_filtered_timestamps.csv"
df_filtered.to_csv(output_path, header=False, index=False)

print(f"✓ Filtered CSV created: overtake_event_filtered_timestamps.csv")
print(f"  Original rows: {len(df)}")
print(f"  Filtered rows: {len(df_filtered)}")
print(f"\n  Timestamps extracted and renumbered:")
for orig, new in timestamp_mapping.items():
    count = (df_filtered['tstID'] == new).sum()
    print(f"    {orig} → {new}  ({count} points)")
print(f"\n  Configurations included: {df_filtered['conID'].unique().tolist()}")
