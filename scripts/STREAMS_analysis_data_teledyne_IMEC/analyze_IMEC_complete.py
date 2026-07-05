"""
Complete analysis of ALL data sources in IMEC dataset.
"""

import pandas as pd
import numpy as np
import os
from glob import glob

# Base path
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "Data_IMEC", "tracked0001", "0001")

print("=" * 70)
print("COMPLETE IMEC DATA ANALYSIS")
print("=" * 70)

# 1. TRACKING DATA (already used)
print("\n" + "=" * 70)
print("1. TRACKING DATA (track_dataframes/tracks_df_1.csv)")
print("=" * 70)
track_file = os.path.join(base_path, "track_dataframes", "tracks_df_1.csv")
df_tracks = pd.read_csv(track_file)
print(f"   Rows: {len(df_tracks)}")
print(f"   Unique tracks: {df_tracks['ID'].nunique()}")
print(f"   Columns: {list(df_tracks.columns)}")
print(f"   Timestamps: {df_tracks['timestamp'].min()} to {df_tracks['timestamp'].max()}")


# 2. GROUND TRUTH LABELS (more accurate!)
print("\n" + "=" * 70)
print("2. GROUND TRUTH LABELS (Streams_labels/)")
print("   Format: X Y Z length width height rotation class occlusion trackID")
print("=" * 70)
labels_dir = os.path.join(base_path, "Streams_labels")
label_files = sorted(glob(os.path.join(labels_dir, "*.txt")))
print(f"   Number of frames: {len(label_files)}")

# Parse all ground truth labels
gt_rows = []
for lf in label_files:
    timestamp = int(os.path.basename(lf).split('_')[1].replace('.txt', ''))
    with open(lf, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 10:
                gt_rows.append({
                    'timestamp': timestamp,
                    'x': float(parts[0]),
                    'y': float(parts[1]),
                    'z': float(parts[2]),
                    'length': float(parts[3]),
                    'width': float(parts[4]),
                    'height': float(parts[5]),
                    'rotation': float(parts[6]),
                    'class': parts[7],
                    'occluded': parts[8] == 'True',
                    'track_id': int(parts[9])
                })

df_gt = pd.DataFrame(gt_rows)
print(f"   Total detections: {len(df_gt)}")
print(f"   Unique tracks: {df_gt['track_id'].nunique()}")
print(f"   Classes: {df_gt['class'].value_counts().to_dict()}")
print(f"   Occluded: {df_gt['occluded'].sum()} ({100*df_gt['occluded'].mean():.1f}%)")


# 3. POINTRCNN 3D PREDICTIONS (full point cloud)
print("\n" + "=" * 70)
print("3. POINTRCNN 3D PREDICTIONS (full_pcd_predictions/pointrcnn/)")
print("   Format: X Y Z l w h rot confidence class")
print("=" * 70)
prcnn_dir = os.path.join(base_path, "full_pcd_predictions", "pointrcnn")
prcnn_files = sorted(glob(os.path.join(prcnn_dir, "*.txt")))
print(f"   Number of frames: {len(prcnn_files)}")

# Count detections
total_det = 0
class_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # Car, Pedestrian, Cyclist, Van
for pf in prcnn_files:
    with open(pf, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                total_det += 1
                cls = int(parts[8])
                class_counts[cls] = class_counts.get(cls, 0) + 1

print(f"   Total detections: {total_det}")
print(f"   Classes: Car={class_counts.get(1,0)}, Pedestrian={class_counts.get(2,0)}, Cyclist={class_counts.get(3,0)}, Van={class_counts.get(4,0)}")


# 4. YOLO 2D PREDICTIONS
print("\n" + "=" * 70)
print("4. YOLO 2D PREDICTIONS (Frames_2_predictions/)")
print("   Format: L T R B confidence class")
print("=" * 70)
yolo_base = os.path.join(base_path, "Frames_2_predictions")
for model in ['yolo11n', 'yolo11m', 'yolo11x']:
    model_dir = os.path.join(yolo_base, model)
    if os.path.exists(model_dir):
        iou_dirs = os.listdir(model_dir)
        print(f"   {model}: {len(iou_dirs)} IoU variants")


# 5. FOREGROUND POINT CLOUDS
print("\n" + "=" * 70)
print("5. FOREGROUND POINT CLOUDS (fg_PointCloud_1/)")
print("=" * 70)
fg_dir = os.path.join(base_path, "fg_PointCloud_1")
fg_files = sorted(glob(os.path.join(fg_dir, "*.npy")))
print(f"   Number of frames: {len(fg_files)}")
if fg_files:
    sample = np.load(fg_files[0])
    print(f"   Sample shape: {sample.shape}")
    print(f"   Sample dtype: {sample.dtype}")


# 6. RAW POINT CLOUDS
print("\n" + "=" * 70)
print("6. RAW POINT CLOUDS (PointCloud_1/)")
print("=" * 70)
pcd_dir = os.path.join(base_path, "PointCloud_1")
pcd_files = sorted(glob(os.path.join(pcd_dir, "*.pcd")))
print(f"   Number of frames: {len(pcd_files)}")


# 7. CAMERA IMAGES
print("\n" + "=" * 70)
print("7. CAMERA IMAGES (Frames_2/)")
print("=" * 70)
img_dir = os.path.join(base_path, "Frames_2")
img_files = sorted(glob(os.path.join(img_dir, "*.png")))
print(f"   Number of frames: {len(img_files)}")


# COMPARE GROUND TRUTH vs TRACKER
print("\n" + "=" * 70)
print("COMPARISON: GROUND TRUTH vs TRACKER")
print("=" * 70)
print(f"   GT unique tracks: {df_gt['track_id'].nunique()} (IDs: {sorted(df_gt['track_id'].unique())})")
print(f"   Tracker unique tracks: {df_tracks['ID'].nunique()} (IDs: {sorted(df_tracks['ID'].unique())})")
print(f"   GT total detections: {len(df_gt)}")
print(f"   Tracker total detections: {len(df_tracks)}")


# SAVE GROUND TRUTH TO PDP FORMAT
print("\n" + "=" * 70)
print("EXPORTING GROUND TRUTH TO PDP FORMAT")
print("=" * 70)

# Map timestamps to sequential tstID
unique_ts = sorted(df_gt['timestamp'].unique())
ts_to_tstid = {ts: idx for idx, ts in enumerate(unique_ts)}

# Map track_id to sequential poiID
unique_tracks = sorted(df_gt['track_id'].unique())
trackid_to_poiid = {tid: idx for idx, tid in enumerate(unique_tracks)}

# Create PDP format
pdp_gt = pd.DataFrame({
    'conID': 0,
    'tstID': df_gt['timestamp'].map(ts_to_tstid),
    'poiID': df_gt['track_id'].map(trackid_to_poiid),
    'x': df_gt['x'],
    'y': df_gt['y'],
    'class': df_gt['class'].map({'Pedestrian': 0, 'Cyclist': 1, 'Car': 2, 'Van': 3})
})

pdp_gt = pdp_gt.sort_values(['conID', 'tstID', 'poiID']).reset_index(drop=True)

output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMEC_GroundTruth_PDP.csv")
pdp_gt.to_csv(output_file, index=False, header=False)

print(f"   Saved to: {output_file}")
print(f"   Configurations: 1")
print(f"   Timesteps: {pdp_gt['tstID'].nunique()}")
print(f"   Objects: {pdp_gt['poiID'].nunique()}")
print(f"   Total rows: {len(pdp_gt)}")

# Also save with full 3D info
df_gt_export = df_gt.copy()
df_gt_export['tstID'] = df_gt_export['timestamp'].map(ts_to_tstid)
df_gt_export['poiID'] = df_gt_export['track_id'].map(trackid_to_poiid)
df_gt_export.to_csv(output_file.replace('.csv', '_full.csv'), index=False)
print(f"   Full version (with z, dims, rotation): {output_file.replace('.csv', '_full.csv')}")

print("\n" + "=" * 70)
print("SUMMARY: DATA SOURCES AVAILABLE")
print("=" * 70)
print("""
| Source                | What it contains                          | Recommended for |
|-----------------------|-------------------------------------------|-----------------|
| Streams_labels        | Ground truth (manual annotations)         | PDP (most accurate) |
| track_dataframes      | Tracker output (x,y,vx,vy filtered)       | PDP (with dynamics) |
| full_pcd_predictions  | PointRCNN 3D detections                   | Raw analysis |
| fg_predictions        | PointRCNN on foreground only              | Raw analysis |
| Frames_2_predictions  | YOLO 2D detections (3 models, 5 IoUs)     | 2D analysis |
| PointCloud_1          | Raw LiDAR point clouds (.pcd)             | 3D visualization |
| fg_PointCloud_1       | Foreground point clouds (.npy)            | 3D visualization |
| Frames_2              | Camera images (anonymized)                | Visual analysis |

For PDP analysis, recommend using:
  1. IMEC_GroundTruth_PDP.csv - Uses GT labels (most accurate positions)
  2. IMEC_PDP_format_interpolated.csv - Uses tracker (has velocity, interpolated)
""")
