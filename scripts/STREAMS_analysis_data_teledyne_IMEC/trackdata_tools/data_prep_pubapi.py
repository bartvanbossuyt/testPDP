#!/usr/bin/env python3
"""
Data preparation script for pubapi tracking data (camera-based object tracking).
Build configurations from pairs of moving objects with overlapping time windows.

Simplified version: finds any two unique moving objects that overlap for at least
TARGET_LENGTH frames, then outputs configurations with exactly TARGET_LENGTH frames each.

Outputs (auto-named):
- C{A}_2OBJ_CL_{D}_F{E}.csv   (with class column)
- C{A}_2OBJ_NC_{D}_F{E}.csv   (no class column)
- C{A}_2OBJ_{D}_F{E}_report.csv
"""

import os
import itertools
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# ===================== CONFIG (edit these) =====================
# Input/Output paths
INPUT_CSV = r"c:\Users\oliverme\OneDrive - UGent\Documents\pythonProject1\pubapi_data_20260119-105631.csv"
OUTDIR = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\Test_data_Teledyne\Output"

# Configuration length (exact number of frames per configuration)
TARGET_LENGTH = 30  # exact number of frames each configuration will have
MAX_CONFIGS = None  # optional cap on number of configurations, or None

# Movement filtering - minimum displacement in meters to be considered "moving"
MIN_MOVEMENT = 1.0  # tracks must move at least this much to be included

# Track identifier column
TRACK_ID_COLUMN = "id"
# ===============================================================

CLASS_ID_NAMES = {
    0: "person",
    1: "bicycle",
    2: "motorcycle",
    5: "car",
    7: "van",
    10: "smalltruck",
    12: "largetruck",
    14: "bus",
    20: "carandtrailer",
    21: "vanandtrailer",
    24: "truckandtrailer",
    25: "scooter",
    26: "unknown"
}


def read_pubapi_data(csv_path: str) -> pd.DataFrame:
    """Read pubapi CSV with optimized dtypes."""
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    df["object_time"] = pd.to_datetime(df["object_time"])
    
    # Ensure numeric columns are proper types
    for col in ["world_x", "world_y", "speed", "gps_latitude", "gps_longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    
    for col in ["class_id", TRACK_ID_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    return df


def assign_sample_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Assign sequential sample indices based on unique timestamps."""
    df = df.sort_values(["object_time", TRACK_ID_COLUMN]).copy()
    
    unique_times = df["object_time"].drop_duplicates().sort_values().reset_index(drop=True)
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    df["sample_idx"] = df["object_time"].map(time_to_idx)
    
    return df


def calculate_track_movement(df: pd.DataFrame, track_id: int) -> float:
    """Calculate total displacement for a track (using world coordinates in meters)."""
    track = df[df[TRACK_ID_COLUMN] == track_id][["sample_idx", "world_x", "world_y"]].sort_values("sample_idx")
    
    if len(track) < 2:
        return 0.0
    
    x_diff = np.diff(track["world_x"].values)
    y_diff = np.diff(track["world_y"].values)
    distances = np.sqrt(x_diff**2 + y_diff**2)
    
    return float(np.nansum(distances))


def get_majority_class(df: pd.DataFrame, track_id: int) -> int:
    """Get the most frequent class_id for a track."""
    classes = df[df[TRACK_ID_COLUMN] == track_id]["class_id"]
    return int(classes.mode().iloc[0]) if len(classes) > 0 else 0


def build_presence_map(df: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, set]]:
    """Build efficient presence maps for overlap computation."""
    frames_by_tid: Dict[int, np.ndarray] = {}
    frameset_by_tid: Dict[int, set] = {}
    
    for tid, g in df.groupby(TRACK_ID_COLUMN, observed=True)["sample_idx"]:
        arr = g.dropna().astype(np.int64).to_numpy()
        if arr.size:
            arr = np.unique(arr)
            tidi = int(tid)
            frames_by_tid[tidi] = arr
            frameset_by_tid[tidi] = set(arr.tolist())
    
    return frames_by_tid, frameset_by_tid


def _longest_run(sorted_unique_indices: np.ndarray) -> Tuple[int, int, int]:
    """Find longest contiguous run in sorted index array."""
    n = sorted_unique_indices.size
    if n == 0:
        return (-1, -1, 0)
    if n == 1:
        f = int(sorted_unique_indices[0])
        return (f, f, 1)
    
    dif = np.diff(sorted_unique_indices)
    breaks = np.where(dif != 1)[0] + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks - 1, n - 1]
    lengths = (ends - starts + 1)
    
    k = int(np.argmax(lengths))
    s = int(sorted_unique_indices[starts[k]])
    e = int(sorted_unique_indices[ends[k]])
    return (s, e, int(lengths[k]))


def longest_common_contiguous(
    combo: Tuple[int, int],
    frameset_by_tid: Dict[int, set]
) -> Tuple[int, int, int]:
    """Find longest common contiguous sample sequence for two tracks."""
    s0 = frameset_by_tid.get(combo[0])
    s1 = frameset_by_tid.get(combo[1])
    
    if not s0 or not s1:
        return (-1, -1, 0)
    
    common = s0 & s1
    if not common:
        return (-1, -1, 0)
    
    arr = np.fromiter(common, dtype=np.int64)
    arr.sort()
    
    return _longest_run(arr)


def rows_for_interval(
    df_indexed: pd.DataFrame,
    combo: Tuple[int, int],
    start_idx: int,
    length: int,
    idx_to_time: Dict[int, datetime],
    track_classes: Dict[int, int]
) -> List[dict]:
    """Materialize rows for a configuration with exactly 'length' frames."""
    end_idx = start_idx + length - 1
    rows = []
    reindex_map = {tid: i for i, tid in enumerate(sorted(combo))}
    
    for tid in sorted(combo):
        try:
            track_data = df_indexed.loc[tid]
            
            if isinstance(track_data, pd.Series):
                idx = track_data.name if hasattr(track_data, 'name') else None
                if idx is not None and start_idx <= idx <= end_idx:
                    timestamp = idx_to_time.get(idx, "")
                    rows.append({
                        "config_index": None,
                        "sample": int(idx - start_idx),
                        "trackId": reindex_map[tid],
                        "x": float(track_data["world_x"]),  # World coords relative to camera
                        "y": float(track_data["world_y"]),  # World coords relative to camera
                        "class_id": track_classes.get(tid, 0),  # Use majority class
                        "speed": float(track_data["speed"]) if pd.notna(track_data["speed"]) else 0.0,
                        "timestamp": str(timestamp),
                    })
            else:
                track_in_range = track_data[
                    (track_data.index >= start_idx) & (track_data.index <= end_idx)
                ]
                
                for idx, row in track_in_range.iterrows():
                    if pd.notna(row["world_x"]) and pd.notna(row["world_y"]):
                        timestamp = idx_to_time.get(idx, "")
                        rows.append({
                            "config_index": None,
                            "sample": int(idx - start_idx),
                            "trackId": reindex_map[tid],
                            "x": float(row["world_x"]),  # World coords relative to camera
                            "y": float(row["world_y"]),  # World coords relative to camera
                            "class_id": track_classes.get(tid, 0),  # Use majority class
                            "speed": float(row["speed"]) if pd.notna(row["speed"]) else 0.0,
                            "timestamp": str(timestamp),
                        })
        except KeyError:
            continue
    
    return rows


def main():
    print("=" * 60)
    print("Building 2-object configurations from pubapi data")
    print("=" * 60)
    
    # Create output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = f"{run_timestamp}_F{TARGET_LENGTH}_minmov{MIN_MOVEMENT}"
    output_dir = os.path.join(OUTDIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSettings:")
    print(f"  Input       : {INPUT_CSV}")
    print(f"  Output      : {output_dir}")
    print(f"  Frame length: {TARGET_LENGTH} (exact)")
    print(f"  Min movement: {MIN_MOVEMENT}m")
    print()
    
    # Read data
    print("Reading data...")
    df = read_pubapi_data(INPUT_CSV)
    total_tracks = df[TRACK_ID_COLUMN].nunique()
    print(f"  Loaded {len(df)} observations, {total_tracks} unique tracks")
    
    # Assign sample indices
    df = assign_sample_indices(df)
    
    # Build index-to-time mapping
    idx_to_time = df.drop_duplicates("sample_idx").set_index("sample_idx")["object_time"].to_dict()
    
    # Filter to moving tracks only
    print(f"\nFiltering tracks with movement >= {MIN_MOVEMENT}m...")
    moving_tracks = []
    track_movements = {}
    track_classes = {}  # Store majority class for each track
    
    for tid in df[TRACK_ID_COLUMN].unique():
        movement = calculate_track_movement(df, int(tid))
        track_movements[int(tid)] = movement
        if movement >= MIN_MOVEMENT:
            moving_tracks.append(int(tid))
            track_classes[int(tid)] = get_majority_class(df, int(tid))
    
    print(f"  {len(moving_tracks)} moving tracks (out of {total_tracks})")
    
    if len(moving_tracks) < 2:
        raise RuntimeError("Need at least 2 moving tracks to create configurations")
    
    # Filter dataframe to moving tracks only
    df = df[df[TRACK_ID_COLUMN].isin(moving_tracks)].copy()
    
    # Show track info
    print(f"\nMoving tracks by class (majority class):")
    class_counts = {}
    for tid in moving_tracks:
        cid = track_classes[tid]
        class_counts[cid] = class_counts.get(cid, 0) + 1
    for cid in sorted(class_counts.keys()):
        name = CLASS_ID_NAMES.get(cid, f"class_{cid}")
        print(f"  {name} (class {cid}): {class_counts[cid]} tracks")
    
    # Build presence maps
    frames_by_tid, frameset_by_tid = build_presence_map(df)
    
    # Generate all pairs of moving tracks
    print(f"\nGenerating track pairs...")
    all_pairs = list(itertools.combinations(moving_tracks, 2))
    print(f"  Total possible pairs: {len(all_pairs)}")
    
    # Find pairs with sufficient overlap
    valid_pairs = []
    for t1, t2 in all_pairs:
        combo = (t1, t2)
        s, e, L = longest_common_contiguous(combo, frameset_by_tid)
        if L >= TARGET_LENGTH:
            valid_pairs.append({
                "combo": combo,
                "overlap_start": s,
                "overlap_end": e,
                "overlap_length": L,
                "classes": (track_classes[t1], track_classes[t2])
            })
    
    print(f"  Pairs with >= {TARGET_LENGTH} overlapping frames: {len(valid_pairs)}")
    
    if not valid_pairs:
        raise RuntimeError(f"No track pairs have >= {TARGET_LENGTH} overlapping frames")
    
    # Sort by overlap length (descending)
    valid_pairs.sort(key=lambda x: x["overlap_length"], reverse=True)
    
    print(f"\nTop 10 pairs by overlap length:")
    for i, p in enumerate(valid_pairs[:10]):
        c1, c2 = p["classes"]
        n1 = CLASS_ID_NAMES.get(c1, f"?{c1}")
        n2 = CLASS_ID_NAMES.get(c2, f"?{c2}")
        print(f"  {p['combo']}: {p['overlap_length']} frames ({n1} + {n2})")
    
    # Build configurations
    print(f"\nBuilding configurations with exactly {TARGET_LENGTH} frames each...")
    
    df_indexed = df.set_index([TRACK_ID_COLUMN, "sample_idx"]).sort_index()
    
    kept_rows = []
    report_rows = []
    cfg_idx = 0
    
    for p in valid_pairs:
        combo = p["combo"]
        os_, oe_, ol_ = p["overlap_start"], p["overlap_end"], p["overlap_length"]
        c1, c2 = p["classes"]
        classes_str = f"{CLASS_ID_NAMES.get(c1, c1)},{CLASS_ID_NAMES.get(c2, c2)}"
        
        # Trim to exactly TARGET_LENGTH frames (take from start of overlap)
        start_candidate = os_
        end_candidate = start_candidate + TARGET_LENGTH - 1
        
        rows = rows_for_interval(df_indexed, combo, start_candidate, TARGET_LENGTH, idx_to_time, track_classes)
        
        if not rows:
            report_rows.append({
                "config_index": None, "combo": combo, "classes": classes_str,
                "kept": False, "reason": "Materialization gap",
                "overlap_length": ol_, "target_length": TARGET_LENGTH,
            })
            continue
        
        # Validate: each track must have exactly TARGET_LENGTH samples
        samples_per_track = {}
        for r in rows:
            tid = r["trackId"]
            samples_per_track[tid] = samples_per_track.get(tid, 0) + 1
        
        if len(samples_per_track) != 2:
            report_rows.append({
                "config_index": None, "combo": combo, "classes": classes_str,
                "kept": False, "reason": f"Missing track ({len(samples_per_track)}/2)",
                "overlap_length": ol_, "target_length": TARGET_LENGTH,
            })
            continue
        
        counts = list(samples_per_track.values())
        if counts[0] != TARGET_LENGTH or counts[1] != TARGET_LENGTH:
            report_rows.append({
                "config_index": None, "combo": combo, "classes": classes_str,
                "kept": False, "reason": f"Wrong sample count: {counts}",
                "overlap_length": ol_, "target_length": TARGET_LENGTH,
            })
            continue
        
        # Valid configuration
        for r in rows:
            r["config_index"] = cfg_idx
        kept_rows.extend(rows)
        
        report_rows.append({
            "config_index": cfg_idx, "combo": combo, "classes": classes_str,
            "kept": True, "reason": "",
            "overlap_length": ol_, "target_length": TARGET_LENGTH,
        })
        cfg_idx += 1
        
        if MAX_CONFIGS is not None and cfg_idx >= MAX_CONFIGS:
            print(f"  Reached MAX_CONFIGS={MAX_CONFIGS}. Stopping.")
            break
    
    # Create output files
    report_df = pd.DataFrame(report_rows)
    
    if not kept_rows:
        report_path = os.path.join(output_dir, f"C0_2OBJ_pubapi_F{TARGET_LENGTH}_report.csv")
        report_df.to_csv(report_path, index=False)
        raise RuntimeError(f"No valid configurations. See report: {report_path}")
    
    cfg_df = pd.DataFrame(kept_rows)
    cfg_df.sort_values(["config_index", "sample", "trackId"], inplace=True, ignore_index=True)
    
    num_configs = cfg_df["config_index"].nunique()
    
    # File names
    fname_with_class = f"C{num_configs}_2OBJ_CL_pubapi_F{TARGET_LENGTH}.csv"
    fname_without_class = f"C{num_configs}_2OBJ_NC_pubapi_F{TARGET_LENGTH}.csv"
    fname_report = f"C{num_configs}_2OBJ_pubapi_F{TARGET_LENGTH}_report.csv"
    
    cfg_path = os.path.join(output_dir, fname_with_class)
    cfg_noclass_path = os.path.join(output_dir, fname_without_class)
    report_path = os.path.join(output_dir, fname_report)
    
    # Output columns
    out_cols_with_class = ["config_index", "sample", "trackId", "x", "y", "class_id", "timestamp"]
    out_cols_no_class = ["config_index", "sample", "trackId", "x", "y", "timestamp"]
    
    cfg_df[out_cols_with_class].to_csv(cfg_path, index=False, header=False)
    cfg_df[out_cols_no_class].to_csv(cfg_noclass_path, index=False, header=False)
    report_df.to_csv(report_path, index=False)
    
    print(f"""
✅ Done!

Output files:
  - {cfg_path}
  - {cfg_noclass_path}
  - {report_path}

Summary:
  - Configurations: {num_configs}
  - Frames per config: {TARGET_LENGTH}
  - Total rows: {len(cfg_df)}
""")


if __name__ == "__main__":
    main()
