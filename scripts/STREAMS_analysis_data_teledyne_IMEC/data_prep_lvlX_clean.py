#!/usr/bin/env python3
"""
Optimized version: Build configurations from class-based combinations with longest-overlap logic.
- No GUI, no command-line args
- Configure everything in the CONFIG section below, then Run.

Outputs (auto-named):
- C{A}_{B}_CL_{D}_F{E}.csv   (with class column)
- C{A}_{B}_NC_{D}_F{E}.csv   (no class column)
- C{A}_{B}_{D}_F{E}_report.csv

Key Optimizations:
- Vectorized operations with NumPy for longest-overlap computation
- Early pruning: only tracks with span >= TARGET_LENGTH considered
- Lean dtypes (float32, categorical), fewer conversions
- Vectorized class lookups and groupby operations (no iterrows)
- Query-based filtering for better performance
- Pre-compute lookups to avoid repeated DataFrame searches
"""

import os
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ===================== CONFIG (edit these) =====================
# Base paths (script will loop through all recordings)
BASE_TRACKS_DIR = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\inD\inD_tracks_only_filtered"
BASE_META_DIR   = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\inD\inD-dataset-v1.1\data"
BASE_OUTDIR     = r"C:\Users\oliverme\OneDrive - UGent\Documents\STREAMS\inD\Data_by_number"

# Recording range to process (e.g., 0 to 32 for all recordings)
START_RECORDING = 0    # Start from recording 00
END_RECORDING = 32     # End at recording 32 (inclusive)

START_FRAME = 0           # inclusive
END_FRAME   = 10000000    # inclusive
LIMIT_TO_RANGE = True     # consider only frames within [START_FRAME..END_FRAME]

CLASS_PATTERN = "car,bicycle"    # e.g. "car,bike" or "car,car,bike"
TARGET_LENGTH = 100      # number of frames each configuration must have (trimmed). Set >=1
MAX_CONFIGS   = 200      # optional cap (int) on number of kept configurations, or None

# Distance filtering (set None to disable)
MIN_DISTANCE = None       # minimum distance in meters (or dataset units) between tracks, or None to disable
MAX_DISTANCE = 4       # maximum distance in meters (or dataset units) between tracks, or None to disable
MIN_CLOSE_FRAMES = 3    # minimum number of frames tracks must be within distance threshold, or None to disable

# Movement filtering (set None to disable)
FILTER_STATIONARY = True  # if True, exclude tracks that don't move enough
MIN_MOVEMENT =  3  # minimum total displacement in meters (or dataset units) for a track to be considered moving
# ===============================================================


def _find_class_column(cols) -> str:
    """Find class column with flexible naming."""
    lower = {c.lower(): c for c in cols}
    if "class" in lower:
        return lower["class"]
    for candidate in ("category", "label", "type"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(
        "Meta must contain a 'class' column (or synonym: category/label/type). "
        f"Found: {list(cols)}"
    )


def read_tracks(tracks_path: str) -> pd.DataFrame:
    """Read tracks CSV with optimized dtypes."""
    df = pd.read_csv(
        tracks_path,
        header=None,
        names=["trackId", "frame", "x", "y"],
        dtype={"trackId": "Int64", "frame": "Int64", "x": "float32", "y": "float32"},
    )
    return df


def read_meta(meta_path: str) -> pd.DataFrame:
    """Read metadata CSV with validation."""
    df = pd.read_csv(meta_path)
    required = {"trackId", "initialFrame", "finalFrame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Meta missing required columns: {sorted(missing)}")
    
    class_col = _find_class_column(df.columns)
    if class_col != "class":
        df = df.rename(columns={class_col: "class"})
    
    for col in ["trackId", "initialFrame", "finalFrame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    return df


def restrict_tracks_to_range(tracks: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    """Filter tracks by frame range using query (faster than boolean indexing)."""
    return tracks.query(f"frame >= {start_frame} & frame <= {end_frame}")


def build_presence_map(tracks_xy: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, set]]:
    """
    Build efficient presence maps for overlap computation.
    
    Returns:
      frames_by_tid: dict[int -> np.ndarray[int]] sorted unique frames per track
      frameset_by_tid: dict[int -> set[int]] for fast intersections
    """
    frames_by_tid: Dict[int, np.ndarray] = {}
    frameset_by_tid: Dict[int, set] = {}
    
    for tid, g in tracks_xy.groupby("trackId", observed=True)["frame"]:
        arr = g.dropna().astype(np.int64).to_numpy()
        if arr.size:
            arr = np.unique(arr)
            tidi = int(tid)
            frames_by_tid[tidi] = arr
            frameset_by_tid[tidi] = set(arr.tolist())
    
    return frames_by_tid, frameset_by_tid


def _longest_run(sorted_unique_frames: np.ndarray) -> Tuple[int, int, int]:
    """Find longest contiguous run in sorted frame array using vectorized operations."""
    n = sorted_unique_frames.size
    if n == 0:
        return (-1, -1, 0)
    if n == 1:
        f = int(sorted_unique_frames[0])
        return (f, f, 1)
    
    # Vectorized: find breaks where diff != 1
    dif = np.diff(sorted_unique_frames)
    breaks = np.where(dif != 1)[0] + 1
    starts = np.r_[0, breaks]
    ends   = np.r_[breaks - 1, n - 1]
    lengths = (ends - starts + 1)
    
    k = int(np.argmax(lengths))
    s = int(sorted_unique_frames[starts[k]])
    e = int(sorted_unique_frames[ends[k]])
    return (s, e, int(lengths[k]))


def longest_common_contiguous_from_sets(
    combo: Tuple[int, ...],
    frameset_by_tid: Dict[int, set],
    frames_by_tid: Dict[int, np.ndarray]
) -> Tuple[int, int, int]:
    """Find longest common contiguous frame sequence for track combination."""
    # Early exit checks
    base_tid = combo[0]
    s0 = frameset_by_tid.get(base_tid)
    if not s0:
        return (-1, -1, 0)
    
    common = s0.copy()
    for tid in combo[1:]:
        s = frameset_by_tid.get(tid)
        if not s:
            return (-1, -1, 0)
        common &= s
        if not common:
            return (-1, -1, 0)
    
    # Convert to sorted array for contiguous check
    arr = np.fromiter(common, dtype=np.int64)
    if arr.size == 0:
        return (-1, -1, 0)
    arr.sort()
    
    return _longest_run(arr)


def build_class_buckets_from_df(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Build class->trackIds mapping using vectorized operations (no iterrows)."""
    df_ok = df.dropna(subset=["trackId", "class"]).copy()
    df_ok["class"] = df_ok["class"].str.strip().str.lower()
    df_ok["trackId"] = df_ok["trackId"].astype(int)
    
    # Vectorized groupby instead of iterrows
    buckets = df_ok.groupby("class", observed=True)["trackId"].apply(
        lambda x: sorted(x.unique().tolist())
    ).to_dict()
    
    return buckets


def parse_class_pattern(pattern: str) -> List[str]:
    """Parse comma-separated class pattern."""
    items = [p.strip().lower() for p in pattern.split(",") if p.strip()]
    if not items:
        raise ValueError("CLASS_PATTERN is empty. Example: 'car,bike'")
    return items


def check_distance_constraint(
    combo: Tuple[int, ...],
    tracks: pd.DataFrame,
    min_dist: float = None,
    max_dist: float = None,
    min_frames: int = None
) -> Tuple[bool, int]:
    """
    Check if tracks in combo meet distance constraints.
    
    Args:
        combo: Tuple of trackIds to check
        tracks: DataFrame with trackId, frame, x, y columns
        min_dist: Minimum distance threshold (or None to skip)
        max_dist: Maximum distance threshold (or None to skip)
        min_frames: Minimum number of frames that must meet distance constraint (or None to skip)
    
    Returns:
        (meets_constraint, num_close_frames): Boolean and count of frames meeting constraint
    """
    if (min_dist is None and max_dist is None) or min_frames is None:
        return (True, 0)  # No constraint, always pass
    
    # Get data for both tracks
    track1_data = tracks[tracks["trackId"] == combo[0]][["frame", "x", "y"]]
    track2_data = tracks[tracks["trackId"] == combo[1]][["frame", "x", "y"]]
    
    if track1_data.empty or track2_data.empty:
        return (False, 0)
    
    # Merge on common frames
    merged = track1_data.merge(
        track2_data, 
        on="frame", 
        suffixes=("_1", "_2")
    )
    
    if merged.empty:
        return (False, 0)
    
    # Calculate Euclidean distance for each frame
    merged["distance"] = np.sqrt(
        (merged["x_1"] - merged["x_2"])**2 + 
        (merged["y_1"] - merged["y_2"])**2
    )
    
    # Filter by distance constraints
    if min_dist is not None and max_dist is not None:
        close_frames = merged[(merged["distance"] >= min_dist) & (merged["distance"] <= max_dist)]
    elif min_dist is not None:
        close_frames = merged[merged["distance"] >= min_dist]
    elif max_dist is not None:
        close_frames = merged[merged["distance"] <= max_dist]
    else:
        close_frames = merged
    
    num_close = len(close_frames)
    meets_constraint = num_close >= min_frames if min_frames is not None else True
    
    return (meets_constraint, num_close)


def calculate_track_movement(tracks: pd.DataFrame, track_id: int) -> float:
    """
    Calculate total displacement for a track.
    
    Args:
        tracks: DataFrame with trackId, frame, x, y columns
        track_id: The track to analyze
    
    Returns:
        Total displacement (sum of distances between consecutive positions)
    """
    track_data = tracks[tracks["trackId"] == track_id][["frame", "x", "y"]].sort_values("frame")
    
    if len(track_data) < 2:
        return 0.0
    
    # Calculate displacement between consecutive frames
    x_diff = np.diff(track_data["x"].values)
    y_diff = np.diff(track_data["y"].values)
    distances = np.sqrt(x_diff**2 + y_diff**2)
    
    return float(np.sum(distances))


def filter_stationary_tracks(
    tracks: pd.DataFrame,
    meta: pd.DataFrame,
    min_movement: float
) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    """
    Filter out tracks that don't move enough.
    
    Args:
        tracks: DataFrame with trackId, frame, x, y columns
        meta: Metadata DataFrame
        min_movement: Minimum total displacement required
    
    Returns:
        (filtered_tracks, filtered_meta, stationary_track_ids): Filtered data and set of removed track IDs
    """
    track_ids = tracks["trackId"].unique()
    stationary_ids = set()
    
    for tid in track_ids:
        movement = calculate_track_movement(tracks, tid)
        if movement < min_movement:
            stationary_ids.add(tid)
    
    if stationary_ids:
        print(f"  Filtered out {len(stationary_ids)} stationary tracks (movement < {min_movement}m)")
        filtered_tracks = tracks[~tracks["trackId"].isin(stationary_ids)].copy()
        filtered_meta = meta[~meta["trackId"].isin(stationary_ids)].copy()
        return filtered_tracks, filtered_meta, stationary_ids
    
    return tracks, meta, stationary_ids


def combos_for_pattern(
    buckets: Dict[str, List[int]], 
    patt: List[str],
    meta: pd.DataFrame,
    target_length: int,
    tracks: pd.DataFrame = None,
    min_dist: float = None,
    max_dist: float = None,
    min_close_frames: int = None
) -> List[Tuple[int, ...]]:
    """
    Generate valid track combinations with early filtering.
    
    Optimization: Pre-filter by checking if tracks can possibly overlap enough
    and optionally by distance constraints.
    """
    cands: List[List[int]] = []
    for label in patt:
        cands.append(buckets.get(label, []))
    
    if any(len(cl) == 0 for cl in cands):
        return []
    
    # Debug: show how many of each class
    print(f"  Found {len(cands[0])} '{patt[0]}' tracks and {len(cands[1])} '{patt[1]}' tracks")
    
    # Pre-compute track spans for quick filtering
    track_spans = meta.set_index("trackId")[["initialFrame", "finalFrame"]].to_dict("index")
    
    valid = []
    distance_filtered = 0
    overlap_too_short = 0
    total_checked = 0
    
    for tup in itertools.product(*cands):
        total_checked += 1
        
        # Skip if not all distinct tracks
        if len(set(tup)) != len(tup):
            continue
        
        # Quick span check: theoretical max overlap
        try:
            min_final = min(track_spans[t]["finalFrame"] for t in tup if t in track_spans)
            max_initial = max(track_spans[t]["initialFrame"] for t in tup if t in track_spans)
            max_possible_overlap = min_final - max_initial + 1
            
            # Only keep if there's a chance of meeting target length
            if max_possible_overlap >= target_length:
                # Check distance constraint if enabled
                if tracks is not None and (min_dist is not None or max_dist is not None):
                    meets_dist, num_close = check_distance_constraint(
                        tup, tracks, min_dist, max_dist, min_close_frames
                    )
                    if not meets_dist:
                        distance_filtered += 1
                        continue
                
                valid.append(tuple(int(x) for x in tup))
            else:
                overlap_too_short += 1
        except (KeyError, ValueError):
            # Skip if any track not found
            continue
    
    print(f"  Checked {total_checked} combinations: {overlap_too_short} too short overlap, {distance_filtered} failed distance filter")
    if distance_filtered > 0:
        print(f"  Filtered out {distance_filtered} combinations due to distance constraints")
    
    return valid


def rows_for_interval(
    tracks_indexed: pd.DataFrame,
    combo: Tuple[int, ...],
    start: int,
    target_len: int
) -> List[dict]:
    """
    Materialize rows for a configuration - extract actual x,y positions for each track.
    
    Each trackId should have its own unique x,y coordinates at each frame.
    """
    end = start + target_len - 1
    frames = np.arange(start, start + target_len, dtype=np.int64)
    
    # Extract data for each track in the combo
    rows = []
    reindex_map = {tid: i for i, tid in enumerate(sorted(combo))}
    
    # For each track, get its actual positions
    for tid in sorted(combo):
        # Get this specific track's data in the frame range
        try:
            track_data = tracks_indexed.loc[tid]
            
            # Filter to our frame range
            if isinstance(track_data, pd.Series):
                # Only one frame for this track
                frame = track_data.name if hasattr(track_data, 'name') else None
                if frame is not None and start <= frame <= end:
                    rows.append({
                        "config_index": None,
                        "frame": int(frame - start),
                        "trackId": reindex_map[tid],
                        "x": float(track_data["x"]),
                        "y": float(track_data["y"]),
                        "class": track_data["class"],
                    })
            else:
                # Multiple frames for this track
                track_in_range = track_data[(track_data.index >= start) & (track_data.index <= end)]
                
                for frame_abs, row in track_in_range.iterrows():
                    if pd.notna(row["x"]) and pd.notna(row["y"]):
                        rows.append({
                            "config_index": None,
                            "frame": int(frame_abs - start),
                            "trackId": reindex_map[tid],
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                            "class": row["class"],
                        })
        except KeyError:
            # Track not in index, skip
            continue
    
    return rows


def _class_to_code(label: str) -> str:
    """Convert class name to short code for file naming."""
    l = label.strip().lower()
    mapping = {
        "car": "C",
        "bicycle": "B",
        "bike": "B",
        "pedestrian": "P",
        "person": "P",
        "bus": "Bu",
        "motorcycle": "M",
        "motorbike": "M",
        "moped": "M",
    }
    return mapping.get(l, l[:2].title() if len(l) > 1 else l.upper())


def _dataset_code(*paths: str) -> str:
    """Infer dataset name from paths."""
    joined = " ".join(paths).lower()
    return "inD" if "ind" in joined else "uniD"


def main():
    assert TARGET_LENGTH is not None and int(TARGET_LENGTH) > 0, "Set TARGET_LENGTH (>0) in CONFIG."
    
    # Loop through all recordings
    for recording_num in range(START_RECORDING, END_RECORDING + 1):
        # Format recording number with leading zero for file paths (00, 01, ..., 32)
        rec_str = f"{recording_num:02d}"
        
        # Build paths for this recording
        TRACKS_PATH = os.path.join(BASE_TRACKS_DIR, f"{rec_str}_tracks.csv")
        META_PATH = os.path.join(BASE_META_DIR, f"{rec_str}_tracksMeta.csv")
        # Output directory uses the actual number (0, 1, 2, ..., 32)
        OUTDIR = os.path.join(BASE_OUTDIR, str(recording_num))
        
        # Check if files exist
        if not os.path.exists(TRACKS_PATH):
            print(f"\n⚠️  Skipping recording {rec_str}: tracks file not found")
            continue
        if not os.path.exists(META_PATH):
            print(f"\n⚠️  Skipping recording {rec_str}: meta file not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing recording {rec_str} → output folder {recording_num}")
        print(f"{'='*70}")
        
        try:
            process_recording(TRACKS_PATH, META_PATH, OUTDIR)
        except Exception as e:
            print(f"\n❌ Error processing recording {rec_str}: {e}")
            print("Continuing to next recording...\n")
            continue
    
    print(f"\n{'='*70}")
    print("✅ All recordings processed!")
    print(f"{'='*70}")


def process_recording(TRACKS_PATH: str, META_PATH: str, OUTDIR: str):
    """Process a single recording with the configured parameters."""
def process_recording(TRACKS_PATH: str, META_PATH: str, OUTDIR: str):
    """Process a single recording with the configured parameters."""
    
    # Create parameter-based subfolder name
    param_parts = []
    if MAX_DISTANCE is not None:
        param_parts.append(f"maxdist{MAX_DISTANCE}")
    if MIN_CLOSE_FRAMES is not None:
        param_parts.append(f"minframes{MIN_CLOSE_FRAMES}")
    if FILTER_STATIONARY and MIN_MOVEMENT is not None:
        param_parts.append(f"minmov{MIN_MOVEMENT}")
    
    # Create subfolder path
    if param_parts:
        subfolder_name = "_".join(param_parts)
        output_dir = os.path.join(OUTDIR, subfolder_name)
    else:
        output_dir = OUTDIR
    
    os.makedirs(output_dir, exist_ok=True)

    print("=== Building configurations (optimized) ===")
    print(f"- Tracks : {TRACKS_PATH}")
    print(f"- Meta   : {META_PATH}")
    print(f"- Outdir : {output_dir}")
    print(f"- Pattern: {CLASS_PATTERN}")
    print(f"- Range  : [{START_FRAME}..{END_FRAME}]  limit_to_range={LIMIT_TO_RANGE}")
    print(f"- Target : {TARGET_LENGTH} frames")
    print(f"- Distance filter: MIN={MIN_DISTANCE}, MAX={MAX_DISTANCE}, MIN_FRAMES={MIN_CLOSE_FRAMES}")
    print(f"- Movement filter: FILTER_STATIONARY={FILTER_STATIONARY}, MIN_MOVEMENT={MIN_MOVEMENT if FILTER_STATIONARY else 'N/A'}\n")

    # Read data
    tracks = read_tracks(TRACKS_PATH)
    meta = read_meta(META_PATH)

    # Early pruning: keep only tracks whose span can support TARGET_LENGTH
    meta["_span"] = (meta["finalFrame"] - meta["initialFrame"] + 1).astype("Int64")
    eligible_meta = meta.loc[
        meta["_span"] >= TARGET_LENGTH, 
        ["trackId", "class", "initialFrame", "finalFrame"]
    ].copy()
    
    if eligible_meta.empty:
        raise RuntimeError(f"No tracks in meta with span >= TARGET_LENGTH={TARGET_LENGTH}.")

    # Restrict tracks to range if requested (BEFORE movement filtering)
    if LIMIT_TO_RANGE:
        tracks = restrict_tracks_to_range(tracks, START_FRAME, END_FRAME)

    # Filter stationary tracks if enabled (AFTER restricting to frame range)
    if FILTER_STATIONARY and MIN_MOVEMENT is not None:
        tracks, meta, stationary_ids = filter_stationary_tracks(tracks, meta, MIN_MOVEMENT)
        if not tracks.empty:
            print(f"Remaining tracks after filtering: {tracks['trackId'].nunique()}\n")
        else:
            raise RuntimeError("All tracks were filtered out as stationary.")
    else:
        stationary_ids = set()

    # Attach classes (only for eligible trackIds) - inner join filters efficiently
    tracks = tracks.merge(eligible_meta[["trackId", "class"]], on="trackId", how="inner")

    # Optimize dtypes for memory efficiency
    tracks["class"] = tracks["class"].astype("category")

    # Build presence maps for overlap computation
    frames_by_tid, frameset_by_tid = build_presence_map(tracks[["trackId", "frame"]])

    # Build class buckets using only eligible tracks
    buckets = build_class_buckets_from_df(eligible_meta)

    patt = parse_class_pattern(CLASS_PATTERN)
    combos = combos_for_pattern(
        buckets, patt, eligible_meta, TARGET_LENGTH,
        tracks=tracks if (MIN_DISTANCE is not None or MAX_DISTANCE is not None) else None,
        min_dist=MIN_DISTANCE,
        max_dist=MAX_DISTANCE,
        min_close_frames=MIN_CLOSE_FRAMES
    )

    # Deduplicate permutations of the same set of trackIds
    unique_combos = []
    seen_sets = set()
    for tup in combos:
        key = frozenset(tup)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        unique_combos.append(tup)
    combos = unique_combos

    if not combos:
        raise RuntimeError(
            f"No valid combinations for pattern '{CLASS_PATTERN}'. "
            f"Available classes (eligible): {sorted(buckets.keys())}"
        )

    print(f"Generated {len(combos)} unique track combinations to evaluate.\n")

    # Pre-compute class lookup dictionary for faster access
    meta_class_dict = meta.set_index("trackId")["class"].to_dict()

    # Compute longest overlaps
    summaries = []
    for combo in combos:
        s, e, L = longest_common_contiguous_from_sets(combo, frameset_by_tid, frames_by_tid)
        
        # Vectorized class lookup instead of repeated DataFrame queries
        classes = [meta_class_dict.get(tid) for tid in combo]
        
        summaries.append({
            "combo": combo,
            "classes": classes,
            "overlap_start": s,
            "overlap_end": e,
            "overlap_length": L
        })

    # Create summary DataFrame
    df_sum = pd.DataFrame([{
        "combo": s["combo"],
        "classes": ",".join([str(c) for c in s["classes"]]),
        "overlap_start": s["overlap_start"],
        "overlap_end": s["overlap_end"],
        "overlap_length": s["overlap_length"]
    } for s in summaries])

    print("Top 20 combinations by overlap length:")
    print(df_sum.sort_values("overlap_length", ascending=False).head(20).to_string(index=False))
    print()

    # Prepare indexed view for efficient materialization
    tracks_idxed = tracks.set_index(["trackId", "frame"]).sort_index()

    # Process configurations
    kept_rows: List[dict] = []
    report_rows = []
    cfg_idx = 0

    for s in summaries:
        combo = s["combo"]
        os_, oe_, ol_ = s["overlap_start"], s["overlap_end"], s["overlap_length"]
        classes_str = ",".join([str(c) for c in s["classes"]])

        if ol_ >= TARGET_LENGTH:
            # Determine trim window
            start_candidate = os_
            end_candidate = start_candidate + TARGET_LENGTH - 1
            if end_candidate > oe_:
                start_candidate = oe_ - TARGET_LENGTH + 1
                end_candidate = oe_

            rows = rows_for_interval(tracks_idxed, combo, start_candidate, TARGET_LENGTH)
            
            if not rows:
                report_rows.append({
                    "config_index": None, "combo": combo, "classes": classes_str,
                    "kept": False, "reason": "Materialization gap",
                    "trimmed_start": start_candidate, "trimmed_end": end_candidate,
                    "target_len": TARGET_LENGTH, "overlap_len": ol_,
                })
                continue

            # Check movement within this specific configuration window if enabled
            if FILTER_STATIONARY and MIN_MOVEMENT is not None:
                config_has_stationary = False
                for tid in combo:
                    # Get track data only in this config's frame range
                    config_track = tracks[(tracks["trackId"] == tid) & 
                                         (tracks["frame"] >= start_candidate) & 
                                         (tracks["frame"] <= (start_candidate + TARGET_LENGTH - 1))][["frame", "x", "y"]].sort_values("frame")
                    
                    if len(config_track) > 1:
                        x_diff = np.diff(config_track["x"].values)
                        y_diff = np.diff(config_track["y"].values)
                        config_movement = float(np.sum(np.sqrt(x_diff**2 + y_diff**2)))
                        
                        if config_movement < MIN_MOVEMENT:
                            config_has_stationary = True
                            break
                    else:
                        config_has_stationary = True
                        break
                
                if config_has_stationary:
                    report_rows.append({
                        "config_index": None, "combo": combo, "classes": classes_str,
                        "kept": False, "reason": "Track stationary in config window",
                        "trimmed_start": start_candidate, "trimmed_end": end_candidate,
                        "target_len": TARGET_LENGTH, "overlap_len": ol_,
                    })
                    continue

            # Assign config index to all rows
            for r in rows:
                r["config_index"] = cfg_idx
            kept_rows.extend(rows)

            report_rows.append({
                "config_index": cfg_idx, "combo": combo, "classes": classes_str,
                "kept": True, "reason": "",
                "trimmed_start": start_candidate, "trimmed_end": end_candidate,
                "target_len": TARGET_LENGTH, "overlap_len": ol_,
            })
            cfg_idx += 1

            if (MAX_CONFIGS is not None) and (cfg_idx >= MAX_CONFIGS):
                print(f"\nReached MAX_CONFIGS={MAX_CONFIGS}. Stopping early.")
                break
        else:
            report_rows.append({
                "config_index": None, "combo": combo, "classes": classes_str,
                "kept": False, "reason": f"Overlap too short ({ol_} < {TARGET_LENGTH})",
                "trimmed_start": None, "trimmed_end": None,
                "target_len": TARGET_LENGTH, "overlap_len": ol_,
            })

    report_df = pd.DataFrame(report_rows)

    # Build file naming components
    patt_order = parse_class_pattern(CLASS_PATTERN)
    B = "".join(_class_to_code(p) for p in patt_order)
    D = _dataset_code(TRACKS_PATH, META_PATH)
    E = f"F{TARGET_LENGTH}"

    if not kept_rows:
        A = "C0"
        report_path = os.path.join(output_dir, f"{A}_{B}_{D}_{E}_report.csv")
        report_df.to_csv(report_path, index=False)
        raise RuntimeError(
            f"No configurations met TARGET_LENGTH={TARGET_LENGTH}. "
            f"Report saved to:\n{report_path}"
        )

    # Build final configuration DataFrame
    cfg_df = pd.DataFrame(kept_rows, columns=["config_index", "frame", "trackId", "x", "y", "class"])
    cfg_df.sort_values(["config_index", "frame", "trackId"], inplace=True, ignore_index=True)

    num_configs = int(cfg_df['config_index'].nunique())
    A = f"C{num_configs}"

    # Generate file names
    fname_with_class    = f"{A}_{B}_CL_{D}_{E}.csv"
    fname_without_class = f"{A}_{B}_NC_{D}_{E}.csv"
    fname_report        = f"{A}_{B}_{D}_{E}_report.csv"

    cfg_path = os.path.join(output_dir, fname_with_class)
    cfg_noclass_path = os.path.join(output_dir, fname_without_class)
    report_path = os.path.join(output_dir, fname_report)

    # Save outputs
    cfg_df.to_csv(cfg_path, index=False, header=False)
    cfg_df.drop(columns=["class"]).to_csv(cfg_noclass_path, index=False, header=False)
    report_df.to_csv(report_path, index=False)

    print(f"""
✅ Wrote:
- {cfg_path}
- {cfg_noclass_path}
- {report_path}

Kept configurations: {cfg_df['config_index'].nunique()}
Target length      : {TARGET_LENGTH} frames
Class pattern      : {CLASS_PATTERN}
""")


if __name__ == "__main__":
    main()
