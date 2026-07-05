"""
config_generator_IMEC.py — Configuration Generator for IMEC tracker data
=========================================================================

Reads IMEC tracker CSVs (24-column format), identifies pairs of co-occurring
tracks with sufficient temporal overlap, extracts "configurations" of exactly
TARGET_LENGTH frames, and saves:
  1. A combined CSV per source file with all configurations (PDP-like format)
  2. A sidecar JSON metadata file *per configuration* with media linking info

The JSON sidecar contains:
  - source_media_path : absolute path to the associated MP4 video
  - media_type        : "video" (or "image" if images were used)
  - start_time        : starting frame index of this config in the video
  - end_time          : ending frame index
  - time_unit         : "frames" or "seconds"
  - fps               : frame rate of the source video

Usage examples:
    # Basic — process a folder of tracker CSVs
    python config_generator_IMEC.py \\
        --input_dir  "Data_IMEC_04/10_10_new/10_10_new/track_dataframes" \\
        --output_dir "output/configs_10_10"

    # With video linking — also writes source_media_path into each JSON
    python config_generator_IMEC.py \\
        --input_dir  "Data_IMEC_04/10_10_new/10_10_new/track_dataframes" \\
        --video_dir  "Data_IMEC_04/10_10_new/10_10_new/Videos" \\
        --output_dir "output/configs_10_10" \\
        --scene_name "10_10_new"

    # Override defaults
    python config_generator_IMEC.py \\
        --input_dir  "Data_IMEC_04/10_55_new/10_55_new/track_dataframes" \\
        --video_dir  "Data_IMEC_04/10_55_new/10_55_new/Videos" \\
        --output_dir "output/configs_10_55" \\
        --target_length 50 \\
        --min_movement 2.0 \\
        --max_configs 20

Depends on: pandas, numpy  (standard scientific-Python stack)
"""

import os
import sys
import json
import argparse
import re
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
#  DEFAULT CONFIGURATION
#  These defaults can all be overridden via CLI arguments.
# ─────────────────────────────────────────────────────────────────────

TARGET_LENGTH = 30       # Number of frames per configuration
MIN_MOVEMENT  = 1.0      # Minimum total track displacement (m) to keep
MAX_CONFIGS   = None      # Maximum configs per CSV (None = unlimited)
FPS           = 15        # Frame rate of the source videos

# Column names in the IMEC 24-column tracker CSVs
COL_TRACK_ID  = "ID"
COL_TIMESTAMP = "timestamp"
COL_X         = "x"
COL_Y         = "y"
COL_VX        = "vx"
COL_VY        = "vy"
COL_CLASS     = "YOLO_cls"

# YOLO class-ID → human-readable name
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ─────────────────────────────────────────────────────────────────────
#  CSV ↔ MP4 MAPPING
# ─────────────────────────────────────────────────────────────────────
#  IMEC data has scene-specific rules linking CSV file names to video
#  file names:
#    10_10_new  →  visual_N.mp4 corresponds to tracks_df_{N+750}.csv
#    10_55_new  →  visual_N.mp4 corresponds to tracks_df_N.csv
# ─────────────────────────────────────────────────────────────────────

def infer_video_path(csv_path: str, video_dir: str, scene_name: str = "") -> str | None:
    """
    Given a CSV file path and a video directory, infer the matching MP4 path.

    The mapping rule depends on the scene name:
      - "10_10…" scenes : CSV index = video index + 750   (offset)
      - "10_55…" scenes : CSV index = video index          (direct)
      - other           : try direct match first

    Args:
        csv_path   : Full path to a tracks_df_*.csv file
        video_dir  : Directory containing visual_*.mp4 files
        scene_name : Scene identifier (e.g. "10_10_new")

    Returns:
        Absolute path to the matching MP4, or None if not found.
    """
    # Extract numeric index from CSV filename  (e.g. "tracks_df_750.csv" → 750)
    csv_basename = os.path.basename(csv_path)
    match = re.search(r"tracks_df_(\d+)\.csv", csv_basename)
    if not match:
        return None
    csv_index = int(match.group(1))

    # Determine video index based on scene mapping rule
    if "10_10" in scene_name.lower():
        video_index = csv_index - 750          # offset rule
    else:
        video_index = csv_index                # direct / default

    # Try the computed path first
    video_path = os.path.join(video_dir, f"visual_{video_index}.mp4")
    if os.path.exists(video_path):
        return os.path.abspath(video_path)

    # Fallback: try direct match for 10_10 scenes
    if "10_10" in scene_name.lower():
        fallback = os.path.join(video_dir, f"visual_{csv_index}.mp4")
        if os.path.exists(fallback):
            return os.path.abspath(fallback)

    return None


# ─────────────────────────────────────────────────────────────────────
#  DATA READING
# ─────────────────────────────────────────────────────────────────────

def read_imec_csv(csv_path: str) -> pd.DataFrame:
    """
    Read an IMEC tracker CSV and return a cleaned DataFrame.

    IMEC CSVs have 24 columns (plus an unnamed row-index column).
    Key columns used downstream: ID, timestamp, x, y, YOLO_cls.

    Args:
        csv_path : Path to the CSV file

    Returns:
        DataFrame with cleaned tracker data (NaN x/y rows removed).

    Raises:
        ValueError : If required columns are missing.
    """
    df = pd.read_csv(csv_path)

    # Drop the auto-generated unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Verify required columns
    required = [COL_TRACK_ID, COL_TIMESTAMP, COL_X, COL_Y]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {missing}")

    # Coerce timestamp to int (each timestamp = one video frame)
    df[COL_TIMESTAMP] = df[COL_TIMESTAMP].astype(int)

    # Drop rows with missing coordinates (cannot materialise these)
    df.dropna(subset=[COL_X, COL_Y], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────
#  TRACK ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────

def calculate_track_movement(df: pd.DataFrame, track_id: int) -> float:
    """
    Total displacement for a track = sum of consecutive Euclidean distances.

    Args:
        df       : DataFrame containing at least COL_TRACK_ID, COL_TIMESTAMP, COL_X, COL_Y
        track_id : Track to measure

    Returns:
        Total displacement in coordinate units (metres for IMEC data).
    """
    track = (df[df[COL_TRACK_ID] == track_id][[COL_TIMESTAMP, COL_X, COL_Y]]
             .sort_values(COL_TIMESTAMP))

    if len(track) < 2:
        return 0.0

    dx = np.diff(track[COL_X].values)
    dy = np.diff(track[COL_Y].values)
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))


def get_majority_class(df: pd.DataFrame, track_id: int) -> int:
    """
    Return the most-frequent YOLO class label for *track_id*.

    Falls back to 0 ("person") when the class column is absent or empty.
    """
    if COL_CLASS not in df.columns:
        return 0

    classes = df[df[COL_TRACK_ID] == track_id][COL_CLASS].dropna()
    if classes.empty:
        return 0
    return int(classes.mode().iloc[0])


# ─────────────────────────────────────────────────────────────────────
#  PRESENCE MAP + CONTIGUOUS-RUN LOGIC
# ─────────────────────────────────────────────────────────────────────

def build_presence_map(df: pd.DataFrame):
    """
    Build per-track presence maps for efficient overlap computation.

    Returns:
        frames_by_tid   : dict  trackId → sorted numpy array of frame indices
        frameset_by_tid : dict  trackId → set of frame indices
    """
    frames_by_tid   = {}
    frameset_by_tid = {}

    for tid, grp in df.groupby(COL_TRACK_ID):
        frames = np.sort(grp[COL_TIMESTAMP].unique())
        frames_by_tid[tid]   = frames
        frameset_by_tid[tid] = set(frames)

    return frames_by_tid, frameset_by_tid


def _longest_run(arr: np.ndarray) -> tuple:
    """
    Find the longest run of *consecutive* integers in a sorted array.

    Uses vectorised np.diff → gap detection → run-length encoding.

    Returns:
        (start, end, length)   or   (None, None, 0) if empty.
    """
    if len(arr) == 0:
        return (None, None, 0)
    if len(arr) == 1:
        return (int(arr[0]), int(arr[0]), 1)

    diffs = np.diff(arr)
    breaks = np.where(diffs != 1)[0]

    if len(breaks) == 0:
        # The whole array is one contiguous run
        return (int(arr[0]), int(arr[-1]), len(arr))

    run_starts  = np.concatenate([[0], breaks + 1])
    run_ends    = np.concatenate([breaks, [len(arr) - 1]])
    run_lengths = run_ends - run_starts + 1

    best = np.argmax(run_lengths)
    return (int(arr[run_starts[best]]),
            int(arr[run_ends[best]]),
            int(run_lengths[best]))


def longest_common_contiguous(combo, frameset_by_tid):
    """
    Longest contiguous frame range where **all** tracks in *combo* are present.

    Steps:
      1. Intersect frame-sets of all tracks.
      2. Sort the intersection.
      3. Find the longest consecutive run via _longest_run().

    Returns:
        (start_frame, end_frame, length)
    """
    common = None
    for tid in combo:
        fs = frameset_by_tid.get(tid, set())
        common = fs.copy() if common is None else common & fs

    if not common:
        return (None, None, 0)

    common_arr = np.sort(np.array(list(common), dtype=np.int64))
    return _longest_run(common_arr)


# ─────────────────────────────────────────────────────────────────────
#  ROW MATERIALISATION
# ─────────────────────────────────────────────────────────────────────

def rows_for_interval(df_indexed, combo, start, target_len, track_classes):
    """
    Extract x, y data for each track in *combo* over a fixed frame window.

    Tracks are re-indexed 0, 1, … and frame numbers become relative to
    *start* (i.e. sample 0, 1, …, target_len-1).

    Args:
        df_indexed    : DataFrame indexed by (COL_TRACK_ID, COL_TIMESTAMP)
        combo         : Tuple of original track IDs
        start         : Absolute start frame
        target_len    : Number of frames to extract
        track_classes : Dict  trackId → majority class ID

    Returns:
        List of row dicts (keys: config_index, sample, trackId, x, y, class_id).
        Empty list if any track is missing from the index.
    """
    end = start + target_len - 1
    reindex_map = {tid: i for i, tid in enumerate(sorted(combo))}
    rows = []

    for tid in sorted(combo):
        try:
            track_data = df_indexed.loc[tid]

            if isinstance(track_data, pd.Series):
                # Edge case: track has a single observation in the index
                frame = track_data.name
                if start <= frame <= end:
                    rows.append({
                        "config_index": None,
                        "sample":   int(frame - start),
                        "trackId":  reindex_map[tid],
                        "x":        float(track_data[COL_X]),
                        "y":        float(track_data[COL_Y]),
                        "class_id": track_classes.get(tid, 0),
                    })
            else:
                # Normal case: multiple frames for this track
                in_range = track_data[
                    (track_data.index >= start) & (track_data.index <= end)
                ]
                for frame_abs, row in in_range.iterrows():
                    if pd.notna(row[COL_X]) and pd.notna(row[COL_Y]):
                        rows.append({
                            "config_index": None,
                            "sample":   int(frame_abs - start),
                            "trackId":  reindex_map[tid],
                            "x":        float(row[COL_X]),
                            "y":        float(row[COL_Y]),
                            "class_id": track_classes.get(tid, 0),
                        })
        except KeyError:
            continue

    return rows


# ─────────────────────────────────────────────────────────────────────
#  PER-CSV PROCESSING
# ─────────────────────────────────────────────────────────────────────

def process_one_csv(
    csv_path: str,
    video_dir: str | None,
    output_dir: str,
    target_length: int,
    min_movement: float,
    max_configs: int | None,
    scene_name: str,
    fps: int,
) -> tuple:
    """
    Process one IMEC tracker CSV → configurations + JSON sidecars.

    Returns:
        (num_configs_created, list_of_report_row_dicts)
    """
    # ── 1. Read & validate ─────────────────────────────────────────
    df = read_imec_csv(csv_path)
    total_tracks = df[COL_TRACK_ID].nunique()
    print(f"  Loaded {len(df)} observations, {total_tracks} unique tracks")

    min_frame = int(df[COL_TIMESTAMP].min())
    max_frame = int(df[COL_TIMESTAMP].max())
    print(f"  Frame range: {min_frame} -> {max_frame} "
          f"({max_frame - min_frame + 1} frames)")

    # ── 2. Filter stationary tracks ────────────────────────────────
    moving_tracks = []
    track_classes: dict[int, int] = {}

    for tid in df[COL_TRACK_ID].unique():
        movement = calculate_track_movement(df, tid)
        if movement >= min_movement:
            moving_tracks.append(tid)
            track_classes[tid] = get_majority_class(df, tid)

    print(f"  Moving tracks (>= {min_movement}m): "
          f"{len(moving_tracks)} / {total_tracks}")

    if len(moving_tracks) < 2:
        print("  Skipping: need at least 2 moving tracks")
        return 0, []

    df = df[df[COL_TRACK_ID].isin(moving_tracks)].copy()

    # ── 3. Build presence maps ─────────────────────────────────────
    frames_by_tid, frameset_by_tid = build_presence_map(df)

    # ── 4. Generate all track pairs & compute overlaps ─────────────
    all_pairs = list(itertools.combinations(moving_tracks, 2))
    print(f"  Track pairs to evaluate: {len(all_pairs)}")

    valid_pairs = []
    for t1, t2 in all_pairs:
        combo = (t1, t2)
        s, e, L = longest_common_contiguous(combo, frameset_by_tid)
        if L >= target_length:
            valid_pairs.append({
                "combo":          combo,
                "overlap_start":  s,
                "overlap_end":    e,
                "overlap_length": L,
                "classes":        (track_classes[t1], track_classes[t2]),
            })

    print(f"  Pairs with >= {target_length} overlapping frames: "
          f"{len(valid_pairs)}")

    if not valid_pairs:
        print("  Skipping: no pairs meet the target length")
        return 0, []

    # Best pairs first (longest overlap)
    valid_pairs.sort(key=lambda x: x["overlap_length"], reverse=True)

    # ── 5. Resolve video path ──────────────────────────────────────
    video_path = None
    if video_dir:
        video_path = infer_video_path(csv_path, video_dir, scene_name)
        if video_path:
            print(f"  Linked video: {os.path.basename(video_path)}")
        else:
            print("  Warning: no matching video found")

    # ── 6. Build configurations ────────────────────────────────────
    df_indexed = df.set_index([COL_TRACK_ID, COL_TIMESTAMP]).sort_index()

    csv_stem = Path(csv_path).stem          # e.g. "tracks_df_750"

    kept_rows: list[dict] = []
    report_rows: list[dict] = []
    cfg_idx = 0

    for p in valid_pairs:
        combo = p["combo"]
        os_, oe_, ol_ = (p["overlap_start"],
                         p["overlap_end"],
                         p["overlap_length"])
        c1, c2 = p["classes"]
        classes_str = (f"{CLASS_NAMES.get(c1, str(c1))},"
                       f"{CLASS_NAMES.get(c2, str(c2))}")

        # Determine the frame window (from the start of the overlap)
        start_candidate = os_
        end_candidate   = start_candidate + target_length - 1
        if end_candidate > oe_:
            start_candidate = oe_ - target_length + 1
            end_candidate   = oe_

        # Materialise rows
        rows = rows_for_interval(
            df_indexed, combo, start_candidate, target_length, track_classes
        )

        if not rows:
            report_rows.append({
                "source_csv": os.path.basename(csv_path),
                "config_index": None,
                "combo": str(combo),
                "classes": classes_str,
                "kept": False,
                "reason": "Materialization gap",
                "overlap_length": ol_,
            })
            continue

        # Validate sample counts per track
        samples_per_track: dict[int, int] = {}
        for r in rows:
            samples_per_track[r["trackId"]] = samples_per_track.get(
                r["trackId"], 0) + 1

        if len(samples_per_track) != 2:
            report_rows.append({
                "source_csv": os.path.basename(csv_path),
                "config_index": None,
                "combo": str(combo),
                "classes": classes_str,
                "kept": False,
                "reason": f"Missing track ({len(samples_per_track)}/2)",
                "overlap_length": ol_,
            })
            continue

        counts = list(samples_per_track.values())
        if counts[0] != target_length or counts[1] != target_length:
            report_rows.append({
                "source_csv": os.path.basename(csv_path),
                "config_index": None,
                "combo": str(combo),
                "classes": classes_str,
                "kept": False,
                "reason": f"Wrong sample count: {counts}",
                "overlap_length": ol_,
            })
            continue

        # ── Valid configuration! ───────────────────────────────────
        config_name = f"{csv_stem}_cfg{cfg_idx:04d}"

        for r in rows:
            r["config_index"] = cfg_idx
        kept_rows.extend(rows)

        # ── Write sidecar JSON metadata ────────────────────────────
        # This JSON links the configuration back to the source media so
        # that media_linker.py can open the video at the right frame.
        metadata = {
            "config_name":        config_name,
            "config_index":       cfg_idx,
            "source_csv":         os.path.basename(csv_path),
            "source_csv_path":    os.path.abspath(csv_path),
            "source_media_path":  video_path,          # None when no video dir
            "media_type":         "video" if video_path else None,
            "start_time":         int(start_candidate),
            "end_time":           int(end_candidate),
            "time_unit":          "frames",
            "fps":                fps,
            "start_time_seconds": round(start_candidate / fps, 4),
            "target_length":      target_length,
            "track_ids_original":   [int(t) for t in combo],
            "track_ids_reindexed":  list(range(len(combo))),
            "track_classes": {
                str(t): CLASS_NAMES.get(track_classes.get(t, 0),
                                        str(track_classes.get(t, 0)))
                for t in combo
            },
            "scene_name": scene_name,
        }

        json_path = os.path.join(output_dir, f"{config_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        report_rows.append({
            "source_csv": os.path.basename(csv_path),
            "config_index": cfg_idx,
            "combo": str(combo),
            "classes": classes_str,
            "kept": True,
            "reason": "",
            "overlap_length": ol_,
        })

        cfg_idx += 1

        if max_configs is not None and cfg_idx >= max_configs:
            print(f"  Reached max_configs={max_configs}. Stopping.")
            break

    # ── 7. Save configuration CSVs ─────────────────────────────────
    if kept_rows:
        cfg_df = pd.DataFrame(kept_rows)
        cfg_df.sort_values(
            ["config_index", "sample", "trackId"],
            inplace=True, ignore_index=True,
        )

        # With class column
        cfg_path_cl = os.path.join(
            output_dir, f"{csv_stem}_configs_CL.csv")
        cfg_df[["config_index", "sample", "trackId", "x", "y", "class_id"]]\
            .to_csv(cfg_path_cl, index=False, header=False)

        # Without class column
        cfg_path_nc = os.path.join(
            output_dir, f"{csv_stem}_configs_NC.csv")
        cfg_df[["config_index", "sample", "trackId", "x", "y"]]\
            .to_csv(cfg_path_nc, index=False, header=False)

        print(f"  Saved {cfg_idx} configs -> {os.path.basename(cfg_path_cl)}")
    else:
        print("  No valid configurations found")

    return cfg_idx, report_rows


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Configuration Generator for IMEC tracker data. "
                    "Finds co-occurring track pairs and extracts "
                    "fixed-length configurations with JSON sidecars.",
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing tracker CSV files (tracks_df_*.csv)")
    parser.add_argument(
        "--video_dir", default=None,
        help="Directory containing MP4 videos (visual_*.mp4). "
             "Omit to skip media linking.")
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for configs + JSON metadata")
    parser.add_argument(
        "--target_length", type=int, default=TARGET_LENGTH,
        help=f"Frames per configuration (default: {TARGET_LENGTH})")
    parser.add_argument(
        "--min_movement", type=float, default=MIN_MOVEMENT,
        help=f"Min track displacement in metres (default: {MIN_MOVEMENT})")
    parser.add_argument(
        "--max_configs", type=int, default=None,
        help="Max configurations per CSV (default: unlimited)")
    parser.add_argument(
        "--scene_name", default="",
        help="Scene name for CSV↔MP4 mapping "
             "(e.g. '10_10_new', '10_55_new'). "
             "Auto-detected from input_dir if omitted.")
    parser.add_argument(
        "--fps", type=int, default=FPS,
        help=f"Source video frame rate (default: {FPS})")

    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Discover tracker CSVs
    csv_pattern = os.path.join(args.input_dir, "tracks_df_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"Error: No tracks_df_*.csv files in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Auto-detect scene name from path when not provided ─────────
    scene_name = args.scene_name
    if not scene_name:
        low = args.input_dir.lower()
        if "10_10" in low:
            scene_name = "10_10_new"
        elif "10_55" in low:
            scene_name = "10_55_new"

    # ── Print header ───────────────────────────────────────────────
    print("=" * 70)
    print("IMEC Configuration Generator")
    print("=" * 70)
    print(f"  Input dir     : {args.input_dir}")
    print(f"  Video dir     : {args.video_dir or '(not provided)'}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Target length : {args.target_length} frames")
    print(f"  Min movement  : {args.min_movement} m")
    print(f"  Max configs   : {args.max_configs or 'unlimited'}")
    print(f"  Scene name    : {scene_name or '(auto-detect failed)'}")
    print(f"  FPS           : {args.fps}")
    print(f"  CSV files     : {len(csv_files)}")
    print()

    # ── Process each CSV ───────────────────────────────────────────
    total_configs = 0
    all_report_rows: list[dict] = []

    for csv_path in csv_files:
        csv_name = os.path.basename(csv_path)
        print(f"\n{'-' * 60}")
        print(f"Processing: {csv_name}")
        print(f"{'-' * 60}")

        try:
            n_configs, rpt = process_one_csv(
                csv_path=csv_path,
                video_dir=args.video_dir,
                output_dir=args.output_dir,
                target_length=args.target_length,
                min_movement=args.min_movement,
                max_configs=args.max_configs,
                scene_name=scene_name,
                fps=args.fps,
            )
            total_configs += n_configs
            all_report_rows.extend(rpt)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # ── Save combined report ───────────────────────────────────────
    if all_report_rows:
        report_df = pd.DataFrame(all_report_rows)
        report_path = os.path.join(args.output_dir, "config_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"\nReport saved: {report_path}")

    print(f"\n{'=' * 70}")
    print(f"Done!  Total configurations generated: {total_configs}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
