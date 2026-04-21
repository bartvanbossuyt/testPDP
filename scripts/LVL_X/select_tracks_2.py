#!/usr/bin/env python3
"""
Hard-coded version of select_tracks_and_configurations.py
Adds 'class' from meta, reindexes trackId per configuration,
and outputs both with and without 'class' columns.
'frame' in configuration outputs now resets to 0..WINDOW_SIZE-1 per config.
"""

import os
import pandas as pd
from typing import List

# ==== USER SETTINGS ====
TRACKS_PATH = "/Users/olivier/Documents/STREAMS/uniD_tracks_only_filtered/00_tracks.csv"
META_PATH   = "/Users/olivier/Documents/STREAMS/uniD-dataset-v1.1 1/data/00_tracksMeta.csv"
OUTDIR      = "/Users/olivier/Documents/STREAMS/Data_prep"

START_FRAME = 0
END_FRAME   = 50
WINDOW_SIZE = 2
MATCH_MODE  = "cover"   # exact | within | overlap | cover
# ========================


def read_tracks(tracks_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        tracks_path,
        header=None,
        names=["trackId", "frame", "x", "y"],
        dtype={"trackId": "Int64", "frame": "Int64"},
    )
    return df


def _find_class_column(cols) -> str:
    lower = {c.lower(): c for c in cols}
    if "class" in lower:
        return lower["class"]
    for candidate in ("category", "label", "type"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(
        "Meta file must contain a 'class' column (or a synonym like 'category'). "
        f"Found columns: {list(cols)}"
    )


def read_meta(meta_path: str) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    required = {"trackId", "initialFrame", "finalFrame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Meta file is missing required columns: {sorted(missing)}")

    class_col = _find_class_column(df.columns)
    if class_col != "class":
        df = df.rename(columns={class_col: "class"})

    for col in ["trackId", "initialFrame", "finalFrame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def select_track_ids(meta: pd.DataFrame, start_frame: int, end_frame: int, match: str) -> List[int]:
    if match == "exact":
        sel = meta[(meta["initialFrame"] == start_frame) & (meta["finalFrame"] == end_frame)]
    elif match == "within":
        sel = meta[(meta["initialFrame"] >= start_frame) & (meta["finalFrame"] <= end_frame)]
    elif match == "overlap":
        sel = meta[(meta["initialFrame"] <= end_frame) & (meta["finalFrame"] >= start_frame)]
    elif match == "cover":
        sel = meta[(meta["initialFrame"] == start_frame) & (meta["finalFrame"] >= end_frame)]
    else:
        raise ValueError("MATCH_MODE must be one of: exact, within, overlap, cover")

    return sel["trackId"].dropna().astype(int).tolist()


def filter_tracks_for_ids_and_range(
    tracks: pd.DataFrame,
    ids: List[int],
    start_frame: int,
    end_frame: int,
    meta_with_class: pd.DataFrame
) -> pd.DataFrame:
    if not ids:
        return tracks.head(0).copy()
    mask = (
        tracks["trackId"].isin(ids)
        & (tracks["frame"] >= start_frame)
        & (tracks["frame"] <= end_frame)
    )
    filtered = tracks.loc[mask].sort_values(["frame", "trackId"]).reset_index(drop=True)
    filtered = filtered.merge(
        meta_with_class[["trackId", "class"]],
        on="trackId", how="left"
    )
    filtered = filtered[["frame", "trackId", "x", "y", "class"]]
    return filtered


def build_config_rows(
    filtered_tracks: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    window_size: int
) -> pd.DataFrame:
    if window_size <= 0:
        raise ValueError("WINDOW_SIZE must be positive")

    total_frames = end_frame - start_frame + 1
    n_configs = total_frames // window_size  # only full windows

    ft = filtered_tracks.copy()
    ft["trackId"] = ft["trackId"].astype(int)

    rows = []
    for cfg_idx in range(n_configs):
        f0 = start_frame + cfg_idx * window_size
        frames = range(f0, f0 + window_size)
        block = ft[ft["frame"].isin(frames)]
        for _, r in block.iterrows():
            rel_frame = int(r["frame"] - f0)  # reset frame to 0..WINDOW_SIZE-1 per config
            rows.append({
                "config_index": cfg_idx,
                "frame": rel_frame,
                "trackId": int(r["trackId"]),   # original for now; reindexed later
                "x": float(r["x"]),
                "y": float(r["y"]),
                "class": r["class"],
            })

    cfg_long = pd.DataFrame(rows, columns=["config_index", "frame", "trackId", "x", "y", "class"])
    if cfg_long.empty:
        return cfg_long

    # Reindex trackId per configuration to 0..n based on sorted unique original IDs
    def _reindex_ids(group: pd.DataFrame) -> pd.DataFrame:
        unique_sorted = sorted(group["trackId"].unique())
        mapping = {tid: i for i, tid in enumerate(unique_sorted)}
        group["trackId"] = group["trackId"].map(mapping).astype(int)
        return group

    cfg_long = cfg_long.groupby("config_index", group_keys=False).apply(_reindex_ids)
    cfg_long.sort_values(["config_index", "frame", "trackId"], inplace=True, ignore_index=True)
    return cfg_long


def save_outputs(filtered_tracks: pd.DataFrame, cfg_long: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    # 1) filtered tracks (reference; absolute frames)
    tracks_out = os.path.join(outdir, "filtered_tracks3.csv")
    filtered_tracks.to_csv(tracks_out, index=False, header=False)

    # 2) configurations with class (frame reset per config)
    cfg_with_class_out = os.path.join(outdir, "configurations_long3.csv")
    cfg_long.to_csv(cfg_with_class_out, index=False, header=False)

    # 3) configurations without class (frame reset per config)
    cfg_no_class = cfg_long.drop(columns=["class"])
    cfg_no_class_out = os.path.join(outdir, "configurations_long_noclass3.csv")
    cfg_no_class.to_csv(cfg_no_class_out, index=False, header=False)

    print(f"""
✅ Wrote:
- {tracks_out}
- {cfg_with_class_out}
- {cfg_no_class_out}
""")


def main():
    print("=== Selecting tracks and building configurations ===")
    tracks = read_tracks(TRACKS_PATH)
    meta = read_meta(META_PATH)

    selected_ids = select_track_ids(meta, START_FRAME, END_FRAME, MATCH_MODE)
    if not selected_ids:
        print("⚠️  No trackIds matched your criteria. Exiting.")
        return

    filtered = filter_tracks_for_ids_and_range(tracks, selected_ids, START_FRAME, END_FRAME, meta)
    cfg_long = build_config_rows(filtered, START_FRAME, END_FRAME, WINDOW_SIZE)
    save_outputs(filtered, cfg_long, OUTDIR)


if __name__ == "__main__":
    main()
