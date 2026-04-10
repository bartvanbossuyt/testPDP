"""
Create a quick overview for a single CSV file (track data or ground truth).

Usage:
    python overview_one_csv.py --csv "Data_IMEC_04/10_10_new/10_10_new/track_dataframes/tracks_df_750.csv"

Output:
    - Console summary (shape, columns, time range, IDs, missing values)
    - One PNG figure with 4 overview plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_track_column(df: pd.DataFrame) -> str | None:
    for col in ["ID", "track_id", "id", "TrackID", "trackId"]:
        if col in df.columns:
            return col
    return None


def detect_time_column(df: pd.DataFrame) -> str | None:
    for col in ["timestamp", "timestep", "frame", "time", "t"]:
        if col in df.columns:
            return col
    return None


def print_overview(df: pd.DataFrame, csv_path: Path, track_col: str | None, time_col: str | None) -> None:
    print("=" * 72)
    print("CSV OVERVIEW")
    print("=" * 72)
    print(f"File: {csv_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")

    if track_col:
        print(f"Track column: {track_col}")
        print(f"Unique tracks: {df[track_col].nunique():,}")

    if time_col:
        tmin = pd.to_numeric(df[time_col], errors="coerce").min()
        tmax = pd.to_numeric(df[time_col], errors="coerce").max()
        print(f"Time column: {time_col}")
        print(f"Time range: {tmin} -> {tmax}")

    key_cols = [c for c in ["x", "y", "vx", "vy", "X_LiDAR", "Y_LiDAR", "class", "YOLO_cls"] if c in df.columns]
    if key_cols:
        print("\nMissing values (%):")
        for c in key_cols:
            miss = df[c].isna().mean() * 100
            print(f"  {c:10s}: {miss:6.2f}%")


def build_plot(df: pd.DataFrame, out_png: Path, track_col: str | None, time_col: str | None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1) Bird's-eye trajectories
    if {"x", "y"}.issubset(df.columns) and track_col:
        ids = df[track_col].dropna().unique()[:120]
        for i, tid in enumerate(ids):
            track = df[df[track_col] == tid].copy()
            if time_col and time_col in track.columns:
                track[time_col] = pd.to_numeric(track[time_col], errors="coerce")
                track = track.sort_values(time_col)
            ax1.plot(track["x"], track["y"], alpha=0.6, linewidth=1)

        ax1.scatter([0], [0], c="red", marker="^", s=120, label="Sensor")
        ax1.set_title("Trajectories (x/y)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid(alpha=0.3)
        ax1.axis("equal")
        ax1.legend(loc="best")
    else:
        ax1.text(0.5, 0.5, "Missing x/y or track ID column", ha="center", va="center")
        ax1.set_title("Trajectories")

    # 2) Active objects over time
    if track_col and time_col and time_col in df.columns:
        tmp = df[[track_col, time_col]].copy()
        tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
        active = tmp.groupby(time_col)[track_col].nunique().sort_index()
        ax2.plot(active.index, active.values, color="tab:blue")
        ax2.fill_between(active.index, active.values, alpha=0.2, color="tab:blue")
        ax2.set_title("Active tracks over time")
        ax2.set_xlabel(time_col)
        ax2.set_ylabel("# active tracks")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Missing time/track columns", ha="center", va="center")
        ax2.set_title("Active tracks over time")

    # 3) Track length distribution
    if track_col:
        lens = df.groupby(track_col).size()
        ax3.hist(lens, bins=30, color="tab:orange", edgecolor="black", alpha=0.8)
        ax3.set_title("Detections per track")
        ax3.set_xlabel("rows per track")
        ax3.set_ylabel("count")
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Missing track column", ha="center", va="center")
        ax3.set_title("Detections per track")

    # 4) Speed distribution
    if {"vx", "vy"}.issubset(df.columns):
        speed = np.sqrt(pd.to_numeric(df["vx"], errors="coerce") ** 2 + pd.to_numeric(df["vy"], errors="coerce") ** 2)
        speed = speed.dropna()
        ax4.hist(speed, bins=40, color="tab:green", edgecolor="black", alpha=0.8)
        ax4.set_title("Speed distribution")
        ax4.set_xlabel("speed")
        ax4.set_ylabel("count")
        ax4.grid(alpha=0.3)
    elif {"x", "y"}.issubset(df.columns) and track_col and time_col:
        tmp = df[[track_col, time_col, "x", "y"]].copy()
        tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
        tmp["x"] = pd.to_numeric(tmp["x"], errors="coerce")
        tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")
        tmp = tmp.sort_values([track_col, time_col])
        dx = tmp.groupby(track_col)["x"].diff()
        dy = tmp.groupby(track_col)["y"].diff()
        dt = tmp.groupby(track_col)[time_col].diff()
        speed = (np.sqrt(dx**2 + dy**2) / dt.replace(0, np.nan)).dropna()
        ax4.hist(speed, bins=40, color="tab:green", edgecolor="black", alpha=0.8)
        ax4.set_title("Approx speed from x/y")
        ax4.set_xlabel("distance per time-step")
        ax4.set_ylabel("count")
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Missing velocity columns", ha="center", va="center")
        ax4.set_title("Speed distribution")

    fig.suptitle(f"Overview: {out_png.stem}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create overview for one CSV file.")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--out", default=None, help="Output PNG path (optional)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop autogenerated index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    track_col = detect_track_column(df)
    time_col = detect_time_column(df)

    out_png = Path(args.out) if args.out else csv_path.with_name(f"{csv_path.stem}_overview.png")

    print_overview(df, csv_path, track_col, time_col)
    build_plot(df, out_png, track_col, time_col)

    print("\nSaved figure:")
    print(out_png)


if __name__ == "__main__":
    main()
