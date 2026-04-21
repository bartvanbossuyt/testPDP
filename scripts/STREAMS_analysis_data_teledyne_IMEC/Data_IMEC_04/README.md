# Data_IMEC_04 Notes

This README summarizes the practical findings about the `Data_IMEC_04` folder structure, CSV overlap, and CSV<->MP4 linking.

## Folder structure

Current scene folders:
- `10_10_new/10_10_new/`
- `10_55_new/10_55_new/`

Each scene contains:
- `track_dataframes/tracks_df_*.csv`
- `visual_*.mp4`

## What is in a `tracks_df_*.csv`

These CSV files contain tracker/tabular data (for example: `ID`, `timestamp`, `x`, `y`, `vx`, `vy`, `L`, `T`, `R`, `B`, `X_LiDAR`, `Y_LiDAR`, etc.).

Important:
- A `tracks_df_*.csv` file does **not** contain image frames.
- Ground-truth images are **not** embedded in the CSV.
- MP4 files are the image/video source.

## CSV <-> MP4 mapping

### Scene `10_55_new`
Direct number match:
- `visual_N.mp4` <-> `tracks_df_N.csv`

Examples:
- `visual_0.mp4` <-> `tracks_df_0.csv`
- `visual_750.mp4` <-> `tracks_df_750.csv`

### Scene `10_10_new`
CSV number has a +750 offset relative to MP4 number:
- `tracks_df_(N+750).csv` <-> `visual_N.mp4`

Examples:
- `visual_0.mp4` <-> `tracks_df_750.csv`
- `visual_750.mp4` <-> `tracks_df_1500.csv`

## Timestamp/frame interpretation

Validated interpretation:
- MP4 frame rate is 15 fps.
- A 60-second clip has about 900 frames (`15 * 60 = 900`).
- Most CSV files span about 900 timestamps.
- File indices advance by 750 (sliding window step), so neighboring windows usually overlap by around 150 timestamps.

So, the practical model is:
- 1 timestamp is approximately 1 frame step.
- 1 CSV is approximately a 900-frame window.
- Consecutive CSVs are shifted by 750 frames, yielding overlap.

## Overlap between CSV files

### `10_10_new`
- Adjacent CSVs overlap consistently by about 148-151 timestamps (~16-17%).
- This is a clean sliding-window pattern.

### `10_55_new`
- Most adjacent CSVs overlap by about 150 timestamps.
- There are irregular cases:
  - Very large overlap between `tracks_df_0.csv` and `tracks_df_750.csv`.
  - A small gap between `tracks_df_4500.csv` and `tracks_df_5250.csv`.
  - Some smaller-than-usual overlaps for specific adjacent pairs.

## Practical takeaway

Yes, CSV and MP4 can be linked reliably at clip level using filename numbering and scene-specific mapping rules.

For strict frame-perfect overlay, verify per scene:
- exact FPS,
- exact meaning of `timestamp` (frame index vs. derived index),
- calibration/projection details when drawing in image pixel space.
