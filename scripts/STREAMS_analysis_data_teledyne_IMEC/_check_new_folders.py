"""Quick comparison of old vs new Data_IMEC_04 folders."""
import pandas as pd
import os

base = os.path.join(os.path.dirname(__file__), "Data_IMEC_04")

# ── 1. Compare CSV file lists ──
scenes = {
    "10_10_old": os.path.join(base, "10_10", "track_dataframes"),
    "10_10_new": os.path.join(base, "10_10_new", "10_10_new", "track_dataframes"),
    "10_55_old": os.path.join(base, "10_55", "track_dataframes"),
    "10_55_new": os.path.join(base, "10_55_new", "10_55_new", "track_dataframes"),
}

print("=" * 60)
print("1. CSV FILE LISTS")
print("=" * 60)
for name, path in scenes.items():
    csvs = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
    nums = [f.replace("tracks_df_", "").replace(".csv", "") for f in csvs]
    print(f"  {name:15s}: {len(csvs)} files  nums={nums}")

# ── 2. MP4 files in new folders ──
print("\n" + "=" * 60)
print("2. MP4 FILES (new folders only)")
print("=" * 60)
for scene in ["10_10_new", "10_55_new"]:
    mp4_dir = os.path.join(base, scene, scene)
    mp4s = sorted([f for f in os.listdir(mp4_dir) if f.endswith(".mp4")])
    nums = [f.replace("visual_", "").replace(".mp4", "") for f in mp4s]
    print(f"  {scene:15s}: {len(mp4s)} mp4s  nums={nums}")

# ── 3. Compare schemas ──
print("\n" + "=" * 60)
print("3. COLUMN COMPARISON (tracks_df_750.csv)")
print("=" * 60)
for label, pair in [("10_10", ("10_10_old", "10_10_new")), ("10_55", ("10_55_old", "10_55_new"))]:
    old = pd.read_csv(os.path.join(scenes[pair[0]], "tracks_df_750.csv"), nrows=0)
    new = pd.read_csv(os.path.join(scenes[pair[1]], "tracks_df_750.csv"), nrows=0)
    old_cols = list(old.columns)
    new_cols = list(new.columns)
    match = old_cols == new_cols
    print(f"\n  {label} columns match: {match}")
    if not match:
        added = set(new_cols) - set(old_cols)
        removed = set(old_cols) - set(new_cols)
        if added: print(f"    NEW has extra cols: {added}")
        if removed: print(f"    OLD has extra cols: {removed}")
    else:
        print(f"    Columns ({len(old_cols)}): {old_cols}")

# ── 4. Compare data content for a shared file ──
print("\n" + "=" * 60)
print("4. DATA COMPARISON (tracks_df_750.csv)")
print("=" * 60)
for label, pair in [("10_10", ("10_10_old", "10_10_new")), ("10_55", ("10_55_old", "10_55_new"))]:
    old = pd.read_csv(os.path.join(scenes[pair[0]], "tracks_df_750.csv"))
    new = pd.read_csv(os.path.join(scenes[pair[1]], "tracks_df_750.csv"))
    print(f"\n  {label}:")
    print(f"    OLD: {old.shape[0]} rows, {old['ID'].nunique()} unique IDs, ts range {old['timestamp'].min()}-{old['timestamp'].max()}")
    print(f"    NEW: {new.shape[0]} rows, {new['ID'].nunique()} unique IDs, ts range {new['timestamp'].min()}-{new['timestamp'].max()}")
    # Check if data is identical
    if old.shape == new.shape:
        cols_to_check = [c for c in old.columns if c in new.columns]
        same = old[cols_to_check].equals(new[cols_to_check])
        print(f"    Data identical: {same}")
    else:
        print(f"    Different row counts -> not identical")

# ── 5. Timestamp ranges per CSV to find MP4 mapping ──
print("\n" + "=" * 60)
print("5. TIMESTAMP RANGES PER CSV (new folders)")
print("=" * 60)
for scene in ["10_10_new", "10_55_new"]:
    td_dir = scenes[scene]
    csvs = sorted([f for f in os.listdir(td_dir) if f.endswith(".csv")])
    print(f"\n  {scene}:")
    for csv in csvs:
        df = pd.read_csv(os.path.join(td_dir, csv))
        num = csv.replace("tracks_df_", "").replace(".csv", "")
        print(f"    {csv:30s}  ts: {df['timestamp'].min():8.0f} - {df['timestamp'].max():8.0f}  rows={df.shape[0]:6d}  IDs={df['ID'].nunique():3d}")

# ── 6. Check MP4-CSV number relationship ──
print("\n" + "=" * 60)
print("6. MP4 <-> CSV NUMBER MAPPING")
print("=" * 60)
for scene in ["10_10_new", "10_55_new"]:
    mp4_dir = os.path.join(base, scene, scene)
    td_dir = scenes[scene]
    mp4_nums = sorted([int(f.replace("visual_", "").replace(".mp4", "")) for f in os.listdir(mp4_dir) if f.endswith(".mp4")])
    csv_nums = sorted([int(f.replace("tracks_df_", "").replace(".csv", "")) for f in os.listdir(td_dir) if f.endswith(".csv")])
    print(f"\n  {scene}:")
    print(f"    MP4 numbers: {mp4_nums}")
    print(f"    CSV numbers: {csv_nums}")
    # Check offset patterns
    if len(mp4_nums) == len(csv_nums):
        offsets = [c - m for c, m in zip(csv_nums, mp4_nums)]
        if len(set(offsets)) == 1:
            print(f"    -> 1:1 mapping with constant offset of {offsets[0]}")
            print(f"       visual_N.mp4 <-> tracks_df_{offsets[0]+0}.csv  (i.e. CSV = MP4_num + {offsets[0]})")
        else:
            print(f"    -> No constant offset: {offsets}")
    else:
        # Try direct match
        shared = sorted(set(mp4_nums) & set(csv_nums))
        only_mp4 = sorted(set(mp4_nums) - set(csv_nums))
        only_csv = sorted(set(csv_nums) - set(mp4_nums))
        print(f"    Shared numbers: {shared}")
        if only_mp4: print(f"    Only in MP4: {only_mp4}")
        if only_csv: print(f"    Only in CSV: {only_csv}")
