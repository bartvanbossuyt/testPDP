# TEST Folder - Track Configuration Tools

This folder contains Python scripts for processing and visualizing traffic trajectory data from the inD drone dataset.

## 🎯 Main Scripts (Active)

### 1. `config_visualizer.py` ⭐
**Purpose**: Interactive visualization of track configurations

**Features**:
- PyQt5-based GUI with matplotlib integration
- Frame-by-frame playback with play/pause controls
- Background image overlay (intersection aerial photos)
- Track trajectory visualization (past/future paths)
- Distance measurement between tracks
- Coordinate conversion from meters to pixels

**Usage**:
```python
# Edit configuration at top of file:
CONFIG_FILE = r"path\to\your\output.csv"
BACKGROUND_IMAGE = r"path\to\XX_background.png"
ORTHO_PX_TO_METER = 0.00814636091724916  # From recordingMeta.csv
SCALE_DOWN_FACTOR = 12  # For inD dataset

# Then run:
python config_visualizer.py
```

**Input Format**: CSV from `data_prep_lvlX_clean.py` output:
```
config_index, frame, trackId, x, y[, class]
```

---

### 2. `data_prep_lvlX_clean.py` ⭐
**Purpose**: Extract track pair configurations from trajectory data

**Features**:
- Class-based filtering (e.g., car+bicycle pairs)
- Longest overlap detection between track pairs
- Distance-based filtering (MIN_DISTANCE, MAX_DISTANCE, MIN_CLOSE_FRAMES)
- Vectorized operations for performance (~5-10x faster than original)
- Deduplication of permutations

**Usage**:
```python
# Edit CONFIG section at top:
TRACKS_PATH = r"path\to\XX_tracks.csv"
META_PATH = r"path\to\XX_tracksMeta.csv"
OUTDIR = r"output\directory"

CLASS_PATTERN = "car,bicycle"  # Track classes to match
TARGET_LENGTH = 50             # Frames per configuration
MAX_CONFIGS = 50               # Optional limit

# Distance filtering (optional)
MIN_DISTANCE = None
MAX_DISTANCE = 10.0
MIN_CLOSE_FRAMES = 5

# Then run:
python data_prep_lvlX_clean.py
```

**Outputs**:
- `C{num}_{classes}_CL_{dataset}_F{length}.csv` - With class column
- `C{num}_{classes}_NC_{dataset}_F{length}.csv` - Without class column
- `C{num}_{classes}_{dataset}_F{length}_report.csv` - Summary report

---

### 3. `alter_csv_lvlX.py`
**Purpose**: Pre-processing to filter original inD tracks CSV

**What it does**:
- Extracts only essential columns: `[trackId, frame, xCenter, yCenter]`
- Removes header and saves as minimal CSV
- Batch processes entire folders

**Usage**:
```python
# Edit settings:
input_folder = "path/to/inD-dataset/data"
output_folder = "path/to/filtered_output"
columns_to_keep = ['trackId', 'frame', 'xCenter', 'yCenter']

# Run:
python alter_csv_lvlX.py
```

**Note**: Currently has Mac paths (`/Users/olivier/...`) - update before use on Windows.

---

## 🗑️ Old/Deprecated Scripts

These are superseded by `data_prep_lvlX_clean.py`:

- `data_prep_lvlX.py` - Original corrupted version (801 lines with duplicates)
- `data_prep_lvlX_backup.py` - Backup of corrupted version
- `data_prep_lvlX_optimized.py` - Intermediate version without distance filtering
- `data_prep_lvlX_distance.py` - Mac version with distance filtering

**Recommendation**: Delete these files to reduce clutter.

---

## 🔄 Alternative Workflow Scripts

### `select_tracks_2.py`
**Different approach**: Selects tracks by exact frame ranges and creates windowed configurations

**Use case**: If you need configurations based on specific time windows rather than track pair overlaps

**Features**:
- Match modes: `exact`, `within`, `overlap`, `cover`
- Reindexes trackIds per configuration
- Resets frame numbers to 0..WINDOW_SIZE-1

### `select_tracks_and_configurations.py`
**CLI version of select_tracks_2.py** with argparse interface

**Recommendation**: Keep `select_tracks_2.py` if you use windowing approach; delete `select_tracks_and_configurations.py`.

---

## 📊 Typical Workflow

1. **Pre-process data** (one-time):
   ```bash
   python alter_csv_lvlX.py
   ```

2. **Extract configurations**:
   ```bash
   python data_prep_lvlX_clean.py
   ```

3. **Visualize results**:
   ```bash
   python config_visualizer.py
   ```

---

## 📁 File Format Reference

### Input: Filtered Tracks CSV
```
trackId, frame, x, y
0, 0, 39.686, -12.169
0, 1, 39.396, -11.832
...
```

### Input: Metadata CSV (with header)
```
trackId, initialFrame, finalFrame, class, width, length, ...
0, 0, 150, car, 1.69, 3.96, ...
1, 5, 180, bicycle, 0.60, 1.80, ...
```

### Output: Configuration CSV
```
config_index, frame, trackId, x, y, class
0, 1000, 45, 12.5, 8.3, car
0, 1000, 67, 15.2, 9.1, bicycle
0, 1001, 45, 12.6, 8.4, car
0, 1001, 67, 15.3, 9.2, bicycle
...
```

---

## 🎨 Visualization Settings

### Coordinate Conversion
The visualizer converts meter coordinates to pixel coordinates for background overlay:

```python
x_pixel = (x_meters / orthoPxToMeter) / scale_down_factor
y_pixel = -(y_meters / orthoPxToMeter) / scale_down_factor
```

**Dataset-specific values**:
- **inD**: `scale_down_factor = 12`
- **rounD**: `scale_down_factor = 10`
- **exiD**: `scale_down_factor = 6`
- **uniD**: `scale_down_factor = 2`

Get `orthoPxToMeter` from `{recording}_recordingMeta.csv` file.

---

## 📦 Dependencies

```bash
pip install pandas numpy matplotlib PyQt5
```

For visualization:
- matplotlib (for image loading and plotting)
- PyQt5 (for interactive GUI)

---

## 🐛 Common Issues

### Issue: Track IDs not visible on background image
**Cause**: Wrong `orthoPxToMeter` or `scale_down_factor`
**Solution**: Check values in `{recording}_recordingMeta.csv` and `visualizer_params.json`

### Issue: Windows path errors
**Cause**: Backslashes interpreted as escape sequences
**Solution**: Use raw strings: `r"C:\path\to\file"`

### Issue: Script has Mac paths
**Cause**: `alter_csv_lvlX.py` and some old scripts have `/Users/olivier/...`
**Solution**: Update paths to Windows format before running

---

## 📝 Notes

- All scripts are configured via editing variables at the top (no command-line arguments in main scripts)
- The `_clean.py` version uses vectorized pandas/numpy operations for better performance
- Distance filtering is optional - set to `None` to disable
- Background images must match the recording number (e.g., `13_background.png` for recording 13)

---

*Last updated: November 19, 2025*
