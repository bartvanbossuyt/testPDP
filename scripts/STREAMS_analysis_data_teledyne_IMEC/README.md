# Olivier's Code - PDP Analysis Scripts

This folder contains Olivier's modified versions of the PDP analysis scripts. The modifications are built upon the base code with several enhancements and adaptations for specific use cases
## Overview

The scripts in this folder maintain the core functionality of the base code while introducing key improvements in configuration flexibility, data handling, and output management. Most notably, these scripts support data with an extra column (classes)

---

## Key Differences from Base Code

### 1. **av.py** - Configuration & Data Loader

**Major Enhancements:**
- **Dual Dataset Support**: Automatically detects and handles both 5-column (no class) and 6-column (with class) CSV files
- **Flexible Class Handling**: 
  - Creates separate `Df_classes` dataframe when class column is present
  - Exports class data to `{dataset_name_exclusive}__Df_classes.csv`
  - Avoids forcing class data into float arrays

  
**Base code differences:**
- Base code only supports 5-column datasets without class information


### 2. **GUI.py** - Graphical User Interface

**Status**: Identical to base code
- Both versions implement a basic Dash web interface for parameter input
- Placeholder implementation for future development

### 3. **N_VA_HeatMap.py** - Heat Map Visualization

**Key Changes:**
- **Figure Creation**: Uses `fig, ax = plt.subplots(figsize=(20, 15), dpi=300)` approach (more explicit)
- **Removed**: `plt.figure(figsize=(20, 15), dpi=300.0)` call that existed in base code
- **Cleaner Structure**: More streamlined figure initialization

**Base code differences:**
- Base code creates figure twice (once with `plt.figure()`, then with `plt.subplots()`)
- Otherwise functionality is identical

### 4. **N_VA_TSNE.py** - t-SNE Dimensionality Reduction

**Status**: Identical to base code
- Performs t-SNE transformation on distance matrices
- Supports all four PDP variants (fundamental, buffer, rough, bufferrough)
- Uses adaptive perplexity calculation: `perp = min(30, M//3)` with minimum of 5
- Saves visualizations to centralized output folder structure (`{OUTPUT_FOLDER}/tsne/`)

### 5. **N_VA_HeatMap_OG.py** - Original Heat Map Version

**Unique to Olivier's Code**: This file doesn't exist in base code
- Appears to be an original/backup version of the heatmap script
- Preserved for reference or rollback purposes


**Other Scripts**: The following scripts appear to be functionally equivalent or very similar to base code:
- `N_Moving_Objects.py`
- `N_PDP.py`
- `N_T_OB.py`
- `N_T_Report.py`
- `N_VA_ClusterMap.py`
- `N_VA_DynamicAbsolute.py`
- `N_VA_HClust.py`
- `N_VA_Inverse.py`
- `N_VA_Mds.py`
- `N_VA_Mds_autoencoder.py`
- `N_VA_StaticAbsolute.py`
- `N_VA_StaticFinetuned.py`
- `N_VA_StaticRelative.py`
- `N_VA_TopK.py`

---

## File Not Present in Olivier's Code

- **N_D_CreateDataset.py** - Only exists in base code, used for dataset creation
- **__init__.py** - Python package initialization file, only in base code

---

## Usage Notes

### Configuration Priority
1. Set dataset paths in `av.py` (currently configured for macOS inD dataset)
2. Configure `OUTPUT_FOLDER` for centralized output location
3. Configure `INPUT_DISTANCE_MATRIX` for distance matrix file location
4. Adjust PDP parameters (window_length_tst, buffer_x/y, rough_x/y) as needed
5. Enable/disable analysis modules using the `N_VA_*` flags

### Dataset Requirements
- This version is optimized for the **inD (Intersection Drone) dataset**
- Supports both classified and non-classified configuration files
- Automatically handles non-integer point IDs through internal mapping

### Output Organization
All outputs are saved to subdirectories under `OUTPUT_FOLDER`:
- `/heatmap/` - Heat map visualizations
- `/tsne/` - t-SNE visualizations
- `/mds/` - MDS visualizations
- Additional folders created by other analysis scripts

---

## Recommended Workflow

1. **Configure**: Update paths in `av.py` for your system and dataset
2. **Select Analysis**: Enable desired analysis flags (`N_VA_*` variables)
3. **Run**: Execute scripts in sequence (starting with data loading/PDP calculation)
4. **Review**: Check output folders for generated visualizations and matrices

---

## Technical Notes

- **Platform**: Primarily developed and tested on macOS
- **Python Environment**: Designed for Python 3.10.9 ('base': conda) according to comments
- **Dependencies**: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, SciPy
- **Distance Matrix**: All visualization scripts support flexible input paths via `av.INPUT_DISTANCE_MATRIX`

---

## Comparison Summary

| Feature | Base Code | Olivier's Code |
|---------|-----------|----------------|
| Class column support | ❌ | ✅ |
| Dataset auto-detection | ❌ | ✅ |
| t-SNE flag | ❌ | ✅ |
| HeatMap_OG variant | ❌ | ✅ |
| CreateDataset script | ✅ | ❌ |
| Window length default | 2 | 3 |


---

*Last updated: October 2025*
