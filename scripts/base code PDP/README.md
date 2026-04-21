# Base Code PDP - PDP Analysis Scripts

This folder contains the cleaned and consolidated base code for PDP (Pairwise Distance Pattern) analysis. These scripts are derived from Olivier's code with improvements and cleanup.

## Overview

The scripts in this folder provide core functionality for analyzing moving object configurations using PDP methodology. They support both 5-column (no class) and 6-column (with class) CSV datasets.

---

## Scripts

### Core Configuration
- **av.py** - Configuration & data loader with settings for all analysis parameters

### Main Runners
- **N_Moving_Objects.py** - Unified runner for PDP variants (fundamental, buffer, rough, buffer+rough)
- **N_PDP.py** - Core PDP transformation from configurations to distance matrices

### Utilities
- **N_T_OB.py** - Transforms datasets to include buffer zones in x/y directions
- **N_T_Report.py** - Generates PDF reports with all visualizations
- **GUI.py** - Basic Dash web interface (placeholder for future development)

### Visual Analytics
- **N_VA_StaticAbsolute.py** - Absolute static visualizations with class support
- **N_VA_StaticAbsolute_color.py** - Absolute static visualizations with class support, enhanced with time-dependent coloring and stationary markers
- **N_VA_StaticRelative.py** - Relative static visualizations with scaling
- **N_VA_StaticFinetuned.py** - Finetuned visualizations (e.g., tennis pitch overlays)
- **N_VA_DynamicAbsolute.py** - Animated trajectory visualizations
- **N_VA_HeatMap.py** - Heat map of distance matrices
- **N_VA_HClust.py** - Hierarchical clustering dendrograms
- **N_VA_ClusterMap.py** - Cluster map visualizations
- **N_VA_Mds.py** - MDS dimensionality reduction
- **N_VA_Mds_autoencoder.py** - Autoencoder-based dimensionality reduction
- **N_VA_TSNE.py** - t-SNE dimensionality reduction
- **N_VA_TopK.py** - Top-K nearest neighbor visualizations
- **N_VA_Inverse.py** - Inverse problem: generate similar configurations

---

## Key Features

1. **Dual Dataset Support**: Automatically detects and handles both 5-column (no class) and 6-column (with class) CSV files

2. **Flexible Class Handling**: 
   - Creates separate `Df_classes` dataframe when class column is present
   - Exports class data to separate files
   - Class-based coloring in visualizations

3. **Centralized Output**: All outputs go to `OUTPUT_FOLDER` defined in av.py

---

## Usage

1. Configure `av.py` with your dataset paths and analysis settings
2. Run `N_Moving_Objects.py` to execute the full analysis pipeline
3. Results are saved to the configured output folder

---

## Changes from Original Code

- Fixed deprecated Dash imports (`dash_core_components` → `dash.dcc`)
- Fixed `N_T_OB.py` to use `av.buffer_x/y` instead of hardcoded values
- Fixed duplicate code in `N_VA_TopK.py`
- Fixed deprecated `df.append()` in `N_VA_Inverse.py` (now uses `pd.concat`)
- Removed duplicate imports across files
- Cleaned up commented-out code and improved documentation
- Removed `N_VA_HeatMap_OG.py` (duplicate older version)
