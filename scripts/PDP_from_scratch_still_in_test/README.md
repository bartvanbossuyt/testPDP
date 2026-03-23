# 📦 PDP Package v2.0 (Modular Implementation)

> 📂 **Previous name:** `pdp/`  
> 📁 **Archived copy:** `scripts/_archive/pdp/`  
> 🔧 **Type:** Clean, modular Python package

A **clean, modular implementation** of (PDP) analysis for moving object trajectories. This is the most modern and maintainable version of the PDP codebase.

## Overview

PDP Analysis compares configurations of moving objects by:
1. Creating inequality matrices that capture positional relationships between points
2. Computing distance matrices between configurations based on these relationships
3. Visualizing results through various methods (MDS, t-SNE, heatmaps, clustering, etc.)

## Installation

```bash
# Install required dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Optional: for autoencoder visualization
pip install tensorflow

# Optional: for animations
pip install python-pptx ffmpeg-python
```

## Quick Start

### Command Line

```bash
# Basic usage
python -m pdp data.csv ./output

# With multiple variants
python -m pdp data.csv ./output --variants fundamental buffer rough

# With configuration file
python -m pdp --config config.json
```

### Python API

```python
from pdp import run_analysis, PDPConfig

# Simple usage
results = run_analysis("data.csv", "./output")

# With custom configuration
config = PDPConfig(
    dataset_path="data.csv",
    output_folder="./output",
    window_length=3,
    run_fundamental=True,
    run_buffer=True,
)

from pdp.main import PDPRunner
runner = PDPRunner(config)
results = runner.run()

# Access results
for variant_name, result in results.items():
    print(f"{variant_name}: {result.distance_matrix.shape}")
```

## Input Format

CSV file with either 5 or 6 columns (no header):

**5 columns (without class):**
```
conID,tstID,poiID,x,y
0,0,0,10.5,20.3
0,0,1,15.2,25.1
...
```

**6 columns (with class):**
```
conID,tstID,poiID,x,y,class
0,0,0,10.5,20.3,pedestrian
0,0,1,15.2,25.1,cyclist
...
```

Where:
- `conID`: Configuration ID (0-indexed)
- `tstID`: Timestamp ID (0-indexed)
- `poiID`: Point/Object ID (0-indexed)
- `x, y`: Spatial coordinates
- `class`: Optional class label for coloring

## PDP Variants

| Variant | Description |
|---------|-------------|
| `fundamental` | Basic PDP comparison |
| `buffer` | Adds buffer points around each position |
| `rough` | Uses tolerance for equality comparisons |
| `buffer_rough` | Combines buffer and rough approaches |

## Output Files

```
output/
├── distance_matrices/
│   └── N_C_PDPg_fundamental_DistanceMatrix.csv
├── heatmap/
│   └── N_C_PDPg_fundamental_heatmap.png
├── hclust/
│   └── N_C_PDPg_fundamental_hclust.png
├── mds/
│   └── N_C_PDPg_fundamental_mds.png
├── tsne/
│   └── N_C_PDPg_fundamental_tsne.png
├── topk/
│   └── N_C_PDPg_fundamental_topk_c0.png
│   └── ...
└── trajectories/
    └── N_C_PDPg_trajectories_sa0.png
    └── ...
```

## Configuration

Create a `config.json` file:

```json
{
  "dataset_path": "/path/to/data.csv",
  "output_folder": "/path/to/output",
  "window_length": 3,
  "run_fundamental": true,
  "run_buffer": false,
  "run_rough": false,
  "run_buffer_rough": false,
  "buffer_x": 15.0,
  "buffer_y": 1.0,
  "rough_x": 30.0,
  "rough_y": 3.0,
  "random_seed": 42,
  "visualizations": {
    "static_absolute": true,
    "static_relative": false,
    "heatmap": true,
    "hclust": true,
    "mds": true,
    "tsne": true,
    "topk": true,
    "autoencoder": false
  },
  "boundaries": {
    "min_x": -150,
    "max_x": 150,
    "min_y": -150,
    "max_y": 150
  }
}
```

## Package Structure

```
pdp/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclasses
├── main.py              # Main entry point & CLI
├── data/
│   ├── __init__.py
│   ├── loader.py        # Dataset loading (5/6 column CSV)
│   └── transforms.py    # Buffer transformation
├── core/
│   ├── __init__.py
│   └── pdp.py           # Core PDP algorithm (vectorized)
└── visualizations/
    ├── __init__.py
    ├── base.py          # Base visualizer class
    ├── heatmap.py       # Distance matrix heatmaps
    ├── clustering.py    # Hierarchical clustering
    ├── mds.py           # MDS dimensionality reduction
    ├── tsne.py          # t-SNE dimensionality reduction
    ├── topk.py          # Top-K nearest neighbors
    ├── trajectories.py  # Trajectory plots
    └── autoencoder.py   # Autoencoder-based reduction
```

## Key Improvements over Legacy Code

1. **Modular Architecture**: Clean separation of data loading, analysis, and visualization
2. **Type Safety**: Dataclasses for configuration with validation
3. **Vectorized Operations**: NumPy broadcasting instead of nested Python loops
4. **No Global State**: Pure functions and classes, easy to test
5. **CLI Interface**: Command-line support with argparse
6. **Extensible**: Easy to add new visualizations or analysis methods
7. **Documentation**: Comprehensive docstrings and type hints

## API Reference

### Core Classes

#### `PDPConfig`
Configuration container with all analysis parameters.

#### `Dataset`
Container for loaded trajectory data with helper methods.

#### `PDPAnalyzer`
Main analysis class that computes inequality and distance matrices.

#### `PDPResult`
Result container with distance matrix and metadata.

### Functions

#### `load_dataset(filepath: str) -> Dataset`
Load a CSV dataset.

#### `run_analysis(dataset_path, output_folder, **kwargs) -> dict`
Run complete PDP analysis.

#### `compute_distance_matrix(dataset, window_length, ...) -> PDPResult`
Compute PDP distance matrix directly.

## License

MIT License - See LICENSE file for details.
