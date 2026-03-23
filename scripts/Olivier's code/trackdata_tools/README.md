# Trackdata Tools

Scripts for processing and analyzing camera-based object tracking data.

## Scripts

| Script | Purpose |
|--------|---------|
| `convert_log_to_csv.py` | Convert JSON/log tracking files to CSV format |
| `analyze_trackdata.py` | Comprehensive analysis of track data (spatial, temporal, class distribution) |
| `analyze_classes.py` | Detailed class analysis with object dimensions |
| `check_class_changes.py` | Detect tracks with inconsistent class assignments |
| `data_prep_pubapi.py` | Build 2-object configurations from tracking data |

## Class ID Mapping

```python
CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Motorcycle",
    5: "Car",
    7: "Van",
    10: "SmallTruck",
    12: "LargeTruck",
    14: "Bus",
    20: "CarAndTrailer",
    21: "VanAndTrailer",
    24: "TruckAndTrailer",
    25: "Scooter",
    26: "Unknown"
}
```

## Usage

### Convert JSON to CSV
```bash
python convert_log_to_csv.py --input "path/to/json_folder" --output "path/to/csv_folder"
```

### Run analysis
```bash
python analyze_trackdata.py
python analyze_classes.py
python check_class_changes.py
```

### Generate configurations
```bash
python data_prep_pubapi.py
```

## Requirements
- pandas
- numpy
- matplotlib (for visualizations)
