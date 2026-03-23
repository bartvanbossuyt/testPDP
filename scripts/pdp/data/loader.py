"""
Dataset loading and preprocessing for PDP Analysis.

Supports both 5-column (no class) and 6-column (with class) CSV formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """
    Container for loaded PDP dataset.
    
    Attributes:
        df: Main dataframe with columns [conID, tstID, poiID, x, y]
        df_classes: Optional dataframe with class labels [conID, tstID, poiID, class]
        array: NumPy array of numeric data (float32)
        num_configurations: Number of unique configurations (con)
        num_timestamps: Number of timestamps per configuration (tst)
        num_points: Number of points per timestamp (poi)
        has_classes: Whether class labels are present
        filepath: Original file path
        name: Dataset name (filename without extension)
    """
    df: pd.DataFrame
    df_classes: Optional[pd.DataFrame]
    array: np.ndarray
    num_configurations: int
    num_timestamps: int
    num_points: int
    has_classes: bool
    filepath: str
    name: str
    
    @property
    def con(self) -> int:
        """Alias for num_configurations (backward compatibility)."""
        return self.num_configurations
    
    @property
    def tst(self) -> int:
        """Alias for num_timestamps (backward compatibility)."""
        return self.num_timestamps
    
    @property
    def poi(self) -> int:
        """Alias for num_points (backward compatibility)."""
        return self.num_points
    
    def get_configuration(self, config_id: int) -> pd.DataFrame:
        """Get data for a specific configuration."""
        return self.df[self.df['conID'] == config_id].copy()
    
    def get_timestamp(self, config_id: int, timestamp_id: int) -> pd.DataFrame:
        """Get data for a specific configuration and timestamp."""
        mask = (self.df['conID'] == config_id) & (self.df['tstID'] == timestamp_id)
        return self.df[mask].copy()
    
    def get_class_for_point(self, config_id: int, timestamp_id: int, point_id: int) -> Optional[str]:
        """Get class label for a specific point, or None if no classes."""
        if not self.has_classes or self.df_classes is None:
            return None
        mask = (
            (self.df_classes['conID'] == config_id) &
            (self.df_classes['tstID'] == timestamp_id) &
            (self.df_classes['poiID'] == point_id)
        )
        result = self.df_classes.loc[mask, 'class']
        return result.iloc[0] if len(result) > 0 else None
    
    def get_unique_classes(self) -> list:
        """Get list of unique class labels, or empty list if no classes."""
        if not self.has_classes or self.df_classes is None:
            return []
        return sorted(self.df_classes['class'].unique(), key=str)


def _detect_columns(filepath: str) -> Tuple[list, bool]:
    """
    Detect column structure from CSV file.
    
    Returns:
        Tuple of (column_names, has_class)
    """
    probe = pd.read_csv(filepath, header=None, nrows=1)
    ncols = probe.shape[1]
    
    if ncols == 5:
        return ['conID', 'tstID', 'poiID', 'x', 'y'], False
    elif ncols == 6:
        return ['conID', 'tstID', 'poiID', 'x', 'y', 'class'], True
    else:
        raise ValueError(
            f"Unexpected number of columns ({ncols}) in {filepath}. "
            "Expected 5 (conID, tstID, poiID, x, y) or 6 (+ class)."
        )


def _normalize_point_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize poiID column to integers if it contains strings.
    
    Returns:
        Tuple of (normalized_df, point_mapping_dict)
    """
    if pd.api.types.is_integer_dtype(df['poiID']):
        return df, {}
    
    point_mapping = {}
    current_id = 0
    
    def map_point(value):
        nonlocal current_id
        try:
            return int(value)
        except (ValueError, TypeError):
            if value not in point_mapping:
                point_mapping[value] = current_id
                current_id += 1
            return point_mapping[value]
    
    df = df.copy()
    df['poiID'] = df['poiID'].apply(map_point).astype(int)
    return df, point_mapping


def load_dataset(filepath: str) -> Dataset:
    """
    Load a PDP dataset from CSV file.
    
    Supports:
    - 5 columns: conID, tstID, poiID, x, y
    - 6 columns: conID, tstID, poiID, x, y, class
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dataset object containing all loaded data
        
    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    """
    filepath = str(Path(filepath).resolve())
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    # Detect column structure
    columns, has_classes = _detect_columns(filepath)
    
    # Load full dataset
    df_raw = pd.read_csv(filepath, header=None, names=columns)
    
    # Force numeric types on core columns
    for col in ['conID', 'tstID', 'poiID', 'x', 'y']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    # Normalize point IDs if needed
    df_raw, _ = _normalize_point_ids(df_raw)
    
    # Split numeric data and class labels
    df_numeric = df_raw[['conID', 'tstID', 'poiID', 'x', 'y']].copy()
    df_classes = None
    if has_classes:
        df_classes = df_raw[['conID', 'tstID', 'poiID', 'class']].copy()
    
    # Convert to numpy array
    array = df_numeric.to_numpy(dtype=np.float32)
    
    # Compute dimensions
    num_configurations = int(df_numeric['conID'].max()) + 1
    num_timestamps = int(df_numeric['tstID'].max()) + 1
    num_points = int(df_numeric['poiID'].max()) + 1
    
    # Get dataset name
    name = Path(filepath).stem
    
    return Dataset(
        df=df_numeric,
        df_classes=df_classes,
        array=array,
        num_configurations=num_configurations,
        num_timestamps=num_timestamps,
        num_points=num_points,
        has_classes=has_classes,
        filepath=filepath,
        name=name,
    )


def save_dataset(dataset: Dataset, filepath: str, include_classes: bool = True):
    """
    Save a dataset to CSV file.
    
    Args:
        dataset: Dataset to save
        filepath: Output file path
        include_classes: Whether to include class column (if available)
    """
    if include_classes and dataset.has_classes and dataset.df_classes is not None:
        # Merge class data with numeric data
        df_out = dataset.df.merge(
            dataset.df_classes[['conID', 'tstID', 'poiID', 'class']],
            on=['conID', 'tstID', 'poiID'],
            how='left'
        )
    else:
        df_out = dataset.df
    
    df_out.to_csv(filepath, index=False, header=False)
