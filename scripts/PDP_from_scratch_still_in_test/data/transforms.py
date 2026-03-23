"""
Data transformations for PDP Analysis.

Includes buffer and rough transformations for creating PDP variants.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from .loader import Dataset


def apply_buffer_transform(
    dataset: Dataset,
    buffer_x: float = 15.0,
    buffer_y: float = 1.0
) -> Dataset:
    """
    Apply buffer transformation to dataset.
    
    Creates 5 points for each original point:
    - Point at (x - buffer_x, y)
    - Point at (x + buffer_x, y)
    - Original point at (x, y)
    - Point at (x, y - buffer_y)
    - Point at (x, y + buffer_y)
    
    Args:
        dataset: Input dataset
        buffer_x: Buffer distance in x direction
        buffer_y: Buffer distance in y direction
        
    Returns:
        New Dataset with buffer points added
    """
    rows = []
    
    for _, row in dataset.df.iterrows():
        con_id = row['conID']
        tst_id = row['tstID']
        poi_id = row['poiID']
        x = row['x']
        y = row['y']
        
        # New point IDs: original * 5 + offset
        base_poi = int(poi_id) * 5
        
        # Add 5 buffer points
        rows.extend([
            [con_id, tst_id, base_poi + 0, x - buffer_x, y],
            [con_id, tst_id, base_poi + 1, x + buffer_x, y],
            [con_id, tst_id, base_poi + 2, x, y],  # Original
            [con_id, tst_id, base_poi + 3, x, y - buffer_y],
            [con_id, tst_id, base_poi + 4, x, y + buffer_y],
        ])
    
    df_buffer = pd.DataFrame(rows, columns=['conID', 'tstID', 'poiID', 'x', 'y'])
    
    # Round to 2 decimal places
    df_buffer['x'] = df_buffer['x'].round(2)
    df_buffer['y'] = df_buffer['y'].round(2)
    df_buffer['poiID'] = df_buffer['poiID'].astype(int)
    
    # Compute new dimensions
    num_points = int(df_buffer['poiID'].max()) + 1
    
    return Dataset(
        df=df_buffer,
        df_classes=None,  # Classes don't transfer to buffer points
        array=df_buffer.to_numpy(dtype=np.float32),
        num_configurations=dataset.num_configurations,
        num_timestamps=dataset.num_timestamps,
        num_points=num_points,
        has_classes=False,
        filepath=dataset.filepath,
        name=f"{dataset.name}_buffer",
    )


def filter_configurations(
    dataset: Dataset,
    config_ids: list
) -> Dataset:
    """
    Filter dataset to include only specified configurations.
    
    Args:
        dataset: Input dataset
        config_ids: List of configuration IDs to keep
        
    Returns:
        Filtered Dataset with remapped configuration IDs
    """
    # Filter and remap configuration IDs
    df_filtered = dataset.df[dataset.df['conID'].isin(config_ids)].copy()
    config_mapping = {old: new for new, old in enumerate(sorted(config_ids))}
    df_filtered['conID'] = df_filtered['conID'].map(config_mapping)
    
    # Filter classes if present
    df_classes = None
    if dataset.has_classes and dataset.df_classes is not None:
        df_classes = dataset.df_classes[dataset.df_classes['conID'].isin(config_ids)].copy()
        df_classes['conID'] = df_classes['conID'].map(config_mapping)
    
    return Dataset(
        df=df_filtered.reset_index(drop=True),
        df_classes=df_classes.reset_index(drop=True) if df_classes is not None else None,
        array=df_filtered[['conID', 'tstID', 'poiID', 'x', 'y']].to_numpy(dtype=np.float32),
        num_configurations=len(config_ids),
        num_timestamps=dataset.num_timestamps,
        num_points=dataset.num_points,
        has_classes=dataset.has_classes,
        filepath=dataset.filepath,
        name=f"{dataset.name}_filtered",
    )


def get_window_data(
    dataset: Dataset,
    config_id: int,
    start_timestamp: int,
    window_length: int
) -> pd.DataFrame:
    """
    Extract data for a sliding window of timestamps.
    
    Args:
        dataset: Input dataset
        config_id: Configuration ID
        start_timestamp: Starting timestamp index
        window_length: Number of timestamps in window
        
    Returns:
        DataFrame with data for the specified window
    """
    mask = (
        (dataset.df['conID'] == config_id) &
        (dataset.df['tstID'] >= start_timestamp) &
        (dataset.df['tstID'] < start_timestamp + window_length)
    )
    return dataset.df[mask].copy()
