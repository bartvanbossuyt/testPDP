# -*- coding: utf-8 -*-
"""
Data Loading Module

Contains utilities for loading and parsing CSV data files.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, coercing bad values to NaN."""
    return pd.to_numeric(s, errors="coerce")


def read_csv_file(
    file_path: Path, 
    names: list = None
) -> pd.DataFrame:
    """
    Read a CSV file handling both standard and custom header formats.
    
    If the first line starts with 'header:', it is skipped and column names are used.
    
    Args:
        file_path: Path to the CSV file
        names: Column names to use (default: ["c", "t", "o", "x", "y"])
    
    Returns:
        Cleaned DataFrame with numeric columns
    """
    if names is None:
        names = ["c", "t", "o", "x", "y"]
    
    # Peek at the first line to detect custom header format
    with file_path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    
    if first.lower().startswith("header:"):
        # Custom header format: skip first line, use fixed names
        df = pd.read_csv(file_path, header=None, names=names, skiprows=1)
    else:
        # Try normal CSV; if columns are not present, fall back to fixed names
        df = pd.read_csv(file_path)
        if not set(names).issubset(df.columns):
            df = pd.read_csv(file_path, header=None, names=names)
    
    # Force numeric columns and drop invalid rows
    for col in names:
        df[col] = to_numeric_series(df[col])
    df = df.dropna(subset=names)
    df = df.reset_index(drop=True)
    
    return df


def read_csv_from_string(
    content: str,
    names: list = None
) -> pd.DataFrame:
    """
    Read CSV data from a string (e.g., uploaded file content).
    
    Args:
        content: CSV content as string
        names: Column names to use (default: ["c", "t", "o", "x", "y"])
    
    Returns:
        Cleaned DataFrame with numeric columns
    """
    from io import StringIO
    
    if names is None:
        names = ["c", "t", "o", "x", "y"]
    
    lines = content.strip().split("\n")
    first_line = lines[0].strip().lower()
    
    if first_line.startswith("header:"):
        # Skip header line
        df = pd.read_csv(StringIO("\n".join(lines[1:])), header=None, names=names)
    else:
        df = pd.read_csv(StringIO(content))
        if not set(names).issubset(df.columns):
            df = pd.read_csv(StringIO(content), header=None, names=names)
    
    # Clean the dataframe
    for col in names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=names)
    df = df.reset_index(drop=True)
    
    return df


def extract_points_for_object(
    df: pd.DataFrame,
    o_val: int,
    c_val: int,
    t_values: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract points for a specific object from the dataframe.
    
    Args:
        df: DataFrame with columns c, t, o, x, y
        o_val: Object ID to filter
        c_val: Configuration ID to filter
        t_values: Optional list of timestamps to include
    
    Returns:
        (points, timestamps): Arrays of (N, 2) coordinates and (N,) timestamps
    """
    sel = df[(df["c"] == c_val) & (df["o"] == o_val)].sort_values("t").reset_index(drop=True)
    
    if t_values is not None:
        sel = sel[sel["t"].isin(t_values)]
    
    if sel.empty:
        return np.array([]).reshape(0, 2), np.array([])
    
    pts = sel[["x", "y"]].values.astype(float)
    ts = sel["t"].values.astype(float)
    return pts, ts


def get_available_configs(df: pd.DataFrame) -> list:
    """Get sorted list of available configuration IDs from dataframe."""
    return sorted(df["c"].dropna().unique().astype(int).tolist())


def get_common_timestamps(df: pd.DataFrame, c_val: int) -> list:
    """Get sorted list of timestamps common to all objects in a configuration."""
    config_df = df[df["c"] == c_val]
    object_ids = config_df["o"].unique()
    
    if len(object_ids) == 0:
        return []
    
    # Find timestamps that exist for all objects
    common_ts = None
    for o_id in object_ids:
        obj_ts = set(config_df[config_df["o"] == o_id]["t"].unique())
        if common_ts is None:
            common_ts = obj_ts
        else:
            common_ts = common_ts.intersection(obj_ts)
    
    return sorted(list(common_ts)) if common_ts else []
