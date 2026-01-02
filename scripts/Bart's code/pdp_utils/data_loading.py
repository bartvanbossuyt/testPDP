# -*- coding: utf-8 -*-
"""
Data loading utilities for PDP inverse.
Handles CSV file reading and DataFrame processing.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, coercing bad values to NaN."""
    return pd.to_numeric(s, errors="coerce")


def read_clean_df(csv_path: Path) -> pd.DataFrame:
    """
    Read a CSV file with columns (c, t, o, x, y).
    If the first line starts with 'header:', it is skipped.
    
    Returns a clean DataFrame with numeric columns and no NaN values.
    """
    with csv_path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    
    names = ["c", "t", "o", "x", "y"]
    
    if first.lower().startswith("header:"):
        df = pd.read_csv(csv_path, header=None, names=names, skiprows=1)
    else:
        df = pd.read_csv(csv_path)
        if not set(names).issubset(df.columns):
            df = pd.read_csv(csv_path, header=None, names=names)
    
    for col in names:
        df[col] = to_numeric_series(df[col])
    
    df = df.dropna(subset=names)
    df = df.reset_index(drop=True)
    return df


def load_points_from_df(
    df: pd.DataFrame, 
    o_val: int = 0, 
    c_val: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract points from a DataFrame for a specific configuration and object.
    
    Args:
        df: DataFrame with columns c, t, o, x, y
        o_val: Object filter value
        c_val: Configuration filter value
    
    Returns:
        pts: (N,2) numpy array [x,y] sorted by t
        ts: (N,) numpy array with t-values (sorted)
    """
    sel = df[(df["c"] == c_val) & (df["o"] == o_val)].sort_values("t").reset_index(drop=True)
    
    if sel.empty:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    
    pts = sel[["x", "y"]].to_numpy(dtype=float)
    ts = sel["t"].to_numpy(dtype=float)
    return pts, ts


def extract_points_from_df(
    df: pd.DataFrame, 
    o_val: int, 
    c_val: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract points from DataFrame, wrapper for load_points_from_df.
    Returns (points, timestamps) arrays.
    """
    return load_points_from_df(df, o_val, c_val)


def get_available_configs(df: pd.DataFrame) -> list[int]:
    """Get list of available configuration values in the DataFrame."""
    return sorted(df["c"].dropna().unique().astype(int).tolist())


def get_available_objects(df: pd.DataFrame, c_val: int) -> list[int]:
    """Get list of available object values for a specific configuration."""
    config_df = df[df["c"] == c_val]
    return sorted(config_df["o"].dropna().unique().astype(int).tolist())


def get_time_range(df: pd.DataFrame, c_val: int) -> tuple[int, int]:
    """Get min and max time values for a specific configuration."""
    config_df = df[df["c"] == c_val]
    if config_df.empty:
        return 0, 0
    t_min = int(config_df["t"].min())
    t_max = int(config_df["t"].max())
    return t_min, t_max


def get_coordinate_bounds(
    df: pd.DataFrame, 
    c_val: int,
    margin: float = 0.1
) -> tuple[float, float, float, float]:
    """
    Calculate coordinate bounds for a configuration with margin.
    
    Returns:
        (x_min, x_max, y_min, y_max) with margin applied
    """
    config_df = df[df["c"] == c_val]
    if config_df.empty:
        return 0.0, 100.0, 0.0, 100.0
    
    x_min, x_max = float(config_df["x"].min()), float(config_df["x"].max())
    y_min, y_max = float(config_df["y"].min()), float(config_df["y"].max())
    
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin
    
    return x_min, x_max, y_min, y_max
