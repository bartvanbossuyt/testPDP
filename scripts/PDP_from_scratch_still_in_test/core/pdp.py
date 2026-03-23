"""
Core PDP (Pairwise Distance Pattern) Algorithm.

This module implements the core PDP algorithm for computing inequality matrices
and distance matrices between configurations of moving objects.

The algorithm:
1. For each configuration and timestamp window, create inequality matrices
   comparing positional relationships between all pairs of points
2. Compare inequality matrices across configurations to compute distances
3. The distance matrix represents similarity between configurations

Key optimizations:
- Vectorized numpy operations instead of nested Python loops
- Pre-grouped data for faster access
- Efficient matrix operations using broadcasting
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import numpy as np
import pandas as pd

from ..data.loader import Dataset


@dataclass
class PDPResult:
    """
    Result of PDP analysis.
    
    Attributes:
        distance_matrix: NxN distance matrix between configurations
        distance_matrix_x: Distance matrix for x-dimension only
        distance_matrix_y: Distance matrix for y-dimension only
        inequality_data: DataFrame with inequality matrices per (conID, tstID)
        identical_groups: List of groups with identical configurations
        variant_name: Name of PDP variant (fundamental, buffer, rough, buffer_rough)
    """
    distance_matrix: np.ndarray
    distance_matrix_x: np.ndarray
    distance_matrix_y: np.ndarray
    inequality_data: pd.DataFrame
    identical_groups: List[List[int]]
    variant_name: str
    
    def save_distance_matrix(self, filepath: str):
        """Save distance matrix to CSV file."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            for row in self.distance_matrix:
                writer.writerow(row.astype(int).tolist())


def compute_inequality_matrix(
    values: np.ndarray,
    roughness: float = 0.0
) -> np.ndarray:
    """
    Compute inequality matrix for a set of position values.
    
    The inequality matrix compares each pair of positions:
    - 0: value_i < value_j (by more than roughness)
    - 1: |value_i - value_j| <= roughness (equal within tolerance)
    - 2: value_i > value_j (by more than roughness)
    
    Args:
        values: 1D array of position values
        roughness: Tolerance for equality comparison
        
    Returns:
        NxN inequality matrix
    """
    n = len(values)
    
    # Compute pairwise differences using broadcasting
    # diff[i,j] = values[j] - values[i]
    diff = values.reshape(1, -1) - values.reshape(-1, 1)
    
    # Create inequality matrix
    matrix = np.zeros((n, n), dtype=np.int8)
    matrix[diff > roughness] = 0   # j > i
    matrix[diff < -roughness] = 2  # i > j
    matrix[np.abs(diff) <= roughness] = 1  # equal
    
    return matrix


def compute_inequality_matrices_for_window(
    df_window: pd.DataFrame,
    num_points: int,
    window_length: int,
    rough_x: float = 0.0,
    rough_y: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inequality matrices for x and y dimensions of a timestamp window.
    
    Args:
        df_window: DataFrame with data for the window (sorted by tstID, poiID)
        num_points: Number of points per timestamp
        window_length: Number of timestamps in window
        rough_x: Roughness parameter for x dimension
        rough_y: Roughness parameter for y dimension
        
    Returns:
        Tuple of (x_inequality_matrix, y_inequality_matrix)
    """
    # Ensure correct ordering
    df_sorted = df_window.sort_values(['tstID', 'poiID'])
    
    x_values = df_sorted['x'].to_numpy()
    y_values = df_sorted['y'].to_numpy()
    
    x_matrix = compute_inequality_matrix(x_values, rough_x)
    y_matrix = compute_inequality_matrix(y_values, rough_y)
    
    return x_matrix, y_matrix


def compute_distance_between_matrices(
    matrix1: np.ndarray,
    matrix2: np.ndarray
) -> int:
    """
    Compute absolute distance between two inequality matrices.
    
    Args:
        matrix1: First inequality matrix
        matrix2: Second inequality matrix
        
    Returns:
        Sum of absolute differences
    """
    return int(np.abs(matrix1.astype(np.int16) - matrix2.astype(np.int16)).sum())


class UnionFind:
    """Union-Find data structure for grouping identical configurations."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """Unite two sets by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_groups(self) -> List[List[int]]:
        """Get all groups with more than one element."""
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return [sorted(g) for g in groups.values() if len(g) > 1]


class PDPAnalyzer:
    """
    Main PDP Analysis class.
    
    Computes inequality matrices and distance matrices for trajectory configurations
    using the Pairwise Distance Pattern methodology.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        window_length: int = 3,
        rough_x: float = 0.0,
        rough_y: float = 0.0
    ):
        """
        Initialize PDP Analyzer.
        
        Args:
            dataset: Dataset to analyze
            window_length: Number of timestamps in sliding window
            rough_x: Roughness parameter for x dimension (0 for fundamental)
            rough_y: Roughness parameter for y dimension (0 for fundamental)
        """
        self.dataset = dataset
        self.window_length = window_length
        self.rough_x = rough_x
        self.rough_y = rough_y
        
        # Validate
        if window_length > dataset.num_timestamps:
            raise ValueError(
                f"Window length ({window_length}) cannot exceed "
                f"number of timestamps ({dataset.num_timestamps})"
            )
        
        # Pre-group data by configuration for faster access
        self._grouped = dataset.df.groupby('conID')
        
        # Number of windows per configuration
        self._num_windows = dataset.num_timestamps - window_length + 1
        
        # Size of inequality matrix
        self._matrix_size = dataset.num_points * window_length
    
    def compute_all_inequality_matrices(self) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        """
        Compute inequality matrices for all (configuration, window) combinations.
        
        Returns:
            Dictionary mapping (config_id, window_start) to (x_matrix, y_matrix)
        """
        result = {}
        
        for config_id in range(self.dataset.num_configurations):
            df_config = self._grouped.get_group(config_id)
            
            for window_start in range(self._num_windows):
                # Get window data
                mask = (
                    (df_config['tstID'] >= window_start) &
                    (df_config['tstID'] < window_start + self.window_length)
                )
                df_window = df_config[mask]
                
                x_matrix, y_matrix = compute_inequality_matrices_for_window(
                    df_window,
                    self.dataset.num_points,
                    self.window_length,
                    self.rough_x,
                    self.rough_y
                )
                
                result[(config_id, window_start)] = (x_matrix, y_matrix)
        
        return result
    
    def compute_distance_matrix(
        self,
        inequality_matrices: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute distance matrices between all configurations.
        
        Args:
            inequality_matrices: Dictionary from compute_all_inequality_matrices
            
        Returns:
            Tuple of (combined_distance_matrix, x_distance_matrix, y_distance_matrix)
        """
        n = self.dataset.num_configurations
        dist_x = np.zeros((n, n), dtype=np.float64)
        dist_y = np.zeros((n, n), dtype=np.float64)
        
        # Compute pairwise distances
        for k in range(n):
            for l in range(n):
                total_x = 0
                total_y = 0
                
                for window_start in range(self._num_windows):
                    x_mat_k, y_mat_k = inequality_matrices[(k, window_start)]
                    x_mat_l, y_mat_l = inequality_matrices[(l, window_start)]
                    
                    total_x += compute_distance_between_matrices(x_mat_k, x_mat_l)
                    total_y += compute_distance_between_matrices(y_mat_k, y_mat_l)
                
                dist_x[k, l] = total_x
                dist_y[k, l] = total_y
        
        # Normalize to percentage scale (0-100)
        # Denominator: 2 * num_windows * (matrix_size^2 - matrix_size) / 100
        matrix_size = self._matrix_size
        denominator = (
            2 * self._num_windows * 
            (matrix_size * matrix_size - matrix_size) / 100
        )
        
        if denominator > 0:
            dist_x = np.round(dist_x / denominator).astype(int)
            dist_y = np.round(dist_y / denominator).astype(int)
        
        # Combined distance (average of x and y)
        dist_combined = np.round((dist_x + dist_y) / 2).astype(int)
        
        return dist_combined, dist_x, dist_y
    
    def find_identical_configurations(
        self,
        distance_matrix: np.ndarray
    ) -> List[List[int]]:
        """
        Find groups of configurations with zero distance (identical PDP patterns).
        
        Uses Union-Find for efficient grouping.
        
        Args:
            distance_matrix: Distance matrix from compute_distance_matrix
            
        Returns:
            List of groups, where each group is a list of configuration IDs
        """
        n = len(distance_matrix)
        uf = UnionFind(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] == 0:
                    uf.union(i, j)
        
        return uf.get_groups()
    
    def run(self, variant_name: str = "fundamental") -> PDPResult:
        """
        Run complete PDP analysis.
        
        Args:
            variant_name: Name for this variant (fundamental, buffer, rough, buffer_rough)
            
        Returns:
            PDPResult containing all analysis results
        """
        # Compute inequality matrices
        inequality_matrices = self.compute_all_inequality_matrices()
        
        # Compute distance matrices
        dist_combined, dist_x, dist_y = self.compute_distance_matrix(inequality_matrices)
        
        # Find identical configurations
        identical_groups = self.find_identical_configurations(dist_combined)
        
        # Create DataFrame for inequality data (for compatibility/debugging)
        inequality_data = pd.DataFrame([
            {
                'conID': con_id,
                'tstID': tst_id,
                'xineqID': x_mat,
                'yineqID': y_mat
            }
            for (con_id, tst_id), (x_mat, y_mat) in inequality_matrices.items()
        ])
        
        return PDPResult(
            distance_matrix=dist_combined,
            distance_matrix_x=dist_x,
            distance_matrix_y=dist_y,
            inequality_data=inequality_data,
            identical_groups=identical_groups,
            variant_name=variant_name,
        )


def compute_distance_matrix(
    dataset: Dataset,
    window_length: int = 3,
    rough_x: float = 0.0,
    rough_y: float = 0.0,
    variant_name: str = "fundamental"
) -> PDPResult:
    """
    Convenience function to compute PDP distance matrix.
    
    Args:
        dataset: Dataset to analyze
        window_length: Number of timestamps in sliding window
        rough_x: Roughness parameter for x dimension
        rough_y: Roughness parameter for y dimension
        variant_name: Name for this variant
        
    Returns:
        PDPResult with all analysis results
    """
    analyzer = PDPAnalyzer(
        dataset=dataset,
        window_length=window_length,
        rough_x=rough_x,
        rough_y=rough_y
    )
    return analyzer.run(variant_name=variant_name)
