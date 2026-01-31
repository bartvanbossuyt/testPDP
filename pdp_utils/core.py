import numpy as np
from typing import TypedDict

# ============= Coordinate Precision Settings =============
# Change these values to adjust coordinate display precision throughout the app
COORD_DISPLAY_PRECISION = 2   # Decimal places for UI display (hover text, status messages)
COORD_CSV_PRECISION = 3       # Decimal places for CSV export (3 digits after decimal point)

# ============= Object Labels for Display =============
OBJECT_LABELS = ["k", "l", "m", "n", "p", "q", "r", "s", "u", "v"]

# Type definition for successful point data in the search process
class SuccessfulPoint(TypedDict):
    point: np.ndarray              # Coordinates of the generated point
    parent_idx: int                # Index in all_pts (may be a generated point)
    parent_point: np.ndarray       # Actual coordinates of the parent point
    original_parent_idx: int       # Index of the ORIGINAL point (k0, k1, k2, l0, l1, l2)
    iteration: int                 # Iteration number when this point was accepted

# ============= PDP Core Functions (from N_PDP.py) =============
def compute_inequality_matrix(points: np.ndarray, dimension: int, roughness: float = 0.0) -> np.ndarray:
    """
    Compute PDP inequality matrix for a set of points along one dimension.
    
    This follows the exact logic from N_PDP.py:
    - Value 0: point j > point i (beyond roughness)
    - Value 1: |point j - point i| <= roughness (equal within tolerance)
    - Value 2: point j < point i (beyond roughness)
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        dimension: 0 for x, 1 for y
        roughness: tolerance for equality (default 0.0)
    
    Returns:
        (N, N) inequality matrix
    """
    n = len(points)
    inequality_matrix = np.zeros((n, n))
    
    values = points[:, dimension]
    
    for i in range(n):
        for j in range(n):
            diff = values[j] - values[i]
            if abs(diff) <= roughness:
                inequality_matrix[i, j] = 1  # Equal (within roughness)
            elif diff > roughness:
                inequality_matrix[i, j] = 0  # Greater than
            else:
                inequality_matrix[i, j] = 2  # Less than
    
    return inequality_matrix

def compare_inequality_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> bool:
    """
    Compare two inequality matrices for equality.
    
    Returns True if matrices are identical (same PDP pattern).
    """
    return np.array_equal(matrix1, matrix2)

def compare_inequality_matrices_with_threshold(matrix1: np.ndarray, matrix2: np.ndarray, threshold: float = 1.0) -> tuple[bool, float]:
    """
    Compare two inequality matrices with a percentage threshold.
    
    Args:
        matrix1: First inequality matrix
        matrix2: Second inequality matrix
        threshold: Minimum required match percentage (0.0 to 1.0), default 1.0 = 100%
    
    Returns:
        (is_match, match_percentage): Tuple of boolean match result and actual match percentage
    """
    if matrix1.shape != matrix2.shape:
        return False, 0.0
    
    total_elements = matrix1.size
    if total_elements == 0:
        return True, 1.0
    
    matching_elements = np.sum(matrix1 == matrix2)
    match_percentage = matching_elements / total_elements
    is_match = match_percentage >= threshold
    
    return is_match, match_percentage

def apply_buffer_transformation(points: np.ndarray, buffer_x: float, buffer_y: float) -> np.ndarray:
    """
    Apply buffer transformation to a set of points.
    
    This creates 5 variants of each point:
    - Original point * 5 + 0: x - buffer_x
    - Original point * 5 + 1: x + buffer_x
    - Original point * 5 + 2: no buffer in x (original x)
    - Original point * 5 + 3: y - buffer_y
    - Original point * 5 + 4: y + buffer_y
    
    The point index is expanded by 5x to accommodate all buffer variants.
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        buffer_x: buffer distance in x-direction
        buffer_y: buffer distance in y-direction
    
    Returns:
        (5*N, 2) array with buffered points
    """
    n = len(points)
    buffered = np.zeros((5 * n, 2))
    
    for i, (x, y) in enumerate(points):
        base_idx = i * 5
        # Variant 0: x - buffer_x
        buffered[base_idx + 0] = [x - buffer_x, y]
        # Variant 1: x + buffer_x
        buffered[base_idx + 1] = [x + buffer_x, y]
        # Variant 2: no buffer in x
        buffered[base_idx + 2] = [x, y]
        # Variant 3: y - buffer_y
        buffered[base_idx + 3] = [x, y - buffer_y]
        # Variant 4: y + buffer_y
        buffered[base_idx + 4] = [x, y + buffer_y]
    
    return buffered
