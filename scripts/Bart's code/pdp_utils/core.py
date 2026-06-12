import numpy as np
from typing import TypedDict, Optional

# ============= Coordinate Precision Settings =============
# Change these values to adjust coordinate display precision throughout the app
COORD_DISPLAY_PRECISION = 2   # Decimal places for UI display (hover text, status messages)
COORD_CSV_PRECISION = 3       # Decimal places for CSV export (3 digits after decimal point)

# ============= Object Labels for Display =============
OBJECT_LABELS = ["A", "B", "m", "n", "p", "q", "r", "s", "u", "v"]

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
    values = points[:, dimension]  # (N,)
    # diff[i, j] = values[j] - values[i]
    diff = values[None, :] - values[:, None]  # (N, N) via broadcasting
    
    # Default to 0 (j > i),  set 1 where equal within roughness, 2 where j < i
    inequality_matrix = np.zeros_like(diff, dtype=float)
    inequality_matrix[diff < -roughness] = 2.0  # j < i
    inequality_matrix[np.abs(diff) <= roughness] = 1.0  # equal within roughness
    # remainder stays 0.0 (j > i)
    
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
    x = points[:, 0]
    y = points[:, 1]
    
    buffered = np.empty((5 * n, 2))
    buffered[0::5, 0] = x - buffer_x
    buffered[0::5, 1] = y
    buffered[1::5, 0] = x + buffer_x
    buffered[1::5, 1] = y
    buffered[2::5, 0] = x
    buffered[2::5, 1] = y
    buffered[3::5, 0] = x
    buffered[3::5, 1] = y - buffer_y
    buffered[4::5, 0] = x
    buffered[4::5, 1] = y + buffer_y
    
    return buffered


# ============= Incremental PDP Checker =============

class IncrementalPDPChecker:
    """Maintains PDP inequality matrices incrementally for O(k*N) checks instead of O(N²).

    Instead of recomputing full N×N inequality matrices on every check,
    this class:
      1. Pre-computes original matrices once during ``__init__``
      2. Maintains generated matrices incrementally: when *k* points change,
         only *k* rows + *k* columns are recomputed → O(k·N)
      3. Tracks running mismatch counts for O(1) match checking

    Typical speedup: N/k ≈ 274× for single-point moves with 274 total points.
    """

    def __init__(
        self,
        original_points: np.ndarray,
        pdp_variant: str = "fundamental",
        buffer_x: float = 0.0,
        buffer_y: float = 0.0,
        rough_x: float = 0.0,
        rough_y: float = 0.0,
        match_threshold: float = 1.0,
        max_mismatches: Optional[int] = None,
    ):
        self.n_original = len(original_points)
        self.pdp_variant = pdp_variant
        self.buffer_x = buffer_x
        self.buffer_y = buffer_y
        self.match_threshold = match_threshold
        self.max_mismatches_limit = max_mismatches

        # Determine roughness (same logic as check_pdp_match)
        if pdp_variant in ("rough", "bufferrough"):
            self.roughness_x = rough_x
            self.roughness_y = rough_y
        elif pdp_variant == "realistic":
            self.roughness_x = 0.0
            self.roughness_y = rough_y
        else:
            self.roughness_x = 0.0
            self.roughness_y = 0.0

        self.uses_buffer = pdp_variant in ("buffer", "bufferrough", "realistic")

        # Store raw original for reset
        self._orig_points_raw = original_points.copy()

        # Transform and compute original matrices (ONCE)
        orig_transformed = self._transform_points(original_points)
        self.N = len(orig_transformed)
        self.orig_x_matrix = compute_inequality_matrix(orig_transformed, 0, self.roughness_x)
        self.orig_y_matrix = compute_inequality_matrix(orig_transformed, 1, self.roughness_y)

        # Initialize generated state as copy of original
        self._gen_points = orig_transformed.copy()
        self.gen_x_matrix = self.orig_x_matrix.copy()
        self.gen_y_matrix = self.orig_y_matrix.copy()

        # Running mismatch counts (start at 0 — identical to original)
        self._d1_mismatches = 0
        self._d2_mismatches = 0
        self._d1_total = int(self.orig_x_matrix.size)
        self._d2_total = int(self.orig_y_matrix.size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform_points(self, points: np.ndarray) -> np.ndarray:
        if self.pdp_variant in ("buffer", "bufferrough"):
            return apply_buffer_transformation(points, self.buffer_x, self.buffer_y)
        elif self.pdp_variant == "realistic":
            return apply_buffer_transformation(points, self.buffer_x, 0.0)
        return points.copy()

    def _get_expanded_indices(self, original_index: int) -> list:
        if self.uses_buffer:
            base = original_index * 5
            return [base, base + 1, base + 2, base + 3, base + 4]
        return [original_index]

    def _compute_expanded_coords(self, coord: np.ndarray) -> list:
        x, y = float(coord[0]), float(coord[1])
        if self.pdp_variant in ("buffer", "bufferrough"):
            bx, by = self.buffer_x, self.buffer_y
            return [
                np.array([x - bx, y]),
                np.array([x + bx, y]),
                np.array([x, y]),
                np.array([x, y - by]),
                np.array([x, y + by]),
            ]
        elif self.pdp_variant == "realistic":
            bx = self.buffer_x
            return [
                np.array([x - bx, y]),
                np.array([x + bx, y]),
                np.array([x, y]),
                np.array([x, y]),
                np.array([x, y]),
            ]
        return [coord.copy()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_to_original(self) -> None:
        """Reset generated to match original (start of new config). O(N²) copy."""
        orig_t = self._transform_points(self._orig_points_raw)
        np.copyto(self._gen_points, orig_t)
        np.copyto(self.gen_x_matrix, self.orig_x_matrix)
        np.copyto(self.gen_y_matrix, self.orig_y_matrix)
        self._d1_mismatches = 0
        self._d2_mismatches = 0

    def update_points(self, positions: dict) -> None:
        """Update generated points and incrementally maintain matrices. O(k·N)."""
        all_affected: list = []
        for orig_idx, coord in positions.items():
            exp_i = self._get_expanded_indices(orig_idx)
            exp_c = self._compute_expanded_coords(coord)
            for ei, ec in zip(exp_i, exp_c):
                if 0 <= ei < self.N:
                    self._gen_points[ei] = ec
                    all_affected.append(ei)
        if not all_affected:
            return
        self._update_matrices_incremental(sorted(set(all_affected)))

    def check_match(self) -> tuple:
        """Check PDP match using running mismatch counts. O(1) for common cases."""
        if self.max_mismatches_limit is not None:
            # max_mismatches mode uses upper triangle — fall back to O(N²)
            triu = np.triu_indices(self.N, k=1)
            d1_mm = int(np.count_nonzero(self.gen_x_matrix[triu] != self.orig_x_matrix[triu]))
            d2_mm = int(np.count_nonzero(self.gen_y_matrix[triu] != self.orig_y_matrix[triu]))
            total = d1_mm + d2_mm
            ok = total <= self.max_mismatches_limit
            return ok, ok

        d1_pct = 1.0 - (self._d1_mismatches / self._d1_total) if self._d1_total > 0 else 1.0
        d2_pct = 1.0 - (self._d2_mismatches / self._d2_total) if self._d2_total > 0 else 1.0

        if self.match_threshold < 1.0:
            avg = (d1_pct + d2_pct) / 2.0
            return avg >= self.match_threshold, avg >= self.match_threshold
        return d1_pct >= self.match_threshold, d2_pct >= self.match_threshold

    def save_state_for_indices(self, original_indices: list) -> dict:
        """Save current state for specified original indices. O(k·N)."""
        exp_set: list = []
        for oi in original_indices:
            exp_set.extend(self._get_expanded_indices(oi))
        exp_set = sorted(set(i for i in exp_set if 0 <= i < self.N))
        return {
            '_d1': self._d1_mismatches,
            '_d2': self._d2_mismatches,
            '_idx': exp_set,
            '_pts': {i: self._gen_points[i].copy() for i in exp_set},
            '_xr': {i: self.gen_x_matrix[i, :].copy() for i in exp_set},
            '_xc': {i: self.gen_x_matrix[:, i].copy() for i in exp_set},
            '_yr': {i: self.gen_y_matrix[i, :].copy() for i in exp_set},
            '_yc': {i: self.gen_y_matrix[:, i].copy() for i in exp_set},
        }

    def restore_saved(self, saved: dict) -> None:
        """Restore state saved by ``save_state_for_indices``. O(k·N)."""
        self._d1_mismatches = saved['_d1']
        self._d2_mismatches = saved['_d2']
        for i in saved['_idx']:
            self._gen_points[i] = saved['_pts'][i]
            self.gen_x_matrix[i, :] = saved['_xr'][i]
            self.gen_x_matrix[:, i] = saved['_xc'][i]
            self.gen_y_matrix[i, :] = saved['_yr'][i]
            self.gen_y_matrix[:, i] = saved['_yc'][i]

    # ------------------------------------------------------------------
    # Incremental matrix update engine
    # ------------------------------------------------------------------

    def _update_matrices_incremental(self, affected_indices: list) -> None:
        """Recompute affected rows/cols and adjust running mismatch counts."""
        N = self.N
        affected_arr = np.array(affected_indices, dtype=np.intp)

        # Mask for non-affected rows (avoids double-counting in col mismatch)
        non_affected_mask = np.ones(N, dtype=bool)
        non_affected_mask[affected_arr] = False

        for dim_idx in range(2):
            gen_mat = self.gen_x_matrix if dim_idx == 0 else self.gen_y_matrix
            orig_mat = self.orig_x_matrix if dim_idx == 0 else self.orig_y_matrix
            roughness = self.roughness_x if dim_idx == 0 else self.roughness_y
            values = self._gen_points[:, dim_idx]

            # --- Count OLD mismatches in affected region ---
            old_mm = 0
            for idx in affected_indices:
                old_mm += int(np.count_nonzero(gen_mat[idx, :] != orig_mat[idx, :]))
            for idx in affected_indices:
                old_mm += int(np.count_nonzero(
                    gen_mat[non_affected_mask, idx] != orig_mat[non_affected_mask, idx]
                ))

            # --- Update affected rows and columns ---
            for idx in affected_indices:
                val_idx = values[idx]
                # Row idx
                diff_row = values - val_idx
                new_row = np.zeros(N)
                new_row[diff_row < -roughness] = 2.0
                new_row[np.abs(diff_row) <= roughness] = 1.0
                gen_mat[idx, :] = new_row
                # Col idx
                diff_col = val_idx - values
                new_col = np.zeros(N)
                new_col[diff_col < -roughness] = 2.0
                new_col[np.abs(diff_col) <= roughness] = 1.0
                gen_mat[:, idx] = new_col

            # --- Count NEW mismatches in affected region ---
            new_mm = 0
            for idx in affected_indices:
                new_mm += int(np.count_nonzero(gen_mat[idx, :] != orig_mat[idx, :]))
            for idx in affected_indices:
                new_mm += int(np.count_nonzero(
                    gen_mat[non_affected_mask, idx] != orig_mat[non_affected_mask, idx]
                ))

            if dim_idx == 0:
                self._d1_mismatches += new_mm - old_mm
            else:
                self._d2_mismatches += new_mm - old_mm
