# -*- coding: utf-8 -*-
"""
PDP Calculations Module

Contains core PDP (Point Distance Precedence) functions:
- Inequality matrix computation
- Matrix comparison with thresholds
- Buffer transformation
- Match checking for various PDP variants
"""

import numpy as np
from typing import Tuple


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


def compare_inequality_matrices_with_threshold(
    matrix1: np.ndarray, 
    matrix2: np.ndarray, 
    threshold: float = 1.0
) -> Tuple[bool, float]:
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


def check_pdp_match(
    original_points: np.ndarray, 
    generated_points: np.ndarray,
    pdp_variant: str = "fundamental",
    buffer_x: float = 25.0,
    buffer_y: float = 10.0,
    rough_x: float = 0.0,
    rough_y: float = 0.0,
    match_threshold: float = 1.0,
    debug: bool = False
) -> Tuple[bool, bool]:
    """
    Check if generated points preserve the PDP pattern of original points.
    
    Supports multiple PDP variants:
    - fundamental: No transformation, strict equality
    - rough: Apply roughness tolerance
    - buffer: Apply buffer transformation (5x points)
    - bufferrough: Both buffer and roughness
    - realistic: Buffer in x only, roughness in y only
    
    Args:
        original_points: (N, 2) array of original coordinates
        generated_points: (N, 2) array of generated coordinates
        pdp_variant: One of "fundamental", "rough", "buffer", "bufferrough", "realistic"
        buffer_x: Buffer distance in x-direction
        buffer_y: Buffer distance in y-direction
        rough_x: Roughness tolerance in x-direction
        rough_y: Roughness tolerance in y-direction
        match_threshold: Minimum match percentage (0.0 to 1.0)
        debug: If True, print debug information
    
    Returns:
        (d1_match, d2_match): Tuple of boolean match results for each dimension
    """
    # Apply buffer transformation if needed
    orig_pts = original_points.copy()
    gen_pts = generated_points.copy()
    
    if pdp_variant in ["buffer", "bufferrough"]:
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, buffer_y)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, buffer_y)
    elif pdp_variant == "realistic":
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, 0.0)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, 0.0)
    
    # Determine roughness values based on variant
    if pdp_variant in ["rough", "bufferrough"]:
        roughness_x = rough_x
        roughness_y = rough_y
    elif pdp_variant == "realistic":
        roughness_x = 0.0
        roughness_y = rough_y
    else:
        roughness_x = 0.0
        roughness_y = 0.0
    
    # Compute inequality matrices
    orig_d1 = compute_inequality_matrix(orig_pts, 0, roughness_x)
    orig_d2 = compute_inequality_matrix(orig_pts, 1, roughness_y)
    gen_d1 = compute_inequality_matrix(gen_pts, 0, roughness_x)
    gen_d2 = compute_inequality_matrix(gen_pts, 1, roughness_y)
    
    # Compare matrices with threshold
    if match_threshold < 1.0:
        # For relaxed thresholds, use average of d1 and d2 percentages
        _, d1_pct = compare_inequality_matrices_with_threshold(orig_d1, gen_d1, 1.0)
        _, d2_pct = compare_inequality_matrices_with_threshold(orig_d2, gen_d2, 1.0)
        avg_pct = (d1_pct + d2_pct) / 2.0
        d1_match = avg_pct >= match_threshold
        d2_match = avg_pct >= match_threshold
    else:
        d1_match, _ = compare_inequality_matrices_with_threshold(orig_d1, gen_d1, match_threshold)
        d2_match, _ = compare_inequality_matrices_with_threshold(orig_d2, gen_d2, match_threshold)
    
    if debug:
        print(f"[PDP DEBUG] Variant: {pdp_variant}, d1_match: {d1_match}, d2_match: {d2_match}")
    
    return d1_match, d2_match


def check_pdp_match_detailed(
    original_points: np.ndarray, 
    generated_points: np.ndarray,
    pdp_variant: str = "fundamental",
    buffer_x: float = 25.0,
    buffer_y: float = 10.0,
    rough_x: float = 0.0,
    rough_y: float = 0.0,
    match_threshold: float = 1.0
) -> dict:
    """
    Extended version of check_pdp_match that returns detailed results for heat map visualization.
    
    Returns:
        Dictionary with:
        - d1_match: Boolean (True if d1 match >= threshold)
        - d2_match: Boolean (True if d2 match >= threshold)
        - d1_percentage: Float (actual d1 match percentage)
        - d2_percentage: Float (actual d2 match percentage)
        - avg_percentage: Float (average of d1 and d2 percentages)
        - original_d1_matrix: N×N inequality matrix for original d1
        - original_d2_matrix: N×N inequality matrix for original d2
        - generated_d1_matrix: N×N inequality matrix for generated d1
        - generated_d2_matrix: N×N inequality matrix for generated d2
    """
    # Apply buffer transformation if needed (for matching calculation)
    orig_pts_matching = original_points.copy()
    gen_pts_matching = generated_points.copy()
    
    if pdp_variant in ["buffer", "bufferrough"]:
        orig_pts_matching = apply_buffer_transformation(orig_pts_matching, buffer_x, buffer_y)
        gen_pts_matching = apply_buffer_transformation(gen_pts_matching, buffer_x, buffer_y)
    elif pdp_variant == "realistic":
        orig_pts_matching = apply_buffer_transformation(orig_pts_matching, buffer_x, 0.0)
        gen_pts_matching = apply_buffer_transformation(gen_pts_matching, buffer_x, 0.0)
    
    # Determine roughness values based on variant
    if pdp_variant in ["rough", "bufferrough"]:
        roughness_x = rough_x
        roughness_y = rough_y
    elif pdp_variant == "realistic":
        roughness_x = 0.0
        roughness_y = rough_y
    else:
        roughness_x = 0.0
        roughness_y = 0.0
    
    # Compute inequality matrices for matching
    orig_d1_matching = compute_inequality_matrix(orig_pts_matching, 0, roughness_x)
    orig_d2_matching = compute_inequality_matrix(orig_pts_matching, 1, roughness_y)
    gen_d1_matching = compute_inequality_matrix(gen_pts_matching, 0, roughness_x)
    gen_d2_matching = compute_inequality_matrix(gen_pts_matching, 1, roughness_y)
    
    # Compute inequality matrices for display (always original points, no buffer)
    original_d1_matrix = compute_inequality_matrix(original_points, 0, roughness_x)
    original_d2_matrix = compute_inequality_matrix(original_points, 1, roughness_y)
    generated_d1_matrix = compute_inequality_matrix(generated_points, 0, roughness_x)
    generated_d2_matrix = compute_inequality_matrix(generated_points, 1, roughness_y)
    
    # Get percentages using matching matrices
    _, d1_percentage = compare_inequality_matrices_with_threshold(orig_d1_matching, gen_d1_matching, 1.0)
    _, d2_percentage = compare_inequality_matrices_with_threshold(orig_d2_matching, gen_d2_matching, 1.0)
    
    # Determine match based on threshold
    if match_threshold < 1.0:
        avg_percentage = (d1_percentage + d2_percentage) / 2.0
        d1_match = avg_percentage >= match_threshold
        d2_match = avg_percentage >= match_threshold
    else:
        d1_match = d1_percentage >= match_threshold
        d2_match = d2_percentage >= match_threshold
    
    return {
        "d1_match": d1_match,
        "d2_match": d2_match,
        "d1_percentage": d1_percentage,
        "d2_percentage": d2_percentage,
        "avg_percentage": (d1_percentage + d2_percentage) / 2.0,
        "original_d1_matrix": original_d1_matrix,
        "original_d2_matrix": original_d2_matrix,
        "generated_d1_matrix": generated_d1_matrix,
        "generated_d2_matrix": generated_d2_matrix,
    }
