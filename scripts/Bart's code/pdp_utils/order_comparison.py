# -*- coding: utf-8 -*-
"""
Order comparison utilities using PDP inequality matrices.
Pure logic functions without Streamlit dependencies.

Supports both Cartesian (x, y) and Frenet (s, n) coordinate systems.
Frenet coordinates are essential for curved road analysis where the
driving direction is not aligned with the x-axis.
"""

import re
import numpy as np
from typing import Optional

from .core import (
    compute_inequality_matrix,
    compare_inequality_matrices,
    compare_inequality_matrices_with_threshold,
    apply_buffer_transformation
)
from .frenet_coordinates import FrenetFrame, compute_inequality_matrix_frenet


def strip_primes(text: str) -> str:
    """Remove prime markers and * markers from a LaTeX-like string."""
    text = re.sub(r"\^\{\*\}", "", text)
    text = re.sub(r"[']+", "", text)
    text = text.replace("*", "")
    return text


def extract_order_string(latex_str: str) -> str:
    """Strip d_1/d_2 prefixes, prime decorations and braces so only the bare order remains."""
    core = latex_str.replace("d_1:", "").replace("d_2:", "").strip()
    core_no_primes = strip_primes(core)
    core_no_braces = re.sub(r"\{([^{}]+)\}", r"\1", core_no_primes)
    return core_no_braces


def check_pdp_match(
    original_points: np.ndarray, 
    generated_points: np.ndarray,
    pdp_variant: str = "fundamental",
    buffer_x: float = 25.0,
    buffer_y: float = 10.0,
    rough_x: float = 0.0,
    rough_y: float = 0.0,
    match_threshold: float = 1.0,
    max_mismatches: Optional[int] = None,
    debug: bool = False
) -> tuple[bool, bool]:
    """
    Check if generated configuration matches original using PDP inequality matrices.

    Supports five variants:
    - fundamental: Basic PDP matching with no tolerance
    - buffer: Apply buffer transformation to both configs
    - rough: Use roughness as equality tolerance
    - bufferrough: Apply buffer transformation AND use roughness tolerance
    - realistic: Buffer ONLY on x, roughness ONLY on y

    Args:
        original_points: Original points (N, 2)
        generated_points: Generated points (N, 2)
        pdp_variant: PDP variant to use
        buffer_x: Buffer distance in x-direction
        buffer_y: Buffer distance in y-direction
        rough_x: Roughness tolerance in x-direction
        rough_y: Roughness tolerance in y-direction
        match_threshold: Minimum match percentage (0.0 to 1.0)
        max_mismatches: If not None, use absolute mismatch mode
        debug: Print debug information

    Returns:
        (d1_match, d2_match): Boolean tuple
    """
    orig_pts = original_points.copy()
    gen_pts = generated_points.copy()

    # DEBUG: Print received parameters
    print(f"[DEBUG CHECK_PDP_MATCH] variant={pdp_variant}, buffer_x={buffer_x}, buffer_y={buffer_y}, rough_x={rough_x}, rough_y={rough_y}")

    # Apply buffer transformation if needed
    if pdp_variant in ["buffer", "bufferrough"]:
        print(f"[DEBUG CHECK_PDP_MATCH] Applying buffer transformation: buffer_x={buffer_x}, buffer_y={buffer_y}")
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, buffer_y)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, buffer_y)
    elif pdp_variant == "realistic":
        print(f"[DEBUG CHECK_PDP_MATCH] Applying realistic buffer: buffer_x={buffer_x}, buffer_y=0")
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, 0.0)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, 0.0)

    # Determine roughness values
    if pdp_variant in ["rough", "bufferrough"]:
        roughness_x = rough_x
        roughness_y = rough_y
        print(f"[DEBUG CHECK_PDP_MATCH] Using rough variant: roughness_x={roughness_x}, roughness_y={roughness_y}")
    elif pdp_variant == "realistic":
        roughness_x = 0.0
        roughness_y = rough_y
        print(f"[DEBUG CHECK_PDP_MATCH] Using realistic variant: roughness_x=0, roughness_y={roughness_y}")
    else:
        roughness_x = 0.0
        roughness_y = 0.0
        print(f"[DEBUG CHECK_PDP_MATCH] Using fundamental variant: roughness_x=0, roughness_y=0")

    # Compute inequality matrices
    original_x_matrix = compute_inequality_matrix(orig_pts, 0, roughness_x)
    original_y_matrix = compute_inequality_matrix(orig_pts, 1, roughness_y)
    generated_x_matrix = compute_inequality_matrix(gen_pts, 0, roughness_x)
    generated_y_matrix = compute_inequality_matrix(gen_pts, 1, roughness_y)

    # Compare matrices
    _, d1_percentage = compare_inequality_matrices_with_threshold(
        original_x_matrix, generated_x_matrix, 1.0
    )
    _, d2_percentage = compare_inequality_matrices_with_threshold(
        original_y_matrix, generated_y_matrix, 1.0
    )

    # Determine match based on mode
    if max_mismatches is not None:
        n = original_x_matrix.shape[0]
        d1_mismatches = 0
        d2_mismatches = 0
        for i in range(n):
            for j in range(i + 1, n):
                if original_x_matrix[i, j] != generated_x_matrix[i, j]:
                    d1_mismatches += 1
                if original_y_matrix[i, j] != generated_y_matrix[i, j]:
                    d2_mismatches += 1
        total_mismatches = d1_mismatches + d2_mismatches
        d1_match = total_mismatches <= max_mismatches
        d2_match = total_mismatches <= max_mismatches
    elif match_threshold < 1.0:
        avg_percentage = (d1_percentage + d2_percentage) / 2.0
        d1_match = avg_percentage >= match_threshold
        d2_match = avg_percentage >= match_threshold
    else:
        d1_match = d1_percentage >= match_threshold
        d2_match = d2_percentage >= match_threshold

    return d1_match, d2_match


def check_pdp_match_detailed(
    original_points: np.ndarray, 
    generated_points: np.ndarray,
    pdp_variant: str = "fundamental",
    buffer_x: float = 25.0,
    buffer_y: float = 10.0,
    rough_x: float = 0.0,
    rough_y: float = 0.0,
    match_threshold: float = 1.0,
    max_mismatches: Optional[int] = None
) -> dict:
    """
    Extended version that returns detailed results for visualization.

    Returns dict with:
    - d1_match, d2_match: Booleans
    - d1_percentage, d2_percentage: Match percentages
    - original/generated matrices for both dimensions
    - mismatch counts
    """
    orig_pts = original_points.copy()
    gen_pts = generated_points.copy()

    if pdp_variant in ["buffer", "bufferrough"]:
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, buffer_y)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, buffer_y)
    elif pdp_variant == "realistic":
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, 0.0)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, 0.0)

    if pdp_variant in ["rough", "bufferrough"]:
        roughness_x = rough_x
        roughness_y = rough_y
    elif pdp_variant == "realistic":
        roughness_x = 0.0
        roughness_y = rough_y
    else:
        roughness_x = 0.0
        roughness_y = 0.0

    original_d1_matrix = compute_inequality_matrix(orig_pts, 0, roughness_x)
    original_d2_matrix = compute_inequality_matrix(orig_pts, 1, roughness_y)
    generated_d1_matrix = compute_inequality_matrix(gen_pts, 0, roughness_x)
    generated_d2_matrix = compute_inequality_matrix(gen_pts, 1, roughness_y)

    _, d1_percentage = compare_inequality_matrices_with_threshold(
        original_d1_matrix, generated_d1_matrix, 1.0
    )
    _, d2_percentage = compare_inequality_matrices_with_threshold(
        original_d2_matrix, generated_d2_matrix, 1.0
    )

    n = original_d1_matrix.shape[0]
    d1_mismatches = 0
    d2_mismatches = 0
    total_cells = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_cells += 1
            if original_d1_matrix[i, j] != generated_d1_matrix[i, j]:
                d1_mismatches += 1
            if original_d2_matrix[i, j] != generated_d2_matrix[i, j]:
                d2_mismatches += 1

    if max_mismatches is not None:
        total_mismatch_count = d1_mismatches + d2_mismatches
        d1_match = total_mismatch_count <= max_mismatches
        d2_match = total_mismatch_count <= max_mismatches
    elif match_threshold < 1.0:
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
        "d1_mismatches": d1_mismatches,
        "d2_mismatches": d2_mismatches,
        "total_mismatches": d1_mismatches + d2_mismatches,
        "total_cells": total_cells,
        "original_d1_matrix": original_d1_matrix,
        "original_d2_matrix": original_d2_matrix,
        "generated_d1_matrix": generated_d1_matrix,
        "generated_d2_matrix": generated_d2_matrix,
    }


def check_pdp_match_frenet(
    original_points: np.ndarray, 
    generated_points: np.ndarray,
    centerline: np.ndarray,
    pdp_variant: str = "fundamental",
    buffer_s: float = 25.0,
    buffer_n: float = 10.0,
    rough_s: float = 0.0,
    rough_n: float = 0.0,
    match_threshold: float = 1.0,
    max_mismatches: Optional[int] = None,
    debug: bool = False
) -> tuple[bool, bool]:
    """
    Check PDP match using Frenet (road-relative) coordinates.
    
    This is the key function for curved road analysis. Instead of using
    absolute (x, y) coordinates, it transforms points to a road-relative
    coordinate system where:
    - s = distance along the road (longitudinal position)
    - n = perpendicular distance from road centerline (lateral position)
    
    This ensures that PDP orderings are computed relative to the driving
    direction, not absolute Cartesian axes.
    
    Args:
        original_points: Original points in Cartesian (x, y)
        generated_points: Generated points in Cartesian (x, y)
        centerline: Road centerline for Frenet transformation
        pdp_variant: PDP variant (fundamental, buffer, rough, bufferrough, realistic)
        buffer_s: Buffer distance along road direction
        buffer_n: Buffer distance perpendicular to road
        rough_s: Roughness tolerance along road direction
        rough_n: Roughness tolerance perpendicular to road
        match_threshold: Minimum match percentage (0.0 to 1.0)
        max_mismatches: If not None, use absolute mismatch mode
        debug: Print debug information
    
    Returns:
        (s_match, n_match): Whether orderings match in s (along road) and n (lateral)
    """
    # Create Frenet frame from centerline
    frame = FrenetFrame(centerline, smooth=True)
    
    # Transform to Frenet coordinates
    orig_frenet = frame.to_frenet(original_points)
    gen_frenet = frame.to_frenet(generated_points)
    
    if debug:
        print(f"[FRENET] Original Cartesian: {original_points}")
        print(f"[FRENET] Original Frenet (s,n): {orig_frenet}")
        print(f"[FRENET] Generated Cartesian: {generated_points}")
        print(f"[FRENET] Generated Frenet (s,n): {gen_frenet}")
    
    # Apply buffer transformation if needed (in Frenet space)
    if pdp_variant in ["buffer", "bufferrough"]:
        from .core import apply_buffer_transformation
        orig_frenet = apply_buffer_transformation(orig_frenet, buffer_s, buffer_n)
        gen_frenet = apply_buffer_transformation(gen_frenet, buffer_s, buffer_n)
    elif pdp_variant == "realistic":
        from .core import apply_buffer_transformation
        orig_frenet = apply_buffer_transformation(orig_frenet, buffer_s, 0.0)
        gen_frenet = apply_buffer_transformation(gen_frenet, buffer_s, 0.0)
    
    # Determine roughness values
    if pdp_variant in ["rough", "bufferrough"]:
        roughness_s = rough_s
        roughness_n = rough_n
    elif pdp_variant == "realistic":
        roughness_s = 0.0
        roughness_n = rough_n
    else:
        roughness_s = 0.0
        roughness_n = 0.0
    
    # Compute inequality matrices in Frenet coordinates
    original_s_matrix = compute_inequality_matrix_frenet(orig_frenet, 0, roughness_s)
    original_n_matrix = compute_inequality_matrix_frenet(orig_frenet, 1, roughness_n)
    generated_s_matrix = compute_inequality_matrix_frenet(gen_frenet, 0, roughness_s)
    generated_n_matrix = compute_inequality_matrix_frenet(gen_frenet, 1, roughness_n)
    
    # Compare matrices
    _, s_percentage = compare_inequality_matrices_with_threshold(
        original_s_matrix, generated_s_matrix, 1.0
    )
    _, n_percentage = compare_inequality_matrices_with_threshold(
        original_n_matrix, generated_n_matrix, 1.0
    )
    
    if debug:
        print(f"[FRENET] s-match percentage: {s_percentage:.2%}")
        print(f"[FRENET] n-match percentage: {n_percentage:.2%}")
    
    # Determine match based on mode
    if max_mismatches is not None:
        n_pts = original_s_matrix.shape[0]
        s_mismatches = 0
        n_mismatches = 0
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                if original_s_matrix[i, j] != generated_s_matrix[i, j]:
                    s_mismatches += 1
                if original_n_matrix[i, j] != generated_n_matrix[i, j]:
                    n_mismatches += 1
        total_mismatches = s_mismatches + n_mismatches
        s_match = total_mismatches <= max_mismatches
        n_match = total_mismatches <= max_mismatches
    elif match_threshold < 1.0:
        avg_percentage = (s_percentage + n_percentage) / 2.0
        s_match = avg_percentage >= match_threshold
        n_match = avg_percentage >= match_threshold
    else:
        s_match = s_percentage >= match_threshold
        n_match = n_percentage >= match_threshold
    
    return s_match, n_match


def check_pdp_match_frenet_detailed(
    original_points: np.ndarray, 
    generated_points: np.ndarray,
    centerline: np.ndarray,
    pdp_variant: str = "fundamental",
    buffer_s: float = 25.0,
    buffer_n: float = 10.0,
    rough_s: float = 0.0,
    rough_n: float = 0.0,
    match_threshold: float = 1.0,
    max_mismatches: Optional[int] = None
) -> dict:
    """
    Extended version of check_pdp_match_frenet that returns detailed results.
    
    Returns dict with:
    - s_match, n_match: Booleans (s=along road, n=lateral)
    - s_percentage, n_percentage: Match percentages
    - original/generated matrices for both dimensions
    - mismatch counts
    - Frenet coordinates for visualization
    """
    frame = FrenetFrame(centerline, smooth=True)
    
    # Transform to Frenet coordinates
    orig_frenet = frame.to_frenet(original_points)
    gen_frenet = frame.to_frenet(generated_points)
    
    # Store original Frenet coords before buffer transformation
    orig_frenet_raw = orig_frenet.copy()
    gen_frenet_raw = gen_frenet.copy()
    
    # Apply transformations
    if pdp_variant in ["buffer", "bufferrough"]:
        from .core import apply_buffer_transformation
        orig_frenet = apply_buffer_transformation(orig_frenet, buffer_s, buffer_n)
        gen_frenet = apply_buffer_transformation(gen_frenet, buffer_s, buffer_n)
    elif pdp_variant == "realistic":
        from .core import apply_buffer_transformation
        orig_frenet = apply_buffer_transformation(orig_frenet, buffer_s, 0.0)
        gen_frenet = apply_buffer_transformation(gen_frenet, buffer_s, 0.0)
    
    if pdp_variant in ["rough", "bufferrough"]:
        roughness_s = rough_s
        roughness_n = rough_n
    elif pdp_variant == "realistic":
        roughness_s = 0.0
        roughness_n = rough_n
    else:
        roughness_s = 0.0
        roughness_n = 0.0
    
    original_s_matrix = compute_inequality_matrix_frenet(orig_frenet, 0, roughness_s)
    original_n_matrix = compute_inequality_matrix_frenet(orig_frenet, 1, roughness_n)
    generated_s_matrix = compute_inequality_matrix_frenet(gen_frenet, 0, roughness_s)
    generated_n_matrix = compute_inequality_matrix_frenet(gen_frenet, 1, roughness_n)
    
    _, s_percentage = compare_inequality_matrices_with_threshold(
        original_s_matrix, generated_s_matrix, 1.0
    )
    _, n_percentage = compare_inequality_matrices_with_threshold(
        original_n_matrix, generated_n_matrix, 1.0
    )
    
    n_pts = original_s_matrix.shape[0]
    s_mismatches = 0
    n_mismatches = 0
    total_cells = 0
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            total_cells += 1
            if original_s_matrix[i, j] != generated_s_matrix[i, j]:
                s_mismatches += 1
            if original_n_matrix[i, j] != generated_n_matrix[i, j]:
                n_mismatches += 1
    
    if max_mismatches is not None:
        total_mismatch_count = s_mismatches + n_mismatches
        s_match = total_mismatch_count <= max_mismatches
        n_match = total_mismatch_count <= max_mismatches
    elif match_threshold < 1.0:
        avg_percentage = (s_percentage + n_percentage) / 2.0
        s_match = avg_percentage >= match_threshold
        n_match = avg_percentage >= match_threshold
    else:
        s_match = s_percentage >= match_threshold
        n_match = n_percentage >= match_threshold
    
    return {
        "s_match": s_match,
        "n_match": n_match,
        "s_percentage": s_percentage,
        "n_percentage": n_percentage,
        "avg_percentage": (s_percentage + n_percentage) / 2.0,
        "s_mismatches": s_mismatches,
        "n_mismatches": n_mismatches,
        "total_mismatches": s_mismatches + n_mismatches,
        "total_cells": total_cells,
        "original_s_matrix": original_s_matrix,
        "original_n_matrix": original_n_matrix,
        "generated_s_matrix": generated_s_matrix,
        "generated_n_matrix": generated_n_matrix,
        "original_frenet": orig_frenet_raw,
        "generated_frenet": gen_frenet_raw,
        "frenet_frame": frame,
    }
