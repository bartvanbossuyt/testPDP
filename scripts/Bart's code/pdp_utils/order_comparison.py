# -*- coding: utf-8 -*-
"""
Order comparison utilities using PDP inequality matrices.
Pure logic functions without Streamlit dependencies.
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

    # Apply buffer transformation if needed
    if pdp_variant in ["buffer", "bufferrough"]:
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, buffer_y)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, buffer_y)
    elif pdp_variant == "realistic":
        orig_pts = apply_buffer_transformation(orig_pts, buffer_x, 0.0)
        gen_pts = apply_buffer_transformation(gen_pts, buffer_x, 0.0)

    # Determine roughness values
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
