# -*- coding: utf-8 -*-
"""
Frenet-Serret Local Coordinate System for PDP in Curved Roads.

This module implements a road-relative coordinate system where:
- s (arc length): distance along the road centerline (replaces d1)
- n (lateral offset): perpendicular distance from centerline (replaces d2)

This solves the problem of PDP analysis in curved roads where the traditional
x/y coordinates don't align with the driving direction.

The Frenet-Serret frame provides:
- T (tangent): unit vector along the road direction
- N (normal): unit vector perpendicular to the road (positive = left)

Reference: PhD Thesis Qayyum (2022) - PDP analysis on curved road networks
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.interpolate import splprep, splev
from scipy.spatial import distance


def compute_centerline_from_trajectory(
    trajectory: np.ndarray,
    smooth: bool = True,
    smoothing_factor: float = 0.0
) -> np.ndarray:
    """
    Compute a smooth centerline from a vehicle trajectory.
    
    Args:
        trajectory: (N, 2) array of [x, y] points
        smooth: Whether to apply spline smoothing
        smoothing_factor: Smoothing factor for spline (0 = interpolate exactly)
    
    Returns:
        Smoothed centerline as (M, 2) array
    """
    if trajectory.shape[0] < 3:
        return trajectory.copy()
    
    if not smooth:
        return trajectory.copy()
    
    try:
        # Fit parametric spline
        tck, u = splprep([trajectory[:, 0], trajectory[:, 1]], s=smoothing_factor, k=min(3, len(trajectory)-1))
        # Evaluate at more points for smoother curve
        u_new = np.linspace(0, 1, max(len(trajectory) * 3, 50))
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except Exception:
        return trajectory.copy()


def compute_arc_length(centerline: np.ndarray) -> np.ndarray:
    """
    Compute cumulative arc length along a centerline.
    
    Args:
        centerline: (N, 2) array of [x, y] points
    
    Returns:
        (N,) array of cumulative arc lengths, starting at 0
    """
    if centerline.shape[0] < 2:
        return np.array([0.0])
    
    # Compute segment lengths
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    
    # Cumulative sum starting from 0
    arc_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return arc_lengths


def compute_tangent_vectors(centerline: np.ndarray) -> np.ndarray:
    """
    Compute unit tangent vectors at each point along the centerline.
    
    Args:
        centerline: (N, 2) array of [x, y] points
    
    Returns:
        (N, 2) array of unit tangent vectors
    """
    if centerline.shape[0] < 2:
        return np.array([[1.0, 0.0]])
    
    tangents = np.zeros_like(centerline)
    
    # Interior points: use central difference
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    
    # Endpoints: use forward/backward difference
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    
    # Normalize
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0  # Avoid division by zero
    tangents = tangents / norms
    
    return tangents


def compute_normal_vectors(tangents: np.ndarray) -> np.ndarray:
    """
    Compute unit normal vectors from tangent vectors.
    Normal is perpendicular to tangent, pointing left (counterclockwise 90°).
    
    Args:
        tangents: (N, 2) array of unit tangent vectors
    
    Returns:
        (N, 2) array of unit normal vectors
    """
    # Rotate tangent 90° counterclockwise: (x, y) -> (-y, x)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def find_closest_centerline_point(
    point: np.ndarray,
    centerline: np.ndarray,
    arc_lengths: Optional[np.ndarray] = None
) -> Tuple[int, float, float]:
    """
    Find the closest point on the centerline to a given point.
    
    Args:
        point: (2,) array [x, y]
        centerline: (N, 2) array of centerline points
        arc_lengths: Optional precomputed arc lengths
    
    Returns:
        (index, s_value, distance): Index of closest point, arc length, and distance
    """
    if arc_lengths is None:
        arc_lengths = compute_arc_length(centerline)
    
    # Find closest point (simple version - can be improved with projection)
    distances = np.linalg.norm(centerline - point, axis=1)
    idx = np.argmin(distances)
    
    # Refine by projecting onto the line segment
    if centerline.shape[0] >= 2:
        # Check segment before
        if idx > 0:
            s_refined, dist_refined = _project_to_segment(
                point, centerline[idx-1], centerline[idx]
            )
            if dist_refined < distances[idx]:
                s = arc_lengths[idx-1] + s_refined * (arc_lengths[idx] - arc_lengths[idx-1])
                return idx, s, dist_refined
        
        # Check segment after
        if idx < centerline.shape[0] - 1:
            s_refined, dist_refined = _project_to_segment(
                point, centerline[idx], centerline[idx+1]
            )
            if dist_refined < distances[idx]:
                s = arc_lengths[idx] + s_refined * (arc_lengths[idx+1] - arc_lengths[idx])
                return idx, s, dist_refined
    
    return idx, arc_lengths[idx], distances[idx]


def _project_to_segment(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray
) -> Tuple[float, float]:
    """
    Project a point onto a line segment.
    
    Args:
        point: (2,) array
        seg_start: Start of segment (2,)
        seg_end: End of segment (2,)
    
    Returns:
        (t, distance): Parameter t ∈ [0,1] and perpendicular distance
    """
    v = seg_end - seg_start
    w = point - seg_start
    
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)
    
    if c2 < 1e-10:
        return 0.0, np.linalg.norm(w)
    
    t = max(0.0, min(1.0, c1 / c2))
    projection = seg_start + t * v
    distance = np.linalg.norm(point - projection)
    
    return t, distance


def cartesian_to_frenet(
    points: np.ndarray,
    centerline: np.ndarray,
    arc_lengths: Optional[np.ndarray] = None,
    tangents: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert Cartesian (x, y) coordinates to Frenet (s, n) coordinates.
    
    s = arc length along centerline (longitudinal position)
    n = signed lateral offset from centerline (positive = left of driving direction)
    
    Args:
        points: (N, 2) array of [x, y] coordinates
        centerline: (M, 2) array of centerline points
        arc_lengths: Optional precomputed arc lengths for centerline
        tangents: Optional precomputed tangent vectors
        normals: Optional precomputed normal vectors
    
    Returns:
        (N, 2) array of [s, n] Frenet coordinates
    """
    if arc_lengths is None:
        arc_lengths = compute_arc_length(centerline)
    if tangents is None:
        tangents = compute_tangent_vectors(centerline)
    if normals is None:
        normals = compute_normal_vectors(tangents)
    
    frenet_coords = np.zeros_like(points)
    
    for i, pt in enumerate(points):
        idx, s, _ = find_closest_centerline_point(pt, centerline, arc_lengths)
        
        # Find the segment index based on s value for correct normal interpolation
        seg_idx = np.searchsorted(arc_lengths, s, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, len(centerline) - 2)
        
        # Interpolate the centerline position and normal at arc length s
        if seg_idx < len(centerline) - 1 and arc_lengths[seg_idx + 1] > arc_lengths[seg_idx]:
            t_param = (s - arc_lengths[seg_idx]) / (arc_lengths[seg_idx + 1] - arc_lengths[seg_idx])
            t_param = np.clip(t_param, 0, 1)
            # Interpolate centerline point
            cl_point = (1 - t_param) * centerline[seg_idx] + t_param * centerline[seg_idx + 1]
            # Interpolate normal (and renormalize)
            normal = (1 - t_param) * normals[seg_idx] + t_param * normals[seg_idx + 1]
            normal = normal / (np.linalg.norm(normal) + 1e-10)
        else:
            cl_point = centerline[seg_idx]
            normal = normals[seg_idx]
        
        # Compute signed lateral offset
        # Vector from interpolated centerline point to our point
        to_point = pt - cl_point
        
        # Project onto normal to get signed distance (positive = left)
        n = np.dot(to_point, normal)
        
        frenet_coords[i] = [s, n]
    
    return frenet_coords


def frenet_to_cartesian(
    frenet_points: np.ndarray,
    centerline: np.ndarray,
    arc_lengths: Optional[np.ndarray] = None,
    tangents: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert Frenet (s, n) coordinates back to Cartesian (x, y).
    
    Args:
        frenet_points: (N, 2) array of [s, n] Frenet coordinates
        centerline: (M, 2) array of centerline points
        arc_lengths: Optional precomputed arc lengths
        tangents: Optional precomputed tangent vectors
        normals: Optional precomputed normal vectors
    
    Returns:
        (N, 2) array of [x, y] Cartesian coordinates
    """
    if arc_lengths is None:
        arc_lengths = compute_arc_length(centerline)
    if tangents is None:
        tangents = compute_tangent_vectors(centerline)
    if normals is None:
        normals = compute_normal_vectors(tangents)
    
    cartesian_coords = np.zeros_like(frenet_points)
    
    for i, (s, n) in enumerate(frenet_points):
        # Find centerline point at arc length s
        idx = np.searchsorted(arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, len(centerline) - 1)
        
        # Interpolate position on centerline
        if idx < len(centerline) - 1 and arc_lengths[idx+1] > arc_lengths[idx]:
            t = (s - arc_lengths[idx]) / (arc_lengths[idx+1] - arc_lengths[idx])
            t = np.clip(t, 0, 1)
            base_point = (1 - t) * centerline[idx] + t * centerline[idx + 1]
            normal = (1 - t) * normals[idx] + t * normals[idx + 1]
            normal = normal / (np.linalg.norm(normal) + 1e-10)
        else:
            base_point = centerline[idx]
            normal = normals[idx]
        
        # Add lateral offset
        cartesian_coords[i] = base_point + n * normal
    
    return cartesian_coords


class FrenetFrame:
    """
    Encapsulates a Frenet-Serret frame for a road centerline.
    
    This class precomputes and caches all necessary values for efficient
    coordinate transformations.
    
    Usage:
        centerline = extract_centerline_from_trajectory(...)
        frame = FrenetFrame(centerline)
        frenet_points = frame.to_frenet(cartesian_points)
        cartesian_back = frame.to_cartesian(frenet_points)
    """
    
    def __init__(self, centerline: np.ndarray, smooth: bool = False):
        """
        Initialize Frenet frame from a centerline.
        
        Args:
            centerline: (N, 2) array of [x, y] centerline points
            smooth: Whether to apply spline smoothing
        """
        if smooth and centerline.shape[0] >= 3:
            self.centerline = compute_centerline_from_trajectory(centerline, smooth=True)
        else:
            self.centerline = centerline.copy()
        
        self.arc_lengths = compute_arc_length(self.centerline)
        self.tangents = compute_tangent_vectors(self.centerline)
        self.normals = compute_normal_vectors(self.tangents)
        self.total_length = self.arc_lengths[-1] if len(self.arc_lengths) > 0 else 0.0
    
    def to_frenet(self, points: np.ndarray) -> np.ndarray:
        """Convert Cartesian points to Frenet coordinates."""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return cartesian_to_frenet(
            points, self.centerline, self.arc_lengths, self.tangents, self.normals
        )
    
    def to_cartesian(self, frenet_points: np.ndarray) -> np.ndarray:
        """Convert Frenet coordinates to Cartesian points."""
        if frenet_points.ndim == 1:
            frenet_points = frenet_points.reshape(1, -1)
        return frenet_to_cartesian(
            frenet_points, self.centerline, self.arc_lengths, self.tangents, self.normals
        )
    
    def get_tangent_at_s(self, s: float) -> np.ndarray:
        """Get the tangent vector at a given arc length s."""
        idx = np.searchsorted(self.arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, len(self.tangents) - 1)
        return self.tangents[idx]
    
    def get_normal_at_s(self, s: float) -> np.ndarray:
        """Get the normal vector at a given arc length s."""
        idx = np.searchsorted(self.arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, len(self.normals) - 1)
        return self.normals[idx]
    
    def get_position_at_s(self, s: float) -> np.ndarray:
        """Get the centerline position at a given arc length s."""
        idx = np.searchsorted(self.arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, len(self.centerline) - 1)
        
        if idx < len(self.centerline) - 1 and self.arc_lengths[idx+1] > self.arc_lengths[idx]:
            t = (s - self.arc_lengths[idx]) / (self.arc_lengths[idx+1] - self.arc_lengths[idx])
            t = np.clip(t, 0, 1)
            return (1 - t) * self.centerline[idx] + t * self.centerline[idx + 1]
        return self.centerline[idx]


def compute_inequality_matrix_frenet(
    points: np.ndarray,
    dimension: int,
    roughness: float = 0.0
) -> np.ndarray:
    """
    Compute PDP inequality matrix for Frenet coordinates.
    
    This is identical to the standard compute_inequality_matrix but operates
    on Frenet (s, n) coordinates instead of (x, y).
    
    Args:
        points: (N, 2) array of Frenet [s, n] coordinates
        dimension: 0 for s (longitudinal), 1 for n (lateral)
        roughness: tolerance for equality
    
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


def check_pdp_match_frenet(
    original_cartesian: np.ndarray,
    generated_cartesian: np.ndarray,
    centerline: np.ndarray,
    roughness_s: float = 0.0,
    roughness_n: float = 0.0
) -> Tuple[bool, bool]:
    """
    Check PDP match using Frenet coordinates.
    
    This is the key function for curved road analysis.
    
    Args:
        original_cartesian: Original points in (x, y)
        generated_cartesian: Generated points in (x, y)
        centerline: Road centerline for Frenet transformation
        roughness_s: Roughness tolerance along road direction
        roughness_n: Roughness tolerance perpendicular to road
    
    Returns:
        (s_match, n_match): Whether orderings match in s and n dimensions
    """
    frame = FrenetFrame(centerline)
    
    # Convert to Frenet coordinates
    orig_frenet = frame.to_frenet(original_cartesian)
    gen_frenet = frame.to_frenet(generated_cartesian)
    
    # Compute inequality matrices
    orig_s_matrix = compute_inequality_matrix_frenet(orig_frenet, 0, roughness_s)
    orig_n_matrix = compute_inequality_matrix_frenet(orig_frenet, 1, roughness_n)
    gen_s_matrix = compute_inequality_matrix_frenet(gen_frenet, 0, roughness_s)
    gen_n_matrix = compute_inequality_matrix_frenet(gen_frenet, 1, roughness_n)
    
    # Compare matrices
    s_match = np.array_equal(orig_s_matrix, gen_s_matrix)
    n_match = np.array_equal(orig_n_matrix, gen_n_matrix)
    
    return s_match, n_match
