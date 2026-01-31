# -*- coding: utf-8 -*-
"""
point_selection.py

Module voor punt-selectie en bewegingsvector generatie voor de PDP inverse analyse.
Geëxtraheerd uit inverse.py voor betere modulariteit.

Functies:
- get_object_info_for_flat_idx: Object info voor een flat index
- is_fixed_point: Check of een punt vast (extern) is
- get_movable_indices: Lijst van verplaatsbare punten
- select_points_for_iteration: Selecteer punten voor een iteratie
- generate_movement_vectors: Genereer bewegingsvectoren
- scale_movement_vectors: Schaal bewegingsvectoren
- apply_movement_vectors: Pas bewegingsvectoren toe
- get_timestamp_for_flat_idx: Timestamp voor een flat index
- get_point_for_flat_idx: Coördinaten voor een flat index
"""

from typing import Any, TypedDict
import numpy as np
import streamlit as st

# ============= Object Labels for Display =============
OBJECT_LABELS = ["k", "l", "m", "n", "p", "q", "r", "s", "u", "v"]


# Type definition for successful point data in the search process
class SuccessfulPoint(TypedDict):
    point: np.ndarray              # Coordinates of the generated point
    parent_idx: int                # Index in all_pts (may be a generated point)
    parent_point: np.ndarray       # Actual coordinates of the parent point
    original_parent_idx: int       # Index of the ORIGINAL point (k0, k1, k2, l0, l1, l2)
    iteration: int                 # Iteration number when this point was accepted


# ============= Module-level references to global state =============
# These will be set by init_point_selection() from the main app
_state: dict[str, Any] = {
    "n_total_points": 0,
    "all_obj_ids_flat": np.array([]),
    "all_local_idx_flat": np.array([]),
    "all_is_fixed_flat": np.array([]),
    "all_pts_flat": np.array([]).reshape(0, 2),
    "all_ts_flat": np.array([]),
    "all_points_plot": {},
    "XLIM": (-50.0, 150.0),
    "YLIM": (-50.0, 150.0),
    "COORD_MIN_X": -50.0,
    "COORD_MAX_X": 150.0,
    "COORD_MIN_Y": -50.0,
    "COORD_MAX_Y": 150.0,
}


def init_point_selection(
    n_total_points: int,
    all_obj_ids_flat: np.ndarray,
    all_local_idx_flat: np.ndarray,
    all_is_fixed_flat: np.ndarray,
    all_pts_flat: np.ndarray,
    all_ts_flat: np.ndarray,
    all_points_plot: dict[int, np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    coord_min_x: float,
    coord_max_x: float,
    coord_min_y: float,
    coord_max_y: float,
) -> None:
    """
    Initialize the module with references to global state from inverse.py.
    Call this once after loading data but before using any other functions.
    """
    _state["n_total_points"] = n_total_points
    _state["all_obj_ids_flat"] = all_obj_ids_flat
    _state["all_local_idx_flat"] = all_local_idx_flat
    _state["all_is_fixed_flat"] = all_is_fixed_flat
    _state["all_pts_flat"] = all_pts_flat
    _state["all_ts_flat"] = all_ts_flat
    _state["all_points_plot"] = all_points_plot
    _state["XLIM"] = xlim
    _state["YLIM"] = ylim
    _state["COORD_MIN_X"] = coord_min_x
    _state["COORD_MAX_X"] = coord_max_x
    _state["COORD_MIN_Y"] = coord_min_y
    _state["COORD_MAX_Y"] = coord_max_y


def get_object_info_for_flat_idx(flat_idx: int) -> tuple[int, int, str]:
    """
    Get object ID, local index, and label for a flat index.
    Returns: (object_id, local_idx_in_object, label_character)
    For external points, object_id is -1 and label is "ext"
    """
    n_total_points: int = _state["n_total_points"]  # type: ignore[assignment]
    all_obj_ids_flat: np.ndarray = _state["all_obj_ids_flat"]  # type: ignore[assignment]
    all_local_idx_flat: np.ndarray = _state["all_local_idx_flat"]  # type: ignore[assignment]
    all_points_plot: dict[int, np.ndarray] = _state["all_points_plot"]  # type: ignore[assignment]
    
    if 0 <= flat_idx < n_total_points:
        o_id = all_obj_ids_flat[flat_idx]
        local_idx = all_local_idx_flat[flat_idx]
        if o_id == -1:
            # External point
            return -1, local_idx, "ext"
        # Find which position this object is in (for label lookup)
        sorted_obj_ids = sorted(all_points_plot.keys())
        obj_position = sorted_obj_ids.index(o_id) if o_id in sorted_obj_ids else 0
        label = OBJECT_LABELS[obj_position % len(OBJECT_LABELS)]
        return o_id, local_idx, label
    return 0, 0, "k"


def is_fixed_point(flat_idx: int) -> bool:
    """Check if a flat index refers to a fixed (external) point."""
    n_total_points: int = _state["n_total_points"]  # type: ignore[assignment]
    all_is_fixed_flat: np.ndarray = _state["all_is_fixed_flat"]  # type: ignore[assignment]
    
    if 0 <= flat_idx < n_total_points:
        return all_is_fixed_flat[flat_idx]
    return False


def get_movable_indices() -> list[int]:
    """Get list of flat indices for movable (non-fixed) points only."""
    n_total_points: int = _state["n_total_points"]  # type: ignore[assignment]
    return [i for i in range(n_total_points) if not is_fixed_point(i)]


def select_points_for_iteration() -> list[int]:
    """
    Select points to move in this iteration based on the point selection mode.
    Returns a list of flat indices of points to move together.
    """
    all_obj_ids_flat: np.ndarray = _state["all_obj_ids_flat"]  # type: ignore[assignment]
    all_ts_flat: np.ndarray = _state["all_ts_flat"]  # type: ignore[assignment]
    
    movable_indices = get_movable_indices()
    if not movable_indices:
        return []
    
    point_selection_mode = st.session_state.get("cfg_point_selection_mode", "Single point")
    
    if point_selection_mode == "Single point":
        # Current default behavior: select one random point
        return [int(np.random.choice(movable_indices))]
    
    elif point_selection_mode == "Multiple random points":
        # Select N random points
        num_points = int(st.session_state.get("cfg_num_random_points", 2))
        num_points = min(num_points, len(movable_indices))  # Can't select more than available
        selected = list(np.random.choice(movable_indices, size=num_points, replace=False))
        return [int(idx) for idx in selected]
    
    elif point_selection_mode == "Consecutive time stamps":
        # Select consecutive timestamps from a single user-chosen object
        selected_object_id = int(st.session_state.get("cfg_consecutive_object_id", 0))
        num_timestamps = int(st.session_state.get("cfg_group_num_timestamps", 2))
        first_timestamp_idx = int(st.session_state.get("cfg_consecutive_first_timestamp", 0))
        
        # Get indices for the selected object, sorted by timestamp
        indices_for_object: list[tuple[int, float]] = []  # (flat_idx, timestamp)
        for flat_idx in movable_indices:
            o_id = all_obj_ids_flat[flat_idx]
            if o_id == selected_object_id:
                t = all_ts_flat[flat_idx]
                indices_for_object.append((flat_idx, t))
        
        # Sort by timestamp
        indices_for_object.sort(key=lambda x: x[1])
        
        if not indices_for_object:
            # Fall back to single point if no points for this object
            return [int(np.random.choice(movable_indices))]
        
        # Clamp first_timestamp_idx to valid range
        max_start = max(0, len(indices_for_object) - num_timestamps)
        first_timestamp_idx = min(first_timestamp_idx, max_start)
        
        # Select consecutive points starting from first_timestamp_idx
        selected_indices = []
        for i in range(num_timestamps):
            if first_timestamp_idx + i < len(indices_for_object):
                selected_indices.append(indices_for_object[first_timestamp_idx + i][0])
        
        return [int(idx) for idx in selected_indices] if selected_indices else [int(np.random.choice(movable_indices))]
    
    # Default fallback
    return [int(np.random.choice(movable_indices))]


def generate_movement_vectors(selected_indices: list[int], base_distance: float) -> dict[int, tuple[float, float]]:
    """
    Generate movement vectors for selected points based on movement direction mode.
    Returns a dict mapping flat_idx -> (delta_x, delta_y)
    
    For multi-point mode, ensures that the chosen direction keeps ALL points within bounds.
    If after max_attempts no valid direction is found, uses the best direction found.
    """
    if not selected_indices:
        return {}
    
    all_pts_flat: np.ndarray = _state["all_pts_flat"]  # type: ignore[assignment]
    XLIM: tuple[float, float] = _state["XLIM"]  # type: ignore[assignment]
    YLIM: tuple[float, float] = _state["YLIM"]  # type: ignore[assignment]
    
    movement_direction = st.session_state.get("cfg_movement_direction", "Same direction")
    
    # Get visualization bounds (XLIM, YLIM) to keep points within the graph
    try:
        coord_min_x, coord_max_x = XLIM
        coord_min_y, coord_max_y = YLIM
    except (TypeError, ValueError):
        # Fallback to session state bounds if XLIM/YLIM not yet computed
        coord_min_x = float(st.session_state.get("coord_min_x", -50.0))
        coord_max_x = float(st.session_state.get("coord_max_x", 150.0))
        coord_min_y = float(st.session_state.get("coord_min_y", -50.0))
        coord_max_y = float(st.session_state.get("coord_max_y", 150.0))
    
    def point_in_bounds(x: float, y: float) -> bool:
        """Check if a point is within visualization bounds."""
        return coord_min_x <= x <= coord_max_x and coord_min_y <= y <= coord_max_y
    
    def get_parent_point(idx: int) -> np.ndarray:
        """Get the current parent point position for a given index."""
        # Check successful_points first for updated parent position
        successful_points: list[SuccessfulPoint] = st.session_state.get("anim_successful_points", [])
        for s in reversed(successful_points):
            if int(s.get("original_parent_idx", -1)) == idx:
                return np.array(s["point"])
        # Fall back to original position
        if 0 <= idx < len(all_pts_flat):
            return all_pts_flat[idx]
        return np.array([0.0, 0.0])
    
    if movement_direction == "Same direction":
        # All points move with the same angle - find a direction that keeps ALL points in bounds
        max_attempts = 50
        best_angle = None
        best_angle: float | None = None
        best_in_bounds_count = 0
        
        for _ in range(max_attempts):
            angle = float(np.random.uniform(0, 2 * np.pi))
            delta_x = base_distance * np.cos(angle)
            delta_y = base_distance * np.sin(angle)
            
            # Check if all points would be in bounds with this direction
            in_bounds_count = 0
            all_in_bounds = True
            for idx in selected_indices:
                parent_pt = get_parent_point(idx)
                new_x = parent_pt[0] + delta_x
                new_y = parent_pt[1] + delta_y
                if point_in_bounds(new_x, new_y):
                    in_bounds_count += 1
                else:
                    all_in_bounds = False
            
            # Track best attempt
            if in_bounds_count > best_in_bounds_count:
                best_in_bounds_count = in_bounds_count
                best_angle = angle
            
            if all_in_bounds:
                # Found a valid direction
                return {int(idx): (delta_x, delta_y) for idx in selected_indices}
        
        # Use best direction found (may not keep all points in bounds, but maximizes in-bounds count)
        if best_angle is not None:
            delta_x = base_distance * np.cos(best_angle)
            delta_y = base_distance * np.sin(best_angle)
            return {int(idx): (delta_x, delta_y) for idx in selected_indices}
        
        # Fallback: use first random angle
        angle = float(np.random.uniform(0, 2 * np.pi))
        delta_x = base_distance * np.cos(angle)
        delta_y = base_distance * np.sin(angle)
        return {int(idx): (delta_x, delta_y) for idx in selected_indices}
    
    else:  # Random directions
        # Each point gets its own random angle - ensure each point stays in bounds
        vectors = {}
        vectors: dict[int, tuple[float, float]] = {}
        max_attempts = 50
        
        for idx in selected_indices:
            parent_pt = get_parent_point(idx)
            best_angle = None
            best_angle: float | None = None
            
            for _ in range(max_attempts):
                angle = float(np.random.uniform(0, 2 * np.pi))
                delta_x = base_distance * np.cos(angle)
                delta_y = base_distance * np.sin(angle)
                
                new_x = parent_pt[0] + delta_x
                new_y = parent_pt[1] + delta_y
                
                if point_in_bounds(new_x, new_y):
                    vectors[int(idx)] = (delta_x, delta_y)
                    break
                elif best_angle is None:
                    best_angle = angle
            else:
                # No valid angle found after max_attempts, use first angle tried
                if best_angle is not None:
                    delta_x = base_distance * np.cos(best_angle)
                    delta_y = base_distance * np.sin(best_angle)
                    vectors[int(idx)] = (delta_x, delta_y)
                else:
                    # Complete fallback
                    angle = float(np.random.uniform(0, 2 * np.pi))
                    delta_x = base_distance * np.cos(angle)
                    delta_y = base_distance * np.sin(angle)
                    vectors[int(idx)] = (delta_x, delta_y)
        
        return vectors


def scale_movement_vectors(vectors: dict[int, tuple[float, float]], scale: float) -> dict[int, tuple[float, float]]:
    """Scale all movement vectors by a factor (e.g., 0.5 to halve distances)."""
    return {int(idx): (dx * scale, dy * scale) for idx, (dx, dy) in vectors.items()}


def apply_movement_vectors(base_points: np.ndarray, vectors: dict[int, tuple[float, float]]) -> dict[int, np.ndarray]:
    """
    Apply movement vectors to base points and return new positions.
    Returns dict mapping flat_idx -> new_position (clipped to visualization bounds)
    """
    XLIM: tuple[float, float] = _state["XLIM"]  # type: ignore[assignment]
    YLIM: tuple[float, float] = _state["YLIM"]  # type: ignore[assignment]
    COORD_MIN_X: float = _state["COORD_MIN_X"]  # type: ignore[assignment]
    COORD_MAX_X: float = _state["COORD_MAX_X"]  # type: ignore[assignment]
    COORD_MIN_Y: float = _state["COORD_MIN_Y"]  # type: ignore[assignment]
    COORD_MAX_Y: float = _state["COORD_MAX_Y"]  # type: ignore[assignment]
    
    # Use visualization bounds (XLIM, YLIM) to keep points within the graph
    try:
        x_min, x_max = XLIM
        y_min, y_max = YLIM
    except (TypeError, ValueError):
        # Fallback to coordinate bounds if XLIM/YLIM not yet computed
        x_min, x_max = COORD_MIN_X, COORD_MAX_X
        y_min, y_max = COORD_MIN_Y, COORD_MAX_Y
    
    new_positions = {}
    for idx, (dx, dy) in vectors.items():
        if 0 <= idx < len(base_points):
            new_x = base_points[idx, 0] + dx
            new_y = base_points[idx, 1] + dy
            # Clip to visualization bounds to keep points within the graph
            new_x = np.clip(new_x, x_min, x_max)
            new_y = np.clip(new_y, y_min, y_max)
            new_positions[idx] = np.array([new_x, new_y])
    return new_positions


def get_timestamp_for_flat_idx(flat_idx: int) -> float:
    """Get the timestamp for a flat index."""
    n_total_points: int = _state["n_total_points"]  # type: ignore[assignment]
    all_ts_flat: np.ndarray = _state["all_ts_flat"]  # type: ignore[assignment]
    
    if 0 <= flat_idx < n_total_points:
        return float(all_ts_flat[flat_idx])
    return 0.0


def get_point_for_flat_idx(flat_idx: int) -> np.ndarray:
    """Get the point coordinates for a flat index."""
    n_total_points: int = _state["n_total_points"]  # type: ignore[assignment]
    all_pts_flat: np.ndarray = _state["all_pts_flat"]  # type: ignore[assignment]
    
    if 0 <= flat_idx < n_total_points:
        return all_pts_flat[flat_idx]
    return np.array([0.0, 0.0])
