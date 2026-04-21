# -*- coding: utf-8 -*-
"""
Lane geometry utilities for computing road centerlines, lane boundaries,
and vehicle direction analysis.

All functions are pure or accept a DataFrame parameter instead of reading globals.
"""

from typing import Any, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from .config import LANE_CONFIGURATIONS
from .drawing import remove_duplicate_points, extract_longest_object_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def safe_normalize(direction: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize a direction vector, returning a fallback for zero-length or NaN vectors."""
    if fallback is None:
        fallback = np.array([1.0, 0.0])
    norm = np.linalg.norm(direction)
    if np.isfinite(norm) and norm > 1e-6:
        return direction / norm
    return fallback


def calculate_vehicle_speeds(config_df: pd.DataFrame) -> dict[int, float]:
    """
    Calculate average speed for each vehicle in km/h.
    Assumes: x,y in meters, t in seconds or deciseconds.
    Returns dict {obj_id: speed_kmh}.
    """
    speeds: dict[int, float] = {}
    for obj_id in config_df['o'].unique():
        obj_df: pd.DataFrame = config_df[config_df['o'] == obj_id].sort_values('t')  # type: ignore[assignment]
        if len(obj_df) < 2:
            speeds[obj_id] = 0.0
            continue

        positions: np.ndarray = obj_df[['x', 'y']].to_numpy()  # type: ignore[assignment]
        times: np.ndarray = obj_df['t'].to_numpy()  # type: ignore[assignment]

        distances: np.ndarray = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        time_diffs: np.ndarray = np.diff(times)

        speeds_ms = distances / time_diffs
        avg_speed_ms = np.mean(speeds_ms) if len(speeds_ms) > 0 else 0.0
        avg_speed_kmh = avg_speed_ms * 3.6
        speeds[obj_id] = avg_speed_kmh

        logger.debug(f"[SPEED] Object {obj_id}: {avg_speed_kmh:.1f} km/h")

    return speeds


def determine_driving_direction(
    config_df: pd.DataFrame, obj_id: int | None = None
) -> np.ndarray:
    """
    Determine the main driving direction based on movement from timestamp 0.
    If *obj_id* is provided, calculate direction for that specific object.
    Returns a unit vector representing the driving direction.
    """
    _default = np.array([1.0, 0.0])

    if obj_id is not None:
        obj_df = config_df[config_df['o'] == obj_id].sort_values('t')
        if len(obj_df) < 2:
            return _default
        p0 = obj_df.iloc[0][['x', 'y']].to_numpy()
        p1 = obj_df.iloc[-1][['x', 'y']].to_numpy()
        return safe_normalize(p1 - p0, _default)

    t_values = sorted(config_df['t'].unique())
    if len(t_values) < 2:
        return _default

    t0_df = config_df[config_df['t'] == t_values[0]]
    t1_df = config_df[config_df['t'] == t_values[1]]

    p0 = np.array([t0_df['x'].mean(), t0_df['y'].mean()])
    p1 = np.array([t1_df['x'].mean(), t1_df['y'].mean()])

    return safe_normalize(p1 - p0, _default)


def vehicles_same_direction(
    config_df: pd.DataFrame, angle_threshold: float = 45.0
) -> bool:
    """
    Determine if vehicles are traveling in roughly the same direction.
    Returns True if angle between vehicle directions < *angle_threshold* degrees.
    """
    object_ids = sorted(config_df['o'].unique())
    if len(object_ids) < 2:
        return True

    directions: list[np.ndarray] = []
    for oid in object_ids:
        direction = determine_driving_direction(config_df, oid)
        directions.append(direction)  # type: ignore[arg-type]
        logger.debug(f"[DIRECTION] Object {oid}: direction={direction}")

    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            dot_product = np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(dot_product)))
            logger.debug(
                f"[DIRECTION] Angle between obj {object_ids[i]} and obj {object_ids[j]}: {angle_deg:.1f}°"
            )
            if angle_deg > angle_threshold:
                return False

    return True


def offset_polyline(points: np.ndarray, offset: float) -> np.ndarray:
    """Offset a polyline perpendicular to its tangent by *offset* units."""
    if points.shape[0] < 2:
        return points.copy()

    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]

    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1.0
    normalized = tangents / norms[:, np.newaxis]
    normals = np.column_stack((-normalized[:, 1], normalized[:, 0]))

    return points + offset * normals


def lane_polylines_bounds(
    lane_polylines: dict[str, Any] | None,
) -> tuple[float, float, float, float] | None:
    """Return (xmin, xmax, ymin, ymax) bounding box of all lane polylines."""
    if not lane_polylines:
        return None

    arrays: list[np.ndarray] = []
    boundaries = lane_polylines.get("boundaries")
    if boundaries:
        arrays.extend(boundaries)  # type: ignore[arg-type]

    centerline = lane_polylines.get("centerline")
    if centerline is not None and centerline.size:
        arrays.append(centerline)  # type: ignore[arg-type]

    if not arrays:
        return None

    stacked = np.vstack(arrays)
    return (
        float(np.min(stacked[:, 0])),
        float(np.max(stacked[:, 0])),
        float(np.min(stacked[:, 1])),
        float(np.max(stacked[:, 1])),
    )


# ---------------------------------------------------------------------------
# Functions that need a DataFrame (formerly read the global ``_df_all``)
# ---------------------------------------------------------------------------

def extract_centerline_from_data(
    c_value: int,
    df_all: pd.DataFrame | None,
    lane_configurations: dict[int, dict[str, Any]] | None = None,
) -> np.ndarray | None:
    """Extract a centerline from data for configuration *c_value*.

    Parameters
    ----------
    c_value : int
        Configuration id.
    df_all : DataFrame or None
        Full dataset (columns ``c, t, o, x, y``).
    lane_configurations : dict, optional
        Override for ``LANE_CONFIGURATIONS``. Uses the package default when *None*.
    """
    if df_all is None:
        return None
    if lane_configurations is None:
        lane_configurations = LANE_CONFIGURATIONS  # type: ignore[assignment]

    config_df = df_all[df_all["c"] == c_value]
    if config_df.empty:
        return None

    lane_cfg: dict[str, Any] = lane_configurations.get(c_value, {})  # type: ignore[union-attr]
    force_horizontal = bool(lane_cfg.get("force_horizontal", False))

    speeds = calculate_vehicle_speeds(config_df)
    driving_direction = determine_driving_direction(config_df)

    slowest_vehicle = None
    if speeds:
        slowest_vehicle = min(speeds.items(), key=lambda x: x[1])[0]

        if c_value in [15]:
            slowest_df = config_df[config_df['o'] == slowest_vehicle].sort_values('t')
            right_lane_path = slowest_df[['x', 'y']].to_numpy(dtype=float)
            right_lane_path = remove_duplicate_points(right_lane_path)
            if right_lane_path.shape[0] >= 2:
                return right_lane_path

    center_samples: list[tuple[float, float, float]] = []
    for t_val, group in config_df.groupby("t"):  # type: ignore[attr-defined]
        center_samples.append(
            (float(t_val), float(group["x"].mean()), float(group["y"].mean()))
        )
    center_samples.sort(key=lambda item: item[0])

    if center_samples:
        centerline = np.array([[row[1], row[2]] for row in center_samples], dtype=float)
        centerline = remove_duplicate_points(centerline)
    else:
        centerline = np.empty((0, 2), dtype=float)

    if centerline.shape[0] < 2:
        return extract_longest_object_path(config_df)

    # Adjust centerline so the slowest vehicle is on the RIGHT lane
    if slowest_vehicle is not None and len(config_df['o'].unique()) > 1:
        slowest_df = config_df[config_df['o'] == slowest_vehicle]
        slowest_positions = slowest_df[['x', 'y']].to_numpy(dtype=float)
        slowest_avg = np.mean(slowest_positions, axis=0)
        centerline_mid = np.mean(centerline, axis=0)
        to_slowest = slowest_avg - centerline_mid
        right_direction = np.array([driving_direction[1], -driving_direction[0]])
        side = np.dot(to_slowest, right_direction)

        if side < 0:
            shift_distance = np.linalg.norm(to_slowest)
            shift_vector = -to_slowest / np.linalg.norm(to_slowest) * shift_distance
            centerline = centerline + shift_vector

    # Force horizontal if configured
    lane_cfg = lane_configurations.get(c_value, {})  # type: ignore[union-attr]
    force_horizontal = lane_cfg.get("force_horizontal", False)

    if force_horizontal and centerline.shape[0] >= 2:
        forced_centerline_y = lane_cfg.get("centerline_y")
        avg_y = float(forced_centerline_y) if forced_centerline_y is not None else float(np.mean(centerline[:, 1]))
        x_min = np.min(centerline[:, 0])
        x_max = np.max(centerline[:, 0])
        centerline = np.array([[x_min, avg_y], [x_max, avg_y]])
        logger.debug(f"[CENTERLINE] Config {c_value}: Forced horizontal lanes at y={avg_y:.2f}")
        return centerline

    # Simplify roughly-straight paths to two points
    if c_value not in [15] and centerline.shape[0] >= 3:
        p_start = centerline[0]
        p_end = centerline[-1]
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            unit_vec = vec / norm
            vecs = centerline - p_start
            cross_products = vecs[:, 0] * unit_vec[1] - vecs[:, 1] * unit_vec[0]
            max_deviation = np.max(np.abs(cross_products))
            if max_deviation < 5.0:
                centerline = np.array([p_start, p_end])

    return centerline


def build_lane_polylines(
    c_value: int,
    lane_width: float,
    lane_count: int,
    df_all: pd.DataFrame | None,
    lane_configurations: dict[int, dict[str, Any]] | None = None,
    xlim: Optional[Tuple[float, float]] = None,
    config_offset: float = 0.0,
) -> dict[str, Any] | None:
    """Build lane polylines for configuration *c_value*.

    This is the extracted version of ``_build_lane_polylines_from_data``.
    The *df_all* parameter replaces the former global ``_df_all``.

    Returns a dict with keys ``boundaries``, ``center_lines``, ``centerline``
    (and optionally ``multi_path``), or *None* on failure.
    """
    if lane_count < 1:
        return None
    if lane_configurations is None:
        lane_configurations = LANE_CONFIGURATIONS  # type: ignore[assignment]
    if df_all is None:
        return None

    config_df = df_all[df_all["c"] == c_value]
    if config_df.empty:
        return None

    lane_cfg: dict[str, Any] = lane_configurations.get(c_value, {})  # type: ignore[union-attr]
    force_horizontal = bool(lane_cfg.get("force_horizontal", False))

    speeds = calculate_vehicle_speeds(config_df)
    max_speed = max(speeds.values()) if speeds else 0.0

    object_ids = sorted(config_df['o'].unique())
    vehicle_y_positions: dict[int, float] = {}
    for oid in object_ids:
        obj_df: pd.DataFrame = config_df[config_df['o'] == oid]  # type: ignore[assignment]
        vehicle_y_positions[oid] = float(obj_df['y'].mean())  # type: ignore[arg-type]

    y_span = 0.0
    if vehicle_y_positions:
        y_vals = list(vehicle_y_positions.values())
        y_span = max(y_vals) - min(y_vals)

    needs_3_lanes = (max_speed > 100.0) or (y_span > 1.5 * lane_width)
    lane_count = 3 if needs_3_lanes else 2
    logger.debug(
        f"[LANE BUILD] Config {c_value}: max_speed={max_speed:.1f} km/h, "
        f"y_span={y_span:.2f}m -> {lane_count} lanes"
    )

    same_dir = vehicles_same_direction(config_df)
    logger.debug(f"[LANE BUILD] Config {c_value}: same_direction={same_dir}")

    if same_dir:
        centerline = extract_centerline_from_data(c_value, df_all, lane_configurations)
        if centerline is None or centerline.shape[0] < 2:
            return None

        # Extend centerline to xlim
        if xlim is not None:
            p1 = centerline[0]
            p2 = centerline[-1]

            if centerline.shape[0] > 2:
                start_tangent = centerline[1] - centerline[0]
                start_norm = np.linalg.norm(start_tangent)
                start_unit = start_tangent / start_norm if start_norm > 1e-6 else np.array([1.0, 0.0])

                end_tangent = centerline[-1] - centerline[-2]
                end_norm = np.linalg.norm(end_tangent)
                end_unit = end_tangent / end_norm if end_norm > 1e-6 else np.array([1.0, 0.0])

                if abs(start_unit[0]) > 1e-6 and xlim[0] < p1[0] - 0.1:
                    t_start = (xlim[0] - p1[0]) / start_unit[0]
                    new_start = p1 + t_start * start_unit
                    centerline = np.vstack([[new_start], centerline])

                if abs(end_unit[0]) > 1e-6 and xlim[1] > p2[0] + 0.1:
                    t_end = (xlim[1] - p2[0]) / end_unit[0]
                    new_end = p2 + t_end * end_unit
                    centerline = np.vstack([centerline, [new_end]])
            else:
                direction = p2 - p1
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    unit_dir = direction / norm
                    if abs(unit_dir[0]) > 1e-6:
                        t_start = (xlim[0] - p1[0]) / unit_dir[0]
                        t_end = (xlim[1] - p1[0]) / unit_dir[0]
                        if t_start > t_end:
                            t_start, t_end = t_end, t_start
                        new_start = p1 + t_start * unit_dir
                        new_end = p1 + t_end * unit_dir
                        centerline = np.array([new_start, new_end])

        half_width = (lane_width * lane_count) / 2.0
        forced_centerline_y = lane_cfg.get("centerline_y")
        lock_centerline = bool(force_horizontal and forced_centerline_y is not None)

        if lock_centerline:
            offset = 0.0
            logger.debug(
                f"[LANE BUILD] Config {c_value}: centerline locked at "
                f"y={float(forced_centerline_y):.2f} (auto-offset disabled)"
            )
        elif len(speeds) == 1:
            single_obj = list(speeds.keys())[0]
            single_df = config_df[config_df['o'] == single_obj]
            avg_y_vehicle = float(single_df['y'].mean())
            target_ref_y = avg_y_vehicle
            current_ref_y = float(np.mean(centerline[:, 1]))
            offset = target_ref_y - current_ref_y
            logger.debug(
                f"[LANE BUILD] Single vehicle obj {single_obj}: "
                f"vehicle_y={avg_y_vehicle:.2f}, target_ref={target_ref_y:.2f}, offset={offset:.2f}m"
            )
        elif len(speeds) > 1:
            avg_vehicle_y = sum(vehicle_y_positions.values()) / len(vehicle_y_positions)
            target_ref_y = avg_vehicle_y
            current_ref_y = float(np.mean(centerline[:, 1]))
            offset = target_ref_y - current_ref_y
            logger.debug(
                f"[LANE BUILD] Multi-vehicle: avg_vehicle_y={avg_vehicle_y:.2f}, "
                f"target_ref={target_ref_y:.2f}, offset={offset:.2f}m"
            )
        else:
            offset = 0.0

        centerline[:, 1] += offset
        logger.debug(
            f"[LANE BUILD] After offset (calculated={offset:.2f}m, "
            f"config_offset={config_offset:.2f}m ignored), "
            f"centerline y-range: [{np.min(centerline[:, 1]):.2f}, {np.max(centerline[:, 1]):.2f}]"
        )

        boundary_offsets = [-half_width + i * lane_width for i in range(lane_count + 1)]
        boundaries = [offset_polyline(centerline, off) for off in boundary_offsets]
        logger.debug(f"[LANE BUILD] Created {len(boundaries)} boundaries with offsets: {boundary_offsets}")
        if boundaries:
            for i, boundary in enumerate(boundaries):
                y_range = [float(np.min(boundary[:, 1])), float(np.max(boundary[:, 1]))]
                logger.debug(f"[LANE BUILD]   Boundary {i} y-range: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
        if centerline.shape[0] > 0:
            centerline_y_dbg = float(np.mean(centerline[:, 1]))
            logger.debug(
                f"[LANE BUILD] Lane y-positions -> lower edge: {centerline_y_dbg - half_width:.2f}, "
                f"dashed divider: {centerline_y_dbg:.2f}, upper edge: {centerline_y_dbg + half_width:.2f}"
            )

        interior_count = max(0, lane_count - 1)
        center_offsets = [-half_width + (i + 1) * lane_width for i in range(interior_count)]
        center_lines = [offset_polyline(centerline, off) for off in center_offsets]

        return {"boundaries": boundaries, "center_lines": center_lines, "centerline": centerline}
    else:
        # Different directions – separate road segments per vehicle
        all_boundaries: list[np.ndarray] = []

        for oid in object_ids:
            obj_df = config_df[config_df['o'] == oid].sort_values('t')
            if len(obj_df) < 2:
                continue

            vehicle_path: np.ndarray = obj_df[['x', 'y']].to_numpy(dtype=float)
            vehicle_path = remove_duplicate_points(vehicle_path)
            if vehicle_path.shape[0] < 2:
                continue

            half_width = (lane_width * lane_count) / 2.0
            rightmost_lane_center = -half_width + lane_width / 2.0
            road_offset = -rightmost_lane_center

            road_centerline = vehicle_path.copy()
            road_centerline[:, 1] += road_offset

            boundary_offsets = [-half_width + i * lane_width for i in range(lane_count + 1)]
            obj_boundaries = [offset_polyline(road_centerline, off) for off in boundary_offsets]
            all_boundaries.extend(obj_boundaries)

            logger.debug(
                f"[LANE BUILD] Case 2: obj {oid} – road offset={road_offset:.2f}m "
                f"to center vehicle in rightmost lane"
            )

        first_obj = object_ids[0]
        centerline_df = config_df[config_df['o'] == first_obj].sort_values('t')
        centerline_arr: np.ndarray = centerline_df[['x', 'y']].to_numpy(dtype=float)
        centerline_arr = remove_duplicate_points(centerline_arr)

        return {
            "boundaries": all_boundaries,
            "center_lines": [],
            "centerline": centerline_arr,
            "multi_path": True,
        }
