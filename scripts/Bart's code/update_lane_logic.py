"""
Script to add lane detection helper functions to inverse.py
"""

# Helper functions to add
helper_functions = '''
def _calculate_vehicle_speeds(config_df: pd.DataFrame) -> dict:
    """
    Calculate average speed for each vehicle based on distance traveled between timestamps.
    Lower values indicate slower vehicles.
    """
    speeds = {}
    for obj_id in config_df['o'].unique():
        obj_df = config_df[config_df['o'] == obj_id].sort_values('t')
        if len(obj_df) < 2:
            speeds[obj_id] = 0.0
            continue
        
        # Calculate distances between consecutive timestamps
        positions = obj_df[['x', 'y']].to_numpy()
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        avg_speed = np.mean(distances) if len(distances) > 0 else 0.0
        speeds[obj_id] = avg_speed
    
    return speeds

def _determine_driving_direction(config_df: pd.DataFrame) -> np.ndarray:
    """
    Determine the main driving direction based on movement from timestamp 0.
    Returns a unit vector representing the driving direction.
    """
    # Get positions at first two timestamps
    t_values = sorted(config_df['t'].unique())
    if len(t_values) < 2:
        return np.array([1.0, 0.0])  # Default to x-direction
    
    t0_df = config_df[config_df['t'] == t_values[0]]
    t1_df = config_df[config_df['t'] == t_values[1]]
    
    # Calculate center of mass for both timestamps
    p0 = np.array([t0_df['x'].mean(), t0_df['y'].mean()])
    p1 = np.array([t1_df['x'].mean(), t1_df['y'].mean()])
    
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        return direction / norm
    return np.array([1.0, 0.0])

'''

# New version of _extract_centerline_from_data
new_centerline_function = '''def _extract_centerline_from_data(c_value: int):
    if _df_all is None:
        return None
    config_df = _df_all[_df_all["c"] == c_value]
    if config_df.empty:
        return None

    # Calculate vehicle speeds to identify slowest vehicle (should be on right)
    speeds = _calculate_vehicle_speeds(config_df)
    
    # Determine driving direction from timestamp 0
    driving_direction = _determine_driving_direction(config_df)
    
    # Find slowest vehicle (should be on the right lane)
    if speeds:
        slowest_vehicle = min(speeds.items(), key=lambda x: x[1])[0]
        
        # For curved roads, use the slowest vehicle's path as reference
        if c_value in [15]:
            slowest_df = config_df[config_df['o'] == slowest_vehicle].sort_values('t')
            right_lane_path = slowest_df[['x', 'y']].to_numpy(dtype=float)
            right_lane_path = _remove_duplicate_points(right_lane_path)
            if right_lane_path.shape[0] >= 2:
                return right_lane_path
    
    # Calculate centerline as average between all vehicles at each timestamp
    center_samples: list[tuple[float, float, float]] = []
    for t_val, group in config_df.groupby("t"):
        center_samples.append((float(t_val), float(group["x"].mean()), float(group["y"].mean())))

    center_samples.sort(key=lambda item: item[0])

    if center_samples:
        centerline = np.array([[row[1], row[2]] for row in center_samples], dtype=float)
        centerline = _remove_duplicate_points(centerline)
    else:
        centerline = np.empty((0, 2), dtype=float)

    if centerline.shape[0] < 2:
        return _extract_longest_object_path(config_df)

    # Check if the path is roughly straight (skip for curved configs)
    # IMPORTANT: Always preserve the original slope angle from start to end point!
    if c_value not in [15] and centerline.shape[0] >= 3:
        p_start = centerline[0]
        p_end = centerline[-1]
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            # Calculate distance of all points to the line segment
            unit_vec = vec / norm
            vecs = centerline - p_start
            # 2D cross product: x1*y2 - x2*y1 gives signed distance * norm
            cross_products = vecs[:, 0] * unit_vec[1] - vecs[:, 1] * unit_vec[0]
            max_deviation = np.max(np.abs(cross_products))
            
            # If deviation is small (e.g. < 5.0m), simplify to straight line
            # Keep original start/end points to preserve the slope angle!
            if max_deviation < 5.0:
                centerline = np.array([p_start, p_end])
    elif centerline.shape[0] == 2:
        pass # Already straight

    return centerline
'''

print("Helper functions and new centerline function prepared.")
print("\nTo apply changes manually:")
print("1. Add helper functions before first _extract_centerline_from_data")
print("2. Replace both _extract_centerline_from_data definitions with the new version")
