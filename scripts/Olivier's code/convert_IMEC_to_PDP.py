"""
Convert IMEC tracking data to PDP format.

IMEC format (tracks_df_1.csv):
    ID, timestamp, x, y, vx, vy, ...

PDP format:
    conID, tstID, poiID, x, y [, class]
    - conID: configuration ID (0 for single scenario)
    - tstID: sequential timestep index (0, 1, 2, ...)
    - poiID: point/object ID (0, 1, 2, ...)
    - x, y: coordinates
    - class: optional object class
"""

import pandas as pd
import numpy as np
import os

def convert_imec_to_pdp(
    input_path: str,
    output_path: str,
    use_lidar_coords: bool = False,
    include_class: bool = False,
    con_id: int = 0
) -> pd.DataFrame:
    """
    Convert IMEC tracking CSV to PDP format.
    
    Parameters:
    -----------
    input_path : str
        Path to IMEC tracks_df_1.csv
    output_path : str
        Path for output PDP-formatted CSV
    use_lidar_coords : bool
        If True, use X_LiDAR/Y_LiDAR instead of filtered x/y
        (Warning: may have NaN values)
    include_class : bool
        If True, include class column (YOLO_cls)
    con_id : int
        Configuration ID (default 0)
    
    Returns:
    --------
    pd.DataFrame with PDP format
    """
    # Read IMEC data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from IMEC data")
    print(f"Unique tracks (IDs): {df['ID'].nunique()}")
    print(f"Timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Choose coordinate columns
    if use_lidar_coords:
        x_col, y_col = 'X_LiDAR', 'Y_LiDAR'
        # Filter out rows with NaN LiDAR coords
        df = df.dropna(subset=[x_col, y_col])
        print(f"After dropping NaN LiDAR coords: {len(df)} rows")
    else:
        x_col, y_col = 'x', 'y'
    
    # Create timestamp to tstID mapping (sequential indices)
    unique_timestamps = sorted(df['timestamp'].unique())
    timestamp_to_tstid = {ts: idx for idx, ts in enumerate(unique_timestamps)}
    
    # Create track ID to poiID mapping (sequential indices)
    unique_ids = sorted(df['ID'].unique())
    id_to_poiid = {track_id: idx for idx, track_id in enumerate(unique_ids)}
    
    # Build PDP dataframe
    pdp_data = {
        'conID': con_id,
        'tstID': df['timestamp'].map(timestamp_to_tstid),
        'poiID': df['ID'].map(id_to_poiid),
        'x': df[x_col],
        'y': df[y_col]
    }
    
    if include_class:
        pdp_data['class'] = df['YOLO_cls']
    
    pdp_df = pd.DataFrame(pdp_data)
    
    # Sort by conID, tstID, poiID for consistency
    pdp_df = pdp_df.sort_values(['conID', 'tstID', 'poiID']).reset_index(drop=True)
    
    # Save without header (PDP format expects no header)
    pdp_df.to_csv(output_path, index=False, header=False)
    
    print(f"\nConverted to PDP format:")
    print(f"  Configurations (con): {pdp_df['conID'].nunique()}")
    print(f"  Timesteps (tst): {pdp_df['tstID'].nunique()}")
    print(f"  Points/Objects (poi): {pdp_df['poiID'].nunique()}")
    print(f"  Total rows: {len(pdp_df)}")
    print(f"\nSaved to: {output_path}")
    
    # Also save mappings for reference
    mapping_path = output_path.replace('.csv', '_mappings.csv')
    mappings = pd.DataFrame({
        'original_timestamp': unique_timestamps,
        'tstID': range(len(unique_timestamps))
    })
    mappings.to_csv(mapping_path, index=False)
    print(f"Saved timestamp mappings to: {mapping_path}")
    
    return pdp_df


def convert_imec_interpolated(
    input_path: str,
    output_path: str,
    use_lidar_coords: bool = False,
    include_class: bool = False,
    con_id: int = 0
) -> pd.DataFrame:
    """
    Convert IMEC data with interpolation to fill gaps.
    
    PDP expects all objects present at all timesteps. This function
    interpolates missing positions for each track.
    """
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from IMEC data")
    
    x_col = 'X_LiDAR' if use_lidar_coords else 'x'
    y_col = 'Y_LiDAR' if use_lidar_coords else 'y'
    
    # Get unique timestamps and IDs
    unique_timestamps = sorted(df['timestamp'].unique())
    unique_ids = sorted(df['ID'].unique())
    
    timestamp_to_tstid = {ts: idx for idx, ts in enumerate(unique_timestamps)}
    id_to_poiid = {track_id: idx for idx, track_id in enumerate(unique_ids)}
    
    # Create complete grid
    rows = []
    for track_id in unique_ids:
        track_data = df[df['ID'] == track_id].copy()
        track_data = track_data.set_index('timestamp')
        
        # Reindex to all timestamps and interpolate
        track_data = track_data.reindex(unique_timestamps)
        track_data[x_col] = track_data[x_col].interpolate(method='linear')
        track_data[y_col] = track_data[y_col].interpolate(method='linear')
        
        # Forward/backward fill for edges
        track_data[x_col] = track_data[x_col].ffill().bfill()
        track_data[y_col] = track_data[y_col].ffill().bfill()
        
        if include_class:
            track_data['YOLO_cls'] = track_data['YOLO_cls'].ffill().bfill()
        
        for ts in unique_timestamps:
            row = {
                'conID': con_id,
                'tstID': timestamp_to_tstid[ts],
                'poiID': id_to_poiid[track_id],
                'x': track_data.loc[ts, x_col],
                'y': track_data.loc[ts, y_col]
            }
            if include_class:
                row['class'] = int(track_data.loc[ts, 'YOLO_cls']) if pd.notna(track_data.loc[ts, 'YOLO_cls']) else 0
            rows.append(row)
    
    pdp_df = pd.DataFrame(rows)
    pdp_df = pdp_df.sort_values(['conID', 'tstID', 'poiID']).reset_index(drop=True)
    
    # Check for any remaining NaN
    if pdp_df[['x', 'y']].isna().any().any():
        print("WARNING: Some NaN values remain after interpolation")
        pdp_df = pdp_df.dropna(subset=['x', 'y'])
    
    pdp_df.to_csv(output_path, index=False, header=False)
    
    print(f"\nConverted to PDP format (interpolated):")
    print(f"  Configurations (con): {pdp_df['conID'].nunique()}")
    print(f"  Timesteps (tst): {pdp_df['tstID'].nunique()}")
    print(f"  Points/Objects (poi): {pdp_df['poiID'].nunique()}")
    print(f"  Total rows: {len(pdp_df)}")
    print(f"\nSaved to: {output_path}")
    
    return pdp_df


if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(
        script_dir,
        "Data_IMEC", "tracked0001", "0001", "track_dataframes", "tracks_df_1.csv"
    )
    
    output_file = os.path.join(script_dir, "IMEC_PDP_format.csv")
    output_file_interp = os.path.join(script_dir, "IMEC_PDP_format_interpolated.csv")
    
    print("=" * 60)
    print("IMEC to PDP Converter")
    print("=" * 60)
    
    # Basic conversion (keeps only rows where data exists)
    print("\n--- Basic Conversion (sparse) ---")
    convert_imec_to_pdp(
        input_path=input_file,
        output_path=output_file,
        use_lidar_coords=False,  # Use filtered x,y from tracker
        include_class=True
    )
    
    # Interpolated conversion (fills gaps)
    print("\n--- Interpolated Conversion (dense) ---")
    convert_imec_interpolated(
        input_path=input_file,
        output_path=output_file_interp,
        use_lidar_coords=False,
        include_class=True
    )
