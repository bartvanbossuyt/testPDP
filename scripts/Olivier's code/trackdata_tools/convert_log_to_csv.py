"""
Convert tracking log/JSON files to structured CSV files.
Each object tracking record is extracted and flattened into a unique row.
Supports batch processing of multiple files in subfolders.
"""
import json
import csv
import re
from pathlib import Path
import argparse
import os

def parse_log_file(log_path):
    """Parse the log file containing JSON data (array or concatenated objects)."""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.strip()
    
    # Try parsing as a JSON array first (new format)
    if content.startswith('['):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    
    # Fallback: The file contains multiple JSON objects concatenated without separators
    # We need to split them by finding the pattern }{ or }{
    json_objects = []
    
    # Add proper separators between JSON objects
    content_fixed = re.sub(r'\}\s*\{', '}|||{', content)
    json_strings = content_fixed.split('|||')
    
    for json_str in json_strings:
        json_str = json_str.strip()
        if json_str:
            try:
                obj = json.loads(json_str)
                json_objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON: {e}")
                continue
    
    return json_objects

def flatten_object_data(record):
    """Flatten a single tracking record into rows, one per object detected."""
    rows = []
    
    # Common fields for the record (handle both old and new field names)
    record_time = record.get('time', '') or record.get('dateTime', '')
    record_type = record.get('type', '') or record.get('bodyType', '')
    camera_type = record.get('cameraType', '')
    data_number = record.get('dataNumber', '')
    interval_time = record.get('intervalTime', '')
    message_type = record.get('messageType', '')
    num_objects = record.get('numObjects', '')
    utc = record.get('utc', '')
    milli_seconds = record.get('milliSeconds', '')
    
    # Handle both 'object' (old format) and 'objects' (new format)
    objects = record.get('object', []) or record.get('objects', [])
    
    for obj in objects:
        # Build timestamp from utc + milliseconds if available
        obj_utc = obj.get('utc', '')
        obj_ms = obj.get('milliSeconds', '')
        obj_time = obj.get('time', '')
        if not obj_time and obj_utc:
            obj_time = f"{obj_utc}.{obj_ms}" if obj_ms else obj_utc
        
        row = {
            # Record-level fields
            'record_time': record_time,
            'record_type': record_type,
            'camera_type': camera_type,
            'data_number': data_number,
            'interval_time': interval_time,
            'message_type': message_type,
            'num_objects': num_objects,
            
            # Object-level fields
            'id': obj.get('id', ''),
            'external_id': obj.get('externalId', ''),
            'class_id': obj.get('classId', ''),
            'class_confidence': obj.get('classConfidence', ''),
            'object_confidence': obj.get('objectConfidence', ''),
            'object_time': obj_time,
            'life_time': obj.get('lifeTime', '') or obj.get('life_time', ''),
            'speed': obj.get('speed', ''),
            'speed_x_confidence': obj.get('speedXConfidence', ''),
            'speed_y_confidence': obj.get('speedYConfidence', ''),
        }
        
        # GPS Coordinates (take first if available)
        gps_coords = obj.get('gpsCoordinates', [{}])
        if gps_coords:
            gps = gps_coords[0]
            row['gps_latitude'] = gps.get('latitude', '')
            row['gps_longitude'] = gps.get('longitude', '')
            row['gps_heading'] = gps.get('heading', '')
        
        # Image Coordinates (take first if available)
        img_coords = obj.get('imageCoordinates', [{}])
        if img_coords:
            img = img_coords[0]
            row['image_x'] = img.get('x', '')
            row['image_y'] = img.get('y', '')
        
        # World Coordinates (take first if available)
        world_coords = obj.get('worldCoordinates', [{}])
        if world_coords:
            world = world_coords[0]
            row['world_x'] = world.get('x', '')
            row['world_y'] = world.get('y', '')
            row['world_z'] = world.get('z', '')
            row['world_heading_3d'] = world.get('heading3d', '')
            row['world_heading_3d_confidence'] = world.get('heading3dConfidence', '')
            row['world_height'] = world.get('height', '')
            row['world_length'] = world.get('length', '')
            row['world_width'] = world.get('width', '')
            row['world_x_confidence'] = world.get('xConfidence', '')
            row['world_y_confidence'] = world.get('yConfidence', '')
        
        rows.append(row)
    
    return rows

def convert_single_file(input_path, output_path):
    """Convert a single JSON/log file to CSV."""
    print(f"\nProcessing: {input_path}")
    
    records = parse_log_file(input_path)
    print(f"  Found {len(records)} records")
    
    if not records:
        print(f"  Skipping - no valid records found")
        return False
    
    # Flatten all records into rows
    all_rows = []
    for record in records:
        rows = flatten_object_data(record)
        all_rows.extend(rows)
    
    print(f"  Total object tracking rows: {len(all_rows)}")
    
    # Remove duplicates based on key fields (id, object_time)
    seen = set()
    unique_rows = []
    for row in all_rows:
        key = (row['id'], row['object_time'], row['external_id'])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    
    print(f"  Unique rows after deduplication: {len(unique_rows)}")
    
    if not unique_rows:
        print(f"  Skipping - no data rows to write")
        return False
    
    # Sort by object_time, then by id
    unique_rows.sort(key=lambda x: (x['object_time'], x['id']))
    
    # Define column order for better readability
    fieldnames = [
        'id', 'external_id', 'class_id', 'object_time', 'life_time',
        'speed', 'class_confidence', 'object_confidence',
        'gps_latitude', 'gps_longitude', 'gps_heading',
        'world_x', 'world_y', 'world_z', 'world_heading_3d',
        'world_height', 'world_length', 'world_width',
        'image_x', 'image_y',
        'speed_x_confidence', 'speed_y_confidence',
        'world_x_confidence', 'world_y_confidence', 'world_heading_3d_confidence',
        'record_time', 'data_number', 'camera_type', 'message_type', 'record_type',
        'interval_time', 'num_objects'
    ]
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    print(f"  Output: {output_path} ({len(unique_rows)} rows)")
    return True

def process_folder(input_folder, output_folder):
    """
    Process all JSON/log files in input_folder (recursively) and 
    output CSVs to output_folder preserving subfolder structure.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Find all .json and .log files recursively
    json_files = list(input_folder.rglob("*.json")) + list(input_folder.rglob("*.log"))
    
    if not json_files:
        print(f"No .json or .log files found in {input_folder}")
        return
    
    print(f"Found {len(json_files)} JSON/log files to process")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for json_file in json_files:
        # Calculate relative path to preserve folder structure
        relative_path = json_file.relative_to(input_folder)
        
        # Create output path with .csv extension
        output_path = output_folder / relative_path.with_suffix('.csv')
        
        try:
            if convert_single_file(json_file, output_path):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {success_count} files converted, {fail_count} failed/skipped")
    print(f"Output folder: {output_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON/log tracking files to CSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a folder (recursive) with output mirroring structure:
  python convert_log_to_csv.py --input "C:\\path\\to\\TEST" --output "C:\\path\\to\\TEST_csv"
  
  # Process a single file:
  python convert_log_to_csv.py --input "C:\\path\\to\\file.json" --output "C:\\path\\to\\file.csv"
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input folder (processes all .json/.log files recursively) or single file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output folder (mirrors input structure) or single output file'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    if input_path.is_file():
        # Single file mode
        convert_single_file(input_path, output_path)
    else:
        # Folder mode
        process_folder(input_path, output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
