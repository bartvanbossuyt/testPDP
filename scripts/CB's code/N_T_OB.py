"""
MODULE: N_T_OB (Outward Buffer Transformation)
VERSION: 230209

PURPOSE:
    Transform original dataset by adding buffer points at specified distances

DESCRIPTION:
    Creates buffer zones around each tracking point by generating additional 
    synthetic points at configurable distances (buffer_x, buffer_y)
    
INPUT:
    - Original dataset: N_C_Dataset.csv
    - Buffer distance parameters: av.buffer_x, av.buffer_y
    
OUTPUT:
    - Buffered dataset: N_C_Dataset_g_buffer.csv
    - 5 points per original point (1 original + 4 buffer points)
    
PARAMETERS:
    - buffer_x: Horizontal distance for buffer points
    - buffer_y: Vertical distance for buffer points
"""

# Import modules
import av  # Global variables and configuration
import csv  # CSV file reading/writing
import os  # File path operations
import time  # Performance timing

# Start time measurement
t_start = time.time()

# Set up output directory for this module
output_dir = av.get_output_dir('OB')

# Load original dataset from root output directory
dataset_path = os.path.join(av.output_base_path, 'N_C_Dataset.csv')

# Read and process dataset
with open(dataset_path, 'r') as csv_file:
  csv_reader = csv.reader(csv_file)
  
  # Create buffered points: for each original point, generate 4 buffer points
  # Structure: original at 5*i, buffers at 5*i+1 to 5*i+4
  # Buffer points created in order: left (-buffer_x), right (+buffer_x), 
  #                                  center (y), below (-buffer_y), above (+buffer_y)
  lines = []
  for line in csv_reader:
    # Original point (no buffer)
    lines.append([line[0], line[1], round((float(line[2])*5+0), 2), round((float(line[3])), 2), line[4]])
    # Left buffer point
    lines.append([line[0], line[1], round((float(line[2])*5+1), 2), round((float(line[3])-av.buffer_x), 2), line[4]])
    # Right buffer point
    lines.append([line[0], line[1], round((float(line[2])*5+2), 2), round((float(line[3])+av.buffer_x), 2), line[4]])
    # Below buffer point
    lines.append([line[0], line[1], round((float(line[2])*5+3), 2), line[3], round((float(line[4])-av.buffer_y), 2)])
    # Above buffer point
    lines.append([line[0], line[1], round((float(line[2])*5+4), 2), line[3], round((float(line[4])+av.buffer_y), 2)])

# Save buffered dataset
with open(os.path.join(output_dir, 'N_C_PDPg_buffer_Dataset.csv'), 'w', newline='') as new_csv_file:
  csv_writer = csv.writer(new_csv_file)

  #write the new lines
  for line in lines:
    csv_writer.writerow(line)

# End and print time
print('Time elapsed for running module "N_T_OB": {:.3f} sec.'.format(time.time() - t_start))