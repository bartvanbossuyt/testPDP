"""
v230209
FUNCTIONALITY
    Transforms the dataset from its original form (O) to a dataset with buffers (buf) in both x- and y-directions
EXPLANATION
    Transforms the dataset from its original form (O) to a dataset with buffers (buf) in x- and y-directions
INPUT
    The dataset
OUTPUT
    A dataset with buffers in x- and y-directions
    csv-file "N_C_PDPg_buffer_Dataset.csv"
INPUT PARAMETERS: buffer distance buffer_x and buffer_y
"""

import csv
import time

# Set buffer distances for x and y directions
buffer_x = 25  # Buffer distance in x-direction
buffer_y = 10  # Buffer distance in y-direction

# Start time
t_start = time.time()

# Open the CSV
with open('N_C_Dataset.csv', 'r') as csv_file:
    # Read the CSV
    csv_reader = csv.reader(csv_file)

    # Add new lines with buffers in x and y directions
    lines = []
    for line in csv_reader:
        # Append original and buffered points for both x and y directions
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 0), 2), round((float(line[3]) - buffer_x), 2), line[4]])
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 1), 2), round((float(line[3]) + buffer_x), 2), line[4]])
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 2), 2), line[3], line[4]])  # No buffer in x-direction
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 3), 2), line[3], round((float(line[4]) - buffer_y), 2)])
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 4), 2), line[3], round((float(line[4]) + buffer_y), 2)])

# Save the new CSV file
with open('N_C_PDPg_buffer_Dataset.csv', 'w', newline='') as new_csv_file:
    csv_writer = csv.writer(new_csv_file)

    # Write the new lines
    for line in lines:
        csv_writer.writerow(line)

# End and print time
print('Time elapsed for running module "N_T_OB": {:.3f} sec.'.format(time.time() - t_start))
