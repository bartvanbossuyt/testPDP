"""
v220316
FUNCTIONALITY
    DATASETS: Automatically creates a random dataset of (moving) points in the defaults format
EXPLANATION
    Creates a random dataset of (moving) points. The number of configurations, time stamps, points and dimensions have to
    be entered via the input parameters in the beginning of the code
    The default format looks like:
    "con", "tst", "poi", "dim1" (, "dim2", ...)
    ...
INPUT
    Parameters of the dataset to be imported
    Parameters
        number of configurations (con)
        number of time stamps (tst)
        number of points (poi)
        number of dimensions (dim)
OUTPUT
    csv-file "N_C_Dataset.csv" containing the random dataset of moving objects
INPUT PARAMETERS:
"""

import csv
import os
import random
import time

# Read parameters from environment variables
con = int(os.environ.get('CON', 12))   # number of configurations
tst = int(os.environ.get('TST', 2))   # number of time stamps
poi = int(os.environ.get('POI', 3))   # number of points
dim = int(os.environ.get('DIM', 2))   # number of descriptors/dimensions

# Start time
t_start = time.time()

# Generate random dataset
L_rows = []   # create empty row
for c in range(con):   # for each configuration
    for t in range(tst):   # for each time stamp
        for p in range(poi):   # for each point
            L_data_arr = [0] * dim
            L_data_arr[0] = round(random.uniform(0, 50),2)  
            L_data_arr[1] = round(random.uniform(0, 50),2)   

            #for d in range(dim):   # for each descriptor/dimension
            #    random_data = random.randrange(100)   # generate random number between 0 and 100
            #    L_data_arr[d] = random_data
            L_rows.append([c, t, p] + L_data_arr)

# Generate meaningful file name
file_name = f"N_C_Dataset_{con}_con_{tst}_tst_{poi}_poi_{dim}_dim.csv"

# Write generated dataset in file
with open(file_name, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for L_row in L_rows:
        wr.writerow(L_row)

# Generate meaningful file name
file_name = "N_C_Dataset.csv"

# Write generated dataset in file
with open(file_name, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for L_row in L_rows:
        wr.writerow(L_row)

# End and print time
print(f'Time elapsed for running module : {round((time.time() - t_start), 3)} sec.')
