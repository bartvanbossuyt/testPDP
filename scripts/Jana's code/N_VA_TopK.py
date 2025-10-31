"""
v220517
FUNCTIONALITY
    Visual Analytics: Creates a heat map based on the distance matrix
EXPLANATION
    Creates a Top K analyses, based on the distance matrix
INPUT
    N_C_DistanceMatrix
OUTPUT
    Visualisation + N_C_TopK ....png
POSSIBLE UPGRADES
INPUT PARAMETERS:
"""

from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import rankdata
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import av
import csv
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import os
import random
import seaborn as sns; sns.set_theme()
import sklearn.datasets as dt
import time

# Start time
t_start = time.time()

#dataset_name = 'N_C_DistanceMatrix.csv'  # filename of csv file
av.L_dataset = []
D_poi_mapping = {}
cur_poi_id = 0
dim = -1

# Set to 1 to see print statements
verbose = 1

# Read in data
#with open(dataset_name) as csv_file:

#with open('N_C_PDPgDistanceMatrix' + str(av.dataset_name_exclusive) + '.csv') as csv_file:
 #   csv_reader = csv.reader(csv_file, delimiter=',')
 #   for L_row in csv_reader:
 #       poi_id = L_row[0]
 #       if dim == -1:
 #           dim = len(L_row) - 3
 #       # Check if poi_id is a string, if it is, map to int
 #       try:
 #           int(poi_id)
 #       except ValueError:
 #           if poi_id not in D_poi_mapping:
 #               D_poi_mapping[poi_id] = cur_poi_id
 #               cur_poi_id += 1
 #           L_row[2] = D_poi_mapping[poi_id]
 #       L_dataset.append(list(map(float, L_row)))


if av.PDPg_fundamental_active == 1:
    file_name = 'N_C_PDPg_fundamental_DistanceMatrix.csv'
elif av.PDPg_buffer_active == 1:
    file_name = 'N_C_PDPg_buffer_DistanceMatrix.csv'
elif av.PDPg_rough_active == 1:
    file_name = 'N_C_PDPg_rough_DistanceMatrix.csv'
elif av.PDPg_bufferrough_active == 1:
    file_name = 'N_C_PDPg_bufferrough_DistanceMatrix.csv'
else:
    print("Variable a does not hold an appropriate value.")
    file_name = None

if file_name is not None:
    # Get the results directory and construct the full path to the distance matrix
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    pdp_dir = os.path.join(results_dir, 'PDP')
    full_path = os.path.join(pdp_dir, file_name)
    
    with open(full_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            if dim == -1:
                dim = len(L_row) - 3
            # Check if poi_id is a string, if it is, map to int
            try:
                int(poi_id)
            except ValueError:
                if poi_id not in D_poi_mapping:
                    D_poi_mapping[poi_id] = cur_poi_id
                    cur_poi_id += 1
                L_row[2] = D_poi_mapping[poi_id]
            av.L_dataset.append(list(map(float, L_row)))

# Transform list to array
av.A_dataset = np.array(av.L_dataset, dtype=np.float32)

# Create the "dir" directory if it doesn't exist
#os.makedirs(av.dir, exist_ok=True)

# Create the top-k visualisations
# Loop over each row of A_dataset and create a bar graph
for i in range(av.con):
    row = av.A_dataset[i]  # Get the i-th row from the A_dataset array
    sorted_indices = np.argsort(row)  # Get sorted indices in ascending order
    sorted_values = row[sorted_indices]  # Sort values in ascending order
    plt.title('TopK wrt Con ' + str(i), fontsize=30)  # Add title with row number
    labels = [str(j) for j in sorted_indices]  # Create labels for each bar
    ax = plt.gca()  # Get the current axes
    ax.spines['bottom'].set_color('black')  # Set the color of the bottom spine (x-axis) to black
    ax.spines['left'].set_color('black')  # Set the color of the left spine (y-axis) to black
    ax.spines['top'].set_visible(False)  # Hide the top spine (x-axis)
    ax.spines['right'].set_visible(False)  # Hide the right spine (y-axis)
    plt.bar(labels, sorted_values, color='white', edgecolor='black')  # Create bar graph with white bars and black borders
    plt.gca().set_facecolor('white')  # Get the current axes
    plt.xlabel('Con', fontsize=20, fontname='Arial')  # Add x-axis label
    plt.ylabel('Distance', fontsize=20, fontname='Arial')  # Add y-axis label
    ax.set_ylim(0, 100)  # Set the y-axis limits
    y_ticks = np.arange(0, 110, 10)  # Generate y-ticks at intervals of 10
    plt.yticks(y_ticks, color='black', fontsize=15)  # Set the y-tick labels to black with fontsize 8
    ax.tick_params(axis='both', labelsize=15, labelcolor='black')  # Change tick 
    ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5, color='black', alpha=0.5)  # Add horizontal grid lines
    
    if av.PDPg_fundamental_active == 1:
        filename = 'N_C_PDPg_fundamental_TopK_c' + str(i) + '.png'
    elif av.PDPg_buffer_active == 1:
        filename = 'N_C_PDPg_buffer_TopK_c' + str(i) + '.png'
    elif av.PDPg_rough_active == 1:
        filename = 'N_C_PDPg_rough_TopK_c' + str(i) + '.png'
    elif av.PDPg_bufferrough_active == 1:
        filename = 'N_C_PDPg_bufferrough_TopK_c' + str(i) + '.png'
        
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    module_dir = os.path.join(results_dir, 'TopK')
    os.makedirs(module_dir, exist_ok=True)
    out_path = os.path.join(module_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.clf()  # Clear the figure to start with a new blank figure

# End and print time
print('Time elapsed for running module "N_VA_TopK": {:.3f} sec.'.format(time.time() - t_start))