"""
v220508
FUNCTIONALITY
    Visual Analytics: Creates a hierarchical cluster tree based on the distance matrix
EXPLANATION
    Creates a hierarchical cluster tree, based on the distance matrix
INPUT
    N_C_DistanceMatrix
OUTPUT
    Visualisation + N_C_HClustering.png
POSSIBLE UPGRADES
    The distance matrix must be symmetrical. You can write code to check this.
INPUT PARAMETERS:
"""

from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import av
import csv
import numpy as np; np.random.seed(0)
import os
import random
import pandas as pd
import scipy.cluster.hierarchy as shc
import seaborn as sns; sns.set_theme()
import sklearn.datasets as dt
import time
import matplotlib

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 12

# Start time
t_start = time.time()

av.L_dataset = []
D_poi_mapping = {}
cur_poi_id = 0
dim = -1

# Set to 1 to see print statements
verbose = 1

# Determine the appropriate file name based on active settings
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

# Read the distance matrix data
if file_name is not None:
    # Resolve full path: if av.INPUT_DISTANCE_MATRIX is a directory, join it;
    # if it's a file path, use that directly. If not set, use file_name as-is.
    full_file_name = file_name
    if getattr(av, 'INPUT_DISTANCE_MATRIX', None):
        inp = av.INPUT_DISTANCE_MATRIX
        if os.path.isdir(inp):
            full_file_name = os.path.join(inp, file_name)
        elif isinstance(inp, str) and inp.lower().endswith('.csv'):
            full_file_name = inp
        else:
            full_file_name = os.path.join(inp, file_name)

    with open(full_file_name) as csv_file:
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
condensed_dist = squareform(av.A_dataset)

# Set the figure size and create the subplot
fig, ax = plt.subplots(figsize=(11, 8), dpi = 100.0)

# Generate new labels
labels = [str(i) for i in range(len(av.A_dataset))]

# Set the linewidth of the dendrogram
matplotlib.rcParams['lines.linewidth'] = 3

# Create the dendrogram
dend = shc.dendrogram(shc.linkage(condensed_dist, method='ward', ), labels=labels, color_threshold=0, above_threshold_color='blue')

# Set the facecolor of the figure and axis to white
fig.set_facecolor('white')
ax.set_facecolor('white')

# Customize the appearance of the axes
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

# Determine the appropriate file name for the output image
#if av.PDPg_fundamental_active == 1:
#    filename = 'N_C_PDPg_fundamental_HClust.png'
#elif av.PDPg_buffer_active == 1:
#    filename = 'N_C_PDPg_buffer_HClust.png'
#elif av.PDPg_rough_active == 1:
#    filename = 'N_C_PDPg_rough_HClust.png'
#elif av.PDPg_bufferrough_active == 1:
#    filename = 'N_C_PDPg_bufferrough_HClust.png'

# Define the output folder once using the central av.OUTPUT_FOLDER
output_folder = os.path.join(av.OUTPUT_FOLDER, 'H_clust')
os.makedirs(output_folder, exist_ok=True)

# Then just choose the filename based on the active variable
if av.PDPg_fundamental_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_fundamental_HClust.png')

elif av.PDPg_buffer_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_buffer_HClust.png')

elif av.PDPg_rough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_rough_HClust.png')

elif av.PDPg_bufferrough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_bufferrough_HClust.png')

# Save the plot as a PNG image
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()  # Clear the figure to start with a new blank figure

# End and print time
print('Time elapsed for running module "N_VA_HClust": {:.3f} sec.'.format(time.time() - t_start))
