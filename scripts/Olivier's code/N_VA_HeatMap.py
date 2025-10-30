"""
v220517
FUNCTIONALITY
    Visual Analytics: Creates a heat map based on the distance matrix
EXPLANATION
    Creates a heat map, based on the distance matrix
INPUT
    N_C_DistanceMatrix
OUTPUT
    Visualisation + N_C_Heatmap.png
POSSIBLE UPGRADES
    The distance matrix must be symmetrical. You can write code to check this.
INPUT PARAMETERS:
"""

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

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 12

# Start time
t_start = time.time()

L_dataset = []
D_poi_mapping = {}
cur_poi_id = 0
dim = -1

# Set to 1 to see print statements
verbose = 1

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

av.L_dataset = []
if file_name is not None:
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
            # Add a delay after reading each row
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

# Set the figure size in inches and specify DPI
fig, ax = plt.subplots(figsize=(20, 15), dpi=300)

# Create heat map
cax = ax.matshow(av.A_dataset, cmap='OrRd', vmin=0, vmax=100)

# Add colorbar
fig.colorbar(cax)

# Set tick marks for grid lines
ax.set_xticks(np.arange(-.5, len(av.A_dataset[0]), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(av.A_dataset), 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

# Loop to annotate each cell with its value
for i in range(av.A_dataset.shape[0]):
    for j in range(av.A_dataset.shape[1]):
        ax.text(j, i, '{:0.2f}'.format(av.A_dataset[i, j]), ha='center', va='center', color='white')

plt.grid(which='both', color='white', linestyle='-', linewidth=0)  # Optional: add grid lines

plt.title('Heatmap of A_dataset')

# Save the plot as a PNG image in the "dir" directory
#if av.PDPg_fundamental_active == 1:
#    filename = 'N_C_PDPg_fundamental_HeatMap.png'
#elif av.PDPg_buffer_active == 1:
#    filename = 'N_C_PDPg_buffer_HeatMap.png'
#elif av.PDPg_rough_active == 1:
#    filename = 'N_C_PDPg_rough_HeatMap.png'
#elif av.PDPg_bufferrough_active == 1:
#    filename = 'N_C_PDPg_bufferrough_HeatMap.png'

# Define output folder once using the central av.OUTPUT_FOLDER
output_folder = os.path.join(av.OUTPUT_FOLDER, 'heatmap')
os.makedirs(output_folder, exist_ok=True)

# Choose filename based on which PDPg variant is active
if av.PDPg_fundamental_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_fundamental_HeatMap.png')

elif av.PDPg_buffer_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_buffer_HeatMap.png')

elif av.PDPg_rough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_rough_HeatMap.png')

elif av.PDPg_bufferrough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_bufferrough_HeatMap.png')

plt.savefig(filename, bbox_inches='tight')
plt.clf()  # clear the figure to start with a new blank figure

# End and print time
print('Time elapsed for running module "N_VA_HeatMap": {:.3f} sec.'.format(time.time() - t_start))
