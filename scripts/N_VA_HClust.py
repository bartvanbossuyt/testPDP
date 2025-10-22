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
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import os
import random
import pandas as pd
import scipy.cluster.hierarchy as shc
import seaborn as sns; sns.set_theme()
import sklearn.datasets as dt
import time

# Start time
t_start = time.time()

av.L_dataset = []
D_poi_mapping = {}
cur_poi_id = 0
dim = -1

# Set to 1 to see print statements
verbose = 1

# Read in data
#with open(dataset_name) as csv_file:

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
    with open(file_name) as csv_file:
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


#with open('N_C_PDPgDistanceMatrix' + str(av.dataset_name_exclusive) + '.csv') as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',')
#    for L_row in csv_reader:
#        poi_id = L_row[0]
#        if dim == -1:
#            dim = len(L_row) - 3
#        # Check if poi_id is a string, if it is, map to int
#        try:
#            int(poi_id)
#        except ValueError:
#            if poi_id not in D_poi_mapping:
#                D_poi_mapping[poi_id] = cur_poi_id
#                cur_poi_id += 1
#            L_row[2] = D_poi_mapping[poi_id]
#        L_dataset.append(list(map(float, L_row)))

# Transform list to array
av.A_dataset = np.array(av.L_dataset, dtype=np.float32)
condensed_dist = squareform(av.A_dataset)

plt.figure(figsize=(10, 7))

# Generate new labels
labels = [str(i) for i in range(len(av.A_dataset))]

dend = shc.dendrogram(shc.linkage(condensed_dist, method='ward'), labels=labels, color_threshold=0, above_threshold_color='black')

# Set the facecolor of the figure to white
plt.gcf().set_facecolor('white')
plt.gcf().set_edgecolor('white')

ax = plt.gca()
ax.set_facecolor('white')

# Change the axes color
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

#av.dataset_name_exclusive = av.dataset_name [:-4] #The dataset without the last four characters ".csv"

if av.PDPg_fundamental_active == 1:
    filename = 'N_C_PDPg_fundamental_HClust.png'
elif av.PDPg_buffer_active == 1:
    filename = 'N_C_PDPg_buffer_HClust.png'
elif av.PDPg_rough_active == 1:
    filename = 'N_C_PDPg_rough_HClust.png'
elif av.PDPg_bufferrough_active == 1:
    filename = 'N_C_PDPg_bufferrough_HClust.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()  # clear the figure to start with a new blank figure

# End and print time
print('Time elapsed for running module "N_VA_HClust": {:.3f} sec.'.format(time.time() - t_start))