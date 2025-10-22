
"""
v230704
FUNCTIONALITY
    Visual Analytics: Creates a cluster map based on the distance matrix
EXPLANATION
    Creates a cluster map, based on the distance matrix
INPUT
    N_C_DistanceMatrix
OUTPUT
    Visualisation + N_C_ClusterMap.png
POSSIBLE UPGRADES
    The distance matrix must be symmetrical. You can write code to check this.
INPUT PARAMETERS:
"""

from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial.distance import pdist, squareform
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

#dataset_name = 'N_C_DistanceMatrix.csv'   # filename of csv file
L_dataset = []
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

av.L_dataset = []
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

# Transform list to array
av.A_dataset = np.array(av.L_dataset, dtype=np.float32)

# Set the figure size in inches
plt.figure(figsize=(12, 8))

# Convert to a condensed distance matrix, which is needed for the sns.clustermap analysis
A_dataset_condensed = squareform(av.A_dataset)
#print(A_dataset_condensed)
# Now you can use condensed_dist_matrix in your clustering function

# Create cluster map
sns.clustermap(av.A_dataset, method= 'ward', cmap= 'OrRd')
#sns.clustermap(A_dataset_condensed, method= 'ward', cmap= 'OrRd')
plt.show()





# Save the plot as a PNG image in the "dir" directory
if av.PDPg_fundamental_active == 1:
    filename = 'N_C_PDPg_fundamental_ClusterMap.png'
elif av.PDPg_buffer_active == 1:
    filename = 'N_C_PDPg_buffer_ClusterMap.png'
elif av.PDPg_rough_active == 1:
    filename = 'N_C_PDPg_rough_ClusterMap.png'
elif av.PDPg_bufferrough_active == 1:
    filename = 'N_C_PDPg_bufferrough_ClusterMap.png'

plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()  # clear the figure to start with a new blank figure








# Save the plot as a PNG image in the "dir" directory
##av.dataset_name_exclusive = av.dataset_name [:-4] #The dataset without the last four characters ".csv"
#file_name = "N_C_ClusterMap" + str(av.dataset_name_exclusive) + ".png"
#plt.savefig(file_name, dpi=300, bbox_inches='tight')
#plt.clf()  # clear the figure to start with a new blank figure

# End and print time
print('Time elapsed for running module "N_VA_ClusterMap": {:.3f} sec.'.format(time.time() - t_start))
