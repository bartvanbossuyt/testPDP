"""
v220505
FUNCTIONALITY
    Visual Analytics: Creates a dimension reduction based on the distance matrix
EXPLANATION
    Creates a dimension reduction, based on the distance matrix
INPUT
    N_C_DistanceMatrix
OUTPUT
    Visualisation + N_C_Mds.png
POSSIBLE UPGRADES
    The distance matrix must be symmetrical. You can write code to check this.
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
import pandas as pd
import os
import random
import seaborn as sns; sns.set_theme()
import sklearn.datasets as dt
import time

# Start time
t_start = time.time()

A_dataset = av.A_dataset
def Transform(A_dataset):
  manifold = MDS(metric=True, n_components=2, dissimilarity='precomputed', random_state=1, normalized_stress= 'auto').fit_transform(av.A_dataset)
  manifold = pd.DataFrame(manifold,columns=['Dimension 1','Dimension 2'])
  return manifold

#dataset_name = 'N_C_DistanceMatrix.csv'   # filename of csv file
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

# Create Dim Reduction
Df_embedding = Transform(av.A_dataset)
A_embedding = Df_embedding.to_numpy()   # transform
sns.set_theme('notebook');
sns.set_style('darkgrid');
plt.figure(figsize=(8,8), dpi=100.0);
mds = sns.scatterplot(data=Df_embedding,x='Dimension 1',y='Dimension 2',markers=True,legend="brief",s=50, color='black');
mds.set(xlabel=None)
mds.set(ylabel=None)

ax = plt.gca()  # Get the current axes
ax.spines['bottom'].set_color('black')  # Set the color of the bottom spine (x-axis) to black
ax.spines['left'].set_color('black')  # Set the color of the left spine (y-axis) to black
ax.spines['top'].set_visible(False)  # Hide the top spine (x-axis)
ax.spines['right'].set_visible(False)  # Hide the right spine (y-axis)

plt.gca().set_facecolor('white') 
# better calculate absolute distances and then you have the same vaue alwaysas max and min... normalised...
x_ticks = np.arange(-30, 30, 5)  # Generate x-ticks at intervals of 5
plt.yticks(x_ticks, color='black', fontsize=8)  # Set the x-tick labels to black with fontsize 8
ax.xaxis.grid(True, linestyle='dotted', linewidth=0.5, color='black', alpha=0.5)  # Add horizontal grid lines

y_ticks = np.arange(-30, 30, 5)  # Generate y-ticks at intervals of 5
plt.yticks(y_ticks, color='black', fontsize=8)  # Set the y-tick labels to black with fontsize 8
ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5, color='black', alpha=0.5)  # Add horizontal grid lines

# Loop for annotation of all points
for i in range(len(av.A_dataset)):
    plt.annotate(i,xy = (A_embedding[i, 0], A_embedding[i, 1]), xytext=(25, 25), textcoords="offset pixels")

if av.PDPg_fundamental_active == 1:
    filename = 'N_C_PDPg_fundamental_Mds.png'
elif av.PDPg_buffer_active == 1:
    filename = 'N_C_PDPg_buffer_Mds.png'
elif av.PDPg_rough_active == 1:
    filename = 'N_C_PDPg_rough_Mds.png'
elif av.PDPg_bufferrough_active == 1:
    filename = 'N_C_PDPg_bufferrough_Mds.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()  # clear the figure to start with a new blank figure

# End and print time
print('Time elapsed for running module "N_VA_Mds": {:.3f} sec.'.format(time.time() - t_start))