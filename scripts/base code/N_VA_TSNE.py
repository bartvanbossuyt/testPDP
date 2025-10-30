"""
vYYYYMMDD
FUNCTIONALITY
    Visual Analytics: Creates a dimension reduction based on a t-SNE transformation
EXPLANATION
    Creates a dimension reduction using t-SNE, based on the distance matrix
INPUT
    N_C_DistanceMatrix (chosen based on the active PDP setting)
OUTPUT
    Visualization saved as a PNG file (with TSNE in the name)
POSSIBLE UPGRADES
    Add more customizable t-SNE parameters as needed
INPUT PARAMETERS:
"""

from matplotlib import pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
from sklearn.manifold import TSNE
import av
import os

# Start time
t_start = time.time()

# Function to perform t-SNE transformation on the distance matrix.
def Transform_tsne(A_dataset):

    M = av.A_dataset.shape[0]           # aantal punten in je matrix
    perp = min(30, M//3)                # bijvoorbeeld: 1/3 van je data, max 30
    perp = max(perp, 5)                 # en niet lager dan 5
    
    # When using a precomputed distance matrix, set metric='precomputed'
    tsne = TSNE(n_components=2,
            perplexity=perp,
            init='random',
            random_state=0,
            metric='precomputed')
    manifold = tsne.fit_transform(A_dataset)
    manifold = pd.DataFrame(manifold, columns=['Dimension 1', 'Dimension 2'])
    return manifold

# Determine the appropriate distance matrix file based on the active setting.
if av.PDPg_fundamental_active == 1:
    file_name = 'N_C_PDPg_fundamental_DistanceMatrix.csv'
elif av.PDPg_buffer_active == 1:
    file_name = 'N_C_PDPg_buffer_DistanceMatrix.csv'
elif av.PDPg_rough_active == 1:
    file_name = 'N_C_PDPg_rough_DistanceMatrix.csv'
elif av.PDPg_bufferrough_active == 1:
    file_name = 'N_C_PDPg_bufferrough_DistanceMatrix.csv'
else:
    print("Variable active settings not appropriately set.")
    file_name = None

# Read the distance matrix data from the chosen file.
av.L_dataset = []
D_poi_mapping = {}
cur_poi_id = 0
dim = -1

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
            poi_id = L_row[0]
            if dim == -1:
                dim = len(L_row) - 3
            try:
                int(poi_id)
            except ValueError:
                if poi_id not in D_poi_mapping:
                    D_poi_mapping[poi_id] = cur_poi_id
                    cur_poi_id += 1
                L_row[2] = D_poi_mapping[poi_id]
            av.L_dataset.append(list(map(float, L_row)))

# Convert list to numpy array for numerical processing.
av.A_dataset = np.array(av.L_dataset, dtype=np.float32)

# Perform the t-SNE transformation.
Df_embedding = Transform_tsne(av.A_dataset)
A_embedding = Df_embedding.to_numpy()


# Plotting the t-SNE results.
fig, ax = plt.subplots(figsize=(11, 8), dpi=100)
ax.scatter(Df_embedding['Dimension 1'], Df_embedding['Dimension 2'], s=50, color='blue')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('t-SNE Visualization')

# Optionally annotate each point with its index.
for i in range(len(av.A_dataset)):
    ax.annotate(i, xy=(A_embedding[i, 0], A_embedding[i, 1]), xytext=(5, 5))

import os


# Define output folder once using the central av.OUTPUT_FOLDER
output_folder = os.path.join(av.OUTPUT_FOLDER, 'tsne')
os.makedirs(output_folder, exist_ok=True)

# Select filename depending on which PDPg mode is active
if av.PDPg_fundamental_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_fundamental_tsne.png')

elif av.PDPg_buffer_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_buffer_tsne.png')

elif av.PDPg_rough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_rough_tsne.png')

elif av.PDPg_bufferrough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_bufferrough_tsne.png')

# Save the plot.
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.clf()  # Clear the figure

# Print elapsed time.
print('Time elapsed for running module "N_VA_TSNE": {:.3f} sec.'.format(time.time() - t_start))