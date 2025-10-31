"""
v241030
FUNCTIONALITY
    Visual Analytics: Creates a dimension reduction based on UMAP
EXPLANATION
    Creates a UMAP dimension reduction, based on the distance matrix
    UMAP (Uniform Manifold Approximation and Projection) is excellent at
    preserving both local and global structure, often faster than t-SNE
INPUT
    N_C_DistanceMatrix
OUTPUT
    Visualisation + N_C_UMAP.png
POSSIBLE UPGRADES
    The distance matrix must be symmetrical. You can write code to check this.
INPUT PARAMETERS:
"""

from matplotlib import pyplot as plt
import umap
import av
import csv
import numpy as np; np.random.seed(0)
import pandas as pd
import os
import seaborn as sns; sns.set_theme()
import time

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 12


# Start time
t_start = time.time()

# Function to perform UMAP transformation
def Transform(A_dataset):
    # UMAP with precomputed distances
    # init='random' is required when using metric='precomputed'
    reducer = umap.UMAP(n_components=2, metric='precomputed', init='random', random_state=42)
    manifold = reducer.fit_transform(A_dataset)
    manifold = pd.DataFrame(manifold, columns=['Dimension 1', 'Dimension 2'])
    return manifold

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
av.L_dataset = []
D_poi_mapping = {}
cur_poi_id = 0
dim = -1

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
A_embedding = Df_embedding.to_numpy()

# Set theme and style for seaborn
sns.set_theme('notebook')
sns.set_style('darkgrid')

# Create figure and axis
fig, ax = plt.subplots(figsize=(11, 8), dpi = 100.0)

# Plot the UMAP results
umap_plot = sns.scatterplot(data=Df_embedding, x='Dimension 1', y='Dimension 2', s=50, color='black')
umap_plot.set(xlabel=None)
umap_plot.set(ylabel=None)

# Customize the axes
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('white')

# Set ticks and grid
x_ticks = np.arange(-30, 30, 5)
y_ticks = np.arange(-30, 30, 5)
plt.xticks(x_ticks, color='black', fontsize=8)
plt.yticks(y_ticks, color='black', fontsize=8)
ax.xaxis.grid(True, linestyle='dotted', linewidth=0.5, color='black', alpha=0.5)
ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5, color='black', alpha=0.5)

# Annotate each point
for i in range(len(av.A_dataset)):
    plt.annotate(i, xy=(A_embedding[i, 0], A_embedding[i, 1]), xytext=(25, 25), textcoords="offset pixels")

# Determine the appropriate file name for the output image
if av.PDPg_fundamental_active == 1:
    filename = 'N_C_PDPg_fundamental_UMAP.png'
elif av.PDPg_buffer_active == 1:
    filename = 'N_C_PDPg_buffer_UMAP.png'
elif av.PDPg_rough_active == 1:
    filename = 'N_C_PDPg_rough_UMAP.png'
elif av.PDPg_bufferrough_active == 1:
    filename = 'N_C_PDPg_bufferrough_UMAP.png'

# Save the plot as a PNG image
results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
module_dir = os.path.join(results_dir, 'UMAP')
os.makedirs(module_dir, exist_ok=True)
out_path = os.path.join(module_dir, filename)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.clf()  # Clear the figure to start with a new blank figure

# End and print time
print('Time elapsed for running module "N_VA_UMAP": {:.3f} sec.'.format(time.time() - t_start))
