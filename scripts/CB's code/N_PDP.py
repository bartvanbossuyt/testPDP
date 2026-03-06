"""
MODULE: N_PDP (Principal Directional Pattern Analysis)
VERSION: 230626

PURPOSE:
    Calculate Principal Directional Pattern (PDP) based distance matrix for all configurations
    References: Amna & Klingele algorithm (see documentation)

DESCRIPTION:
    Transforms all configurations of tracking point trajectories to PDP distance matrix
    using inequality relationships between all point pairs across time windows
    
INPUT:
    - Dataset: N_C_Dataset.csv (or derived buffer/rough versions)
    - Configuration: window_length_tst, buffer/rough distance parameters
    
OUTPUT:
    - Distance matrix: N_C_PDPgDistanceMatrix.csv
    - Heatmaps: InequalityMatrix visualizations for each configuration
    - Index file: Df_con_tst_xineq_yineq.csv (configuration-timestamp-inequality mapping)
    
FUTURE ENHANCEMENTS:
    - Support for 1D and 3D analysis
    - Long-period trajectory support
    - Dimension reduction for high-dimensional data
"""

# Importing required libraries
from matplotlib.colors import ListedColormap

import av
import csv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 12

# ==================================================================================
# UTILITY FUNCTIONS: Disjoint-Set (Union-Find) Data Structure
# ==================================================================================
# Used for grouping equivalent inequality matrices

def make_set(x):
    """Create a singleton set containing element x"""
    return {x}

def find_set(disjoint_sets, elem):
    """Find and return the set containing element elem"""
    for s in disjoint_sets:
        if elem in s:
            return s
    return None

def union(disjoint_sets, set1, set2):
    """Union two sets: remove both and add their union"""
    disjoint_sets.remove(set1)
    disjoint_sets.remove(set2)
    disjoint_sets.append(set1.union(set2))

# ==================================================================================
# INITIALIZATION
# ==================================================================================

# Performance timing
t_start = time.time()

# Set up output directory for PDP module results
output_dir = av.get_output_dir('PDP')

# Load dataset with standard column structure
Df_dataset = pd.read_csv(os.path.join(av.output_base_path, 'N_C_Dataset.csv'), header=None)
Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']

# Initialize data structures for inequality matrices
Df_con_tst_xineq_yineq = pd.DataFrame(columns=['conID' , 'tstID', 'xineqID', 'yineqID'])
D_inequality = {}  # Dictionary to store inequality matrices (key: (con_id, tst_id))
new_index = 0

# Initialize roughness parameters (overridden below if active)
rough_x = 0
rough_y = 0

# Load roughness parameters if rough-type PDP is active
if av.PDPg_rough_active == 1: 
    rough_x = av.rough_x
    rough_y = av.rough_y

if av.PDPg_bufferrough_active == 1: 
    rough_x = av.rough_x
    rough_y = av.rough_y

# Pre-group dataset by configuration for efficient iteration
group_by_con_id = av.Df_dataset.groupby('conID')

# ==================================================================================
# MAIN LOOP: PROCESS ALL CONFIGURATIONS
# ==================================================================================

for con_id in range(av.con):  # Process each configuration
    # Get data for this configuration
    Df_con_id = group_by_con_id.get_group(con_id)

    # ==================================================================================
    # STEP 1: Create inequality matrices for each time window
    # ==================================================================================
    for tst_id in range(av.tst-(av.window_length_tst-1)):  # Process each time window
        
        # Filter data for current time window (includes window_length_tst consecutive timestamps)
        conditions = [Df_con_id['tstID'] == tst_id + i for i in range(av.window_length_tst)]
        mask = np.logical_or.reduce(conditions)
        Df_tst_id = Df_con_id[mask]

        i = 0
        j = 0

        # Process each dimension (x and y) separately
        L_tst_id_dfs = []
        for dim_id in ['x', 'y']:
            # Load roughness parameter for current dimension
            rough = rough_x if dim_id == 'x' else rough_y
            
            # Create inequality matrix: compare all point positions pairwise
            A_inequality_matrix = np.zeros((int(av.poi*av.window_length_tst), int(av.poi*av.window_length_tst)))
            
            # Compare each point with every other point in dimension dim_id
            for i in range(int(av.poi*av.window_length_tst)):
                for j in range(int(av.poi*av.window_length_tst)):
                    # Calculate position difference
                    diff = Df_tst_id[dim_id].iloc[j] - Df_tst_id[dim_id].iloc[i]
                    
                    # Classify inequality based on roughness threshold
                    if abs(diff) <= rough:
                        A_inequality_matrix[i, j] = 1  # Equal (within roughness)
                    elif diff > rough:
                        A_inequality_matrix[i, j] = 0  # Greater than (j > i)
                    else:
                        A_inequality_matrix[i, j] = 2  # Less than (j < i)
            
            # Store inequality matrix information
            Df_con_tst_xineq_yineq.at[new_index, 'conID'] = con_id
            Df_con_tst_xineq_yineq.at[new_index, 'tstID'] = tst_id
            if dim_id == "x":
                Df_con_tst_xineq_yineq.at[new_index, 'xineqID'] = A_inequality_matrix
            elif dim_id == "y":
                Df_con_tst_xineq_yineq.at[new_index, 'yineqID'] = A_inequality_matrix
            
            # Convert to DataFrame for visualization
            Df_inequality = pd.DataFrame(A_inequality_matrix)
            L_tst_id_dfs.append(Df_inequality)
            
            if av.N_VA_InequalityMatrices == 1:
                # Generate axis labels for inequality matrix visualization
                ticks = [f"c{con_id}_t{tst_id}_d{dim_id}_p{var2}_w{var1}" for var1 in range(int(av.window_length_tst)) for var2 in range(int(av.poi))]

                # Create colormap: green (<=), yellow (=), red (>)
                cmap = ListedColormap(["green", "yellow", "red"])
                
                # Set normalization for the colormap with specified boundaries
                cNorm  = plt.matplotlib.colors.BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
                

                # --- NEW: Reorder rows and columns ---
                num_time = int(av.window_length_tst)
                num_obj = int(av.poi)
                # Get original order labels (ticks are the actual values)
                labels = ticks

                # New order: all time points for p0 first, then all time points for p1
                # (works for poi=2, can be extended to any poi)
                reordered_labels = []
                for p in range(num_obj):
                    for w in range(num_time):
                        reordered_labels.append(f"c{con_id}_t{tst_id}_d{dim_id}_p{p}_w{w}")

                # Get new indices of labels in ticks
                reordered_indices = [labels.index(lbl) for lbl in reordered_labels]

                # Reorder the matrix according to new indices
                Df_inequality_reordered = Df_inequality.iloc[reordered_indices, reordered_indices]
                # --- NEW section ends ---





                # Display heatmap with reordered matrix for clarity
                plt.figure(figsize=(11, 8), dpi=300.0)
                plt.imshow(Df_inequality_reordered, cmap=cmap, norm=cNorm)
                
                # Set and rotate axis labels for readability
                plt.xticks(range(len(reordered_labels)), reordered_labels, rotation=45, ha='right')
                plt.yticks(range(len(reordered_labels)), reordered_labels)

                # Add diagonal visualization lines to mark quadrants
                # ------- NEW: Add white diagonal lines in upper-right & lower-left quadrants -------
                split_idx = num_time  # P0 occupies first half; P1 starts at split_idx
                line_x = np.arange(split_idx)
                line_y = np.arange(split_idx)

                # Upper-right diagonal (P0 vs P1)
                plt.plot(split_idx + line_x, line_y, color='white', linewidth=2)

                # Lower-left diagonal (P1 vs P0)
                plt.plot(line_x, split_idx + line_y, color='white', linewidth=2)
                # ------- NEW section ends -------

                plt.grid(which='both', color='white', linestyle='-', linewidth=0)

                # Create and add legend for inequality values
                patches = [mpatches.Patch(color=cmap(i), label="{l}".format(l=label)) for i, label in zip(range(3), ['<', '=', '>'])]
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Inequality')
                
                # Determine dimension index for filename
                dim = 0 if dim_id == 'x' else 1
                
                # Generate filename based on analysis type
                if av.PDPg_fundamental_active == 1:
                    filename = "N_C_PDPg_fundamental_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"
                elif av.PDPg_buffer_active == 1:
                    filename = "N_C_PDPg_buffer_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"
                elif av.PDPg_rough_active == 1:
                    filename = "N_C_PDPg_rough_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"
                elif av.PDPg_bufferrough_active == 1:
                    filename = "N_C_PDPg_bufferrough_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"

                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()  # close the figure to release memory
            
        new_index = len(Df_con_tst_xineq_yineq)  # Add the index +1 that stores the line in the dataframe


# ==================================================================================
# STEP 2: Store and index inequality matrices
# ==================================================================================

# Store processed matrices in dictionary
D_inequality[(con_id, tst_id)] = tuple(L_tst_id_dfs)

# Save index file mapping configurations and timestamps to inequality matrices
Df_con_tst_xineq_yineq.to_csv(os.path.join(output_dir, "Df_con_tst_xineq_yineq.csv"), index=False)

# ==================================================================================
# STEP 3: Group equivalent inequality matrices
# ==================================================================================

if av.N_VA_InequalityMatrices == 1:
    
    # Convert DataFrames to hashable tuples for grouping
    def df_to_tuple(df):
        """Convert DataFrame to tuple-of-tuples for hashing"""
        return tuple(map(tuple, df.values))

    # Dictionary to group identical matrices
    matrix_dict = {}

    # Group matrices: identical matrices appear in multiple time windows
    for tst_id, df_tuple in D_inequality.items():
        # Convert DataFrame tuple to hashable tuple
        df_tuple_hashable = tuple(map(df_to_tuple, df_tuple))
        # Track all time windows with this matrix pattern
        if df_tuple_hashable in matrix_dict:
            matrix_dict[df_tuple_hashable].append(tst_id)
        else:
            matrix_dict[df_tuple_hashable] = [tst_id]

    # List to collect grouped matrix information
    output_entries = []

    # Create output entries with frequency statistics
    for df_tuple_hashable, tstID in matrix_dict.items():
        # Reconstruct DataFrames from tuples
        df_tuple = tuple(pd.DataFrame(df) for df in df_tuple_hashable)
        
        # Create summary for this matrix pattern
        output_entry = {
            'times': len(tstID),  # Frequency of this matrix pattern
            'tst_id': tstID,  # All timestamps with this pattern
            'x_dimension': df_tuple[0],  # X-dimension inequality matrix
            'y_dimension': df_tuple[1]   # Y-dimension inequality matrix
        }
        output_entries.append(output_entry)

    # Sort matrix patterns by frequency (descending)
    output_entries.sort(key=lambda x: x['times'], reverse=True)
    
    # Optional: Print matrix pattern statistics
    # for entry in output_entries:
    #     freq = entry['times']
    #     msg = f"{freq} time" if freq == 1 else f"{freq} times"
    #     print(f"{msg} for (con_id, tst_id) : {' , '.join(map(str, entry['tst_id']))}")

# ==================================================================================
# STEP 4: Calculate distance matrix for all configuration pairs
# ==================================================================================

# Initialize distance matrices for x and y dimensions
A_rel_distance_matrix_x = np.empty((av.con, av.con))
A_rel_distance_matrix_y = np.empty((av.con, av.con))

# Compare all pairs of configurations
for k in range(av.con):  # First configuration index
    if k % 100 == 0:
        # print(f"Processing configuration {k}/{av.con}")
        pass
    
    for l in range(av.con):  # Second configuration index
        # Initialize distance accumulators for this configuration pair
        abs_distance_x = 0
        abs_distance_y = 0
        rel_distance_x = 0
        rel_distance_y = 0

        # Compare inequality matrices across all time windows
        for tst_id in range(av.tst-(av.window_length_tst-1)):
            # Retrieve inequality matrices for this configuration pair and time window
            mat0_x = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'xineqID'].values[0]
            mat1_x = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == l) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'xineqID'].values[0]

            # Calculate element-wise differences
            for i in range(int(av.poi*av.window_length_tst)):
                for j in range(int(av.poi*av.window_length_tst)):
                    abs_distance_x += abs(mat0_x[i][j] - mat1_x[i][j])
        #rel_distance_x = int(round (abs_distance_x / ((2*(((av.tst-(av.window_length_tst-1))*(av.poi * av.window_length_tst) * (av.poi * av.window_length_tst)) - (av.poi * av.window_length_tst)))/100), 0))
        rel_distance_x = int(round(abs_distance_x / ((2*(av.tst-(av.window_length_tst-1))*(((av.poi * av.window_length_tst) * (av.poi * av.window_length_tst)) - (av.poi * av.window_length_tst)))/100), 0))


        A_rel_distance_matrix_x[k][l] = rel_distance_x

#FOR y: 
A_rel_distance_matrix_y = np.empty((av.con, av.con))
k = 0
l = 0

for k in range(av.con):  # Loop over all configurations con_id
    for l in range(av.con):  # Loop over all configurations con_id

        # Initialize distances to 0
        abs_distance_y = 0
        rel_distance_y = 0
        nordist = 0

        for tst_id in range(av.tst-(av.window_length_tst-1)):  # Loop over all time stamps, dependant of the window length
            # Retrieve y-dimension inequality matrices
            mat0_y = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'yineqID'].values[0]
            mat1_y = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == l) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'yineqID'].values[0]
            
            # Sum element-wise differences for y-dimension
            for i in range(int(av.poi*av.window_length_tst)):
                for j in range(int(av.poi*av.window_length_tst)):
                    abs_distance_y += abs(mat0_y[i][j] - mat1_y[i][j])
        
        # Normalize distance to 0-100 scale
        max_possible_diff = 2 * (av.tst-(av.window_length_tst-1)) * ((av.poi * av.window_length_tst) ** 2 - (av.poi * av.window_length_tst))
        rel_distance_y = int(round(abs_distance_y / (max_possible_diff / 100), 0))
        
        A_rel_distance_matrix_y[k][l] = rel_distance_y

# ==================================================================================
# STEP 5: Combine and normalize distance matrices
# ==================================================================================

# Average distance matrices from x and y dimensions
A_rel_distance_matrix = np.empty((av.con, av.con))
A_rel_distance_matrix = np.round((A_rel_distance_matrix_x + A_rel_distance_matrix_y) / 2).astype(int)

# ==================================================================================
# STEP 6: Identify equivalent configurations using Union-Find
# ==================================================================================

# Initialize each configuration as its own disjoint set
disjoint_sets = [make_set(i) for i in range(av.con)]

# Find all configuration pairs with distance = 0 (equivalent patterns)
for i in range(av.con):
    for j in range(i + 1, av.con):
        if A_rel_distance_matrix[i, j] == 0:
            set_i = find_set(disjoint_sets, i)
            set_j = find_set(disjoint_sets, j)
            
            # Union sets if they are different
            if set_i is not set_j:
                union(disjoint_sets, set_i, set_j)

# Extract and sort unique equivalence sets
unique_sets = [sorted(list(s)) for s in disjoint_sets if len(s) > 1]

# Create filtered datasets for each equivalence class
file_paths = []
for idx, unique_set in enumerate(unique_sets):
    # Filter dataset to include only equivalent configurations
    filtered_df = Df_dataset[Df_dataset.iloc[:, 0].isin(unique_set)]
    
    # Save filtered dataset
    file_path = f"Filtered_Dataset_{idx+1}.csv"
    filtered_df.to_csv(file_path, index=False, header=None)
    file_paths.append(file_path)

# Initialize lists for data processing
conversion_mappings = []  # Store ID remapping for each equivalence class
file_paths = []           # Store paths of generated filtered datasets

# Process each equivalence class
for idx, unique_set in enumerate(unique_sets):
    # Filter dataset for this equivalence class
    filtered_df = Df_dataset[Df_dataset['conID'].isin(unique_set)].copy()
    
    # Create new ID mapping: original_id -> sequential new_id
    conversion_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_set)}
    conversion_mappings.append(conversion_mapping)
    
    # Remap configuration IDs to sequential indices
    filtered_df['conID'] = filtered_df['conID'].map(conversion_mapping)
    
    # Save filtered and remapped dataset
    file_path = f"Filtered_Dataset_{idx+1}.csv"
    filtered_df.to_csv(file_path, index=False, header=None)
    file_paths.append(file_path)

# Save ID conversion mapping for reference
conversion_df = pd.DataFrame(conversion_mappings)
conversion_df.to_csv(os.path.join(output_dir, "Conversion_Mapping.csv"), index=False)

# ==================================================================================
# STEP 7: Save results and output files
# ==================================================================================

# Generate output filename based on active PDP type
if av.N_VA_Inverse == 1:
    filename = 'N_C_PDPg_DistanceMatrix.csv'
elif av.PDPg_fundamental_active == 1:
    filename = 'N_C_PDPg_fundamental_DistanceMatrix.csv'
elif av.PDPg_buffer_active == 1:
    filename = 'N_C_PDPg_buffer_DistanceMatrix.csv'
elif av.PDPg_rough_active == 1:
    filename = 'N_C_PDPg_rough_DistanceMatrix.csv'
elif av.PDPg_bufferrough_active == 1:
    filename = 'N_C_PDPg_bufferrough_DistanceMatrix.csv'

# Save distance matrix to CSV
with open(os.path.join(output_dir, filename), 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for L_row in A_rel_distance_matrix:
        wr.writerow(L_row.tolist())

# Print execution time
print('Module "N_PDP" completed in {:.3f} seconds.'.format(time.time() - t_start))