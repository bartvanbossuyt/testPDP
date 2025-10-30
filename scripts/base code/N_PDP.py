# N_PDP.py
"""
v230626
FUNCTIONALITY
    Transformations: From set of all configurations of (s,t)-trajectories to PDP
EXPLANATION
    Transforms all configurations of moving objects to a distance matrix based on PDP
INPUT
    Uses av.Df_dataset prepared by N_Moving_Objects (5 numeric cols).
OUTPUT
    csv-file "*_DistanceMatrix.csv" containing PDPg distance matrix; optional filtered datasets.
"""

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

# ---------------- Disjoint-Set (Union-Find) ----------------
def make_set(x): return {x}

def find_set(disjoint_sets, elem):
    for s in disjoint_sets:
        if elem in s:
            return s
    return None

def union(disjoint_sets, set1, set2):
    disjoint_sets.remove(set1)
    disjoint_sets.remove(set2)
    disjoint_sets.append(set1.union(set2))

# ---------------- Start ----------------
t_start = time.time()

# IMPORTANT: use the dataset prepared by N_Moving_Objects (always 5 numeric cols).
# Do NOT re-read CSV or rename columns here.
Df_dataset = av.Df_dataset.copy()  # columns: ['conID','tstID','poiID','x','y']

# Create containers
Df_con_tst_xineq_yineq = pd.DataFrame(columns=['conID', 'tstID', 'xineqID', 'yineqID']) 
D_inequality = {}
new_index = 0

# Roughness handling
rough_x = 0
rough_y = 0
if av.PDPg_rough_active == 1 or av.PDPg_bufferrough_active == 1:
    rough_x = av.rough_x
    rough_y = av.rough_y

# Pre-group by conID
group_by_con_id = Df_dataset.groupby('conID')

for con_id in range(av.con):
    Df_con_id = group_by_con_id.get_group(con_id)

    # For each starting tst index (windowed)
    for tst_id in range(av.tst - (av.window_length_tst - 1)):
        # Mask: include frames tst_id..tst_id+window-1
        conditions = [Df_con_id['tstID'] == tst_id + i for i in range(av.window_length_tst)]
        mask = np.logical_or.reduce(conditions)
        Df_tst_id = Df_con_id[mask]

        L_tst_id_dfs = []
        for dim_id in ['x', 'y']:
            rough = rough_x if dim_id == 'x' else rough_y

            size = int(av.poi * av.window_length_tst)
            A_inequality_matrix = np.zeros((size, size))

            # Use underlying numpy arrays for speed
            vals = Df_tst_id[dim_id].to_numpy()
            # Ensure ordering is consistent with expected indexing
            # (Assumes Df_tst_id rows are already ordered by tstID then poi)
            for i in range(size):
                di = vals[i]
                for j in range(size):
                    dj = vals[j]
                    if abs(dj - di) <= rough:
                        A_inequality_matrix[i, j] = 1
                    elif dj - di > rough:
                        A_inequality_matrix[i, j] = 0
                    else:
                        A_inequality_matrix[i, j] = 2

            # Store to row
            Df_con_tst_xineq_yineq.at[new_index, 'conID'] = con_id
            Df_con_tst_xineq_yineq.at[new_index, 'tstID'] = tst_id
            if dim_id == "x":
                Df_con_tst_xineq_yineq.at[new_index, 'xineqID'] = A_inequality_matrix
            else:
                Df_con_tst_xineq_yineq.at[new_index, 'yineqID'] = A_inequality_matrix

            # Keep as DF for optional plotting
            Df_inequality = pd.DataFrame(A_inequality_matrix)
            L_tst_id_dfs.append(Df_inequality)

            # Optional visualization
            if av.N_VA_InequalityMatrices == 1:
                ticks = [f"c{con_id}_t{tst_id}_d{dim_id}_p{var2}_w{var1}"
                         for var1 in range(int(av.window_length_tst))
                         for var2 in range(int(av.poi))]
                cmap = ListedColormap(["green", "yellow", "red"])
                cNorm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

                plt.figure(figsize=(11, 8), dpi=300.0)
                plt.imshow(Df_inequality, cmap=cmap, norm=cNorm)
                plt.xticks(range(len(ticks)), ticks, rotation=45, ha='right')
                plt.yticks(range(len(ticks)), ticks)
                plt.grid(which='both', color='white', linestyle='-', linewidth=0)
                patches = [mpatches.Patch(color=cmap(i), label=label)
                           for i, label in zip(range(3), ['<', '=', '>'])]
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Values')

                output_folder = os.path.join(av.OUTPUT_FOLDER, 'ineq_matrix')
                os.makedirs(output_folder, exist_ok=True)

                dim_idx = 0 if dim_id == 'x' else 1
                if av.PDPg_fundamental_active == 1:
                    fname = f"N_C_PDPg_fundamental_InequalityMatrix_c{con_id}_t{tst_id}_d{dim_idx}.png"
                elif av.PDPg_buffer_active == 1:
                    fname = f"N_C_PDPg_buffer_InequalityMatrix_c{con_id}_t{tst_id}_d{dim_idx}.png"
                elif av.PDPg_rough_active == 1:
                    fname = f"N_C_PDPg_rough_InequalityMatrix_c{con_id}_t{tst_id}_d{dim_idx}.png"
                elif av.PDPg_bufferrough_active == 1:
                    fname = f"N_C_PDPg_bufferrough_InequalityMatrix_c{con_id}_t{tst_id}_d{dim_idx}.png"
                else:
                    fname = f"N_C_PDPg_InequalityMatrix_c{con_id}_t{tst_id}_d{dim_idx}.png"

                plt.savefig(os.path.join(output_folder, fname), dpi=300, bbox_inches='tight')
                plt.close()

        # store per (con_id, tst_id)
        D_inequality[(con_id, tst_id)] = tuple(L_tst_id_dfs)
        new_index += 1

# Save mapping table
Df_con_tst_xineq_yineq.to_csv("Df_con_tst_xineq_yineq.csv", index=False)

# ---------------- Count identical matrices across timestamps ----------------
if av.N_VA_InequalityMatrices == 1:
    def df_to_tuple(df):  # make hashable
        return tuple(map(tuple, df.values))

    matrix_dict = {}
    for key_tuple, df_tuple in D_inequality.items():
        df_tuple_hashable = tuple(map(df_to_tuple, df_tuple))
        if df_tuple_hashable in matrix_dict:
            matrix_dict[df_tuple_hashable].append(key_tuple)
        else:
            matrix_dict[df_tuple_hashable] = [key_tuple]

    output_entries = []
    for df_tuple_hashable, keys in matrix_dict.items():
        df_tuple = tuple(pd.DataFrame(df) for df in df_tuple_hashable)
        output_entries.append({
            'times': len(keys),
            'tst_id': keys,
            'x_dimension': df_tuple[0],
            'y_dimension': df_tuple[1],
        })
    output_entries.sort(key=lambda x: x['times'], reverse=True)

# ---------------- Distance matrices ----------------
A_rel_distance_matrix_x = np.empty((av.con, av.con))
A_rel_distance_matrix_y = np.empty((av.con, av.con))

# For x
for k in range(av.con):
    for l in range(av.con):
        abs_distance_x = 0
        for tst_id in range(av.tst - (av.window_length_tst - 1)):
            mat0_x = Df_con_tst_xineq_yineq.loc[
                (Df_con_tst_xineq_yineq['conID'] == k) &
                (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'xineqID'
            ].values[0]
            mat1_x = Df_con_tst_xineq_yineq.loc[
                (Df_con_tst_xineq_yineq['conID'] == l) &
                (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'xineqID'
            ].values[0]
            abs_distance_x += np.abs(mat0_x - mat1_x).sum()

        denom = (2 * (av.tst - (av.window_length_tst - 1)) *
                 ((av.poi * av.window_length_tst) * (av.poi * av.window_length_tst) - (av.poi * av.window_length_tst)) / 100)
        A_rel_distance_matrix_x[k, l] = int(round(abs_distance_x / denom, 0))

# For y
for k in range(av.con):
    for l in range(av.con):
        abs_distance_y = 0
        for tst_id in range(av.tst - (av.window_length_tst - 1)):
            mat0_y = Df_con_tst_xineq_yineq.loc[
                (Df_con_tst_xineq_yineq['conID'] == k) &
                (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'yineqID'
            ].values[0]
            mat1_y = Df_con_tst_xineq_yineq.loc[
                (Df_con_tst_xineq_yineq['conID'] == l) &
                (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'yineqID'
            ].values[0]
            abs_distance_y += np.abs(mat0_y - mat1_y).sum()

        denom = (2 * (av.tst - (av.window_length_tst - 1)) *
                 ((av.poi * av.window_length_tst) * (av.poi * av.window_length_tst) - (av.poi * av.window_length_tst)) / 100)
        A_rel_distance_matrix_y[k, l] = int(round(abs_distance_y / denom, 0))

# Combine
A_rel_distance_matrix = np.round(
    (A_rel_distance_matrix_x + A_rel_distance_matrix_y) / 2
).astype(int)

# ---------------- Union-Find over identical configs ----------------
disjoint_sets = [make_set(i) for i in range(av.con)]
for i in range(av.con):
    for j in range(i + 1, av.con):
        if A_rel_distance_matrix[i, j] == 0:
            si = find_set(disjoint_sets, i)
            sj = find_set(disjoint_sets, j)
            if si is not sj:
                union(disjoint_sets, si, sj)

unique_sets = [sorted(list(s)) for s in disjoint_sets if len(s) > 1]

# Optional: write filtered datasets & conversion map (using 5-col Df_dataset)
file_paths = []
conversion_mappings = []
for idx, unique_set in enumerate(unique_sets):
    filtered_df = Df_dataset[Df_dataset['conID'].isin(unique_set)].copy()
    conv_map = {orig: new for new, orig in enumerate(unique_set)}
    conversion_mappings.append(conv_map)
    filtered_df['conID'] = filtered_df['conID'].map(conv_map)
    file_path = f"Filtered_Dataset_{idx+1}.csv"
    filtered_df.to_csv(file_path, index=False, header=None)
    file_paths.append(file_path)

if conversion_mappings:
    conversion_df = pd.DataFrame(conversion_mappings)
    conversion_df.to_csv("Conversion_Mapping.csv", index=False)

# Define output folder once
output_folder = av.OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Select filename depending on which PDPg mode is active
if av.N_VA_Inverse == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_DistanceMatrix.csv')
elif av.PDPg_fundamental_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_fundamental_DistanceMatrix.csv')
elif av.PDPg_buffer_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_buffer_DistanceMatrix.csv')
elif av.PDPg_rough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_rough_DistanceMatrix.csv')
elif av.PDPg_bufferrough_active == 1:
    filename = os.path.join(output_folder, 'N_C_PDPg_bufferrough_DistanceMatrix.csv')
else:
    filename = os.path.join(output_folder, 'N_C_PDPg_DistanceMatrix.csv')

# Write distance matrix to CSV
with open(filename, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for L_row in A_rel_distance_matrix:
        wr.writerow(L_row.tolist())


print('Time elapsed for running module "N_PDP": {:.3f} sec.'.format(time.time() - t_start))
