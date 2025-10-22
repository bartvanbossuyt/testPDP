#create the PDP-symbol: see amna and klingele... use algorithm of amna.

"""
v230626
FUNCTIONALITY
    Transformations: From set of all configurations of (s,t)-trajectories to PDP
EXPLANATION
    Transforms all configurations of moving objects to a distance matrix based on PDP
INPUT
    N_C_Dataset.csv
OUTPUT
    csv-file "N_C_PDPgDistanceMatrix.csv" containing PDPg distance matrix
POSSIBLE UPGRADES
    also for 1 dimension and for 3 dimensions
    also for long periods
INPUT PARAMETERS:
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

# Disjoint-Set (Union-Find) Functions
def make_set(x):
    return {x}

def find_set(disjoint_sets, elem):
    for s in disjoint_sets:
        if elem in s:
            return s
    return None

def union(disjoint_sets, set1, set2):
    disjoint_sets.remove(set1)
    disjoint_sets.remove(set2)
    disjoint_sets.append(set1.union(set2))

# Start time
t_start = time.time()

Df_dataset = pd.read_csv(av.dataset_name, header=None)
Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']

# Create data structures to store information
Df_con_tst_xineq_yineq = pd.DataFrame(columns=['conID' , 'tstID', 'xineqID', 'yineqID']) 
D_inequality = {}  # Initialize an empty dictionary to store DataFrames
new_index = 0

rough = 0

if av.PDPg_rough_active == 1: 
    rough = av.rough

if av.PDPg_bufferrough_active == 1: 
    rough = av.rough

group_by_con_id = av.Df_dataset.groupby('conID') #pre-group your DataFrame by 'conID' so that you can quickly access all rows for a given con_id without having to filter the DataFrame over and over in each iteration of the loop
for con_id in range(av.con):  # Loop over all configurations con_id
    #Df_con_id = av.Df_dataset[av.Df_dataset['conID'] == con_id]  # Filter the DataFrame for the current con_id
    Df_con_id = group_by_con_id.get_group(con_id) # pre-group your DataFrame by 'conID' so that you can quickly access all rows for a given con_id without having to filter the DataFrame over and over in each iteration of the loop

    # Create inequality matrix for each time stamp and each dimension (x and y)
    for tst_id in range(av.tst-(av.window_length_tst-1)):  # Loop over all time stamps, dependant of the window length
        
        # Filter the dataframe for current timestamp tst_id with the current window_length_tst
        conditions = [Df_con_id['tstID'] == tst_id + i for i in range(av.window_length_tst)]
        mask = np.logical_or.reduce(conditions)
        Df_tst_id = Df_con_id[mask]

        i = 0
        j = 0

        L_tst_id_dfs = []  # Create a list to hold the dataframes for each dimension
        for dim_id in ['x', 'y']:  # Loop over all dimensions (x and y)            
            A_inequality_matrix = np.zeros((int(av.poi*av.window_length_tst), int(av.poi*av.window_length_tst))) # Create an empty matrix of appropriate size
            #for i in range(av.poi*av.window_length_tst):  
            for i in range(int(av.poi*av.window_length_tst)): # Loop over all points, dependant of the window length
                for j in range(int(av.poi*av.window_length_tst)):  # Loop over all points, dependant of the window length
                    if (abs(Df_tst_id[dim_id].iloc[j] - Df_tst_id[dim_id].iloc[i]) <= rough):
                        A_inequality_matrix[i, j] = 1
                    elif ((Df_tst_id[dim_id].iloc[j] - Df_tst_id[dim_id].iloc[i]) > rough):
                        A_inequality_matrix[i, j] = 0
                    else:
                        A_inequality_matrix[i, j] = 2
                    #if ((Df_tst_id[dim_id].iloc[j] - Df_tst_id[dim_id].iloc[i]) < rough):
                    #    A_inequality_matrix[i, j] = 2
                    #elif ((Df_tst_id[dim_id].iloc[i] < Df_tst_id[dim_id].iloc[j]) > rough):
                    #    A_inequality_matrix[i, j] = 0
                    #else:
                    #    A_inequality_matrix[i, j] = 1
            # Add a line to the dataframe containing all configurations conID, all time stamps ID and all inequality arrays for each dimension
            # new_index = len(Df_con_tst_xineq_yineq)
            Df_con_tst_xineq_yineq.at[new_index, 'conID'] = con_id
            Df_con_tst_xineq_yineq.at[new_index, 'tstID'] = tst_id
            if dim_id == "x":
                Df_con_tst_xineq_yineq.at[new_index, 'xineqID'] = A_inequality_matrix
            elif dim_id == "y":
                Df_con_tst_xineq_yineq.at[new_index, 'yineqID'] = A_inequality_matrix

            Df_inequality = pd.DataFrame(A_inequality_matrix)  # Convert the numpy array to a dataframe for better visualization
            
            L_tst_id_dfs.append(Df_inequality)  # Add the dataframe to the list  
            
            if av.N_VA_InequalityMatrices == 1:

                #ticks = [f"window_id {var1}, point_id {var2}" for var1 in range(av.window_length_tst) for var2 in range(av.poi)]  # The variables var1 and var2 can vary between 0 and 3
                #!!!!! var1 moet niet alleen window_id 0 zijn maar ook soms window_id 1
                #ticks = [f"window_id {var1}, point_id {var2}" for var1 in range(int(av.window_length_tst)) for var2 in range(int(av.poi))] # The variables var1 and var2 can vary between 0 and 3
                ticks = [f"c{con_id}_t{tst_id}_d{dim_id}_p{var2}_w{var1}" for var1 in range(int(av.window_length_tst)) for var2 in range(int(av.poi))] # The variables var1 and var2 can vary between 0 and 3

                # Create a colormap with specified colors (red, yellow, green)
                cmap = ListedColormap(["green", "yellow", "red"])
                
                # Set normalization for the colormap with specified boundaries. 
                cNorm  = plt.matplotlib.colors.BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
                
                # Create the heatmap with the specified colormap
                plt.figure(figsize=(11, 8))  # adjust as needed
                #plt.imshow(Df_inequality, cmap=cmap, norm=cNorm, interpolation='none', aspect='equal')
                plt.imshow(Df_inequality, cmap=cmap, norm=cNorm)
                
                # Set the x and y ticks
                #plt.xticks(range(len(ticks)), ticks)
                plt.xticks(range(len(ticks)), ticks, rotation=45, ha='right')  # Rotate labels at 45 degrees and align them to the righ
                plt.yticks(range(len(ticks)), ticks)
                
                plt.grid(which='both', color='white', linestyle='-', linewidth=0)  # Optional: add grid lines

                # Create legend handles manually for the colors we're interested in
                patches = [mpatches.Patch(color=cmap(i), label="{l}".format(l=label)) for i, label in zip(range(3), ['<', '=', '>'])]
                
                # Put those patched as legend-handles into the legend
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Values')
                
                # Save the plot as a PNG image in the "dir" directory
                
                if dim_id == 'x':
                    dim = 0
                elif dim_id == 'y':
                    dim = 1
            
                if av.PDPg_fundamental_active == 1:
                    filename = "N_C_PDPg_fundamental_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"
                elif av.PDPg_buffer_active == 1:
                    filename = "N_C_PDPg_buffer_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"
                elif av.PDPg_rough_active == 1:
                    filename = "N_C_PDPg_rough_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"
                elif av.PDPg_bufferrough_active == 1:
                    filename = "N_C_PDPg_bufferrough_InequalityMatrix_c" + str(con_id) + "_t" + str(tst_id) + "_d" + str(dim) + ".png"

                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()  # close the figure to release memory
            
        new_index = len(Df_con_tst_xineq_yineq)  # Add the index +1 that stores the line in the dataframe

        # Store the list of dataframes in the dictionary with the (con_id, tst_id) as the key
        D_inequality[(con_id, tst_id)] = tuple(L_tst_id_dfs)
        
# Save dataframe "Df_con_tst_xineq_yineq"
Df_con_tst_xineq_yineq.to_csv("Df_con_tst_xineq_yineq.csv", index=False)

#CALCULATE INEQUALITY MATRICES
if av.N_VA_InequalityMatrices == 1:

    # this function will be used to make the dataframes hashable
    def df_to_tuple(df):
        return tuple(map(tuple, df.values))

    # Create a dictionary where the keys are tuples of tuples and the values are lists
    matrix_dict = {}

    for tst_id, df_tuple in D_inequality.items():  # Iterate over D_inequality items
        
        # Convert each DataFrame in the tuple to a tuple of tuples
        df_tuple_hashable = tuple(map(df_to_tuple, df_tuple))
        # Append the tst_id to the list of timestamps tstID for this matrix pair
        if df_tuple_hashable in matrix_dict:
            matrix_dict[df_tuple_hashable].append(tst_id)
        else:
            matrix_dict[df_tuple_hashable] = [tst_id]

    # List to hold output entries
    output_entries = []

    # Populate the output_entries list
    for df_tuple_hashable, tstID in matrix_dict.items():
        # Convert the tuples back to DataFrames for pretty printing
        df_tuple = tuple(pd.DataFrame(df) for df in df_tuple_hashable)
        
        # Create a dictionary for this output entry
        output_entry = {
            'times': len(tstID),
            'tst_id': tstID,
            'x_dimension': df_tuple[0],
            'y_dimension': df_tuple[1]
        }
        
        # Add the output_entry dictionary to the output_entries list
        output_entries.append(output_entry)

    # Sort output_entries in descending order of 'times'
    output_entries.sort(key=lambda x: x['times'], reverse=True)
    
    """
    # Print the sorted output_entries:
    for entry in output_entries:
        if f"{entry['times']}" == "1":
            print(f"{entry['times']} time for (con_id, tst_id) : {' , '.join(map(str, entry['tst_id']))}")
        if f"{entry['times']}" > "1":
            print(f"{entry['times']} times for (con_id, tst_id) : {' , '.join(map(str, entry['tst_id']))}")
    """        

#CALCULATE DISTANCE MATRIX

#FOR X: 
A_rel_distance_matrix_x = np.empty((av.con, av.con))
k = 0
l = 0

for k in range(av.con):  # Loop over all configurations con_id
    if k == 300:
        print(k)
    for l in range(av.con):  # Loop over all configurations con_id
        
        # Add this code to print possible values
        #filtered_rows = Df_con_tst_xineq_yineq[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == 0)]
        #possible_values = filtered_rows['xineqID'].values
        # Retrieve matrices
        #print("Possible values for mat0_x:", possible_values)
        
    #    print(Df_con_tst_xineq_yineq['tstID'])
    #    print(Df_con_tst_xineq_yineq['conID'])
    ##    #print(loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == 0), 'xineqID'])
      #  print(Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == 0), 'xineqID'])
      #  print(Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == 0), 'xineqID'].values[0])

       # mat0_x = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == 0), 'xineqID'].values[0]  # Retrieve inequality matrix of first configuration for dimension x
       # mat1_x = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == l) & (Df_con_tst_xineq_yineq['tstID'] == 0), 'xineqID'].values[0]  # Retrieve inequality matrix of second configuration for dimension x

        # Initialize distances to 0
        abs_distance_x = 0
        rel_distance_x = 0
        nordist = 0

        for tst_id in range(av.tst-(av.window_length_tst-1)):  # Loop over all time stamps, dependant of the window length
            #Loop over each entry in the matrices
            #for i in range(av.poi*av.window_length_tst):
            
            mat0_x = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'xineqID'].values[0]  # Retrieve inequality matrix of first configuration for dimension x
            mat1_x = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == l) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'xineqID'].values[0]  # Retrieve inequality matrix of second configuration for dimension x

            for i in range(int(av.poi*av.window_length_tst)):
                for j in range(int(av.poi*av.window_length_tst)):
                    # Add absolute difference of corresponding entries to distance
                    abs_distance_x += abs(mat0_x[i][j] - mat1_x[i][j])
            # To normalise the distances between 0 a,d 100, we need to devide the distances by the maximum difference and then multiply with 100
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
            #Loop over each entry in the matrices
            
            # Retrieve matrices
            mat0_y = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == k) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'yineqID'].values[0]  # Retrieve inequality matrix of first configuration for dimension y
            mat1_y = Df_con_tst_xineq_yineq.loc[(Df_con_tst_xineq_yineq['conID'] == l) & (Df_con_tst_xineq_yineq['tstID'] == tst_id), 'yineqID'].values[0]  # Retrieve inequality matrix of second configuration for dimension y
            
            for i in range(int(av.poi*av.window_length_tst)):
                for j in range(int(av.poi*av.window_length_tst)):
                    # Add absolute difference of corresponding entries to distance
                    abs_distance_y += abs(mat0_y[i][j] - mat1_y[i][j])
            # To normalise the distances between 0 a,d 100, we need to devide the distances by the maximum difference and then multiply with 100
        #rel_distance_y = int(round (abs_distance_y / ((2*(((av.tst-(av.window_length_tst-1))*(av.poi * av.tst) * (av.poi * av.tst)) - (av.poi * av.tst)))/100), 0))
        rel_distance_y = int(round(abs_distance_y / ((2*(av.tst-(av.window_length_tst-1))*(((av.poi * av.window_length_tst) * (av.poi * av.window_length_tst)) - (av.poi * av.window_length_tst)))/100), 0))



        A_rel_distance_matrix_y[k][l] = rel_distance_y

A_rel_distance_matrix = np.empty((av.con, av.con))
A_rel_distance_matrix = np.round((A_rel_distance_matrix_x + A_rel_distance_matrix_y) / 2).astype(int)

# Initialize each CID as its own set
disjoint_sets = [make_set(i) for i in range(av.con)]

# Loop through the distance matrix to find pairs with distance 0
for i in range(av.con):
    for j in range(i + 1, av.con):  # start from i+1 to avoid duplicate pairs and self-comparison
        if A_rel_distance_matrix[i, j] == 0:
            set_i = find_set(disjoint_sets, i)
            set_j = find_set(disjoint_sets, j)
            
            if set_i is not set_j:
                # Union the sets of i and j
                union(disjoint_sets, set_i, set_j)

# Eliminate duplicate sets and sort them
unique_sets = [sorted(list(s)) for s in disjoint_sets if len(s) > 1]

# Create separate CSV files for each unique set
file_paths = []
for idx, unique_set in enumerate(unique_sets):
    # Filter the original dataframe Df_dataset to only include rows with configuration IDs in the unique set
    filtered_df = Df_dataset[Df_dataset.iloc[:, 0].isin(unique_set)]
    
    # Create a new CSV file for this filtered dataframe in the working directory
    file_path = f"Filtered_Dataset_{idx+1}.csv"
    filtered_df.to_csv(file_path, index=False, header=None)
    file_paths.append(file_path)

file_paths

#file_paths = []
#for idx, unique_set in enumerate(unique_sets):
 #   # Filter the original dataframe Df_dataset to only include rows with configuration IDs in the unique set
  #  filtered_df = Df_dataset[Df_dataset.iloc[:, 0].isin(unique_set)]
    
    # Create a new CSV file for this filtered dataframe
   # file_path = f"/mnt/data/Filtered_Dataset_{idx+1}.csv"
    #filtered_df.to_csv(file_path, index=False, header=None)
    #file_paths.append(file_path)

# Initialize an empty list to store the conversion mappings
conversion_mappings = []

# Initialize an empty list to store the file paths
file_paths = []

# Loop through each unique set to create the filtered datasets and conversion files
for idx, unique_set in enumerate(unique_sets):
    
    # Filter the original dataframe Df_dataset to only include rows with configuration IDs in the unique set
    filtered_df = Df_dataset[Df_dataset['conID'].isin(unique_set)].copy()
    
    ## Filter the original dataframe Df_dataset to only include rows with configuration IDs in the unique set
    #filtered_df = Df_dataset[Df_dataset['conID'].isin(unique_set)]
    
    # Generate the conversion mapping for this unique set
    conversion_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_set)}
    conversion_mappings.append(conversion_mapping)
    
    # Update the 'conID' in the filtered dataframe
    filtered_df['conID'] = filtered_df['conID'].map(conversion_mapping)
    
    # Create a new CSV file for this filtered dataframe in the working directory
    file_path = f"Filtered_Dataset_{idx+1}.csv"
    filtered_df.to_csv(file_path, index=False, header=None)
    file_paths.append(file_path)

# Create a conversion CSV file
conversion_df = pd.DataFrame(conversion_mappings)
conversion_df.to_csv("Conversion_Mapping.csv", index=False)

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
    
with open(filename, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for L_row in A_rel_distance_matrix:
        wr.writerow(L_row.tolist())

# End and print time
print('Time elapsed for running module "N_PDP": {:.3f} sec.'.format(time.time() - t_start))