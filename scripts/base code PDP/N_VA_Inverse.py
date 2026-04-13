"""
v220517
FUNCTIONALITY
    Generates similar configurations based on modifying existing ones
EXPLANATION
    Creates new configurations by modifying existing ones and checking if they 
    maintain the same inequality structure (inverse problem approach)
INPUT
    N_C_Dataset.csv - The base configuration dataset
OUTPUT
    N_C_similar_configurations.csv - Dataset with similar configurations
"""

import av
import copy
import csv
import numpy as np
import N_PDP
import pandas as pd
import random
import shutil
import time

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

L_basic_configuration = []

def inequalities(array_of_1_configuration):
    """
    This function takes an array of configuration, processes it, and 
    returns a DataFrame storing the inequalities of the configuration.
    """
    Df_dataset = pd.DataFrame(array_of_1_configuration)
    Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
        
    # Create data structures to store information
    Df_con_tst_xineq_yineq = pd.DataFrame(columns=['conID', 'tstID', 'xineqID', 'yineqID']) 
    D_inequality = {}
    new_index = 0

    group_by_con_id = Df_dataset.groupby('conID')
    
    for con_id in range(av.con):
        Df_con_id = group_by_con_id.get_group(con_id)

        for tst_id in range(av.tst-(av.window_length_tst-1)):
            conditions = [Df_con_id['tstID'] == tst_id + i for i in range(av.window_length_tst)]
            mask = np.logical_or.reduce(conditions)
            Df_tst_id = Df_con_id[mask]

            L_tst_id_dfs = []
            for dim_id in ['x', 'y']:
                A_inequality_matrix = np.zeros((int(av.poi*av.window_length_tst), int(av.poi*av.window_length_tst)))
                for i in range(int(av.poi*av.window_length_tst)):
                    for j in range(int(av.poi*av.window_length_tst)):
                        if Df_tst_id[dim_id].iloc[i] > Df_tst_id[dim_id].iloc[j]:
                            A_inequality_matrix[i, j] = 2
                        elif Df_tst_id[dim_id].iloc[i] < Df_tst_id[dim_id].iloc[j]:
                            A_inequality_matrix[i, j] = 0
                        else:
                            A_inequality_matrix[i, j] = 1
                            
                Df_con_tst_xineq_yineq.at[new_index, 'conID'] = con_id
                Df_con_tst_xineq_yineq.at[new_index, 'tstID'] = tst_id
                if dim_id == "x":
                    Df_con_tst_xineq_yineq.at[new_index, 'xineqID'] = A_inequality_matrix
                elif dim_id == "y":
                    Df_con_tst_xineq_yineq.at[new_index, 'yineqID'] = A_inequality_matrix

                Df_inequality = pd.DataFrame(A_inequality_matrix)
                L_tst_id_dfs.append(Df_inequality)
                    
            new_index = len(Df_con_tst_xineq_yineq)
                
    Df_con_tst_xineq_yineq.to_csv(av.get_output_path("Df_con_tst_xineq_yineq.csv"), index=False)
    return Df_con_tst_xineq_yineq

def modify_selected_point(selected_point, av_dim):
    """
    Function to modify a selected point based on the specified conditions.
    """
    new_point = list(selected_point)
        
    # Always modify column 3 (x)
    new_point[3] += round(random.uniform(T_min, T_max), 2)

    # If av.dim = 2, also modify column 4 (y)
    if av_dim == 2:
        new_point[4] += round(random.uniform(T_min, T_max), 2)
            
    # If av.dim = 3, also modify column 5
    if av_dim == 3:
        new_point[5] += round(random.uniform(T_min, T_max), 2)

    return new_point

def load_and_check_configuration(file_name):
    """
    Load the configuration from a CSV file and check if the first column
    of every line contains only "0" values.
    """
    is_first_column_only_zero = True
    D_poi_mapping = {}
    cur_poi_id = 0

    with open(file_name, 'r') as file:
        for line in file:
            if line.split(',')[0].strip().replace('"','') != "0":
                is_first_column_only_zero = False
                break

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            if av.dim == -1:
                dim = len(L_row) - 3
            try:
                int(poi_id)
            except ValueError:
                if poi_id not in D_poi_mapping:
                    D_poi_mapping[poi_id] = cur_poi_id
                    cur_poi_id += 1
                L_row[2] = D_poi_mapping[poi_id]
            L_basic_configuration.append(list(map(float, L_row)))
    A_basic_configuration = np.array(L_basic_configuration, dtype=np.float32)

    return is_first_column_only_zero, L_basic_configuration, A_basic_configuration

# Set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Start time
t_start = time.time()

# Define the threshold T as a range between T_min and T_max
T_min, T_max = av.min_boundary_x, av.max_boundary_x

# Load configuration
file_name = 'N_C_Dataset.csv'
is_first_column_only_zero, L_basic_configuration, A_basic_configuration = load_and_check_configuration(file_name)

# Create the inverse dataset
teller = 0
for configuration in range(av.num_similar_configurations * av.new_configuration_step):
    teller += 1    
    
    # Determine the row to select in the configuration
    row_in_con = random.randint(1, av.poi * av.tst) - 1
    selected_point = L_basic_configuration[row_in_con]

    # Modify the selected point based on av.dim
    new_selected_point = modify_selected_point(selected_point, av.dim)

    A_new_configuration = copy.deepcopy(A_basic_configuration)

    # Replace the specific row with new_point
    A_new_configuration[row_in_con] = new_selected_point
    
    if av.dim == 2:
        # As long as outside of the boundaries
        while (A_new_configuration[row_in_con, 3] < av.min_boundary_x or 
               A_new_configuration[row_in_con, 3] > av.max_boundary_x or 
               A_new_configuration[row_in_con, 4] < av.min_boundary_y or 
               A_new_configuration[row_in_con, 4] > av.max_boundary_y):
            old_x_difference = A_new_configuration[row_in_con, 3] - A_basic_configuration[row_in_con, 3]
            new_x_difference = old_x_difference / 2
            A_new_configuration[row_in_con, 3] = round(A_basic_configuration[row_in_con, 3] + new_x_difference, 2)
            old_y_difference = A_new_configuration[row_in_con, 4] - A_basic_configuration[row_in_con, 4]
            new_y_difference = old_y_difference / 2
            A_new_configuration[row_in_con, 4] = round(A_basic_configuration[row_in_con, 4] + new_y_difference, 2)
    else:
        print("Adapt the code for dimensions other than 2")
    
    while True:
        result = inequalities(A_new_configuration)
        N_PDP.Df_con_tst_xineq_yineq.to_csv(av.get_output_path("basic_inequalities_representation.csv"), index=False)
        result.to_csv(av.get_output_path("new_inequalities_representation.csv"), index=False)
        Df_new_xineq_yineq = result

        # Check if the new generated configuration is similar to the basic configuration
        if N_PDP.Df_con_tst_xineq_yineq.equals(Df_new_xineq_yineq):
            if configuration == 0:
                source_file_path = 'N_C_Dataset.csv'
                destination_file_path = av.get_output_path('N_C_similar_configurations.csv')
                shutil.copy(source_file_path, destination_file_path)
                with open(destination_file_path, 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    data = list(reader)
                df_dataset = pd.DataFrame(data[1:], columns=data[0])
                
            # Convert A_new_configuration to a DataFrame
            df_A_new_configuration = pd.DataFrame(A_new_configuration, columns=df_dataset.columns)
            df_A_new_configuration = df_A_new_configuration.round(2)

            # Append the new DataFrame to df_dataset (using pd.concat instead of deprecated append)
            if teller % av.new_configuration_step == 0:
                df_dataset = pd.concat([df_dataset, df_A_new_configuration], ignore_index=True)
                df_dataset.to_csv(av.get_output_path("N_C_similar_configurations.csv"), index=False)
            break  # Exit the while loop once a similar configuration is found

# End and print time
print('Time elapsed for running module "N_VA_Inverse": {:.3f} sec.'.format(time.time() - t_start))
