"""
v220517
FUNCTIONALITY

EXPLANATION
!!!
#*put the Df_con_tst_xineq_yineq of the old configuration in a variable
*create new version of old configuration
-load current configuration
-apply changes to the configuration
--randomly select a number of points (between 1 and av.poi) and the loacation in the fulle set of points
--randomly select a number of time stamps (between 1 and av.tst) and the location in the full set of time stamps
--randomly select a number of dimensions (between 1 and av.dim) and the location in the full set of dimensions
--define the changes based on the above
-save new configuration
*check if the new configuration is similar to the old
-create the Df_con_tst_xineq_yineq of the new configuration
-save the Df_con_tst_xineq_yineq of the new configuration
-check if Df_con_tst_xineq_yineq of the new configuration is similar to the old
--if it is, then the new configuration is saved
--if it is not, then the new configuration is discarded
!!!

INPUT

OUTPUT

POSSIBLE UPGRADES

INPUT PARAMETERS:
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

pd.set_option('display.max_rows', None)  # To show all rows
pd.set_option('display.max_columns', None)  # To show all columns
pd.set_option('display.width', None)  # To remove constraint on the display width
pd.set_option('display.max_colwidth', None)  # To show full width of each column

L_basic_configuration = []

def inequalities(array_of_1_configuration):
    """
    This function takes an array of configuration, processes it, and 
    returns a DataFrame storing the inequalities of the configuration.
    """
    Df_dataset = pd.DataFrame(array_of_1_configuration)
    Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
        
    # Create data structures to store information
    Df_con_tst_xineq_yineq = pd.DataFrame(columns=['conID' , 'tstID', 'xineqID', 'yineqID']) 
    D_inequality = {}  # Initialize an empty dictionary to store DataFrames
    new_index = 0

    group_by_con_id = Df_dataset.groupby('conID') #pre-group your DataFrame by 'conID' so that you can quickly access all rows for a given con_id without having to filter the DataFrame over and over in each iteration of the loop
    
    for con_id in range(av.con):  # Loop over all configurations con_id
        Df_con_id = group_by_con_id.get_group(con_id) # pre-group your DataFrame by 'conID' so that you can quickly access all rows for a given con_id without having to filter the DataFrame over and over in each iteration of the loop

        # Create inequality matrix for each time stamp and each dimension (x and y)
        for tst_id in range(av.tst-(av.window_length_tst-1)):  # Loop over all time stamps, dependant of the window length
               
            # Filter the dataframe for current timestamp tst_id with the current window_length_tst
            conditions = [Df_con_id['tstID'] == tst_id + i for i in range(av.window_length_tst)]
                
             #print(conditions)
            
            mask = np.logical_or.reduce(conditions)
            
            Df_tst_id = Df_con_id[mask]
            #print(Df_tst_id)

            i = 0
            j = 0

            L_tst_id_dfs = []  # Create a list to hold the dataframes for each dimension
            for dim_id in ['x', 'y']:  # Loop over all dimensions (x and y)            
                A_inequality_matrix = np.zeros((int(av.poi*av.window_length_tst), int(av.poi*av.window_length_tst))) # Create an empty matrix of appropriate size
                #for i in range(av.poi*av.window_length_tst):  
                for i in range(int(av.poi*av.window_length_tst)): # Loop over all points, dependant of the window length
                    for j in range(int(av.poi*av.window_length_tst)):  # Loop over all points, dependant of the window length
                        #print("")
                        #print(Df_tst_id[dim_id].iloc[i])
                        #print(Df_tst_id[dim_id].iloc[j])
                        #print("")
                        if Df_tst_id[dim_id].iloc[i] > Df_tst_id[dim_id].iloc[j]:
                            A_inequality_matrix[i, j] = 2
                        elif Df_tst_id[dim_id].iloc[i] < Df_tst_id[dim_id].iloc[j]:
                            A_inequality_matrix[i, j] = 0
                        else:
                            A_inequality_matrix[i, j] = 1
                # Add a line to the dataframe containing all configurations conID, all time stamps ID and all inequality arrays for each dimension
                # new_index = len(Df_con_tst_xineq_yineq)
                Df_con_tst_xineq_yineq.at[new_index, 'conID'] = con_id
                #print(Df_con_tst_xineq_yineq)
                Df_con_tst_xineq_yineq.at[new_index, 'tstID'] = tst_id
                #print(Df_con_tst_xineq_yineq)
                if dim_id == "x":
                    Df_con_tst_xineq_yineq.at[new_index, 'xineqID'] = A_inequality_matrix
                    #print(Df_con_tst_xineq_yineq)
                elif dim_id == "y":
                    Df_con_tst_xineq_yineq.at[new_index, 'yineqID'] = A_inequality_matrix
                    #print(Df_con_tst_xineq_yineq)

                Df_inequality = pd.DataFrame(A_inequality_matrix)  # Convert the numpy array to a dataframe for better visualization
                    
                L_tst_id_dfs.append(Df_inequality)  # Add the dataframe to the list  
                    
            new_index = len(Df_con_tst_xineq_yineq)  # Add the index +1 that stores the line in the dataframe
                
    # Save dataframe "Df_con_tst_xineq_yineq"
    Df_con_tst_xineq_yineq.to_csv("Df_con_tst_xineq_yineq.csv", index=False)

    return Df_con_tst_xineq_yineq

def modify_selected_point(selected_point, av_dim):
    # Function to modify a selected point based on the specified conditions
    # Copy the selected point to avoid modifying the original
    new_point = list(selected_point)
        
    # Always modify column 3
    new_point[3] += round(random.uniform(T_min, T_max), 2)

    # If av.dim = 2, also modify column 4
    if av_dim == 2:
        new_point[4] += round(random.uniform(T_min, T_max), 2)
            
    # If av.dim = 3, also modify column 5
    if av_dim == 3:
        new_point[5] += round(random.uniform(T_min, T_max), 2)

    return new_point

def load_and_check_configuration(file_name):
    """
    Load the configuration from a CSV file and check if the first column
    of every line contains only "0" values. Also, load the configuration into
    a numpy array.

    :param file_name: The path to the CSV file to be loaded.
    :return: A tuple containing a flag indicating if the first column is only zeros,
             the list of basic configurations, and the numpy array of the configuration.
    """
    # Initialize necessary variables
    is_first_column_only_zero = True  # Initialize the flag to True
    #L_basic_configuration = []
    D_poi_mapping = {}
    cur_poi_id = 0

    # Check if the first column of every line in a CSV file contains only "0" values.
    # If a non-"0" value is found, it sets a flag to False and stops processing the file.
    with open(file_name, 'r') as file:
        for line in file:
            # Split the line by comma and check the first element
            if line.split(',')[0].strip().replace('"','') != "0":
                is_first_column_only_zero = False
                break  # Exit the loop if any non-"0" is found

    # Load the configuration 
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            if av.dim == -1:
                dim = len(L_row) - 3
            # Check if poi_id is a string, if it is, map to int
            try:
                int(poi_id)
            except ValueError:
                if poi_id not in D_poi_mapping:
                    D_poi_mapping[poi_id] = cur_poi_id
                    cur_poi_id += 1
                L_row[2] = D_poi_mapping[poi_id]
            L_basic_configuration.append(list(map(float, L_row)))
    A_basic_configuration = np.array(L_basic_configuration, dtype=np.float32) # Transform list to array

    return is_first_column_only_zero, L_basic_configuration, A_basic_configuration

# Set display options to a manageable number
pd.set_option('display.max_rows', 500)  # Display up to 500 rows
pd.set_option('display.max_columns', 500)  # Display up to 500 columns

# Start time
t_start = time.time()

# Define the threshold T as a range between T_min and T_max
T_min, T_max = av.min_boundary_x, av.max_boundary_x  # Example values, adjust according to your needs

# Load configuration
file_name = 'N_C_Dataset.csv'
#is_zero_only, basic_config_list, config_array = load_and_check_configuration(file_name)
is_first_column_only_zero, L_basic_configuration, A_basic_configuration = load_and_check_configuration(file_name)

# Create the inverse dataset
teller = 0
for configuration in range(av.num_similar_configurations*av.new_configuration_step):
    teller += 1    
    #-apply changes to the configuration
    # Determine the row to select in the configuration
    row_in_con = random.randint(1, av.poi*av.tst) - 1
    # Select the point for that row
    selected_point = L_basic_configuration[row_in_con]

    # Modify the selected point based on av.dim
    new_selected_point = modify_selected_point(selected_point, av.dim)

    # Assuming A_basic_configuration is a list of lists
    A_new_configuration = copy.deepcopy(A_basic_configuration)
    print()
    print(A_basic_configuration)    
    print()
    print(A_new_configuration)    
    print()

    # Replace the specific row with new_point
    A_new_configuration[row_in_con] = new_selected_point
    if av.dim == 2:
        #as long as outside of the boundaries 
        while A_new_configuration[row_in_con, 3]<av.min_boundary_x or A_new_configuration[row_in_con, 3]>av.max_boundary_x or A_new_configuration[row_in_con, 4]<av.min_boundary_y or A_new_configuration[row_in_con, 4]>av.max_boundary_y:
            # Assuming A_new_configuration and A_basic_configuration are numpy arrays
            old_x_difference = A_new_configuration[row_in_con, 3] - A_basic_configuration[row_in_con, 3]
            new_x_difference = old_x_difference/2
            A_new_configuration[row_in_con, 3] = round(A_basic_configuration[row_in_con, 3] + new_x_difference, 2)
            old_y_difference = A_new_configuration[row_in_con, 4] - A_basic_configuration[row_in_con, 4]
            new_y_difference = old_y_difference/2
            A_new_configuration[row_in_con, 4] = round(A_basic_configuration[row_in_con, 4] + new_y_difference, 2)
            print()
            print(A_basic_configuration)    
            print()
            print(A_new_configuration)    
            print()
        print()
        print(A_basic_configuration)  
        print()  
        print(A_new_configuration)    
        print()
    else:
        print("Adapt the code")
    
    while True:
        print()
        print(A_basic_configuration)  
        print()  
        print(A_new_configuration)    
        print()

        result = inequalities(A_new_configuration)
        N_PDP.Df_con_tst_xineq_yineq.to_csv("basic_inequalities_representation.csv", index=False)
        result.to_csv("new_inequalities_representation.csv", index=False)
        Df_new_xineq_yineq=result

        #add a similar configuration
        #print()
        #print("BASIC INEQUALITY: ")
        #print(str(N_PDP.Df_con_tst_xineq_yineq))
        #print("BASIC CONFIGURATION: ")
        #print(A_basic_configuration) 
        #print()
        #print("NEW INEQUALITY: ")
        #print(str(Df_new_xineq_yineq))
        #print("NEW CONFIGURATION: ")
        #print(A_new_configuration) 
        if N_PDP.Df_con_tst_xineq_yineq.equals(Df_new_xineq_yineq): #if the new generated configuration is similar to the basic configuration
            #print (N_PDP.Df_con_tst_xineq_yineq)
            #print (Df_new_xineq_yineq)
            if configuration == 0:
                # Define the source and destination file paths
                source_file_path = 'N_C_Dataset.csv'
                destination_file_path = 'N_C_similar_configurations.csv'
                # Copy the file
                shutil.copy(source_file_path, destination_file_path)
                # Load the existing dataset from N_C_similar_configurations.csv
                #df_ataset = pd.read_csv("N_C_similar_configurations.csv")
                with open("N_C_similar_configurations.csv", 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    data = list(reader)
                df_dataset = pd.DataFrame(data[1:], columns=data[0])
                
            # Convert A_new_configuration to a DataFrame
            df_A_new_configuration = pd.DataFrame(A_new_configuration, columns=df_dataset.columns)
            df_A_new_configuration = df_A_new_configuration.round(2)
            #print(df_A_new_configuration)

            # Append the new DataFrame to df_dataset
            if teller % av.new_configuration_step == 0:
                df_dataset = df_dataset.append(df_A_new_configuration, ignore_index=True)
            #print(df_dataset)
            # Save the updated dataset back to N_C_Dataset.csv
            #the following line has to be done only after the step length
                df_dataset.to_csv("N_C_similar_configurations.csv", index=False)
            #print("Added similar configurations to the dataset")

            break  # Exit the loop
        else: #if the new generated configuration is not similar to the basic configuration
            old_x_difference = A_new_configuration[row_in_con, 3] - A_basic_configuration[row_in_con, 3]
            new_x_difference = old_x_difference/av.division_factor
            A_new_configuration[row_in_con, 3] = round(A_basic_configuration[row_in_con, 3] + new_x_difference, 2)
            old_y_difference = A_new_configuration[row_in_con, 4] - A_basic_configuration[row_in_con, 4]
            new_y_difference = old_y_difference/av.division_factor
            A_new_configuration[row_in_con, 4] = round(A_basic_configuration[row_in_con, 4] + new_y_difference, 2)
            result = inequalities(A_new_configuration)
            Df_new_xineq_yineq=result
            print()
            print(A_basic_configuration) 
            print()   
            print(A_new_configuration)    
            print() 
    # Before the next iteration, update basic configuration with the new one
    A_basic_configuration = A_new_configuration.copy()  # Important to use copy if dealing with complex data structures
    print()
    print(A_basic_configuration) 
    print()   
    print(A_new_configuration)    
    print()

# Change the datatypes of the filevalues.
file_name = 'N_C_similar_configurations.csv' #load current configuration
# Temporarily store the modified rows
modified_rows = []
loop_counter = 0
if file_name is not None:
    # Open and read the CSV file
    with open(file_name, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # Convert the first three columns to integers
            row[:3] = [int(float(value)) for value in row[:3]]
            # Convert the first columns to the value of the configuration
            #!!!! row in this is still not ok. This must become the variable presenting the configuration.
            row[0] = loop_counter // (av.tst * av.poi)
            #row[:1] = [row for value in row[:1]]
            # Convert the next two columns to floats with two decimal places
            row[3:5] = [round(float(value), 2) for value in row[3:5]]
            # Add the modified row to the list
            modified_rows.append(row)
            loop_counter += 1

    # Write the modified data back into the CSV file
    with open(file_name, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(modified_rows)

print(f'File "{file_name}" has been updated successfully.')

print()