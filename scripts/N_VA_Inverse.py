"""
!!then rough checken for inverse: sits in a never ending loop... why?

This is a module that is not really well integrated in the others.

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
    returns a DataFrame storing the inequalities of the configuration. Depending on the PDPg setting different code will be run
    """
    if ((av.PDPg_fundamental_active == 1) or (av.PDPg_rough_active == 1) or (av.PDPg_bufferrough_active == 1)):  # If the inverse dataset needs to be calculated for the fundamental PDPg
        
        # Initialize dataset and results DataFrame
        Df_dataset = pd.DataFrame(array_of_1_configuration, columns=['conID', 'tstID', 'poiID', 'x', 'y'])
        Df_con_tst_xineq_yineq = pd.DataFrame(columns=['conID', 'tstID', 'xineqID', 'yineqID']) 
        new_index = 0
        rough = 0
        if av.PDPg_rough == 1: 
            rough = av.rough
        if av.PDPg_bufferrough == 1: 
            rough = av.rough

        # Group dataset by 'conID' for efficient lookup
        group_by_con_id = Df_dataset.groupby('conID') 
        
        for con_id in range(av.con):  # Loop over all configurations con_id
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

                    # Store results in DataFrame
                    Df_con_tst_xineq_yineq.at[new_index, 'conID'] = con_id
                    Df_con_tst_xineq_yineq.at[new_index, 'tstID'] = tst_id
                    if dim_id == "x":
                        Df_con_tst_xineq_yineq.at[new_index, 'xineqID'] = A_inequality_matrix
                    elif dim_id == "y":
                        Df_con_tst_xineq_yineq.at[new_index, 'yineqID'] = A_inequality_matrix

                    Df_inequality = pd.DataFrame(A_inequality_matrix)  # Convert the numpy array to a dataframe for better visualization
                        
                    L_tst_id_dfs.append(Df_inequality)  # Add the dataframe to the list  
                        
                new_index = len(Df_con_tst_xineq_yineq)  # Add the index +1 that stores the line in the dataframe
                    
        # Save dataframe "Df_con_tst_xineq_yineq"
        Df_con_tst_xineq_yineq.to_csv("Df_con_tst_xineq_yineq.csv", index=False)

        return Df_con_tst_xineq_yineq    
    
def modify_selected_point(selected_point, av_dim):
    # Function to modify the selected point based on the specified conditions
    # Copy the selected point to avoid modifying the original

    random_value = random.randint(0, 1)
    new_point = list(selected_point)
        
    # Always modify column 3
    if random_value == 0:
        new_point[3] -= round(random.uniform(T_min_X, T_max_X), 2)
    elif random_value == 1:
        new_point[3] += round(random.uniform(T_min_X, T_max_X), 2)
    
    # If av.dim = 2, also modify column 4
    if av_dim == 2:
    #    new_point[4] += round(random.uniform(T_min_Y, T_max_Y), 2)
        if random_value == 0:
            new_point[4] -= round(random.uniform(T_min_Y, T_max_Y), 2)
        elif random_value == 1:
            new_point[4] += round(random.uniform(T_min_Y, T_max_Y), 2)

    # If av.dim = 3, also modify column 5
    if av_dim == 3:
    #    new_point[5] += round(random.uniform(T_min_Z, T_max_Z), 2)
        if random_value == 0:
            new_point[5] -= round(random.uniform(T_min_Z, T_max_Z), 2)
        elif random_value == 1:
            new_point[5] += round(random.uniform(T_min_Z, T_max_Z), 2)

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
T_min_X, T_max_X = av.min_boundary_x, av.max_boundary_x  # Example values, adjust according to your needs
T_min_Y, T_max_Y = av.min_boundary_y, av.max_boundary_y  # Example values, adjust according to your needs

# Load configuration
file_name = 'N_C_Dataset.csv'
is_first_column_only_zero, L_basic_configuration, A_basic_configuration = load_and_check_configuration(file_name)

# Create the inverse dataset
teller = 0

if av.PDPg_fundamental_active == 1:  # If the inverse dataset needs to be calculated for the fundamental PDPg
    for configuration in range(av.num_similar_configurations*av.new_configuration_step): 
        total = (av.num_similar_configurations*av.new_configuration_step) - 1
        print(f'Calculating configuration {configuration} of {total} configurations')
        teller += 1     
        row_in_con = random.randint(1, av.poi*av.tst) - 1  # Randomly select a row in the configuration
        selected_point = L_basic_configuration[row_in_con]   # Select the point for that row
        new_selected_point = modify_selected_point(selected_point, av.dim)   # Modify the selected point based on av.dim

        # Create a new configuration
        A_new_configuration = copy.deepcopy(A_basic_configuration) 
        A_new_configuration[row_in_con] = new_selected_point

        if av.dim == 2:
            # Adjust while out of bounds
            while (A_new_configuration[row_in_con, 3]<(av.min_boundary_x-0.1) or A_new_configuration[row_in_con, 3]>(av.max_boundary_x+0.1) or A_new_configuration[row_in_con, 4]<(av.min_boundary_y-0.1) or A_new_configuration[row_in_con, 4]>(av.max_boundary_y+0.1)): #if new coordinate outside "boundary+ 0.01"
                # Assuming A_new_configuration and A_basic_configuration are numpy arrays
                old_x_difference = A_new_configuration[row_in_con, 3] - A_basic_configuration[row_in_con, 3]
                new_x_difference = old_x_difference/2
                A_new_configuration[row_in_con, 3] = round(A_basic_configuration[row_in_con, 3] + new_x_difference, 5)
                old_y_difference = A_new_configuration[row_in_con, 4] - A_basic_configuration[row_in_con, 4]
                new_y_difference = old_y_difference/2
                A_new_configuration[row_in_con, 4] = round(A_basic_configuration[row_in_con, 4] + new_y_difference, 5)
        else:
            print("Adapt the code")      
        while True:
            result = inequalities(A_new_configuration)
            N_PDP.Df_con_tst_xineq_yineq.to_csv("basic_inequalities_representation.csv", index=False)
            result.to_csv("new_inequalities_representation.csv", index=False)
            Df_new_xineq_yineq=result
            #print (N_PDP.Df_con_tst_xineq_yineq)
            #print (Df_new_xineq_yineq)
            if N_PDP.Df_con_tst_xineq_yineq.equals(Df_new_xineq_yineq): #if the new generated configuration is similar to the basic configuration
                if configuration == 0:
                    # Define the source and destination file paths
                    source_file_path = 'N_C_Dataset.csv'
                    destination_file_path = 'N_C_similar_configurations.csv'
                    # Copy the file
                    shutil.copy(source_file_path, destination_file_path)
                    # Load the existing dataset from N_C_similar_configurations.csv
                    with open("N_C_similar_configurations.csv", 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        data = list(reader)
                    df_dataset = pd.DataFrame(data[1:], columns=data[0])
                # Convert A_new_configuration to a DataFrame
                df_A_new_configuration = pd.DataFrame(A_new_configuration, columns=df_dataset.columns)
                df_A_new_configuration = df_A_new_configuration.round(2)
                # Append the new DataFrame to df_dataset
                if teller % av.new_configuration_step == 0:
                    df_dataset = pd.concat([df_dataset, df_A_new_configuration], ignore_index=True)
                # Save the updated dataset back to N_C_Dataset.csv
                #the following line has to be done only after the step length
                    df_dataset.to_csv("N_C_similar_configurations.csv", index=False)
                break  # Exit the loop
            else: #if the new generated configuration is not similar to the basic configuration
                old_x_difference = A_new_configuration[row_in_con, 3] - A_basic_configuration[row_in_con, 3]
                old_y_difference = A_new_configuration[row_in_con, 4] - A_basic_configuration[row_in_con, 4]
                if ((abs(old_x_difference) < 0.1) or (abs(old_y_difference) < 0.1)) :   # If the distance between new and old is extremely small            
                    if configuration == 0:
                        # Define the source and destination file paths
                        source_file_path = 'N_C_Dataset.csv'
                        destination_file_path = 'N_C_similar_configurations.csv'
                        # Copy the file
                        shutil.copy(source_file_path, destination_file_path)
                        # Load the existing dataset from N_C_similar_configurations.csv
                        with open("N_C_similar_configurations.csv", 'r') as csv_file:
                            reader = csv.reader(csv_file)
                            data = list(reader)
                        df_dataset = pd.DataFrame(data[1:], columns=data[0])
                    A_new_configuration = A_basic_configuration   # Keep the basic configuration
                    # Convert A_new_configuration to a DataFrame
                    df_A_new_configuration = pd.DataFrame(A_new_configuration, columns=df_dataset.columns)
                    df_A_new_configuration = df_A_new_configuration.round(2)
                    # Append the new DataFrame to df_dataset
                    if teller % av.new_configuration_step == 0:
                        df_dataset = pd.concat([df_dataset, df_A_new_configuration], ignore_index=True)
                    # Save the updated dataset back to N_C_Dataset.csv
                    #the following line has to be done only after the step length
                        df_dataset.to_csv("N_C_similar_configurations.csv", index=False)
                    break  # Exit the loop                    
                    
                    
                    
                    
                    
                #    result = inequalities(A_basic_configuration)   # Keep the old, thus keep the previous configuration
                #    print (N_PDP.Df_con_tst_xineq_yineq)
                #    print (Df_new_xineq_yineq)
                
                
                
                
                
                else:
                    new_x_difference = old_x_difference/av.division_factor
                    new_y_difference = old_y_difference/av.division_factor
                    A_new_configuration[row_in_con, 3] = round(A_basic_configuration[row_in_con, 3] + new_x_difference, 2)
                    A_new_configuration[row_in_con, 4] = round(A_basic_configuration[row_in_con, 4] + new_y_difference, 2)
                    result = inequalities(A_new_configuration)
                    Df_new_xineq_yineq=result
                    #print (N_PDP.Df_con_tst_xineq_yineq)
                    #print (Df_new_xineq_yineq)
        # Before the next iteration, update basic configuration with the new one
        A_basic_configuration = A_new_configuration.copy()  # Important to use copy if dealing with complex data structures


if av.PDPg_buffer_active == 1:  # If the inverse dataset needs to be calculated for the buffer PDPg
    print("inverse buffer needs to be implemented")
if av.PDPg_rough_active == 1:  # If the inverse dataset needs to be calculated for the rough PDPg
    print("inverse rough needs to be implemented")

    for configuration in range(av.num_similar_configurations*av.new_configuration_step): 
        total = (av.num_similar_configurations*av.new_configuration_step) - 1
        print(f'Calculating configuration {configuration} of {total} configurations')
        teller += 1     
        row_in_con = random.randint(1, av.poi*av.tst) - 1  # Randomly select a row in the configuration
        selected_point = L_basic_configuration[row_in_con]   # Select the point for that row
        new_selected_point = modify_selected_point(selected_point, av.dim)   # Modify the selected point based on av.dim

        # Create a new configuration
        A_new_configuration = copy.deepcopy(A_basic_configuration) 
        A_new_configuration[row_in_con] = new_selected_point

        if av.dim == 2:
            # Adjust while out of bounds
            while (A_new_configuration[row_in_con, 3]<(av.min_boundary_x-0.1) or A_new_configuration[row_in_con, 3]>(av.max_boundary_x+0.1) or A_new_configuration[row_in_con, 4]<(av.min_boundary_y-0.1) or A_new_configuration[row_in_con, 4]>(av.max_boundary_y+0.1)): #if new coordinate outside "boundary+ 0.01"
                # Assuming A_new_configuration and A_basic_configuration are numpy arrays
                old_x_difference = A_new_configuration[row_in_con, 3] - A_basic_configuration[row_in_con, 3]
                new_x_difference = old_x_difference/2
                A_new_configuration[row_in_con, 3] = round(A_basic_configuration[row_in_con, 3] + new_x_difference, 5)
                old_y_difference = A_new_configuration[row_in_con, 4] - A_basic_configuration[row_in_con, 4]
                new_y_difference = old_y_difference/2
                A_new_configuration[row_in_con, 4] = round(A_basic_configuration[row_in_con, 4] + new_y_difference, 5)
        else:
            print("Adapt the code")      
        while True:
            result = inequalities(A_new_configuration)
            N_PDP.Df_con_tst_xineq_yineq.to_csv("basic_inequalities_representation.csv", index=False)
            result.to_csv("new_inequalities_representation.csv", index=False)
            Df_new_xineq_yineq=result
            #print (N_PDP.Df_con_tst_xineq_yineq)
            #print (Df_new_xineq_yineq)
            if N_PDP.Df_con_tst_xineq_yineq.equals(Df_new_xineq_yineq): #if the new generated configuration is similar to the basic configuration
                if configuration == 0:
                    # Define the source and destination file paths
                    source_file_path = 'N_C_Dataset.csv'
                    destination_file_path = 'N_C_similar_configurations.csv'
                    # Copy the file
                    shutil.copy(source_file_path, destination_file_path)
                    # Load the existing dataset from N_C_similar_configurations.csv
                    with open("N_C_similar_configurations.csv", 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        data = list(reader)
                    df_dataset = pd.DataFrame(data[1:], columns=data[0])
                # Convert A_new_configuration to a DataFrame
                df_A_new_configuration = pd.DataFrame(A_new_configuration, columns=df_dataset.columns)
                df_A_new_configuration = df_A_new_configuration.round(2)
                # Append the new DataFrame to df_dataset
                if teller % av.new_configuration_step == 0:
                    df_dataset = pd.concat([df_dataset, df_A_new_configuration], ignore_index=True)
                # Save the updated dataset back to N_C_Dataset.csv
                #the following line has to be done only after the step length
                    df_dataset.to_csv("N_C_similar_configurations.csv", index=False)
                break  # Exit the loop
            else: #if the new generated configuration is not similar to the basic configuration
                old_x_difference = A_new_configuration[row_in_con, 3] - A_basic_configuration[row_in_con, 3]
                old_y_difference = A_new_configuration[row_in_con, 4] - A_basic_configuration[row_in_con, 4]
                if ((abs(old_x_difference) < 0.1) or (abs(old_y_difference) < 0.1)) :   # If the distance between new and old is extremely small            
                    A_new_configuration = A_basic_configuration   # Keep the basic configuration
                    # Convert A_new_configuration to a DataFrame
                    df_A_new_configuration = pd.DataFrame(A_new_configuration, columns=df_dataset.columns)
                    df_A_new_configuration = df_A_new_configuration.round(2)
                    # Append the new DataFrame to df_dataset
                    if teller % av.new_configuration_step == 0:
                        df_dataset = pd.concat([df_dataset, df_A_new_configuration], ignore_index=True)
                    # Save the updated dataset back to N_C_Dataset.csv
                    #the following line has to be done only after the step length
                        df_dataset.to_csv("N_C_similar_configurations.csv", index=False)
                    break  # Exit the loop                    
                    
                    
                    
                    
                    
                #    result = inequalities(A_basic_configuration)   # Keep the old, thus keep the previous configuration
                #    print (N_PDP.Df_con_tst_xineq_yineq)
                #    print (Df_new_xineq_yineq)
                
                
                
                
                
                else:
                    new_x_difference = old_x_difference/av.division_factor
                    new_y_difference = old_y_difference/av.division_factor
                    A_new_configuration[row_in_con, 3] = round(A_basic_configuration[row_in_con, 3] + new_x_difference, 2)
                    A_new_configuration[row_in_con, 4] = round(A_basic_configuration[row_in_con, 4] + new_y_difference, 2)
                    result = inequalities(A_new_configuration)
                    Df_new_xineq_yineq=result
                    #print (N_PDP.Df_con_tst_xineq_yineq)
                    #print (Df_new_xineq_yineq)
        # Before the next iteration, update basic configuration with the new one
        A_basic_configuration = A_new_configuration.copy()  # Important to use copy if dealing with complex data structures






if av.PDPg_bufferrough_active == 1:  # If the inverse dataset needs to be calculated for the bufferrough PDPg
    print("inverse bufferrough needs to be implemented")

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