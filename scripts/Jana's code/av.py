"""
!!!Df_dataset, dataset_name, dataset_name_exclusive, L_dataset, A_dataset, con, tst, poi are those in the active setting, if nothing is active, then it is the original. The original has to be saved as nothing or original and the other active serttings as their active setting.
"""

# Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI threading issues

# Import necessary libraries
import csv  # For reading and writing csv files
import numpy as np  # For numerical calculations
import os  # For file handling
import pandas as pd  # For data manipulation
import time  # For timing the code
import tkinter as tk  # For creating a graphical user interface
from tkinter import ttk  # For creating a graphical user interface

# Record the start time of the script
t_start = time.time()

# Environment flag: if set to '1' the heavy dataset loading / GUI mainloop will be skipped.
# This allows external GUIs to import this module for defaults without triggering file I/O.
AV_SKIP_LOAD = os.environ.get('AV_SKIP_LOAD', '0')

# Function to update GUI settings and close the window
def update_settings():   
    """
    Updates global settings based on GUI input and closes the GUI.
    """
    global PDPg_fundamental, PDPg_buffer, PDPg_rough, PDPg_bufferrough
    PDPg_fundamental = fundamental_var.get()
    PDPg_buffer = buffer_var.get()
    PDPg_rough = rough_var.get()  
    PDPg_bufferrough = bufferrough_var.get()   
    root.destroy()  # Close the window

# Activation of GUI
graphical_user_interface = 0 

# !!! Default setting for PDP types; this has to be changed to say which PDP to calculate.
PDPg_fundamental = 1 # !!!must always be 1 , so this always has to be calculated.
PDPg_buffer = 0  
PDPg_rough = 0 
PDPg_bufferrough = 0

# Set the parameters for the calculations to be included in the report
N_VA_Report = 1
N_VA_StaticAbsolute = 1
N_VA_StaticRelative = 0
N_VA_StaticFinetuned = 0
N_VA_DynamicAbsolute = 0  #have to doublecheck this code
N_PDP = 1  # The calculation of PDP and matrices  
N_VA_InequalityMatrices = 0
N_VA_HeatMap = 0
N_VA_HClust = 0
N_VA_ClusterMap = 0
N_VA_Mds = 0
N_VA_TSNE = 0
N_VA_UMAP = 0
N_VA_Mds_autoencoder = 0
N_VA_TopK = 0
N_VA_Inverse = 0

# Window length in time stamps for PDP analysis
window_length_tst = 2

# The distance that is taken for buffer if buffer is active; original point is extended with two points in each dimension at a buffer distance of the original point; if two points are less than double the buffer distance from each other then this makes already a difference with the fundamental
buffer_x = 0  # Buffer distance in x-direction
buffer_y = 0  # Buffer distance in y-direction

# The distance that is taken for rough if rough is active; two points that are originally different but have a distance less than this rough-distance will be 'the same' after transformation
rough_x = 0  # Rough distance in x-direction
rough_y = 0   # Rough distance in y-direction


# Dimensions and/or descriptors
DD = 2  # If dimension and descriptor is the same
des	= 2  # Number of descriptors
dim	= 2  # Number of dimensions

# Min/max x/y values
min_boundary_x = 0
max_boundary_x = 50
min_boundary_y = 0
max_boundary_y = 50

# Number of frames for interpolation if there is an interpolation
num_frames = 20  

# Set the number of similar configurations to add
num_similar_configurations = 5  # Number of similar configurations to add
new_configuration_step = 3  # The number of new configurations that has to be generated before an update  
division_factor = 5  # The factor that is used to devide the difference between the basis and the new point to speed up the similarity between the inequality matrices and to avoid conversion of points which results in an issue with the stopping mechanism. The higher the factor the quicker the code but the smaller the changes. Value 2 is of the original code for the fast algorithm.

# Mapping for non-integer poi IDs (initialized empty)
D_point_mapping = {}
curr_point_id = 0

# Initialize the GUI if enabled
if graphical_user_interface == 1:
    # Create a basic tkinter window
    root = tk.Tk()
    root.title("Settings")

    # Create IntVar for checkboxes to hold integer value
    fundamental_var = tk.IntVar(value=PDPg_fundamental)
    buffer_var = tk.IntVar(value=PDPg_buffer)
    rough_var = tk.IntVar(value=PDPg_rough)
    bufferrough_var = tk.IntVar(value=PDPg_bufferrough)
    
    # Create Checkbuttons (Checkboxes)
    fundamental_check = ttk.Checkbutton(root, text="PDPg_fundamental", variable=fundamental_var)
    fundamental_check.pack(pady=10)
    buffer_check = ttk.Checkbutton(root, text="PDPg_buffer", variable=buffer_var)
    buffer_check.pack(pady=10)
    rough_check = ttk.Checkbutton(root, text="PDPg_rough", variable=rough_var)
    rough_check.pack(pady=10)
    bufferrough_check = ttk.Checkbutton(root, text="PDPg_bufferrough", variable=bufferrough_var)
    bufferrough_check.pack(pady=10)
    
    # Submit button to update the settings and close the window
    submit_btn = ttk.Button(root, text="Update Settings", command=update_settings)
    submit_btn.pack(pady=20)
    root.mainloop()

if AV_SKIP_LOAD != '1':
    # Output the current working directory and dataset details
    # Allow the dataset filename to be overridden by an environment variable (used by GUI)
    dataset_name = os.environ.get('AV_DATASET', 'N_C_Dataset.csv')  # The name of the dataset file
    dataset_name_exclusive = dataset_name[:-4]  # The name of the dataset file without the extension

    # Read and process the dataset
    Df_dataset = pd.read_csv(dataset_name, header=None)  # Read the dataset
    Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']  # Set the column names

    # Load the dataset into a list for further processing
    L_dataset = []  # Initialize an empty list for dataset entries
    with open(dataset_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        # Read each row from the CSV file
        for L_row in csv_reader:
            poi_id = L_row[0]
            try:
                # Attempt to convert poiID to integer
                int(poi_id)
            except ValueError:
                # Handle non-integer poiID by mapping it to a unique integer
                if poi_id not in D_point_mapping:
                    D_point_mapping[poi_id] = curr_point_id
                    curr_point_id += 1
                L_row[2] = D_point_mapping[poi_id]
            L_dataset.append(list(map(float, L_row)))

    # Convert list to a numpy array for efficient numerical processing
    A_dataset = np.array(L_dataset, dtype=np.float32)

    # Save the processed dataset back to a CSV file (write into results dir if set)
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    os.makedirs(results_dir, exist_ok=True)
    Df_dataset.to_csv(os.path.join(results_dir, "Df_dataset.csv"), index=False)

    # Detect variables
    con = Df_dataset['conID'].max() + 1  # The number of configurations
    tst = Df_dataset['tstID'].max() + 1  # The number of time stamps
    poi = Df_dataset['poiID'].max() + 1  # The number of points

    # !!! Activation status of each PDP type; this has to be left here to 0 for all. It is just important that this can be activated during the code running.
    PDPg_fundamental_active = 0
    PDPg_buffer_active = 1
    PDPg_rough_active = 0
    PDPg_bufferrough_active = 0

    # Check if the window length exceeds the number of timestamps
    if window_length_tst > tst: 
        print("ERROR IN VALUE OF VARIABLE: window_length_tst > tst")

    # Final output to indicate the duration the script has run
    print('Time elapsed for running module "av": {:.3f} sec.'.format(time.time() - t_start))