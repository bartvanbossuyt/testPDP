"""
!!!Df_dataset, dataset_name, dataset_name_exclusive, L_dataset, A_dataset, con, tst, poi are those in the active setting, if nothing is active, then it is the original. The original has to be saved as nothing or original and the other active serttings as their active setting.
"""
"""
Not 100% sure, but I think that the interpreter needs to be on "Python 3.10.9 ('base': conda)"
"""

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
N_VA_Inverse = 0 # only fundamental is implemented
N_VA_Report = 0
N_VA_StaticAbsolute = 1
N_VA_StaticRelative = 0
N_VA_StaticFinetuned = 0
N_VA_DynamicAbsolute = 0  #have to doublecheck this code
N_PDP = 1 # The calculation of PDP and matrices  
N_VA_InequalityMatrices = 1
N_VA_HeatMap = 1
N_VA_HClust = 1
N_VA_ClusterMap = 0
N_VA_Mds = 1
N_VA_Mds_autoencoder = 0
N_VA_TopK = 1

# Window length in time stamps for PDP analysis
window_length_tst = 2

# Distance details for different PDP types
buffer = 20  # The distance that is taken for buffer if buffer is active; original point is extended with two points in each dimension at a buffer distance of the original point; if two points are less than double the buffer distance from each other then this makes already a difference with the fundamental
rough = 2.9  # The distance that is taken for rough if rough is active; two points that are originally different but have a distance less than this rough-distance will be 'the same' after transformation

# Dimensions and/or descriptors
DD = 2  # If dimension and descriptor is the same
des	= 2  # Number of descriptors
dim	= 2  # Number of dimensions

# Min/max x/y values
min_boundary_x = 0
max_boundary_x = 100
min_boundary_y = 0
max_boundary_y = 100

# Number of frames for interpolation if there is an interpolation
num_frames = 20

# Set the number of similar configurations to add
num_similar_configurations = 5  # Number of similar configurations to add
new_configuration_step = 5  # The number of new configurations that has to be generated before an update  
division_factor = 2  # The factor that is used to devide the difference between the basis and the new point to speed up the similarity between the inequality matrices and to avoid conversion of points which results in an issue with the stopping mechanism. The higher the factor the quicker the code but the smaller the changes. Value 2 is of the original code for the fast algorithm.

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

# ---------------- Choose active dataset ----------------
DATA_NOCLASS   = ""
DATA_WITHCLASS = ""

# Pick which one you want active right now:
dataset_name = DATA_WITHCLASS # ← switch to DATA_NOCLASS if you want the 5-col file
dataset_name_exclusive = os.path.splitext(os.path.basename(dataset_name))[0]

# Central output folder for distance matrices, images and other generated files.
# Change this single variable if you move the output folder.
# Use an absolute path or a path relative to the project root. Example:
# OUTPUT_FOLDER = '/Users/olivier/Documents/STREAMS/Test_datalvlX_output'
OUTPUT_FOLDER = os.path.expanduser('')

# Path or folder where distance-matrix CSV files live. You can set this to either:
# - an absolute path to a directory (then scripts will look for filenames inside that dir),
# - or a full path to a single CSV file (then that file will be used directly),
# - or None to keep existing relative-file behavior.
# Example (directory): av.INPUT_DISTANCE_MATRIX = '/Users/olivier/Documents/STREAMS/Test_datalvlX_output'
# Example (single file): av.INPUT_DISTANCE_MATRIX = '/path/to/N_C_PDPg_fundamental_DistanceMatrix.csv'
INPUT_DISTANCE_MATRIX = ''

# ---------------- Load & normalize data ----------------
# Detect number of columns: 5 (no class) or 6 (with class)
_probe = pd.read_csv(dataset_name, header=None, nrows=1)
ncols = _probe.shape[1]

if ncols == 5:
    colnames = ['conID', 'tstID', 'poiID', 'x', 'y']
    has_class = False
elif ncols == 6:
    colnames = ['conID', 'tstID', 'poiID', 'x', 'y', 'class']
    has_class = True
else:
    raise ValueError(f"Unexpected number of columns ({ncols}) in {dataset_name}. Expected 5 or 6.")

# Read full dataset with proper names
Df_raw = pd.read_csv(dataset_name, header=None, names=colnames)

# Force numeric on core columns; leave 'class' untouched
for c in ['conID', 'tstID', 'poiID', 'x', 'y']:
    Df_raw[c] = pd.to_numeric(Df_raw[c], errors='ignore')

if not pd.api.types.is_integer_dtype(Df_raw['poiID']):
    D_point_mapping = {}
    _state = {"curr": 0}

    def map_poi(v, _state=_state):
        try:
            return int(v)
        except (ValueError, TypeError):
            if v not in D_point_mapping:
                D_point_mapping[v] = _state["curr"]
                _state["curr"] += 1
            return D_point_mapping[v]

    Df_raw['poiID'] = Df_raw['poiID'].apply(map_poi).astype(int)


# Split into numeric base and optional classes
Df_dataset = Df_raw[['conID', 'tstID', 'poiID', 'x', 'y']].copy()
Df_dataset.to_csv(f"{dataset_name_exclusive}__Df_dataset.csv", index=False)

Df_classes = None
if has_class:
    Df_classes = Df_raw[['conID', 'tstID', 'poiID', 'class']].copy()
    Df_classes.to_csv(f"{dataset_name_exclusive}__Df_classes.csv", index=False)

# Legacy outputs you already rely on
L_dataset = Df_dataset.values.tolist()
A_dataset = Df_dataset[['conID','tstID','poiID','x','y']].to_numpy(dtype=np.float32)

# Detect counts (max+1)
con = int(Df_dataset['conID'].max()) + 1
tst = int(Df_dataset['tstID'].max()) + 1
poi = int(Df_dataset['poiID'].max()) + 1
print("poi =", poi)

# Keep standard export name if other modules expect it
Df_dataset.to_csv("Df_dataset.csv", index=False)

# ---------------- PDP activation flags (runtime) ----------------
PDPg_fundamental_active = 0
PDPg_buffer_active = 0
PDPg_rough_active = 0
PDPg_bufferrough_active = 0

# ---------------- Basic validity checks ----------------
if window_length_tst > tst: 
    print("ERROR IN VALUE OF VARIABLE: window_length_tst > tst")

# ---------------- Done ----------------
print('Time elapsed for running module: {:.3f} sec.'.format(time.time() - t_start))
