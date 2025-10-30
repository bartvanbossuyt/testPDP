"""
Flexible PDP data loader + settings

- Supports both:
  * STREAMS/configurations_long_noclass2.csv  (5 cols: conID,tstID,poiID,x,y)
  * STREAMS/configurations_long2.csv          (6 cols: conID,tstID,poiID,x,y,class)

- Exposes the same variables you already use:
  Df_dataset, dataset_name, dataset_name_exclusive, L_dataset, A_dataset, con, tst, poi
  (and optionally Df_classes when a class column is present)

- Fixes:
  * Correctly maps non-integer poiID (uses column index 2 / 'poiID', not 0)
  * Defines D_point_mapping / curr_point_id
  * Avoids forcing 'class' into float arrays
"""

# ---------------- Imports ----------------
import csv
import numpy as np
import os
import pandas as pd
import time
import tkinter as tk
from tkinter import ttk

# ---------------- Timer ----------------
t_start = time.time()

# ---------------- GUI helper ----------------
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

# ---------------- Runtime toggles & analysis settings ----------------
graphical_user_interface = 0  # 1 to show the Tk GUI

# PDP types (keep fundamental always 1 as you noted)
PDPg_fundamental = 1  # must always be 1
PDPg_buffer = 0  
PDPg_rough = 0
PDPg_bufferrough = 0

# Report/visual analytics switches
N_VA_Report = 0
N_VA_StaticAbsolute = 1
N_VA_StaticRelative = 0
N_VA_StaticFinetuned = 0
N_VA_DynamicAbsolute = 0
N_PDP = 1
N_VA_InequalityMatrices = 0
N_VA_HeatMap = 1
N_VA_HClust = 1
N_VA_ClusterMap = 0
N_VA_Mds = 1
N_VA_Mds_autoencoder = 0
N_VA_TopK = 1
N_VA_Inverse = 0

# NEW: Flag for t-SNE analysis (set to 1 to run, or 0 otherwise)
N_VA_TSNE = 1

# PDP window and geometry params
window_length_tst = 3

buffer_x = 15
buffer_y = 1
rough_x = 30
rough_y = 3

DD  = 2
des = 2
dim = 2

min_boundary_x = -150
max_boundary_x = 150
min_boundary_y = -150
max_boundary_y = 150

num_frames = 20

# Simulation/extension params
num_similar_configurations = 5
new_configuration_step = 3
division_factor = 5

# ---------------- GUI (optional) ----------------
if graphical_user_interface == 1:
    root = tk.Tk()
    root.title("Settings")

    fundamental_var = tk.IntVar(value=PDPg_fundamental)
    buffer_var = tk.IntVar(value=PDPg_buffer)
    rough_var = tk.IntVar(value=PDPg_rough)
    bufferrough_var = tk.IntVar(value=PDPg_bufferrough)
    
    ttk.Checkbutton(root, text="PDPg_fundamental", variable=fundamental_var).pack(pady=6)
    ttk.Checkbutton(root, text="PDPg_buffer", variable=buffer_var).pack(pady=6)
    ttk.Checkbutton(root, text="PDPg_rough", variable=rough_var).pack(pady=6)
    ttk.Checkbutton(root, text="PDPg_bufferrough", variable=bufferrough_var).pack(pady=6)
    
    ttk.Button(root, text="Update Settings", command=update_settings).pack(pady=12)
    root.mainloop()

# ---------------- Choose active dataset ----------------
DATA_NOCLASS   = "/Users/olivier/Documents/STREAMS/inD/Data/C16_CBP_CL_inD_F100/C16_CBP_NC_inD_F100.csv"
DATA_WITHCLASS = "/Users/olivier/Documents/STREAMS/inD/Data/C16_CBP_CL_inD_F100/C16_CBP_CL_inD_F100.csv"

# Pick which one you want active right now:
dataset_name = DATA_WITHCLASS # ← switch to DATA_NOCLASS if you want the 5-col file
dataset_name_exclusive = os.path.splitext(os.path.basename(dataset_name))[0]

# Central output folder for distance matrices, images and other generated files.
# Change this single variable if you move the output folder.
# Use an absolute path or a path relative to the project root. Example:
# OUTPUT_FOLDER = '/Users/olivier/Documents/STREAMS/Test_datalvlX_output'
OUTPUT_FOLDER = os.path.expanduser('/Users/olivier/Documents/STREAMS/inD/Data/test')

# Path or folder where distance-matrix CSV files live. You can set this to either:
# - an absolute path to a directory (then scripts will look for filenames inside that dir),
# - or a full path to a single CSV file (then that file will be used directly),
# - or None to keep existing relative-file behavior.
# Example (directory): av.INPUT_DISTANCE_MATRIX = '/Users/olivier/Documents/STREAMS/Test_datalvlX_output'
# Example (single file): av.INPUT_DISTANCE_MATRIX = '/path/to/N_C_PDPg_fundamental_DistanceMatrix.csv'
INPUT_DISTANCE_MATRIX = '/Users/olivier/Documents/STREAMS/inD/Data/test'

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
