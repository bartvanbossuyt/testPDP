# TODO: N_Moving_Objects execution and bug fixes
# - Disable N_VA_DynamicAbsolute (currently has error) and fix it
# - Disable N_VA_ClusterMap (currently has error) and fix it
# - Suppress array/interaction count prints, but save and display in Report

# ENHANCEMENT: Rough and buffer distance calculations
# - Fine-tune variable assignments for "fundamental" and "buffer" distance types
# - Integrate buffer calculations into Report module
# - Ensure normal and buffer calculations are comparable with configuration toggle

# DOCUMENTATION: Visualization clarity
# - Ensure clear titles for all visualizations
# - Add page numbers to all report pages
# - Extend fuzzy and buffer calculations to Report

import av  # Import all variables and configuration settings
import csv  # For reading and writing CSV files
import importlib  # For reloading modules
import numpy as np  # For numerical calculations  
import os  # For file and directory handling
import pandas as pd  # For data manipulation and DataFrame operations
import shutil  # For file system operations
import time  # For timing and performance measurement

t_start = time.time()

# Set up output directory for Moving Objects visualizations
output_dir = av.get_output_dir('Moving_Objects')

# Conditionally import visualization modules based on configuration settings
# - Each module handles different visualization types (static/dynamic, absolute/relative)
# - Organized by dimension: absolute (column) vs relative (column pair) and color scheme
if av.N_VA_StaticAbsolute == 1:
    import N_VA_StaticAbsolute  # Original absolute static visualizations
if av.N_VA_StaticAbsolute_color == 1:
    import N_VA_StaticAbsolute_color  # Colored version of absolute static visualizations
if av.N_VA_StaticRelative == 1: 
    import N_VA_StaticRelative  # Relative static visualizations
if av.N_VA_StaticFinetuned == 1: 
    import N_VA_StaticFinetuned  # Finetuned static visualizations
if av.N_VA_DynamicAbsolute == 1: 
    import N_VA_DynamicAbsolute  # Dynamic visualizations

# Function: Create working dataset from original data source
# Purpose: Load, preprocess, and structure dataset for PDP analysis
# Parameters:
# - data_filename: Path to the CSV file containing raw tracking data
# - D_point_mapping: Dictionary mapping original point IDs to standardized indices
# - curr_point_id: Counter for generating sequential point IDs
# - window_length_tst: Configuration parameter for time window validation
# Returns: Dataframe, numpy array, and dataset dimensions (con, tst, poi)
def SetDataForPDPType(data_filename, D_point_mapping, curr_point_id, window_length_tst):
    # Load data as Dataframe with header
    Df_dataset = pd.read_csv(data_filename, header=None)
    Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
    
    # Load data as list for preprocessing
    L_dataset = []  # Initialize empty list for raw data
    with open(data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            try:
                int(poi_id)
            except ValueError:
                if poi_id not in D_point_mapping:
                    D_point_mapping[poi_id] = curr_point_id
                    curr_point_id += 1
                L_row[2] = D_point_mapping[poi_id]
            L_dataset.append(list(map(float, L_row)))
    
    # Convert list to numpy array for efficient computation
    A_dataset = np.array(L_dataset, dtype=np.float32)
    
    # Save processed dataframe to output directory
    Df_dataset.to_csv(os.path.join(output_dir, "Df_dataset.csv"), index=False)
    
    # Auto-detect dataset dimensions from data structure
    con = Df_dataset['conID'].max() + 1  # Number of configurations (game instances)
    tst = Df_dataset['tstID'].max() + 1  # Number of timestamps per configuration
    poi = Df_dataset['poiID'].max() + 1  # Number of tracking points
    
    # Validate time window parameter
    if av.window_length_tst > tst: 
        print("ERROR IN VALUE OF VARIABLE: window_length_tst > tst")
    
    return Df_dataset, A_dataset, con, tst, poi

if av.PDPg_fundamental == 1:
    av.PDPg_fundamental_active = 1
    
    # Copy dataset file to output directory
    source_file = av.get_input_file(av.dataset_name)
    target_file = os.path.join(output_dir, "N_C_PDPg_fundamental_Dataset.csv")
    shutil.copyfile(source_file, target_file) 
    av.dataset_name = 'N_C_PDPg_fundamental_Dataset.csv'
    av.dataset_name_exclusive = av.dataset_name [:-4]
    
    # Load dataset as Dataframe with header
    dataset_path = av.get_input_file(av.dataset_name, ['Moving_Objects'])
    av.Df_dataset = pd.read_csv(dataset_path, header=None)
    av.Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
    
    # Load dataset as list for preprocessing
    av.L_dataset = []  # Initialize empty list for raw data
    # Load dataset as list for preprocessing with point ID mapping
    av.L_dataset = []
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            int(poi_id)
            av.L_dataset.append(list(map(float, L_row)))
    
    # Convert list to numpy array for efficient computation
    av.A_dataset = np.array(av.L_dataset, dtype=np.float32)
    
    # Save processed dataframe to output directory
    av.Df_dataset.to_csv(os.path.join(output_dir, "Df_dataset.csv"), index=False)
    
    # Auto-detect dataset dimensions
    av.con = av.Df_dataset['conID'].max() + 1  # Number of configurations
    av.tst = av.Df_dataset['tstID'].max() + 1  # Number of timestamps
    av.poi = av.Df_dataset['poiID'].max() + 1  # Number of tracking points

    # Execute analysis modules configured for fundamental dataset
    if av.N_PDP == 1: 
         import N_PDP  # Principal Directional Pattern analysis
    if av.N_VA_HeatMap == 1: 
        import N_VA_HeatMap  # Heatmap visualization from distance matrix
    if av.N_VA_HClust == 1: 
        import N_VA_HClust  # Hierarchical clustering tree from distance matrix
    if av.N_VA_ClusterMap == 1: 
        import N_VA_ClusterMap  # Cluster map visualization from distance matrix
    if av.N_VA_Mds == 1:
        import N_VA_Mds  # Multidimensional scaling dimension reduction
    if av.N_VA_Mds_autoencoder == 1:
        import N_VA_Mds_autoencoder  # Autoencoder-based dimension reduction
    if av.N_VA_TopK == 1:
        import N_VA_TopK  # Top-K similar configurations selection
    if av.N_VA_Inverse == 1:
        import N_VA_Inverse  # Inverse analysis to find similar patterns
    if av.N_VA_Report == 1:
        import N_T_Report  # Generate comprehensive analysis report
    
    av.PDPg_fundamental_active = 0
        
if av.PDPg_buffer == 1:
    av.PDPg_buffer_active = 1
    import N_T_OB
    
    # Set up buffer dataset for analysis
    av.dataset_name = 'N_C_PDPg_buffer_Dataset.csv'  # Buffer dataset filename
    av.dataset_name_exclusive = av.dataset_name [:-4]  # Dataset name without extension
    dataset_path = av.get_input_file(av.dataset_name, ['OB', 'Moving_Objects'])
    av.Df_dataset = pd.read_csv(dataset_path, header=None)
    av.Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
    # Load dataset into a list
    av.L_dataset = []  # Initialize empty list
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            int(float(poi_id))
            av.L_dataset.append(list(map(float, L_row)))
    # Transform list to array
    av.A_dataset = np.array(av.L_dataset, dtype=np.float32)
    # Save dataframe "Df_dataset"
    av.Df_dataset.to_csv(os.path.join(output_dir, "Df_dataset.csv"), index=False)
    # Automatically detected variables
    # Detect the number of configurations in the dataset
    av.con = av.Df_dataset['conID'].max() + 1
    # Detect the number of time stamps in the dataset
    av.tst = av.Df_dataset['tstID'].max() + 1
    # Detect the number of points in the dataset
    av.poi = av.Df_dataset['poiID'].max() + 1
    
    if av.N_PDP == 1: 
        importlib.reload(N_PDP)
    if av.N_VA_HeatMap == 1: 
        importlib.reload(N_VA_HeatMap)
    if av.N_VA_HClust == 1: 
        importlib.reload(N_VA_HClust)
    if av.N_VA_ClusterMap == 1: 
        importlib.reload(N_VA_ClusterMap)
    if av.N_VA_Mds == 1:
        importlib.reload(N_VA_Mds)
    if av.N_VA_TopK == 1:
        importlib.reload(N_VA_TopK)
    if av.N_VA_Inverse == 1:
        importlib.reload(N_VA_Inverse)
    if av.N_VA_Report == 1:
        importlib.reload(N_T_Report)
    av.PDPg_buffer_active = 0

if av.PDPg_rough == 1:    
    av.PDPg_rough_active = 1
    #import N_T_OR

    #shutil.copyfile(av.dataset_name, "N_C_PDPg_rough_Dataset.csv") 
    
#    av.dataset_name = 'N_C_Dataset.csv'  # Filename of csv file when no buffer and rough borders
    av.dataset_name = 'N_C_PDPg_fundamental_Dataset.csv'  # Filename of csv file when no buffer and rough borders; this is just the original file because the roughness is taken into account when the inequality matrix values are calculated.
    av.dataset_name_exclusive = av.dataset_name [:-4] # The dataset without the last four characters ".csv"
    
    # Open the original file as a dataframe with a header in the current working directory
    dataset_path = av.get_input_file(av.dataset_name, ['Moving_Objects'])
    av.Df_dataset = pd.read_csv(dataset_path, header=None)
    #av.Df_dataset = pd.read_csv("N_C_Dataset.csv", header=None)
    av.Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
    # Open the file as a list in the current working directory
    av.L_dataset = []  # Create an empty list
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            int(float(poi_id))
            av.L_dataset.append(list(map(float, L_row)))
    # Transform list to array
    av.A_dataset = np.array(av.L_dataset, dtype=np.float32)
    # Save dataframe "Df_dataset"
    av.Df_dataset.to_csv(os.path.join(output_dir, "Df_dataset.csv"), index=False)
    # Automatically detected variables
    # Detect the number of configurations in the dataset
    av.con = av.Df_dataset['conID'].max() + 1
    # Detect the number of time stamps in the dataset
    av.tst = av.Df_dataset['tstID'].max() + 1
    # Detect the number of points in the dataset
    av.poi = av.Df_dataset['poiID'].max() + 1
    
    if av.N_PDP == 1: 
        importlib.reload(N_PDP)
    if av.N_VA_HeatMap == 1: 
        importlib.reload(N_VA_HeatMap)
    if av.N_VA_HClust == 1: 
        importlib.reload(N_VA_HClust)
    if av.N_VA_ClusterMap == 1: 
        importlib.reload(N_VA_ClusterMap)
    if av.N_VA_Mds == 1:
        importlib.reload(N_VA_Mds)
    if av.N_VA_TopK == 1:
        importlib.reload(N_VA_TopK)
    if av.N_VA_Inverse == 1:
        importlib.reload(N_VA_Inverse)
    if av.N_VA_Report == 1:
        importlib.reload(N_T_Report)
    av.PDPg_rough_active = 0
    
if av.PDPg_bufferrough == 1:
    av.PDPg_bufferrough_active = 1
    #import N_T_OBR
    import N_T_OB
    av.dataset_name = 'N_C_PDPg_buffer_Dataset.csv'  # Use buffered dataset for bufferrough calculation
    av.dataset_name_exclusive = av.dataset_name [:-4]  # Dataset name without extension
    
    # Locate dataset from OB or Moving_Objects output folders
    dataset_path = av.get_input_file(av.dataset_name, ['OB', 'Moving_Objects'])
    av.Df_dataset = pd.read_csv(dataset_path, header=None)
    av.Df_dataset.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
    # Load dataset into a list
    av.L_dataset = []  # Initialize empty list
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for L_row in csv_reader:
            poi_id = L_row[0]
            int(float(poi_id))
            av.L_dataset.append(list(map(float, L_row)))
    # Transform list to array
    av.A_dataset = np.array(av.L_dataset, dtype=np.float32)
    # Save dataframe "Df_dataset"
    av.Df_dataset.to_csv(os.path.join(output_dir, "Df_dataset.csv"), index=False)
    # Automatically detected variables
    # Detect the number of configurations in the dataset
    av.con = av.Df_dataset['conID'].max() + 1
    # Detect the number of time stamps in the dataset
    av.tst = av.Df_dataset['tstID'].max() + 1
    # Detect the number of points in the dataset
    av.poi = av.Df_dataset['poiID'].max() + 1

    if av.N_PDP == 1: 
        importlib.reload(N_PDP)
    if av.N_VA_HeatMap == 1: 
        importlib.reload(N_VA_HeatMap)
    if av.N_VA_HClust == 1: 
        importlib.reload(N_VA_HClust)
    if av.N_VA_ClusterMap == 1: 
        importlib.reload(N_VA_ClusterMap)
    if av.N_VA_Mds == 1:
        importlib.reload(N_VA_Mds)
    if av.N_VA_TopK == 1:
        importlib.reload(N_VA_TopK)
    if av.N_VA_Inverse == 1:
        importlib.reload(N_VA_Inverse)
    if av.N_VA_Report == 1:
        importlib.reload(N_T_Report)
    av.PDPg_bufferrough_active = 0

# End and print time
print('Time elapsed for running module "N_Moving_Objects": {:.3f} sec.'.format(time.time() - t_start))