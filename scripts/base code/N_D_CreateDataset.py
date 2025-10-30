import csv
import time
import numpy as np

# ------------- CONFIGURABLE SETTINGS -------------
CON = 10   # Number of configurations
TST = 2   # Number of time stamps
POI = 2   # Number of points
DIM = 2   # Number of dimensions (coordinates)
MIN_COORD_X = 0   # Minimum X coordinate
MAX_COORD_X = 100    # Maximum X coordinate
MIN_COORD_Y = 0     # Minimum Y coordinate
MAX_COORD_Y = 100      # Maximum Y coordinate
FILENAME = "N_C_Dataset.csv"  # Default output CSV file name
# -------------------------------------------------

# Function to generate a random dataset
def generate_random_dataset(con, tst, poi, dim, min_x, max_x, min_y, max_y):
    """
    Generates a random dataset of moving points. Each point has a random value for each dimension.
    
    Parameters:
        con (int): Number of configurations
        tst (int): Number of time stamps
        poi (int): Number of points
        dim (int): Number of dimensions (coordinates)
        min_x (float): Minimum value for the X coordinate
        max_x (float): Maximum value for the X coordinate
        min_y (float): Minimum value for the Y coordinate
        max_y (float): Maximum value for the Y coordinate
        
    Returns:
        dataset (list): A list of rows, where each row corresponds to [conID, tstID, poiID, dim1, dim2, ...]
    """
    dataset = []
    
    for c in range(con):
        for t in range(tst):
            for p in range(poi):
                random_coords = [
                    round(np.random.uniform(min_x, max_x), 0), 
                    round(np.random.uniform(min_y, max_y), 0)
                ]
                dataset.append([c, t, p] + random_coords)
    
    return dataset

# Function to write dataset to CSV
def write_to_csv(filename, data):
    """
    Writes the dataset to a CSV file.
    
    Parameters:
        filename (str): The name of the output file.
        data (list): The dataset to be written.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerows(data)

# Main routine
if __name__ == "__main__":
    # Start timer
    t_start = time.time()

    # Generate dataset with configurable settings
    dataset = generate_random_dataset(CON, TST, POI, DIM, MIN_COORD_X, MAX_COORD_X, MIN_COORD_Y, MAX_COORD_Y)

    # Generate meaningful filename
    custom_file_name = f"N_C_Dataset_{CON}_con_{TST}_tst_{POI}_poi_{DIM}_dim.csv"

    # Write the dataset to the generated file name
    write_to_csv(custom_file_name, dataset)
    
    # Write the dataset to the default CSV
    write_to_csv(FILENAME, dataset)

    # End timer and print elapsed time
    print(f"Time elapsed: {time.time() - t_start:.3f} sec")