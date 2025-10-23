# Load the necessary libraries
from matplotlib import cm  # Colormaps
import matplotlib.patches as patches  # For drawing shapes
import matplotlib.pyplot as plt  # Plotting library
import matplotlib as mpl  # Matplotlib settings
import os
import pandas as pd
import time

# Import custom attributes
import av

# Start time
t_start = time.time()

# Set the default unit of length to centimeters
mpl.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 12

# Load the dataset
df = pd.read_csv(av.dataset_name, header=None)

# Get the unique values of the first column
configurations = df[0].unique()

# Create a list of colors
if av.poi == 3:
    colors = ['black', 'blue', 'magenta']
else:
    colors = [plt.cm.cividis(i / av.poi) for i in range(av.poi)]

# Create the scatterplot, including arrows
for config in configurations:
    config_data = df[df[0] == config]  # Get the data for the current configuration

    # Get the x/y-values
    x = config_data[3]
    y = config_data[4]

    # Calculate scaling factor
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    scaling_factor = min(x_range, y_range) / 10

    # Initialize lists to store arrow coordinates
    x_coords = []
    y_coords = []

    plt.figure(figsize=(18, 18))  # Set the size of the figure

    # Check if there's only one timestamp
    if av.tst == 1:
        for point_index, (x_val, y_val) in enumerate(zip(x, y)):
            poi_idx = point_index
            color_to_use = colors[poi_idx % len(colors)]
            plt.scatter(x_val, y_val, color=color_to_use, s=200)  # s is the marker size
    else:
        vector_index = 0
        # Add vectors between points
        for p in range(av.poi):  # for each point
            for i in range(av.tst - 1):  # for each interval
                x1 = x.iloc[p + i * av.poi]
                y1 = y.iloc[p + i * av.poi]
                x2 = x.iloc[p + i * av.poi + av.poi]
                y2 = y.iloc[p + i * av.poi + av.poi]

                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])

                # Calculate increments
                x_increment = abs(x2 - x1)
                y_increment = abs(y2 - y1)

                # Use poi-based color
                color_to_use = colors[p % len(colors)]
                plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True,
                          head_width=0.2,
                          head_length=1,
                          linewidth=2, color=color_to_use)

                # Add label for the first timestamp of each point
                if i == 0:
                    plt.text(x1, y1, f'p{p}', fontsize=30, ha='right')

                vector_index += 1

    # Adjust the limits of the axes based on arrow coordinates
    plt.xlim(min(x_coords) - 1, max(x_coords) + 1)
    plt.ylim(min(y_coords) - 1, max(y_coords) + 1)

    # Set the labels of the axes
    plt.xlabel('X-Axis (m)', fontsize=30, fontname='monospace')
    plt.ylabel('Y-Axis (m)', fontsize=30, fontname='monospace')
    ax = plt.gca()  # Get the current axes
    ax.tick_params(axis='both', labelsize=30, labelcolor='black')  # Set the size and color of the tick labels

    plt.title(f"Configuration {config}", fontname="monospace", fontsize=40)
    
    output_folder = os.path.join(av.OUTPUT_FOLDER, 'configuraties')
    os.makedirs(output_folder, exist_ok=True)

    file_name = os.path.join(output_folder, "N_C_Csa{}.png".format(config))
    #file_name = f"N_C_Csa{config}.png"  # csa from configuration static absolute
    plt.savefig(file_name, dpi=100, bbox_inches='tight')
    plt.close()  # close the figure to release memory

# End and print time
print('Time elapsed for running module "N_VA_StaticAbsolute": {:.3f} sec.'.format(time.time() - t_start))
