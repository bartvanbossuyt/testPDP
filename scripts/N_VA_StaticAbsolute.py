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
mpl.rcParams['figure.dpi'] = 2.54

# Load the dataset
df = pd.read_csv(av.dataset_name, header=None)  

# Get the unique values of the first column
configurations = df[0].unique() 

# Create a list of colors
if av.poi == 2:
    colors= ['red', 'blue']
elif av.poi == 3:
    colors = ['black', 'blue', 'magenta']
else:
    #colors = [plt.cm.cividis(i/av.poi) for i in range(av.poi)]
    colors = [plt.cm.cividis(i / int(av.poi)) for i in range(int(av.poi))]

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
    #plt.figure(figsize=(12, 8))  # Set the size of the figure
    plt.figure(figsize=(18, 18))  # Set the size of the figure
    
    # Set the limits of the axes
    plt.xlim(av.min_boundary_x, av.max_boundary_x)
    plt.ylim(av.min_boundary_y, av.max_boundary_y)
    
    # Set the labels of the axes
    plt.xlabel('X-Axis (m)', fontsize=30, fontname='Arial')  
    plt.ylabel('Y-Axis (m)', fontsize=30, fontname='Arial')
    ax = plt.gca()  # Get the current axes
    ax.tick_params(axis='both', labelsize=30, labelcolor='black')  # Set the size and color of the tick labels
    
    # Check if there's only one timestamp
    if av.tst == 1:
        #colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple']
        #plt.scatter(x, y, color=colors[i % 10], s=100)  # s is the marker size
        for point_index, (x_val, y_val) in enumerate(zip(x, y)):
            color_to_use = colors[point_index % len(colors)]
            plt.scatter(x_val, y_val, color=color_to_use, s=200)  # s is the marker size
    else:
        vector_index = 0
        # Add vectors between points
        for p in range(int(av.poi)):  # for each point
            for i in range(av.tst - 1):  # for each interval
                x1 = x.iloc[int(p + i * av.poi)]
                y1 = y.iloc[int(p + i * av.poi)]
                x2 = x.iloc[int(p + i * av.poi + av.poi)]
                y2 = y.iloc[int(p + i * av.poi + av.poi)]
                # Use the custom color list to assign a un@ique color to each point
                color_to_use = colors[p % len(colors)]
                
                 # Only draw an arrow for the last interval
                if i == av.tst - 2:  # Last interval
                    plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True, head_width=(((x.max()+1)-(x.min()-1))/40), head_length=(((x.max()+1)-(x.min()-1))/20), linewidth=10, color=color_to_use)
                else:
                    plt.plot([x1, x2], [y1, y2], linewidth=10, color=color_to_use)
                
                #plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True, head_width=(((x.max()+1)-(x.min()-1))/40), head_length=(((x.max()+1)-(x.min()-1))/20), linewidth=10, color=color_to_use)
                #plt.arrow(x1, y1, x2 - x1, y2 - y1, linewidth=0.4, color=color_to_use)
                
                # Add label for the first timestamp of each point
                if i == 0:
                    plt.text(x1, y1, f'p{p}', fontsize=30, ha='right')
                
                vector_index += 1
    
    """
    # Draw the tennis pitch
    plt.hlines(0, -5.485, 5.485, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(6.4, -4.115, 4.115, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(-6.4, -4.115, 4.115, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(11.885, -5.485, 5.485, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(-11.885, -5.485, 5.485, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(0, -6.4, 6.4, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(-5.485, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(5.485, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(-4.115, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(4.115, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    """

    plt.title("Configuration {}".format(config),fontname="Arial", fontsize=40)
    file_name = "N_C_Csa{}.png".format(config) # csa from configuration static absolute    
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()  # close the figure to release memory



# End and print time
print('Time elapsed for running module "N_VA_StaticAbsolute": {:.3f} sec.'.format(time.time() - t_start))