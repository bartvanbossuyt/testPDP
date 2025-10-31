from matplotlib import cm
import av
import matplotlib as mpl
import matplotlib.patches as patches  # For drawing shapes
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

# Start time
t_start = time.time()

# Set the default unit of length to centimeters
mpl.rcParams['figure.dpi'] = 2.54

# Load the dataset
df = pd.read_csv(av.dataset_name, header=None)

# Define your custom colors
if av.poi == 3:
    colors = ['black', 'red', 'blue']
else:
    colors = [plt.cm.cividis(i/av.poi) for i in range(av.poi)]

# Make a list of the configurations
configurations = df[0].unique()

# Create the scatterplot, including arrows
for config in configurations:
    config_data = df[df[0] == config]
    x = config_data[3]
    y = config_data[4]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    # Calculate the max of range x and rang y
    max_range = max(x_range,y_range)
    
    #plt.figure(figsize=(12, 8), dpi=100.0)
    plt.figure(figsize=(7, 18), dpi=100.0)  # Set the size of the figure

    #finetuned for microtraffic
    plt.xlim((((x.min()+x.max())/2)-100),(((x.min()+x.max())/2)+100))
    plt.ylim(7.25,18)
    
    plt.xlabel('X-Axis (m)', fontsize=30, fontname='monospace')
    plt.ylabel('Y-Axis (m)', fontsize=30, fontname='monospace')
    ax = plt.gca()  # Get current axes
    ax.tick_params(axis='both', labelsize=30, labelcolor='black')  # Change tick labels on both x and y axis
    
    # Check if there's only one timestamp
    if av.tst == 1:
        #plt.scatter(x, y, color=colors[i % 10], s=100)  # s is the marker size
        for point_index, (x_val, y_val) in enumerate(zip(x, y)):
            color_to_use = colors[point_index % len(colors)]
            plt.scatter(x_val, y_val, color=color_to_use, s=100)  # s is the marker size
    else:
        vector_index = 0
        # Add vectors between points
        for p in range(av.poi):  # for each point
            for i in range(av.tst - 1):  # for each interval
                x1 = x.iloc[p + i * av.poi]
                y1 = y.iloc[p + i * av.poi]
                x2 = x.iloc[p + i * av.poi + av.poi]
                y2 = y.iloc[p + i * av.poi + av.poi]
                # Use the custom color list to assign a un@ique color to each point
                color_to_use = colors[p % len(colors)]
                plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True, head_width=(((x.max()+1)-(x.min()-1))/40), head_length=(((x.max()+1)-(x.min()-1))/20), linewidth=2, color=color_to_use)
                
                # Add label for the first timestamp of each point
                if i == 0:
                    plt.text(x1, y1, f'p{p}', fontsize=12, ha='right')
                
                vector_index += 1
        
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

    plt.title('Configuration ' + str(config),fontname="monospace", fontsize=40)  # increment the configuration by 1
    file_name = 'N_C_Csf' + str(config) + '.png'  # increment the filename by 1
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()  # close the figure to release memory

# End and print time
print('Time elapsed for running module "N_VA_StaticFinetuned": {:.3f} sec.'.format(time.time() - t_start))