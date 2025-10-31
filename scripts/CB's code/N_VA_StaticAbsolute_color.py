# Load the necessary libraries
from matplotlib import cm  # Colormaps
import matplotlib.patches as patches  # For drawing shapes
import matplotlib.pyplot as plt  # Plotting library
import matplotlib as mpl  # Matplotlib settings
import os
import pandas as pd
import time
import numpy as np  # Numerical operations
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For aligned colorbar

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

# Create a list of colors for scatter when only one timestamp
if av.poi == 3:
    colors = ['black', 'blue', 'magenta']
else:
    colors = [plt.cm.cividis(i/av.poi) for i in range(av.poi)]

# Define colormap for time sequence (plasma)
cmap_time = cm.plasma
num_intervals = av.tst - 1 if av.tst > 1 else 1  # number of arrow intervals per point

# Parameters for stationary marker sizing
min_marker_size = 100  # minimum red dot size
max_marker_size = 500  # maximum red dot size

# Create the scatterplot, including arrows and stationary markers
for config in configurations:
    config_data = df[df[0] == config]  # Get the data for the current configuration
    
    # Get the x/y-values
    x = config_data[3]
    y = config_data[4]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(18, 18), dpi=100.0)

    # Set the limits of the axes
    ax.set_xlim(av.min_boundary_x, av.max_boundary_x)
    ax.set_ylim(av.min_boundary_y, av.max_boundary_y)
    
    # Set the labels of the axes
    ax.set_xlabel('X-Axis (m)', fontsize=30, fontname='monospace')  
    ax.set_ylabel('Y-Axis (m)', fontsize=30, fontname='monospace')
    ax.tick_params(axis='both', labelsize=30, labelcolor='black')
    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Track stationary durations and coords per point
    static_counts = {}
    static_coords = {}

    # Plot data
    if av.tst == 1:
        for point_index, (x_val, y_val) in enumerate(zip(x, y)):
            color_to_use = colors[point_index % len(colors)]
            ax.scatter(x_val, y_val, color=color_to_use, s=200)
    else:
        for p in range(av.poi):
            for i in range(av.tst - 1):
                idx = p + i * av.poi
                x1, y1 = x.iloc[idx], y.iloc[idx]
                x2, y2 = x.iloc[idx + av.poi], y.iloc[idx + av.poi]
                dx, dy = x2 - x1, y2 - y1
                step_len = np.hypot(dx, dy)
                if step_len == 0:
                    static_counts[p] = static_counts.get(p, 0) + 1
                    static_coords[p] = (x1, y1)
                else:
                    # Determine head sizes based on step length
                    head_width = step_len * 4
                    head_length = step_len * 8
                    color_to_use = cmap_time(i / num_intervals)
                    ax.arrow(
                        x1, y1, dx, dy,
                        length_includes_head=True,
                        head_width=head_width,
                        head_length=head_length,
                        linewidth=2 + step_len * 0.05,
                        color=color_to_use
                    )
                if i == 0:
                    ax.text(x1, y1, f'p{p}', fontsize=30, ha='right')

    # Plot stationary markers scaled by duration relative to total span
    stationary_plotted = False
    total_intervals = num_intervals
    for p, count in static_counts.items():
        x1, y1 = static_coords[p]
        # Compute marker size scaled by count / total_intervals
        size = min_marker_size + (count / total_intervals) * (max_marker_size - min_marker_size)
        label = 'Stationary' if not stationary_plotted else None
        ax.scatter(x1, y1, color='red', s=size, label=label)
        stationary_plotted = True

    # Add colorbar for time sequence with same height as y-axis and aligned
    if av.tst > 1:
        sm = plt.cm.ScalarMappable(cmap=cmap_time, norm=plt.Normalize(vmin=0, vmax=num_intervals))
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.2)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Time step', fontsize=30, fontname='monospace')
        cbar.ax.tick_params(labelsize=30)

    # Legend for stationary points
    if stationary_plotted:
        ax.legend(fontsize=30, loc='upper right')

    # Title, save, and render
    ax.set_title(f"Configuration {config}", fontname="monospace", fontsize=40)
    # Save the figure
    fig.savefig(f"N_C_Csa{config}.png", dpi=300, bbox_inches='tight')
    # Display the figure
    # plt.show()
    plt.close(fig)

# End and print time
print(f"Time elapsed for running module \"N_VA_StaticAbsolute\": {time.time() - t_start:.3f} sec.")