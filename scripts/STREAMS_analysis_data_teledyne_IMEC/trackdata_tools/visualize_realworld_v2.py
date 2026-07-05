import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Motorcycle",
    3: "MotorcyclePlus",
    4: "VRU",
    5: "Car",
    6: "SmallVehicle",
    7: "Van",
    8: "LargeVehicle",
    9: "Vehicle",
    10: "SmallTruck",
    11: "MiddleTruck",
    12: "LargeTruck",
    13: "Truck",
    14: "Bus",
    15: "DoubleBus",
    16: "CarTrailer",
    17: "Box",
    18: "Cone",
    19: "ObjectOfInterest",
    20: "CarAndTrailer",
    21: "VanAndTrailer",
    22: "TruckTrailer",
    23: "TruckHead",
    24: "TruckAndTrailer",
    25: "Scooter",
    26: "MiddleTruckSmall",
    27: "MiddleTruckLarge"
}

CLASS_COLORS = {
    'Person': '#E74C3C',
    'Bicycle': '#27AE60',
    'Motorcycle': '#9B59B6',
    'Car': '#3498DB',
    'Van': '#E67E22',
    'SmallTruck': '#F39C12',
    'LargeTruck': '#D35400',
    'Bus': '#795548',
    'CarAndTrailer': '#1ABC9C',
    'VanAndTrailer': '#16A085',
    'TruckAndTrailer': '#2980B9',
    'Scooter': '#8E44AD',
    'Unknown': '#AAAAAA'
}

# Create output folder
output_folder = Path(__file__).parent / 'realworld_visualizations'
output_folder.mkdir(exist_ok=True)

# Load all data
trackdata_csv_folder = Path(__file__).parent / 'TrackData_csv'
csv_files = list(trackdata_csv_folder.rglob('*.csv'))

all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    parts = csv_file.relative_to(trackdata_csv_folder).parts
    df['scenario'] = parts[0]
    df['file_name'] = csv_file.stem
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df['class_name'] = combined_df['class_id'].map(CLASS_NAMES).fillna('unknown')

print("Creating real-world position visualizations (per scenario)...")
print(f"Output folder: {output_folder}")

scenarios = sorted(combined_df['scenario'].unique())

# ============================================================================
# Helper function to create detailed trajectory plot
# ============================================================================
def plot_trajectories(ax, data, title, show_legend=True, padding=1.5):
    """Plot trajectories with good zoom and detail."""
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F8F8')
    
    # Track classes present
    classes_present = set()
    
    for (file_name, track_id), track_data in data.groupby(['file_name', 'id']):
        track_data = track_data.sort_values('object_time')
        class_name = track_data['class_name'].iloc[0]
        classes_present.add(class_name)
        color = CLASS_COLORS.get(class_name, '#AAAAAA')
        
        x = track_data['world_x'].values
        y = track_data['world_y'].values
        
        # Plot trajectory line with better visibility
        ax.plot(x, y, color=color, alpha=0.8, linewidth=2.5, solid_capstyle='round')
        
        # Start point (green circle)
        ax.scatter(x[0], y[0], color='green', s=60, marker='o', alpha=0.9, zorder=7, 
                  edgecolors='white', linewidth=1.5)
        
        # End point with direction arrow
        if len(x) > 1:
            ax.scatter(x[-1], y[-1], color='red', s=60, marker='s', alpha=0.9, zorder=7,
                      edgecolors='white', linewidth=1.5)
            
            # Direction arrow
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
            arrow_scale = 0.5
            ax.annotate('', xy=(x[-1] + dx*arrow_scale, y[-1] + dy*arrow_scale), 
                       xytext=(x[-1], y[-1]),
                       arrowprops=dict(arrowstyle='-|>', color=color, lw=2, 
                                      mutation_scale=15),
                       zorder=8)
        
        # Add track ID label at midpoint
        mid_idx = len(x) // 2
        ax.annotate(f'{track_id}', xy=(x[mid_idx], y[mid_idx]), fontsize=7, 
                   color='black', alpha=0.7, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))
    
    # Set limits with padding for better zoom
    x_min, x_max = data['world_x'].min(), data['world_x'].max()
    y_min, y_max = data['world_y'].min(), data['world_y'].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add scale bar
    scale_length = max(1, int((x_range + y_range) / 10))  # Dynamic scale
    scale_x = x_min - padding + 0.5
    scale_y = y_min - padding + 0.5
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 'k-', linewidth=4)
    ax.text(scale_x + scale_length/2, scale_y + 0.3, f'{scale_length}m', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Add meter markers on axes
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    
    ax.set_xlabel('World X (meters)', fontsize=11)
    ax.set_ylabel('World Y (meters)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_aspect('equal')
    
    if show_legend:
        # Legend for classes
        legend_handles = []
        for class_name in sorted(classes_present):
            color = CLASS_COLORS.get(class_name, '#AAAAAA')
            count = data[data['class_name'] == class_name].groupby(['file_name', 'id']).ngroups
            legend_handles.append(mpatches.Patch(color=color, label=f'{class_name} ({count})'))
        
        # Add start/end markers to legend
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                         markersize=10, label='Start'))
        legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                                         markersize=10, label='End'))
        
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)
    
    return classes_present

# ============================================================================
# Generate individual scenario figures
# ============================================================================
for scenario in scenarios:
    scenario_data = combined_df[combined_df['scenario'] == scenario]
    files_in_scenario = sorted(scenario_data['file_name'].unique())
    
    print(f"\nProcessing {scenario}...")
    
    # --------------------------------------------------------------------------
    # Figure 1: All files in scenario combined (overview)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 11))
    
    n_objects = scenario_data.groupby(['file_name', 'id']).ngroups
    class_counts = scenario_data.groupby(['file_name', 'id', 'class_name']).size().reset_index()
    class_summary = class_counts.groupby('class_name').size().to_dict()
    class_str = ', '.join([f"{k}: {v}" for k, v in sorted(class_summary.items())])
    
    plot_trajectories(ax, scenario_data, 
                     f'{scenario} - All Recordings Combined\n({n_objects} objects: {class_str})')
    
    plt.tight_layout()
    fig.savefig(output_folder / f'{scenario}_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {scenario}_overview.png")
    
    # --------------------------------------------------------------------------
    # Figure 2: Each file in scenario as separate subplot
    # --------------------------------------------------------------------------
    n_files = len(files_in_scenario)
    n_cols = min(2, n_files)
    n_rows = (n_files + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows))
    if n_files == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, file_name in enumerate(files_in_scenario):
        ax = axes[idx]
        file_data = scenario_data[scenario_data['file_name'] == file_name]
        
        n_obj = file_data.groupby('id').ngroups
        plot_trajectories(ax, file_data, f'{file_name}\n({n_obj} objects)', padding=1.0)
    
    # Hide unused subplots
    for idx in range(len(files_in_scenario), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{scenario} - Individual Recordings', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_folder / f'{scenario}_by_file.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {scenario}_by_file.png")
    
    # --------------------------------------------------------------------------
    # Figure 3: Heatmap for this scenario
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x_all = scenario_data['world_x'].values
    y_all = scenario_data['world_y'].values
    
    # Determine bin size based on data range
    x_range = x_all.max() - x_all.min()
    y_range = y_all.max() - y_all.min()
    n_bins = max(30, int(max(x_range, y_range) * 3))
    
    heatmap, xedges, yedges = np.histogram2d(x_all, y_all, bins=n_bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlOrRd', aspect='equal')
    plt.colorbar(im, ax=ax, label='Detection Density', shrink=0.8)
    
    ax.set_xlabel('World X (meters)', fontsize=11)
    ax.set_ylabel('World Y (meters)', fontsize=11)
    ax.set_title(f'{scenario} - Object Detection Heatmap', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    fig.savefig(output_folder / f'{scenario}_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {scenario}_heatmap.png")
    
    # --------------------------------------------------------------------------
    # Figure 4: Scene view with camera reference
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 11))
    
    ax.set_facecolor('#E8E8E8')
    ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5, color='white')
    
    # Draw meter grid
    x_min, x_max = scenario_data['world_x'].min() - 2, scenario_data['world_x'].max() + 2
    y_min, y_max = scenario_data['world_y'].min() - 2, scenario_data['world_y'].max() + 2
    
    for x in range(int(x_min), int(x_max) + 1, 2):
        ax.axvline(x=x, color='white', linestyle='-', alpha=0.7, linewidth=0.5)
    for y in range(int(y_min), int(y_max) + 1, 2):
        ax.axhline(y=y, color='white', linestyle='-', alpha=0.7, linewidth=0.5)
    
    classes_present = set()
    
    for (file_name, track_id), track_data in scenario_data.groupby(['file_name', 'id']):
        track_data = track_data.sort_values('object_time')
        class_name = track_data['class_name'].iloc[0]
        classes_present.add(class_name)
        color = CLASS_COLORS.get(class_name, '#AAAAAA')
        
        x = track_data['world_x'].values
        y = track_data['world_y'].values
        
        # Draw trajectory with shadow effect
        ax.plot(x, y, color='black', alpha=0.2, linewidth=4)  # Shadow
        ax.plot(x, y, color=color, alpha=0.9, linewidth=2.5)  # Main line
        
        # Start/end markers
        ax.scatter(x[0], y[0], color='green', s=80, marker='o', zorder=7,
                  edgecolors='black', linewidth=1.5)
        
        if len(x) > 1:
            ax.scatter(x[-1], y[-1], color='red', s=80, marker='o', zorder=7,
                      edgecolors='black', linewidth=1.5)
    
    # Camera marker at origin
    ax.scatter(0, 0, color='yellow', s=300, marker='^', zorder=10, 
              edgecolors='black', linewidth=2)
    ax.annotate('CAMERA', xy=(0, 0), xytext=(1.5, -1.5),
               fontsize=11, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Axis reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    
    # Compass
    compass_x = x_max - 1
    compass_y = y_max - 1
    ax.annotate('', xy=(compass_x, compass_y + 1.5), xytext=(compass_x, compass_y),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(compass_x, compass_y + 2, 'Y+', ha='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(compass_x + 1.5, compass_y), xytext=(compass_x, compass_y),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(compass_x + 2, compass_y, 'X+', ha='left', fontsize=10, fontweight='bold')
    
    # Legend
    legend_handles = []
    for class_name in sorted(classes_present):
        color = CLASS_COLORS.get(class_name, '#AAAAAA')
        count = scenario_data[scenario_data['class_name'] == class_name].groupby(['file_name', 'id']).ngroups
        legend_handles.append(mpatches.Patch(color=color, label=f'{class_name} ({count})'))
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                     markersize=12, label='Start', markeredgecolor='black'))
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                     markersize=12, label='End', markeredgecolor='black'))
    legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', 
                                     markersize=14, label='Camera', markeredgecolor='black'))
    ax.legend(handles=legend_handles, loc='upper left', fontsize=10, framealpha=0.95)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('World X (meters from camera)', fontsize=11)
    ax.set_ylabel('World Y (meters from camera)', fontsize=11)
    ax.set_title(f'{scenario} - Scene View (Camera Reference Frame)', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(output_folder / f'{scenario}_scene.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {scenario}_scene.png")

# ============================================================================
# Summary figure comparing all scenarios (SKIPPED - causes timeout)
# ============================================================================
# print("\n📊 Creating summary comparison figure...")
# fig, axes = plt.subplots(2, 3, figsize=(18, 13))
# axes = axes.flatten()
# for idx, scenario in enumerate(scenarios):
#     ax = axes[idx]
#     scenario_data = combined_df[combined_df['scenario'] == scenario]
#     n_objects = scenario_data.groupby(['file_name', 'id']).ngroups
#     plot_trajectories(ax, scenario_data, f'{scenario}\n({n_objects} objects)', 
#                      show_legend=True, padding=1.0)
# if len(scenarios) < 6:
#     axes[5].axis('off')
# plt.suptitle('All Scenarios Comparison - Real World Trajectories', fontsize=16, fontweight='bold', y=1.01)
# plt.tight_layout()
# fig.savefig(output_folder / 'all_scenarios_comparison.png', dpi=150, bbox_inches='tight')
# plt.close(fig)
# print("✅ all_scenarios_comparison.png")

print("\n" + "=" * 60)
print(f"All visualizations saved to: {output_folder}")
print("=" * 60)
print("\nGenerated files:")
for f in sorted(output_folder.glob('*.png')):
    print(f"  {f.name}")
