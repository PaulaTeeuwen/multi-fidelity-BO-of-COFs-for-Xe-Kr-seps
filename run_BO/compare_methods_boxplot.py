import pickle
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Methods to compare - specify folder names relative to search_results/mfbo/
methods = [
    "Original",
    "Run3_10Dec_Saloni_POI",
    "Run3_10Dec_Saloni_POI_RedoOnHPC",
    "Run5_10Dec_UCB_B004_new",
    "Run5_10Dec_UCB_B2_new",
    "Run5_10Dec_UCB_B4_new",
    "Run5_10Dec_UCB_B25_new",
    "Run5_10Dec_UCB_B100_new",
]

# Custom labels for each method (same order as methods list)
method_labels = [
    "Original",
    "POI",
    "POI Redo",
    "UCB β=0.04",
    "UCB β=2",
    "UCB β=4",
    "UCB β=25",
    "UCB β=100",
]

# Get the path to mfbo directory from current location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
mfbo_base = os.path.join(project_root, 'search_results', 'mfbo')

# Dictionary to store accumulated costs per method
method_costs = {}

# Loop through each method and collect final accumulated costs
for method in methods:
    method_dir = os.path.join(mfbo_base, method)
    
    if not os.path.exists(method_dir):
        print(f"Skipping {method}: directory not found")
        continue
    
    # Get all pickle files (exclude ablation)
    pkl_files = [f for f in os.listdir(method_dir) if f.endswith(".pkl") and "_ablation" not in f]
    
    if len(pkl_files) == 0:
        print(f"Skipping {method}: no pickle files found")
        continue
    
    final_costs = []
    
    for pkl_file in sorted(pkl_files):
        try:
            with open(os.path.join(method_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)
                # Get the final accumulated cost (last value in the array)
                final_cost = data['accumulated_cost'][-1]
                final_costs.append(final_cost)
        except Exception as e:
            print(f"Error reading {pkl_file} in {method}: {e}")
            continue
    
    if final_costs:
        method_costs[method] = final_costs
        print(f"{method}: {len(final_costs)} runs, mean cost = {np.mean(final_costs):.2f}, std = {np.std(final_costs):.2f}")

# Create box plot comparing all methods
plt.figure(figsize=(12, 6))
methods_to_plot = [m for m in methods if m in method_costs]
costs_to_plot = [method_costs[m] for m in methods_to_plot]

# Get corresponding labels for methods that have data
labels_to_plot = [method_labels[methods.index(m)] for m in methods_to_plot]

bp = plt.boxplot(costs_to_plot, tick_labels=labels_to_plot, patch_artist=True, showfliers=True)

# Color the boxes
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

# Add number of runs on top of each boxplot
ax = plt.gca()
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
offset = y_range * 0.02  # 2% of the y-axis range
for i, method in enumerate(methods_to_plot):
    n_runs = len(method_costs[method])
    # Get the maximum value for this boxplot to position text above it
    y_max = max(method_costs[method])
    ax.text(i + 1, y_max + offset, f'n={n_runs}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xlabel('Method')
plt.ylabel('Final Accumulated Cost [hours]')
plt.title('Distribution of Accumulated Cost by Method')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

output_file = os.path.join(mfbo_base, 'method_comparison_boxplot.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {output_file}")
plt.close()
