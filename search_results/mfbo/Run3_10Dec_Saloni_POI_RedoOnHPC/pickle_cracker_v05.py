import pickle
import glob
import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# Path to pickle folder
pickle_dir = "/Users/paulateeuwen/GitHub/AIChemy/multi-fidelity-BO-of-COFs-for-Xe-Kr-seps/search_results/mfbo/Run3_10Dec_Saloni_POI_RedoOnHPC"
files = [os.path.join(pickle_dir, f) for f in os.listdir(pickle_dir) if f.endswith(".pkl")]
print(f"Found {len(files)} pickle files")
print(files)

# Output CSV file
csv_file = f"{pickle_dir}/iterations_accumulated_cost.csv"

# Dictionary to store all runs
all_runs_iterations = {}  # key: run_number, value: iteration numbers
all_runs_cost = {}  # key: run_number, value: accumulated_cost
all_runs_selectivity = {}  # key: run_number, value: y_max_acquired

# Loop through all pickle files
for file in files:
    with open(file, "rb") as f:
        data = pickle.load(f)
        print(f"\n{file}:")
        print(f"  Keys: {list(data.keys())}")
        print(f"  accumulated_cost shape: {len(data['accumulated_cost'])}")

    # Extract data
    accumulated_cost = data["accumulated_cost"]
    y_max_acquired = data["y_max_acquired"]
    iterations = list(range(len(y_max_acquired)))

    # Extract run number from filename
    match = re.search(r'run_(\d+)', os.path.basename(file))
    run_number = int(match.group(1)) if match else f"NA_{os.path.basename(file)}"

    # Store data in dictionaries
    all_runs_iterations[run_number] = iterations
    all_runs_cost[run_number] = accumulated_cost
    all_runs_selectivity[run_number] = y_max_acquired

# Determine the maximum number of iterations across all runs
max_iterations = max(len(vals) for vals in all_runs_selectivity.values())

# Prepare CSV rows: each row = one iteration across all runs
rows = []
for i in range(max_iterations):
    row = []
    for run in sorted(all_runs_selectivity.keys()):  # optional: sort runs by number
        run_vals = all_runs_selectivity[run]
        # If this run doesn't have this iteration, fill with empty string
        row.append(run_vals[i] if i < len(run_vals) else "")
    rows.append(row)

# Write to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    # Header = run numbers
    writer.writerow(sorted(all_runs_selectivity.keys()))
    # Write all iterations
    writer.writerows(rows)

print(f"CSV saved to {csv_file} with {len(rows)} iterations and {len(all_runs_selectivity)} runs.")

# Create Plot 1: Selectivity vs Iterations
plt.figure(figsize=(10, 6))
for run in sorted(all_runs_iterations.keys()):
    x_data = all_runs_iterations[run]
    y_data = all_runs_selectivity[run]
    plt.plot(x_data, y_data, alpha=0.5, linewidth=1)

plt.xlabel('Iteration')
plt.ylabel('Xe/Kr Selectivity')
plt.title('Xe/Kr Selectivity vs Iteration')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file_1 = f"{pickle_dir}/selectivity_vs_iterations.png"
plt.savefig(plot_file_1, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_file_1}")
plt.close()

# Create Plot 2: Selectivity vs Accumulated Cost
plt.figure(figsize=(10, 6))
for run in sorted(all_runs_cost.keys()):
    x_data = all_runs_cost[run]
    y_data = all_runs_selectivity[run]
    plt.plot(x_data, y_data, alpha=0.5, linewidth=1)

plt.xlabel('Accumulated Cost [hours]')
plt.ylabel('Xe/Kr Selectivity')
plt.title('Xe/Kr Selectivity vs Accumulated Cost')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file_2 = f"{pickle_dir}/selectivity_vs_cost.png"
plt.savefig(plot_file_2, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_file_2}")
plt.close()

# Create Plot 3: Average Selectivity with Standard Deviation vs Iterations
# Pad all runs to same length by extending the last value
max_iterations = max(len(vals) for vals in all_runs_selectivity.values())
padded_selectivity_iterations = []

for run in sorted(all_runs_selectivity.keys()):
    y_vals = all_runs_selectivity[run]
    
    # Pad by extending the last value
    if len(y_vals) < max_iterations:
        y_padded = np.concatenate([y_vals, [y_vals[-1]] * (max_iterations - len(y_vals))])
    else:
        y_padded = y_vals
    
    padded_selectivity_iterations.append(y_padded)

padded_array_iter = np.array(padded_selectivity_iterations)

# Calculate mean and std
mean_selectivity = np.mean(padded_array_iter, axis=0)
std_selectivity = np.std(padded_array_iter, axis=0)
iterations_mean = np.arange(len(mean_selectivity))

plt.figure(figsize=(10, 6))
plt.plot(iterations_mean, mean_selectivity, color='black', linewidth=2, label='Mean')
plt.fill_between(iterations_mean, 
                 mean_selectivity - std_selectivity, 
                 mean_selectivity + std_selectivity, 
                 alpha=0.3, color='blue', label='±1 Std Dev')
plt.xlabel('Iteration')
plt.ylabel('Xe/Kr Selectivity')
plt.title('Mean Xe/Kr Selectivity with Standard Deviation (vs Iteration)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file_3 = f"{pickle_dir}/selectivity_mean_std_iterations.png"
plt.savefig(plot_file_3, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_file_3}")
plt.close()

# Create Plot 4: Average Selectivity with Standard Deviation vs Accumulated Cost
# For this, we need to interpolate to a common cost axis
cost_min = min(min(costs) for costs in all_runs_cost.values())
cost_max = max(max(costs) for costs in all_runs_cost.values())
common_cost = np.linspace(cost_min, cost_max, 100)

interpolated_selectivity = []
for run in sorted(all_runs_cost.keys()):
    x_cost = all_runs_cost[run]
    y_vals = all_runs_selectivity[run]
    # Interpolate this run's selectivity to common cost axis
    y_interp = np.interp(common_cost, x_cost, y_vals)
    interpolated_selectivity.append(y_interp)

interpolated_array = np.array(interpolated_selectivity)
mean_selectivity_cost = np.mean(interpolated_array, axis=0)
std_selectivity_cost = np.std(interpolated_array, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(common_cost, mean_selectivity_cost, color='black', linewidth=2, label='Mean')
plt.fill_between(common_cost, 
                 mean_selectivity_cost - std_selectivity_cost, 
                 mean_selectivity_cost + std_selectivity_cost, 
                 alpha=0.3, color='blue', label='±1 Std Dev')
plt.xlabel('Accumulated Cost [hours]')
plt.ylabel('Xe/Kr Selectivity')
plt.title('Mean Xe/Kr Selectivity with Standard Deviation (vs Accumulated Cost)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file_4 = f"{pickle_dir}/selectivity_mean_std_cost.png"
plt.savefig(plot_file_4, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_file_4}")
plt.close()
