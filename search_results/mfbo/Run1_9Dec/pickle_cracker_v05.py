import pickle
import glob
import os
import re
import csv
import matplotlib.pyplot as plt

# Path to pickle folder
pickle_dir = "/Users/paulateeuwen/GitHub/AIChemy/multi-fidelity-BO-of-COFs-for-Xe-Kr-seps/search_results/mfbo/Run1_9Dec"
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
