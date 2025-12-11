import pickle
import glob
import os
import re
import csv

# Path to pickle folder
pickle_folder = "Run3_10Dec_Saloni_POI/*.pkl"

# Output CSV file
csv_file = "iterations.csv"

# Dictionary to store all runs
all_runs = {}  # key: run_number, value: list of y_acquired

# Loop through all pickle files
for file in glob.glob(pickle_folder):
    with open(file, "rb") as f:
        data = pickle.load(f)

    # Extract y values
    y_max_acquired = data["y_max_acquired"]

    # Extract run number from filename
    match = re.search(r'run_(\d+)', os.path.basename(file))
    run_number = int(match.group(1)) if match else f"NA_{os.path.basename(file)}"

    # Store list of y values in dictionary
    all_runs[run_number] = y_max_acquired

# Determine the maximum number of iterations across all runs
max_iterations = max(len(vals) for vals in all_runs.values())

# Prepare CSV rows: each row = one iteration across all runs
rows = []
for i in range(max_iterations):
    row = []
    for run in sorted(all_runs.keys()):  # optional: sort runs by number
        run_vals = all_runs[run]
        # If this run doesn't have this iteration, fill with empty string
        row.append(run_vals[i] if i < len(run_vals) else "")
    rows.append(row)

# Write to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    # Header = run numbers
    writer.writerow(sorted(all_runs.keys()))
    # Write all iterations
    writer.writerows(rows)

print(f"CSV saved to {csv_file} with {len(rows)} iterations and {len(all_runs)} runs.")
