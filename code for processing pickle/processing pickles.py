#code for opening pickle and processing in excel 

#%% 
#!pip install pandas openpyxl
#%%

import pickle
import glob
import os
import re
import csv 

#%% 
# # Open file in read-binary mode
# with open('../search_results/sfbo/Original/sfbo_results_run_1.pkl', 'rb') as file:
#     data = pickle.load(file)

# print('Retrieved pickled data:')
# for i, item in enumerate(data):
#     print(f"\nData {i}:")
#     print(item)



#%%

# Path to pickle folder 

###CHANGE HERE 
pickle_folder = r'../search_results/mfbo/Run3_10Dec_Saloni_POI'

# Output CSV file
csv_file = "iterations.csv"

# Dictionary to store all runs
all_runs = {}  # key: run_number, value: list of y_acquired

# Loop through all pickle files
for file in os.listdir(pickle_folder):
    with open(os.path.join(pickle_folder, file), "rb") as f:
        data = pickle.load(f) 
    # Extract y values
    y_max_acquired = data["y_max_acquired"] 
    print(y_max_acquired)

    # Extract run number from filename
    match = re.search(r'run_(\d+)', os.path.basename(file))
    run_number = int(match.group(1)) if match else f"NA_{os.path.basename(file)}"

    # Store list of y values in dictionary
    all_runs[run_number] = y_max_acquired   


#%%

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

#%%

# Write to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    # Header = run numbers
    writer.writerow(sorted(all_runs.keys()))
    # Write all iterations
    writer.writerows(rows)

#print(f"CSV saved to {csv_file} with {len(rows)} iterations and {len(all_runs)} runs.")

#%%v


###CHANGE EXCEL FILE NAME 

import pandas as pd 

df = pd.read_csv("iterations.csv")
subfolder = '../processed_files' 
excel_file = os.path.join(subfolder, "output_POI_nonpadded.xlsx") 
df.to_excel(excel_file, index=False)  # index=False avoids writing the row numbers



#%% 

#print(df) 
df = df.fillna(18.5345)
df


# %% 

#####CHANGE EXCEL FILE NAME

subfolder = '../processed_files' 
excel_file = os.path.join(subfolder, "output_POI.xlsx") 
df.to_excel(excel_file, index=False)  # index=False avoids writing the row numbers



# %%
