import csv, os
import mne 
import json
from tabulate import tabulate
file_list = []
with open("data_exploration.txt", 'w') as out_file: 
    with open("/home/schepasc/tueg_json" , 'r') as file:
        file_list = json.load(file)
    out_file.write("Correctness-Checks: \n\n" )
    out_file.write("There are this many EDF-Files:  " + str(len(file_list)) + "\n")

    durations = [f.get('duration') for f in file_list]
    out_file.write("average duration is:    "  + str  (sum(durations) / len(durations)) + "\n\n\n")
    out_file.write("********************************************************************\n")
    for i in range(0, 1600, 240):
        minutes = i // 60
        longer_than = [f for f in file_list if (f.get('duration') > i)]
        number = len(longer_than)
        out_file.write(f" {number} files are longer than {minutes} minutes\n")

    
    out_file.write("********************************************************************\n")

    

    longer_than_six = [f for f in file_list if (f.get('duration') > 256 and f.get('ref') == 'AR'  and f.get('sr') == 256)]
    out_file.write(f"There are {len(longer_than_six)} edf files which are longer than 256 seconds, have SR of 256 and use average reference\n")

    sum_of_channels = sum(len(f.get('channels' , [])) for f in longer_than_six)
    out_file.write(f"There are {sum_of_channels} EEG-Channels in files longer than 256 seconds, SR=256, AR\n")


    longer_than_six = [f for f in file_list if (f.get('duration') > 256 and f.get('ref') == 'AR'  and f.get('sr') == 250)]
    out_file.write(f"There are {len(longer_than_six)} edf files which are longer than 256 seconds, have SR of 250 and use average reference\n")

    sum_of_channels = sum(len(f.get('channels' , [])) for f in longer_than_six)
    out_file.write(f"There are {sum_of_channels} EEG-Channels in files longer than 256 seconds, SR=250, AR\n")

    unique_sr_values = set(entry["sr"] for entry in file_list)
    print(unique_sr_values)
   
    # Create a dictionary to store counts
    count_dict = {}

    # Count the occurrences of each (sr, duration interval) combination
    for entry in file_list:
        sr = entry["sr"]
        duration = entry["duration"]
        
        # Group duration into intervals of 120 seconds
        duration_interval = min((duration // 120) * 120, 1600)

        if (sr, duration_interval) in count_dict:
            count_dict[(sr, duration_interval)] += 1
        else:
            count_dict[(sr, duration_interval)] = 1

    # Extract unique 'sr' and 'duration_interval' values
    unique_sr = sorted(set(sr for sr, duration_interval in count_dict.keys()))
    unique_duration_intervals = sorted(set(duration_interval for sr, duration_interval in count_dict.keys()))

    # Create a table with counts
    table_data = []
    for duration_interval in unique_duration_intervals:
        row = [duration_interval]
        for sr in unique_sr:
            count = count_dict.get((sr, duration_interval), 0)
            row.append(count)
        table_data.append(row)

    # Add headers to the table
    headers = ["Duration Interval"] + [f"SR {sr}" for sr in unique_sr]

    # Use tabulate to get the formatted table
    formatted_table = tabulate(table_data, headers, tablefmt="grid")

    # Save the formatted table to a file
    with open("tueg_table.txt", "w") as file:
        file.write(formatted_table)

    print("Table saved to 'tueg_table.txt'")


