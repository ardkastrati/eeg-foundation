import os

# Specify the directory path
directory_path = '/itet-stor/kard/deepeye_storage/foundation_clinical/CLI_UNM_D011/data/PD Gait/ALL_DATA/RAW_DATA'

# Initialize an empty list to store the first three characters of filenames
mat_files_prefixes = []

# Loop through all files in the specified directory
for filename in os.listdir(directory_path):
    # Check if the file ends with .mat
    if filename.endswith('.eeg'):
        # Extract the first three characters of the filename and add to the list
        mat_files_prefixes.append(filename[:-3])

# Display the list
print(set(mat_files_prefixes))