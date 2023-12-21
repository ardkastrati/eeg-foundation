import json
import mne 
import time
path_prefix = "/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf" 


with open('first_run_json', 'r') as file:
    data = json.load(file)

data = data[0:200]

time_to_load = {}
j = 0
for i in range(5):
    print(i)
    for edf_file in data:
        j = j + 1
        if j % 50 == 0:
            print(j)
        open_start = time.time()
        path = edf_file['path']
        
        raw_data = mne.io.read_raw_edf(path_prefix+path, preload = True)
        
        
        open_end = time.time() 
        open_time = open_end - open_start
        
        if path in time_to_load:
            time_to_load[path].append(open_time)
        else:
            time_to_load[path] = [open_time]
    with open('load_times' + str(i), 'w') as file:
        json.dump(time_to_load, file)

