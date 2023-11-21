import csv, os
import mne 
import json

def generate_lookup_file(data_dir, lookup_name):

    prefix = "/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf"
    mne.set_log_level("WARNING")
    counter = 0
    edf_paths = []
    sample_index = []
    
    for root, dirs, files in os.walk(top=data_dir):
            for name in files:
                
                if name.endswith('.edf'):
                    p = os.path.join(root, name)
                    
                    edf_paths.append(p)


    print("done getting the paths")

   
        

    size = len(edf_paths)
    for path in edf_paths:
        


        info = mne.io.read_raw_edf(path, preload=False)
        channels = []
        for channel_name in info.info['ch_names']:
            if "EEG" in channel_name:
                channels.append(channel_name)
            
        
        
        counter += 1
        if counter % 10000 == 0:
             print(size - counter)
             print(path)

        ref = ""
        if "01_tcp_ar" in path:
             ref = "AR"
        if "02_tcp_le" in path:
             ref = "LE"
        if "03_tcp_ar_a" in path:
             ref = "ARA"
        if "04_tcp_le_a" in path:
             ref = "LEA"
        shortpath = path.replace(prefix, "")
        sample_index.append((shortpath, channels, ref))



    with open(lookup_name, 'w', ) as file:

        json.dump(sample_index, file)

if __name__ == "__main__":
    generate_lookup_file("/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf/000", "debug_json")
    #generate_lookup_file("/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf", "tueg_index")