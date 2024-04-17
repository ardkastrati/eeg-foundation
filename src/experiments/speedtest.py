import mne

import json
import scipy
import scipy.signal
import random
import numpy as np
import time
import os
import csv
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchaudio
from torch.nn.functional import interpolate

filename = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf//000/aaaaaadj/s003_2004_05_13/02_tcp_le/aaaaaadj_s003_t001.edf"
path_prefix = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"
data = mne.io.read_raw_edf(filename)
with open("/home/maxihuber/eeg-foundation/src/data/000_json", 'r') as file:
    edf_data =  json.load(file)
    
    print(len(edf_data))
with open("/home/maxihuber/tueg_json", 'r') as file:
    whole_data = json.load(file)

index = []
sr = data.info['sfreq']
n_per_window = int(sr * 4.0)
noverlap = int(sr * 0.25)
channel_data_scipy = channel_data =  data.get_data()[0]
channel_data_torch = data[data.ch_names[0]][0]
#convert to my_volt
channel_data = channel_data * 1000000


def build_channel_index(min_duration = 0.0, specific_sr = -1.0):
        
    #read the json_dictionaries and generate an index. can filter by minimum duration (in seconds)
    # or select only specific sampling freq.
    
    for edf_file in whole_data:
       
        channel_names = edf_file['channels']
        path = edf_file['path']
        sampling_rate = edf_file['sr']
        ref = edf_file['ref']
        duration = edf_file['duration']

        if(duration > 360 and (sampling_rate == 250 or sampling_rate == 256)):
            chn = channel_names[0]
            index.append((path, chn, ref))

        

def test_scipy(data):
    print("Starting SCIPY Test:")
    start_time = time.time()
    num_exec = 200
    for i in range (num_exec):
        #avoid caching
        data = data + 1
        window = scipy.signal.windows.hamming(n_per_window)
        _, _, spg = scipy.signal.spectrogram(x=data, fs=sr, window=window, nperseg = n_per_window, noverlap=n_per_window - noverlap)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time to compute {num_exec} spectrograms with scipy: {total_time} seconds")

def test_torch(data):
    print("Starting torch test:")
    torch_fft = torchaudio.transforms.Spectrogram(n_fft=1000, win_length=1000, hop_length=62, normalized=True)
    torch_fft = torch_fft.cuda()
    start_time = time.time()
    num_exec = 200
    for i in range (num_exec):
        #avoid caching
        data = data + 1
        channel_tensor = torch.from_numpy(data)
        channel_tensor = channel_tensor.cuda()
        spectro = torch_fft(channel_tensor)
        spectro = spectro.cpu()
      
        
        spectro = 20 * np.log10(spectro)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time to compute {num_exec} spectrograms with Torchaudio(Cuda): {total_time} seconds")
build_channel_index()
num_files = len(index)
print(f"Loading {num_files} channels")
start_time = time.time()
indices = list(range(0, len(index)))
random.shuffle(indices)
print("opening this many edf files:" + str(len(index)))
count = 0
for idx in indices:
    if count % 1000 == 0:
        print("at" + str(count))
    count = count + 1
    path, _ ,_ = index[idx]
    loaded_data = mne.io.read_raw_edf(path_prefix + path)
end_time = time.time()
total_time = end_time - start_time
print(f"Loading {num_files} edf files took {total_time} seconds")






test_scipy(channel_data_scipy)
test_torch(channel_data_torch)


