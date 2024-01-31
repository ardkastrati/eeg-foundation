import sys

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

import torchaudio
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
import multiprocessing
from torchvision import transforms


    
def load_channel_data():


    def __call__(self, data, precrop = True, crop_idx = [60, 316]):
        
        
        #takes as input a touple of path and string and returns the channaldata cropped to the timeframe [1min:5min15seconds]
        path = data['path']
        chn = data['chn']
        
        #include = chn -> only loads the data for the channel we want the sample from.
        edf_data = mne.io.read_raw_edf(path, include=chn, preload=False)

        data = edf_data.get_data()
        sr = int(edf_data.info['sfreq'])
        
        channel_data = edf_data[chn][0]

        #convert to u_volt
        channel_data = channel_data * 1000000
        
        channel_data = torch.from_numpy(channel_data)
        
        channel_data = channel_data.squeeze(0)
        #cropping, removing first minute
        if precrop: 
            channel_data = channel_data[sr * crop_idx[0]:sr * crop_idx[1]]
            
            
        
        #interpolate to 256sr     
        if sr == 250:
            
            new_length = (len(channel_data) // 250) * 256
            channel_data = interpolate(channel_data.unsqueeze(0).unsqueeze(0), new_length, mode='nearest').squeeze(0).squeeze(0)
        
        return channel_data
    

def build_channel_index(data_dir, path_prefix, max_duration=1200, min_duration = 0.0, select_sr = [256], select_ref = ['AR']):
            
    #read the json_dictionaries and generate an index. can filter by minimum duration (in seconds)
    # or select only specific sampling freq.
    index = []
    with open(data_dir, 'r') as file:
        data =  json.load(file)
    for edf_file in data: 

        channel_names = edf_file['channels']
        path = edf_file['path']
        sampling_rate = edf_file['sr']
        ref = edf_file['ref']
        duration = edf_file['duration']

        if sampling_rate in select_sr and ref in select_ref and duration >= min_duration and duration <= max_duration:
            for chn in channel_names:
                index.append({'path' : path_prefix+path, 'chn' : chn})
        
        return index
            
            
            
    