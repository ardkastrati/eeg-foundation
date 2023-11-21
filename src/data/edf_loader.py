import mne

import json
import scipy
import scipy.signal

import numpy as np

import os
import csv
import torch
from torch.utils.data import Dataset


def edf2spectrogram(filename, channel_name, window_size = 1.0, overlap = 0.1, debug = True):

   

    data = mne.io.read_raw_edf(filename, preload = False)
    
    #channel_names = data.ch_names
    
    #compute parameters for scipy.signal.spectgram

    sr = data.info['sfreq']
    n_per_window = int(sr * window_size)
    window = scipy.signal.hamming(n_per_window)
    noverlap = int(sr * overlap)

    spectrograms = []
    
    
    channel_index = data.ch_names.index(channel_name)
    channel_data =  data.get_data()[channel_index]
            
    #compute spectrogram and add to list
    frequencies, times, spg = scipy.signal.spectrogram(x=channel_data, fs=sr, window=window, nperseg = n_per_window, noverlap=n_per_window - noverlap)

    

    
    return spg


    
    

class EDFTESTDataset(Dataset):

    def __init__(self, data_dir, window_size = 1.0, overlap = 0.1, debug= False, img_size = (1024, 128)):

        
        

        #load dataset
        with open(data_dir, 'r') as file:
            data =  json.load(file)

        self.data = data
        #self.edf_paths = get_edf_paths(edf_dir)
        self.overlap = overlap
        self.window_size = window_size
        self.img_size = img_size
        self.debug = debug
        self.width = self.img_size[0]
        self.height = self.img_size[1]
        self.path_prefix = "/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf"  

    def __len__(self):
         return len(self.data)
    
    def __getitem__(self, idx):
        

        path, channels, ref = self.data[idx]

        actual_path = self.path_prefix + path

        spectrogram = edf2spectrogram(actual_path,channel_name=channels[0], window_size=self.window_size, overlap=self.overlap, debug=self.debug)
            
        data_sample = []
                 
        h, w = spectrogram.shape
        spectrogram = spectrogram[0:self.height, :]


        if h < self.height: 

            padding_size = self.height - h
            spectrogram = np.pad(spectrogram, ((padding_size,0), (0, 0)), mode='constant')

        if w < 1024:

            padding_size = 1024 - w
            spectrogram = np.pad(spectrogram, ( (0, 0), (0,padding_size)), mode='constant')
            w = 1024
        
        num_samples = w // self.width
        
        samples = []
        start_ind = 0

        for i in range(num_samples):
                    
            slice = spectrogram[:, i * start_ind:start_ind+self.width]
            samples.append(np.transpose(slice))

        #a channel, transformed to spectrograms that are cut up into the desired shape of transformer input. and the channel name to generate token.
        #data_sample.append((samples, channel_name))

        #for now just trim it to single sample size. 
         
        return samples[0]




        



if __name__ == "__main__":

    dset = EDFTESTDataset("/home/schepasc/eeg-foundation/src/data/debug_json")
    print("getting first spectrogram")
    print(dset[200].shape)
    


    