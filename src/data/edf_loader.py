import mne

import json
import scipy
import scipy.signal
import random
import numpy as np

import os
import csv
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchaudio


def edf2spectrogram(filename, channel_name, window_size = 4.0, overlap = 0.25, debug = True):

   

    data = mne.io.read_raw_edf(filename, preload = False)
    
    #channel_names = data.ch_names
    
    #compute parameters for scipy.signal.spectgram

    sr = data.info['sfreq']
    n_per_window = int(sr * window_size)
    window = scipy.signal.windows.hamming(n_per_window)
    noverlap = int(sr * overlap)

    spectrograms = []
    
   
    
    channel_index = data.ch_names.index(channel_name)
    channel_data =  data.get_data()[channel_index]
    #convert to volt
    channel_data = channel_data * 1000000

    if len(channel_data) < n_per_window:
        channel_data = np.pad(channel_data, (1, n_per_window - len(channel_data)), mode = 'constant')
        
                              
    #compute spectrogram and add to list
    
    frequencies, times, spg = scipy.signal.spectrogram(x=channel_data, fs=sr, window=window, nperseg = n_per_window, noverlap=n_per_window - noverlap)

    #convert to decibel, add small value to avoid divide by 0 exception 
    spg += 1e-5
    spg = 20 * np.log10(spg)
    """
    spectransform = torchaudio.transforms.Spectrogram(n_fft=256, win_length=n_per_window, hop_length=noverlap)
    spg = spectransform(channel_data)
    """
    
    return spg


    
    

class EDFDataset(Dataset):

    def __init__(self,
         data_dir, 
         window_size = 4.0, 
         overlap = 0.25, 
         debug= False, 
         img_size = (1024, 128),
         min_duration = 200.0, 
         specific_sr = 256.0,  
         random_sample = False, 
         fixed_sample = True):

        
        print("=====================")
        print(data_dir)
        #load dataset
        with open(data_dir, 'r') as file:
            data =  json.load(file)
        self.random_sample = random_sample
        self.data = data
        self.index = []
        #self.edf_paths = get_edf_paths(edf_dir)
        self.overlap = overlap
        self.window_size = window_size
        self.img_size = img_size
        self.debug = debug
        self.img_width = self.img_size[0]
        self.img_height = self.img_size[1]
        self.path_prefix = "/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf"  
        self.fixed_sample = fixed_sample
        
        self.min_duration = min_duration
        self.specific_sr = specific_sr
        self.build_channel_index(min_duration= self.min_duration, specific_sr=self.specific_sr)
    def __len__(self):
        
        return len(self.index)
    
    def build_channel_index(self, min_duration = 0.0, specific_sr = -1.0):
        
        #read the json_dictionaries and generate an index. can filter by minimum duration (in seconds)
        # or select only specific sampling freq.
        for edf_file in self.data: 

            channel_names = edf_file['channels']
            path = edf_file['path']
            sampling_rate = edf_file['sr']
            ref = edf_file['ref']
            duration = edf_file['duration']

            if (specific_sr == -1 or sampling_rate == specific_sr) and duration >= min_duration:
                for chn in channel_names:
                    self.index.append((path, chn, ref))

        print(len(self.index))

    def __getitem__(self, idx):
        

        path, channel, ref = self.index[idx]

        actual_path = self.path_prefix + path
        spectrogram = edf2spectrogram(actual_path,channel_name=channel, window_size=self.window_size, overlap=self.overlap, debug=self.debug)
            
        data_sample = []

        
        h, w = spectrogram.shape
        
        spectrogram = spectrogram[0:self.img_height, :]

        
        if h < self.img_height: 

            padding_size = self.img_height - h
            spectrogram = np.pad(spectrogram, ((0, padding_size), (0, 0)), mode='constant')

        if w < self.img_width:

            padding_size = self.img_width - w
            spectrogram = np.pad(spectrogram, ( (0, 0), (0,padding_size)), mode='constant')
            
        
        
        
        samples = []
        
        #if random sample, select a slice starting anywhere that fits
        if self.random_sample: 
            
            max_startpoint = spectrogram.shape[1] - self.img_width

            if max_startpoint == 0:
                ret = spectrogram
            else:
                startpoint = random.randint(0, max_startpoint)
                ret = spectrogram[:, startpoint:startpoint+self.img_width]

                #norm the spg on per sample basis.
                ret = (ret - np.mean(ret)) / (2 * np.std(ret))

        
        else:
            start_ind = 0
            num_samples =spectrogram.shape[1] // self.img_width
            
            for i in range(num_samples):
                    
                slice = spectrogram[:, i * start_ind:start_ind+self.img_width]
                samples.append(slice)
                ret = samples
            #this is for overfitting testing
            if self.fixed_sample:

                ret = samples[1]
                #norm the spg on per sample basis.
                ret = (ret - np.mean(ret)) / (2 * np.std(ret))
                
        #for now just trim it to single sample size. 
        
        #return spectrogram
        
        #add img_channels, in our case only one
        ret = np.expand_dims(ret, axis=0)
        return ret




        
def calc_mean_std(num_samples, data_set):

    #not used at the moment, using per sample normalizing.
    mean = 0
    for i in range(num_samples):

        smp = data_set[i][0]
        m = np.mean(smp)
        mean += m

    mean = mean / num_samples
    standard_deviation = 0

    for i in range(num_samples):
        smp = data_set[i][0]
        std = np.sqrt(np.sum((smp  - mean)**2) / 131072)
        standard_deviation += std
    standard_deviation = standard_deviation / num_samples
    return mean, standard_deviation

class one_image_dataset(Dataset):
    #class to overfit on single images.
    def __init__(self):
        with open('/home/schepasc/eeg-foundation/nice_spectro', 'r') as file:
            self.data= json.load(file)
            self.data = np.transpose(self.data)
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        ret = np.expand_dims(self.data, axis=0)
        return ret


if __name__ == "__main__":

    dset = EDFDataset("/home/schepasc/eeg-foundation/src/data/000_json")
    print("getting first spectrogram")
    spectro = dset[7]
    mean = np.mean(spectro[0])
    print(spectro.shape)
    print(spectro)
    print(mean)
    
    