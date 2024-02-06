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
from torch.utils.data import DataLoader
import multiprocessing
from torchvision import transforms

class EDFDataset(Dataset):

    def __init__(self,
         data_dir, 
         window_size = 4.0, 
         window_shift = 0.25, 
         debug= False, 
         target_size = (128, 1024),
         min_duration = 316, 
         select_sr = [250, 256],  
         select_ref = ['AR'],
         random_sample = False, 
         fixed_sample = True,
         use_cache = False,
         interpolate_250to256 = True):

        
        print("=====================")
        print(data_dir)
        #load dataset
        with open(data_dir, 'r') as file:
            data =  json.load(file)

        self.data = data
        self.index = []

        self.build_channel_index(min_duration=min_duration, select_sr=select_sr, select_ref=select_ref)

        #parameters for fft
        self.window_shift = window_shift
        self.window_size = window_size
        self.target_size = target_size

        self.fft_256 = torchaudio.transforms.Spectrogram(n_fft=int(window_size * 256), win_length=int(window_size * 256), hop_length= int(window_shift * 256)
                                                         , normalized=True)
        
        self.img_height = self.target_size[0]
        self.target_width = self.target_size[1]
        
        self.crop = CropSpectrogram(target_size=target_size)
        self.transform = transforms.Compose([self.fft_256, self.crop])
        self.interpolate_250to256 = interpolate_250to256
        self.select_sr = select_sr


        self.path_prefix = "/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf" 

        self.load_times = []
        self.transform_times = []

        self.shmpath = "/dev/shm/spec/"
    def get_times(self):

        return np.mean(self.load_times), np.mean(self.transform_times)

    def build_channel_index(self, min_duration = 0.0, select_sr = [256], select_ref = ['AR']):
        
        #read the json_dictionaries and generate an index. can filter by minimum duration (in seconds)
        # or select only specific sampling freq.
        
        for edf_file in self.data: 

            channel_names = edf_file['channels']
            path = edf_file['path']
            sampling_rate = edf_file['sr']
            ref = edf_file['ref']
            duration = edf_file['duration']

            if sampling_rate in select_sr and ref in select_ref and duration >= min_duration and duration <= 1260:
                for chn in channel_names:
                    self.index.append((path, chn, ref))

        print(len(self.index))



    def load_channel_data(self, path, chn, pre_crop = True):

        #takes as input an index and returns a tensor containing the channel_data associated with that index
        #open_start = time.time()
        full_path = self.path_prefix + path
        #include = chn -> only loads the data for the channel we want the sample from.
        edf_data = mne.io.read_raw_edf(full_path, include=chn, preload=False)
        
        #open_end = time.time()
        #open_time = open_end - open_start
        data = edf_data.get_data()
        #print("opening took" + str(open_time))

        sr = int(edf_data.info['sfreq'])
        
        channel_data = edf_data[chn][0]
        #convert to myvolt
        channel_data = channel_data * 1000000
        
        channel_data = torch.from_numpy(channel_data)
        
        channel_data = channel_data.squeeze(0)
        #cropping, removing first minute
        if pre_crop:
            
            target_length = self.window_shift * self.target_width * sr
            
            
            channel_data = channel_data[sr:int(sr + target_length)]
            
            
        
        #interpolate to 256sr     
        if self.interpolate_250to256 and sr == 250:
            
            new_length = (len(channel_data) // 250) * 256
            channel_data = interpolate(channel_data.unsqueeze(0).unsqueeze(0), new_length, mode='nearest').squeeze(0).squeeze(0)
        
        
        
        return channel_data


    
    def __len__(self):
        
        return len(self.index)
    
    def __getitem__(self, idx):
        
       
        

        


        path, chn, ref = self.index[idx]
        channel_data = self.load_channel_data(path, chn)
       

        #apply transforms
        
        spectrogram = self.transform(channel_data)

        
        #convert to DB
        
        spectrogram = 20 * torch.log10(spectrogram)
        
        #normalize on per sample basis
        spectrogram = (spectrogram - torch.mean(spectrogram)) / (torch.std(spectrogram) * 2)
        
        
        #transform_time = transform_end_time - transform_start_time
        

        

        return spectrogram    

class CropSpectrogram(object):

    def __init__(self, target_size):

        self.target_height = target_size[0]
        self.target_width = target_size[1]

    def __call__(self, spectrogram):

        #crop vertically
        spectrogram = spectrogram[4:68, :]

        #crop&pad horizontally (padding shouldn't be necessary because of our sample selection but just in case)

        width_pad = self.target_width - spectrogram.shape[1]

        if width_pad > 0:
            pad= torch.nn.ZeroPad2d((0, width_pad, 0, 0))
            spectrogram = pad(spectrogram)
        elif width_pad < 0:
            spectrogram = spectrogram[:, 0:self.target_width]
        #add 1 'image channel'
        return spectrogram.unsqueeze(0)
    

class EDFCacheDataset(Dataset):

    def __init__(self, cache):
        
        self.cache = cache

    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx): 
        return self.cache[idx] 
    










class EDFDiscDataset(Dataset):

    def __init__(self, file_index):

        self.file_index = file_index
    
    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):

        path = self.file_index[idx]

        spectro = np.load(path)
        spectro = torch.from_numpy(spectro)
        
        return spectro.unsqueeze(0)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    dset = EDFDataset("/home/schepasc/eeg-foundation/src/data/000_json", select_sr=[250, 256], min_duration=316)
    
    
  
    spectro = dset[0]
    
            
          
    
    
    print(spectro)
    
    load, trs = dset.get_times()
    print("load time average:" + str(load))
    print("transformtime average: " + str(trs))