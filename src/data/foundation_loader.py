import mne

import scipy
import scipy.signal

import numpy as np

import os

import torch
from torch.utils.data import Dataset

def get_edf_paths(data_path): 
        
        #make mne stop spamming the console

        mne.set_log_level("WARNING")

        #save the paths in a textfile
        edf_paths = []
        debug_file = "/home/schepasc/eeg-foundation/debug_paths"

        if os.path.exists(debug_file):
            with open(debug_file, 'r') as file:
                edf_paths = [line.strip() for line in file.readlines()]
        else:
        
            for root, dirs, files in os.walk(top=data_path):
                for name in files:
                    if name.endswith('.edf'):
                        edf_paths.append(os.path.join(root, name))

            with open(debug_file, 'w') as file:
                for path in edf_paths:
                    file.write(f"{path}\n")

        return edf_paths

def edf2spectrogram(filename, window_size = 1.0, overlap = 0.1, single_channel = True):

   

    data = mne.io.read_raw_edf(filename)
    raw_data = data.get_data()

    channel_names = data.ch_names
    

    #compute parameters for scipy.signal.spectgram

    sr = data.info['sfreq']
    n_per_window = int(sr * window_size)
    window = scipy.signal.hamming(n_per_window)
    noverlap = int(sr * overlap)

    spectrograms = []
    
    for chn in channel_names:
        
        #array of voltages
        channel_index = data.ch_names.index(chn)
        channel_data =  data.get_data()[channel_index]
            
        #compute spectrogram and add to list
        frequencies, times, spg = scipy.signal.spectrogram(x=channel_data, fs=sr, window=window, nperseg = n_per_window, noverlap=n_per_window - noverlap)

        #padding with 0 so it fits transformer
        if len(spg) < 128:

            padding_size = 128-len(spg)
            spg = np.pad(spg, ((0, padding_size), (0, 0)), mode='constant')


        spectrograms.append((spg, chn))

        
        
        #only compute spectrogram of one channel
        if single_channel:
            break


    return spectrograms

class EDFTESTDataset(Dataset):

    def __init__(self, edf_dir):

        self.edf_dir = edf_dir
        self.edf_paths = get_edf_paths(edf_dir)
          

    def __len__(self):
         return len(self.edf_paths)
    
    def __getitem__(self, idx):
         
        path = self.edf_paths[idx]

        #for testing just take one spectrogram
        spg, channel = edf2spectrogram(path)[0]

        #making it fit the dimensions wanted by MAE-transformer, padding it if sample is too short in time/ cutting if too long

        spg = spg[0:128, 0:1024]
        spg = np.transpose(spg)
        expected_shape = (1024, 128)

        if spg.shape != expected_shape:
           
           h, w = spg.shape
           padding_size = 1024 - h
           spg = np.pad(spg, ((padding_size,0), (0, 0)), mode='constant')
           
        return spg



if __name__ == "__main__":

    dset = EDFTESTDataset("/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf/000")
    print("getting first spectrogram")
    print(dset[0].shape)


