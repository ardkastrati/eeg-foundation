
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import scipy.io as io
import uuid
from datasets.datastructure import datastructure

def mi_limb_read_file(file):
    def translate_Event(i):
        if(i == 1): 
            return 'left hand imagined movement'
        elif(i == 2): 
            return 'right hand imagined movement'
        elif(i == 3):
            return 'left foot right foot imagined movement'
        elif(i == 4):
            return 'left hand and right hand imagined movement'
        elif(i == 5):
            return 'left hand and right foot imagined movement'
        elif(i == 6):
            return 'right hand and left foot imagined movement'
        elif(i == 7):
            return 'rest'
    columns =['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
    event_annotation = ["The participants were asked to concentrate mind on performing the indicated motor imagery task kinesthetically rather than a visual type of imagery while avoiding any motion during imagination."]

    mat = io.loadmat(file)
    matrices = mat.get('data')
    label = mat.get('label')
    usr_id = str(uuid.uuid1())
    raw_files, events = [], []
    print(matrices.shape)

    for i in tqdm(range(0, matrices.shape[2])): 
        data = matrices[:,:,i]
        data = np.transpose(data)
        raw_file, raw_events = datastructure(data, columns, file, usr_id, metadata='Right handed, healthy, 23-25 years old, sitting in a chair at one-meter distance in front of a computer screen', 
                        timeseries=translate_Event(label[0,0]), session=1, 
                        filtering='Sampling rate = 1000Hz, band-pass filtering range = 0.5 - 100Hz, additional 50-Hz notch filter was used during data acquisition. Thereafter, the original EEG signals were band-pass filtered between 1 and 40Hz, and then downsampled at 200Hz', 
                        rate=200, 
                        hardware='EEG data recorded from 64 Ag/AgCl scalp electrodes placed according to the International 10/20 System referenced to nose and grounded prefrontal lobe. EEG signals were acquired by a Neuroscan SynAmps2 amplifier',
                        reference="CAR", 
                        dataset="MI_Limb", section="MI", eeg_channels=64, trial=i)
        
        raw_events['Event Description'] = 'Participant imagines the following action: ' + translate_Event(label[0,0])
        raw_events.loc[0:399, ['Event Description']] = 'Participant looks at a white circle at the center of the monitor'
        raw_events.loc[400:599, ['Event Description']] = 'Participant looks at a red circle at the center of the monitor'
        raw_events.loc[1400:1599, ['Event Description']] = 'Participant is in a resting state'
        raw_files.append(raw_file)
        events.append(raw_events)
    return raw_files, events, "standard_1020", 50