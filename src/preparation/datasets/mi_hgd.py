import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
from decimal import Decimal
from braindecode.datasets.bbci import  BBCIDataset
from mne.io import concatenate_raws,read_raw_edf
import mne
from datasets.datastructure import datastructure


def mi_hgd_read_file(file):
    columns = ['FP1', 'FP2', 'FPZ', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 'EOG EOGh', 'EOG EOGv', 'EMG EMG_RH', 'EMG EMG_LH', 'EMG EMG_RF', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h', 'PO9', 'PO10', 'P9', 'P10', 'AFF1', 'AFZ', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h']
    event_annotation = 'Visual cues (arrows) were presented using a monitor. Depending on the direction of a gray arrow that was shown on black background, the subjects had to repetitively clench their toes (downward arrow), perform sequential finger-tapping of their left (leftward arrow) or right (rightward arrow) hand, or relax (upward arrow). Arrows displayed for 4 s each, with 3 to 4 s of continuous random inter-trial interval'
    def translate_Event(i): 
        if(i == 1):
            return 'left foot and right foot movement clenching toes'
        elif(i == 2):
            return 'left hand movement sequential finger tapping'
        elif(i == 3):
            return 'rest'
        elif(i == 4):
            return 'right hand movement sequential finger tapping'
    usr_id = str(uuid.uuid1())
    raw = read_raw_edf(file, preload = False)
    events, event_dict = mne.events_from_annotations(raw)
    raw_file, event_df = datastructure(raw.get_data()[:133, :].T*1e6, columns, file, usr_id, # We take only EEG for now
                        session=1, 
                        metadata='healthy subjects, 6 female, 2 left-handed, age 27.2±3.6 (mean±std)',
                        rate=5000,
                        hardware='Data recorded with BCI2000. EEG setup: active electromagnetic shielding, shielded window, ventilation & cable feedthrough, high-resolution and low-noise amplifiers, actively shielded EEG caps: 128 channels (WaveGuard Original, ANT, Enschede, NL) and full optical decoupling',
                        dataset="eegmmidb", section="MI", eeg_channels=3, event_annotation=event_annotation)
    previous = 0
    for j in range(0, len(events)):
        this_event = events[j]
        end = this_event[0]
        raw_event = this_event[2]
        event = translate_Event(raw_event)
        if(event == 'left foot and right foot movement clenching toes' or event == 'left hand movement sequential finger tapping' or event == 'right hand movement sequential finger tapping'):
            event_df.loc[previous:previous+1999, 'Event Description'] = 'Participant is performing the following action: ' + event
        else:
            event_df.loc[previous:previous+1999, 'Event Description'] = 'Participant is performing the following action: rest'
        event_df.loc[previous:end-1, ['Trial']] = j + 1
        previous = end
    return [raw_file], [event_df], "standard_1005", 50