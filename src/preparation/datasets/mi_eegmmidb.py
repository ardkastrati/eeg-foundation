import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
from decimal import Decimal
from mne.io import concatenate_raws,read_raw_edf
import mne
from datasets.datastructure import datastructure
import os

def mi_eegmmidb_read_file(file):
    def translate_Event(i, run):
        if(i == 1):
            return "rest"
        elif(i == 2):
            if(run == 3 or run == 7 or run == 11):
                return 'left hand movement left fist open and close'
            elif(run == 4 or run == 8 or run == 12):
                return 'left hand imagined movement left fist imagined open and close'
            elif(run == 5 or run == 9 or run == 13):
                return 'left hand and right hand movement left fist and right fist open and close'
            elif(run == 6 or run == 10 or run == 14):
                return 'left hand and right hand movement left fist and right fist imagined open and close'
            else:
                return 'According to dataset description, there might be a fault for event description in this record'

        elif(i == 3):
            if(run == 3 or run == 7 or run == 11):
                return 'right hand movement right fist open and close'
            elif(run == 4 or run == 8 or run == 12):
                return 'right hand imagined movement right fist imagined open and close'
            elif(run == 5 or run == 9 or run == 13):
                return 'left foot and right foot movement open and close'
            elif(run == 6 or run == 10 or run == 14):
                return 'left foot and right foot imagined movement open and close'
            else:
                return 'According to dataset descriptiom, there might be a fault for event description in this record'

    columns = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
                'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10',
                'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
                ]
    event_annotation = 'For events that subjects are asked to open and close or imagine opening and closing one fist, A target appears on either the left or the right side of the screen to indicate left or right fist. For events that subjects need to open and close or imagine opening and closing both fists or both feet, a target appears on either the top or the bottom of the screen to indicate both fists(target on top) or both feet(target on bottom)'

    usr_id = str(uuid.uuid1())
    raw = read_raw_edf(file, preload = False)
    events, event_dict = mne.events_from_annotations(raw)
    raw_file, event_df = datastructure(raw.get_data()[:64, :].T*1e6, columns, file, usr_id, # We take only EEG for now
                        session=int(os.path.basename(file)[5:7]), 
                        rate=160,
                        hardware='64-channel EEG were recorded using the BCI2000 system',
                        dataset="eegmmidb", section="MI", eeg_channels=3, event_annotation=event_annotation)

    event_df['Event Description'] = ''
    for k in range(0, len(events)):
        this_event = events[k]
        start = this_event[0]
        raw_event = this_event[2]
        event = translate_Event(raw_event, int(os.path.basename(file)[5:7]))
        event_df.loc[start:, ['Event Description']] = 'Participant is performing the following action: ' + event
    return [raw_file], [event_df], "standard_1020", 60