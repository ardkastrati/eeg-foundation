import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
import mne
from datasets.datastructure import datastructure




def mi_ulm_read_file(file):
    columns = ['F3', 'F1', 'FZ', 'F2', 'F4', 'FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h',
           'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h',
           'FCC6h', 'FTT8h', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h',
           'CCP4h', 'CCP6h', 'TTP8h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'CPP5h', 'CPP3h', 'CPP1h',
           'CPP2h', 'CPP4h', 'CPP6h', 'P3', 'P1', 'Pz', 'P2', 'P4', 'PPO1h', 'PPO2h']
    event_annotation='At second 0 a cross appeared together with a beep sound; at second 2 the cue was presented and subjects executed a sustained movement or avoided any movement, respectively. After the trial, a break with arandom duration of 2s to 3s followed'
    usr_id = str(uuid.uuid1())
    raw = mne.io.read_raw_gdf(file)
    events, event_dict = mne.events_from_annotations(raw)
    new_event_dict = {v:k for k,v in event_dict.items()}
    raw_file, event_df = datastructure(raw.get_data()[:61, :].T*1e6, columns, file, usr_id, # We take only EEG for now
                        session=1, 
                        rate=512, 
                        metadata='Subjects are healthy and aged between 22 and 40 years with a mean age of 27 years (standard deviation 5 years). Nine subjects were female, and all the subjects except s1 were right-handed.',
                        reference='Reference was placed on the right mastoid, ground on AFz',
                        filtering='Sampled with 512 Hz and bandpass filtered from 0.01 Hz to 200 Hz. Power line interference was suppressed with a notch filter at 50 Hz',
                        hardware='The EEG was measured from 61 channels covering frontal, central, parietal and temporal areas using active electrodes (g.tec medical engineering GmbH, Austria)',
                        dataset="ULM", section="MI", eeg_channels=61, event_annotation=event_annotation)
    k = 0
    trial_counter = 1
    while k < len(events)-8:
        start = events[k][0]
        end = events[k+7][0]
        code = new_event_dict.get(events[k+3][2])
        
        event_df.loc[start:end-1, ["Trial"]] = trial_counter
        event_df.loc[0, 'Event Description'] = 'There is a beep and a cross pops up on the computer screen'
        if(code == '1536'):
            part.loc[1024, 'Event Description'] = 'Participant performs elbow flexion' if "exection" in file else 'Participant imagines performing elbow flexion'
        elif(code == '1537'):
            part.loc[1024, 'Event Description'] = 'Participant performs elbow extension' if "exection" in file else 'Participant imagines performing elbow extension'
        elif(code == '1538'):
            part.loc[1024, 'Event Description'] = 'Participant performs supination' if "exection" in file else 'Participant imagines performing supination'
        elif(code == '1539'):
            part.loc[1024, 'Event Description'] = 'Participant performs pronation'  if "exection" in file else 'Participant imagines performing pronation'
        elif(code == '1540'):
            part.loc[1024, 'Event Description'] = 'Participant performs hand closing' if "exection" in file else 'Participant imagines performing hand closing'
        elif(code == '1541'):
            part.loc[1024, 'Event Description'] = 'Participant performs hand opening' if "exection" in file else 'Participant imagines performing hand opening'
        elif(code == '1542'):
            part.loc[1024, 'Event Description'] = 'Participant takes a rest'
        part.loc[2560, 'Event Description'] = 'Participant moves back to the starting position and takes a break'
        k += 7
        trial_counter += 1


        
