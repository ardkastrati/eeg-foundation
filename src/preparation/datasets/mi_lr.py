from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import scipy.io as io
import uuid
from datasets.datastructure import datastructure

def mi_lr_read_file(file):
    columns =['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', "EMG-1", "EMG-2", "EMG-3", "EMG-4"]
    event_annotation = "We asked each subject to move his/her fingers, starting from the index finger and proceeding to the little finger and touching each to their thumb within 3 seconds after onset. Each subject practiced these actual finger movements, and then performed the MI experiment. When imagining the movement, we asked subjects to imagine the kinesthetic experience, rather than imagining the visual experience. 'Monitor showed a black screen with a fixation cross for 2 seconds; the subject was then ready to perform hand movements. One of 2 instructions (“left hand” or “right hand”) appeared randomly on the screen for 3 seconds, and subjects were asked to perform/imagine moving the appropriate hand"

    def translate_Noise(i):
        if(i == 0): 
            return 'eye blink'
        elif(i == 1):
            return 'eye saccade vertical'
        elif(i == 2):
            return 'eye saccade horizontal'
        elif(i == 3):
            return 'head movement'
        elif(i == 4):
            return 'jaw clenching'
        
    # Load the Excel file
    file_path = '/itet-stor/kard/deepeye_storage/foundation/MI_LR/ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/Questionnaire_results_of_52_subjects.xlsx'

    data = pd.read_excel(file_path)
    # Extract the relevant rows for sex, handedness, and age
    sex = data.iloc[5, 7:59].values
    handedness = data.iloc[3, 7:59].values
    age = data.iloc[4, 7:59].values
    # Replace the sex and handedness values with appropriate labels
    sex = ['Female' if s == 0.0 else 'Male' for s in sex]
    handedness = ['Left' if h == 0.0 else 'Right' if h == 1.0 else 'Both' for h in handedness]
    # Create a structured dataframe with formatted subject labels
    subjects = [f'/itet-stor/kard/deepeye_storage/foundation/MI_LR/ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/mat_data/s{i:02d}.mat' for i in range(1, 53)]
    quuestionnaire = pd.DataFrame({
        'Sex': sex,
        'Handedness': handedness,
        'Age': age
    }, index=subjects)
    mat = io.loadmat(file)
    usr_id = str(uuid.uuid1())
    raw_files, events = [], []
    data = mat['eeg']

    struct = data[0,0] #17 elements of mat file
    noise = struct[0] 
    for j in range(0, 5): 
        noise_j = noise[j][0]
        noise_j = np.transpose(noise_j)
        raw_file, event = datastructure(noise_j, columns, file, usr_id, metadata='19 females, mean age ± SD age = 24.8 ± 3.86 years', 
                      timeseries=translate_Noise(j), session=1, rate=512, hardware='EEG data collected by 64 Ag/AgCl active electrodes and BCI2000 system 3.0.2 , EEG signals recorded by 64-channel montage based on the international 10-10 system, EMG: same system as EEG', 
                      reference="CAR", dataset="MI_LR", section="MI", eeg_channels=64, event_annotation="Missing")
        event['Event Description'] = 'Participant is performing the following action: ' + translate_Noise(j)
        event["Age"] = quuestionnaire.loc[file]["Age"]
        event["Handedness"] = quuestionnaire.loc[file]["Handedness"]
        event["Sex"] = quuestionnaire.loc[file]["Sex"]
        raw_files.append(raw_file)
        events.append(event)
        
    rest = struct[1]
    rest = np.transpose(rest)
    raw_file, event = datastructure(rest, columns, file, usr_id, metadata='19 females, mean age ± SD age = 24.8 ± 3.86 years', 
                      timeseries=f"rest", session=1, rate=512, hardware='EEG data collected by 64 Ag/AgCl active electrodes and BCI2000 system 3.0.2 , EEG signals recorded by 64-channel montage based on the international 10-10 system, EMG: same system as EEG', 
                      reference="CAR", dataset="MI_LR", section="MI", eeg_channels=64, event_annotation="Missing")
    event['Event Description'] = 'Participant is performing the following action: rest'
    event["Age"] = quuestionnaire.loc[file]["Age"]
    event["Handedness"] = quuestionnaire.loc[file]["Handedness"]
    event["Sex"] = quuestionnaire.loc[file]["Sex"]
    raw_files.append(raw_file)
    events.append(event)
    
    movement_left = struct[3]
    movement_left = np.transpose(movement_left)
    n_movement = struct[6][0][0]
    part_len = int(len(movement_left)/n_movement)
    previous = 0 
    for j in range(0, n_movement): 
        part = movement_left[previous:previous+part_len, :]
        previous = previous + part_len
        raw_file, event = datastructure(part, columns, file, usr_id, metadata='19 females, mean age ± SD age = 24.8 ± 3.86 years', 
                      timeseries=f"left hand movement", session=1, rate=512, hardware='EEG data collected by 64 Ag/AgCl active electrodes and BCI2000 system 3.0.2 , EEG signals recorded by 64-channel montage based on the international 10-10 system, EMG: same system as EEG', 
                      reference="CAR", dataset="MI_LR", section="MI", eeg_channels=64, event_annotation=event_annotation, trial=j)
        event['Event Description'] = 'Participant is performing the following action: left hand'
        event["Age"] = quuestionnaire.loc[file]["Age"]
        event["Handedness"] = quuestionnaire.loc[file]["Handedness"]
        event["Sex"] = quuestionnaire.loc[file]["Sex"]
        raw_files.append(raw_file)
        events.append(event)
        
    movement_right = struct[4]
    movement_right = np.transpose(movement_right)
    part_len = int(len(movement_right)/n_movement)
    previous = 0 
    for j in range(0, n_movement): 
        part = movement_right[previous:previous+part_len, :]
        previous = previous + part_len
        raw_file, event = datastructure(part, columns, file, usr_id, metadata='19 females, mean age ± SD age = 24.8 ± 3.86 years', 
                        timeseries="right hand movement", session=1, rate=512, hardware='EEG data collected by 64 Ag/AgCl active electrodes and BCI2000 system 3.0.2 , EEG signals recorded by 64-channel montage based on the international 10-10 system, EMG: same system as EEG', 
                        reference="CAR", dataset="MI_LR", section="MI", eeg_channels=64, event_annotation="Missing", trial=j)
        event['Event Description'] = 'Participant is performing the following action: right hand'
        event["Age"] = quuestionnaire.loc[file]["Age"]
        event["Handedness"] = quuestionnaire.loc[file]["Handedness"]
        event["Sex"] = quuestionnaire.loc[file]["Sex"]
        raw_files.append(raw_file)
        events.append(event)

    img_left = struct[7]
    img_left = np.transpose(img_left)
    n_img = struct[9][0][0]
    part_len = int(len(img_left) /n_img)
    previous = 0 
    for j in range(0, n_img): 
        part = img_left[previous:previous+part_len, :]
        previous = previous + part_len
        raw_file, event = datastructure(part, columns, file, usr_id, metadata='19 females, mean age ± SD age = 24.8 ± 3.86 years', 
                        timeseries="left hand imagined movement", session=1, rate=512, hardware='EEG data collected by 64 Ag/AgCl active electrodes and BCI2000 system 3.0.2 , EEG signals recorded by 64-channel montage based on the international 10-10 system, EMG: same system as EEG', 
                        reference="CAR", dataset="MI_LR", section="MI", eeg_channels=64,  event_annotation=event_annotation, trial=j)
        event['Event Description'] = 'Participant is imagining the following action: left hand'
        event["Age"] = quuestionnaire.loc[file]["Age"]
        event["Handedness"] = quuestionnaire.loc[file]["Handedness"]
        event["Sex"] = quuestionnaire.loc[file]["Sex"]
        raw_files.append(raw_file)
        events.append(event)

    img_right = struct[8]
    img_right = np.transpose(img_right)
    part_len = int(len(img_right) /n_img)
    previous = 0 
    for j in range(0, n_img): 
        part = img_right[previous:previous+part_len, :]
        previous = previous + part_len
        raw_file, event = datastructure(part, columns, file, usr_id, metadata='19 females, mean age ± SD age = 24.8 ± 3.86 years', 
                        timeseries="right hand imagined movement", session=1, rate=512, hardware='EEG data collected by 64 Ag/AgCl active electrodes and BCI2000 system 3.0.2 , EEG signals recorded by 64-channel montage based on the international 10-10 system, EMG: same system as EEG', 
                        reference="CAR", dataset="MI_LR", section="MI", eeg_channels=64, event_annotation=event_annotation, trial=j)
        event['Event Description'] = 'Participant is imagining the following action: right hand'
        event["Age"] = quuestionnaire.loc[file]["Age"]
        event["Handedness"] = quuestionnaire.loc[file]["Handedness"]
        event["Age"] = quuestionnaire.loc[file]["Age"]
        raw_files.append(raw_file)
        events.append(event)
    return raw_files, events, "standard_1020", 60