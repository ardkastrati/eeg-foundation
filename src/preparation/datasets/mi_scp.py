import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
from datasets.datastructure import datastructure

def mi_scp_read_file(file):
    def translate_5FEvent(i):
        if(i == 1):
            return 'thumb imagined movement'
        elif(i == 2):
            return 'index finger imagined movement'
        elif(i == 3):
            return 'middle finger imagined movement'
        elif(i == 4):
            return 'ring finger imagined movement'
        elif(i == 5):
            return 'pinkie finger imagined movement'
        elif(i == 99):
            return 'relax'
        elif(i == 91):
            return 'rest'
        elif(i == 92):
            return 'end experiment'
        else:
            return 'unknown'

    def translate_CLA_HaLT_FreeFormEvent(i):
        if(i == 1):
            return 'left hand imagined movement'
        elif(i == 2):
            return 'right hand imagined movement'
        elif(i == 3):
            return 'passive state'
        elif(i == 4):
            return 'left foot imagined movement'
        elif(i == 5):
            return 'tongue imagined movement'
        elif(i == 6): 
            return 'right foot imagined movement'
        elif(i == 91):
            return 'rest'
        elif(i == 92):
            return 'end experiment'
        elif(i == 99):
            return 'relax'
        else:
            return 'unknown'
    
    def translate_FreeFormEvent(i):
        if(i == 1):
            return 'left hand movement pressing key'
        elif(i == 2):
            return 'right hand movement pressing key'
        elif(i == 91):
            return 'rest'
        elif(i == 92):
            return 'end experiment'
        elif(i == 99):
            return 'relax'
        else:
            return 'unknown'
    

    def translate_NoMotorEvent(i):
        if(i == 1):
            return 'watches screen (signal indicates left hand)'
        elif(i == 2):
            return 'watches screen (signal indicates right hand)'
        elif(i == 3):
            return 'passive'
        elif(i == 4):
            return 'watches screen (signal indicates left leg)'
        elif(i == 5):
            return 'watches screen (signal indicates tongue)'
        elif(i == 6): 
            return 'watches screen (signal indicates right leg)'
        elif(i == 91):
            return 'rest'
        elif(i == 92):
            return 'end experiment'
        elif(i == 99):
            return 'relax'
        else:
            return 'unknown'


    columns = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'Cz', 'Pz', 'X5']

    usr_id = str(uuid.uuid1())
    subject_meta = ""
    age = ""
    gender = ""
    if "SubjectA" in file:
        subject_meta = "Male, 20-25 y.o."
        age = "20-25"
        gender = "Male"
    elif "SubjectB" in file:
        subject_meta = "Male, 20-25 y.o."
        age = "20-25"
        gender = "Male"
    elif "SubjectC" in file:
        subject_meta = "Male, 25-30 y.o."
        age = "25-30"
        gender = "Male"
    elif "SubjectD" in file:
        subject_meta = "Male, 25-30 y.o."
        age = "25-30"
        gender = "Male"
    elif "SubjectE" in file:
        subject_meta = "Female, 20-25 y.o."
        age = "20-25"
        gender = "Female"
    elif "SubjectF" in file:
        subject_meta = "Male, 30-35 y.o."
        age = "30-35"
        gender = "Male"
    elif "SubjectG" in file:
        subject_meta = "Male, 30-35 y.o."
        age = "30-35"
        gender = "Male"
    elif "SubjectH" in file:
        subject_meta = "Male, 20-25 y.o."
        age = "20-25"
        gender = "Male"
    elif "SubjectI" in file:
        subject_meta = "Female, 20-25 y.o."
        age = "20-25"
        gender = "Female"
    elif "SubjectJ" in file:
        subject_meta = "Female, 20-25 y.o."
        age = "20-25"
        gender = "Female"
    elif "SubjectK" in file:
        subject_meta = "Male, 20-25 y.o."
        age = "20-25"
        gender = "Male"
    elif "SubjectL" in file:
        subject_meta = "Female, 20-25 y.o."
        age = "20-25"
        gender = "Female"
    elif "SubjectM" in file:
        subject_meta = "Female, 20-25 y.o."
        age = "20-25"
        gender = "Female"
    raw_files = []
    event_dfs = []
    ###5F
    if "5F-" in file:
        mat = io.loadmat(file)
        data = mat['o']
        eeg = data[0][0][5]
        mark = data[0][0][4]
        mark = mark[:,0]
        raw_file, event_df = datastructure(eeg, columns, file, usr_id, # We take only EEG for now
                                session="5F", 
                                metadata=subject_meta,
                                rate=1000 if "HFREQ" in file else 200,
                                reference='two ground leads labeled A1 and A2 (placed at the earbuds)',
                                filtering='a band-pass filter of 0.53-100 Hz was present' if "HFREQ" in file else 'a band-pass filter of 0.53-70 Hz was present',
                                hardware='EEG data were acquired using an EEG-1200 JE-921A EEG system with the help of a standard 10/20 EEG cap with 19 bridge electrodes in the 10/20 international configuration',
                                dataset="MI_SCP", section="MI", eeg_channels=22, event_annotation='Each recording session was organized as a sequence of BCI interaction segments separated by 2 min breaks. At the beginning of each recording session, a 2.5 min initial relaxation period was administered. At the beginning of each trial, an action signal appeared (represented by a number from 1 to 5) directly above the finger whose movement imagery was to be implemented. The action signal remained on for 1s, during which time the participants implemented the corresponding imagery once. A pause of variable duration of 1.5-2.5 s followed, concluding the trial')        
    
        for i in range(0, len(event_df)):
            event = translate_5FEvent(mark[i])
            event_df.loc[i, 'Event Description'] = 'Participant is performing the following action: ' + event
        
        if "HFREQ" not in file:
            unit_length = 600
        else:
            unit_length = 3000

        start = 0
        trial_counter = 1
     
        while(start+unit_length < len(event_df)):
            end = start + unit_length
            event_df.loc[start:end-1, "Trial"] = trial_counter
            start = end
            trial_counter += 1
        raw_files.append(raw_file)
        event_dfs.append(event_df)
            
    ###CLA
    elif "CLA-" in file:
        mat = io.loadmat(file)
        data = mat['o']
        eeg = data[0][0][5]
        mark = data[0][0][4]
        mark = mark[:,0]
        raw_file, event_df = datastructure(eeg, columns, file, usr_id, # We take only EEG for now
                                session="CLA", 
                                metadata=subject_meta,
                                rate=200,
                                reference='two ground leads labeled A1 and A2 (placed at the earbuds)',
                                filtering='a band-pass filter of 0.53-70 Hz was present',
                                hardware='EEG data were acquired using an EEG-1200 JE-921A EEG system with the help of a standard 10/20 EEG cap with 19 bridge electrodes in the 10/20 international configuration',
                                dataset="MI_SCP", section="MI", eeg_channels=22, event_annotation='Each recording session was organized as a sequence of BCI interaction segments separated by 2 min breaks. At the beginning of each recording session, a 2.5 min initial relaxation period was administered. At the beginning of each trial, an action signal appeared (represented by a number from 1 to 5) directly above the finger whose movement imagery was to be implemented. The action signal remained on for 1s, during which time the participants implemented the corresponding imagery once. A pause of variable duration of 1.5-2.5 s followed, concluding the trial')        
        for i in range(0, len(event_df)):
            event = translate_CLA_HaLT_FreeFormEvent(mark[i])
            event_df.loc[i, 'Event Description'] = 'Participant is performing the following action: ' + event
        unit_length = 600
        start = 0
        trial_counter = 1
        while(start + unit_length < len(event_df)):
            end = start + unit_length
            event_df.loc[start:end-1, "Trial"] = trial_counter
            start = end
            trial_counter += 1
        raw_files.append(raw_file)
        event_dfs.append(event_df)

    elif "HaLT-" in file:
        mat = io.loadmat(file)
        data = mat['o']
        eeg = data[0][0][5]
        mark = data[0][0][4]
        mark = mark[:,0]
        raw_file, event_df = datastructure(eeg, columns, file, usr_id, # We take only EEG for now
                                session="HaLT", 
                                metadata=subject_meta,
                                rate=200,
                                reference='two ground leads labeled A1 and A2 (placed at the earbuds)',
                                filtering='a band-pass filter of 0.53-70 Hz was present',
                                hardware='EEG data were acquired using an EEG-1200 JE-921A EEG system with the help of a standard 10/20 EEG cap with 19 bridge electrodes in the 10/20 international configuration',
                                dataset="MI_SCP", section="MI", eeg_channels=22, event_annotation='Each recording session was organized as a sequence of BCI interaction segments separated by 2 min breaks. At the beginning of each recording session, a 2.5 min initial relaxation period was administered. At the beginning of each trial, an action signal appeared (represented by a number from 1 to 5) directly above the finger whose movement imagery was to be implemented. The action signal remained on for 1s, during which time the participants implemented the corresponding imagery once. A pause of variable duration of 1.5-2.5 s followed, concluding the trial')        
        for i in range(0, len(event_df)):
            event = translate_CLA_HaLT_FreeFormEvent(mark[i])
            event_df.loc[i, 'Event Description'] = 'Participant is performing the following action: ' + event
        unit_length = 600
        start = 0
        trial_counter = 1
        while(start+unit_length < len(event_df)):
            end = start + unit_length
            event_df.loc[start:end-1, "Trial"] = trial_counter
            start = end
            trial_counter += 1
        raw_files.append(raw_file)
        event_dfs.append(event_df)
    elif "FREEFORM-" in file:
        mat = io.loadmat(file)
        data = mat['o']
        eeg = data[0][0][5]
        mark = data[0][0][4]
        mark = mark[:,0]
        raw_file, event_df = datastructure(eeg, columns, file, usr_id, # We take only EEG for now
                                session="FREEFORM", 
                                metadata=subject_meta,
                                rate=200,
                                reference='two ground leads labeled A1 and A2 (placed at the earbuds)',
                                filtering='a band-pass filter of 0.53-70 Hz was present',
                                hardware='EEG data were acquired using an EEG-1200 JE-921A EEG system with the help of a standard 10/20 EEG cap with 19 bridge electrodes in the 10/20 international configuration',
                                dataset="MI_SCP", section="MI", eeg_channels=22, event_annotation='Each recording session was organized as a sequence of BCI interaction segments separated by 2 min breaks. At the beginning of each recording session, a 2.5 min initial relaxation period was administered. At the beginning of each trial, an action signal appeared (represented by a number from 1 to 5) directly above the finger whose movement imagery was to be implemented. The action signal remained on for 1s, during which time the participants implemented the corresponding imagery once. A pause of variable duration of 1.5-2.5 s followed, concluding the trial')        
        
        for i in range(0, len(event_df)):
            event = translate_FreeFormEvent(mark[i])
            event_df.loc[i, 'Event Description'] = 'Participant is performing the following action: ' + event
        unit_length = 600
        start = 0
        trial_counter = 1
        while(start + unit_length < len(event_df)):
            end = start + unit_length
            event_df.loc[start:end-1, "Trial"] = trial_counter
            start = end
            trial_counter += 1
        raw_files.append(raw_file)
        event_dfs.append(event_df)
    elif "NoMT-" in file:
        mat = io.loadmat(file)
        data = mat['o']
        eeg = data[0][0][5]
        mark = data[0][0][4]
        mark = mark[:,0]
        raw_file, event_df = datastructure(eeg, columns, file, usr_id, # We take only EEG for now
                                session="NoMT", 
                                metadata=subject_meta,
                                rate=200,
                                reference='two ground leads labeled A1 and A2 (placed at the earbuds)',
                                filtering='a band-pass filter of 0.53-70 Hz was present',
                                hardware='EEG data were acquired using an EEG-1200 JE-921A EEG system with the help of a standard 10/20 EEG cap with 19 bridge electrodes in the 10/20 international configuration',
                                dataset="MI_SCP", section="MI", eeg_channels=22, event_annotation='Each recording session was organized as a sequence of BCI interaction segments separated by 2 min breaks. At the beginning of each recording session, a 2.5 min initial relaxation period was administered. At the beginning of each trial, an action signal appeared (represented by a number from 1 to 5) directly above the finger whose movement imagery was to be implemented. The action signal remained on for 1s, during which time the participants implemented the corresponding imagery once. A pause of variable duration of 1.5-2.5 s followed, concluding the trial')        
        
        for i in range(0, len(event_df)):
            event = translate_NoMotorEvent(mark[i])
            event_df.loc[i, 'Event Description'] = 'Participant is performing the following action: ' + event
        unit_length = 600
        start = 0
        trial_counter = 1
        while(start + unit_length < len(event_df)):
            end = start + unit_length
            event_df.loc[start:end-1, "Trial"] = trial_counter            
            start = end
            trial_counter += 1
        raw_files.append(raw_file)
        event_dfs.append(event_df)

    return raw_files, event_dfs, "standard_1020", 50













    


    

