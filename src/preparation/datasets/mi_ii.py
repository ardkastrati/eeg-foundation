import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
import mne
from datasets.datastructure import datastructure


def mi_ii_read_file(file):
    columns = ['AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'CP3', 'CPz', 'CP4', 'P7', 
           'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2']
    event_annotation = 'At t = 0 s, a cross was presented in the middle of the screen. Participants were asked to relax and fixate the cross to avoid eye movements. At t = 3 s, a beep was sounded to get the attention of participant. The cue indicating the requested imagery task, one out of five graphical symbols, was presented from t = 3 s to t = 4.25 s. At t = 10 s, a second beep was sounded and the fixation-cross disappeared, which indicated the end of the trial. A variable break (inter-trial- interval, ITI) lasting between 2.5 s and 3.5 s occurred before the start of the next trial. Participants were asked to avoid movements during the imagery period, and to move and blink during the ITI'

    def translate_Event(i): 
        if(i == 1):
            return 'imagined mental word association'
        elif(i == 2):
            return 'imagined mental subtraction'
        elif(i == 3):
            return 'imagined spatial navigation'
        elif(i == 4):
            return 'right hand imagined movement'
        elif(i == 5):
            return 'left foot and right foot imagined movement'

    usr_id = str(uuid.uuid1())
    mat = io.loadmat(file)
    data = mat['data']
    session1 = data[0,0][0][0]
    rData = session1[0]
    raw_file, event_df = datastructure(rData, columns, file, usr_id, # We take only EEG for now
                            session=1, 
                            metadata='Participants with disability (spinal cord injury and stroke)',
                            rate=256,
                            reference='Reference and ground were placed at the left and right mastoid, respectively',
                            filtering='EEG was band pass filtered 0.5-100 Hz (notch filter at 50 Hz) and sampled at a rate of 256 Hz',
                            hardware='The g.tec GAMMAsys system with g.LADYbird active electrodes and two g.USBamp biosignal amplifiers (Guger Technolgies, Graz, Austria) was used for recording',
                            dataset="MI_II", section="MI", eeg_channels=30, event_annotation=event_annotation)
    trials1 = session1[2]
    labels1 = session1[1]
    for i in range(0, len(trials1)):
        start = trials1[i][0]
        if(i != len(trials1) - 1):
           end = trials1[i+1][0]
        else: 
           end = len(event_df) - 1
        event_df.loc[start:end-1, "Trial"] = i+1
        event_df.loc[start:end-1, "Event Description"] = 'Participant is relaxing and fixating the cross on the screen to avoid eye movements'
        event_df.loc[start+768:end-1, "Event Description"] = 'Participant is imagining the following event: ' + translate_Event(labels1[i]) 
        event_df.loc[start+2560:end-1, "Event Description"] = 'Participant is in inter-trial interval'


    session2 = data[0,1][0][0]
    rData = session2[0]
    raw_file2, event_df2 = datastructure(rData, columns, file, usr_id, # We take only EEG for now
                            session=2, 
                            metadata='Participants with disability (spinal cord injury and stroke)',
                            rate=256,
                            reference='Reference and ground were placed at the left and right mastoid, respectively',
                            filtering='EEG was band pass filtered 0.5-100 Hz (notch filter at 50 Hz) and sampled at a rate of 256 Hz',
                            hardware='The g.tec GAMMAsys system with g.LADYbird active electrodes and two g.USBamp biosignal amplifiers (Guger Technolgies, Graz, Austria) was used for recording',
                            dataset="MI_II", section="MI", eeg_channels=30, event_annotation=event_annotation)
    trials1 = session2[2]
    labels1 = session2[1]
    for i in range(0, len(trials1)):
        start = trials1[i][0]
        if(i != len(trials1) - 1):
           end = trials1[i+1][0]
        else: 
           end = len(event_df2) - 1
        event_df2.loc[start:end-1, "Trial"] = i+1
        event_df2.loc[start:end-1, "Event Description"] = 'Participant is relaxing and fixating the cross on the screen to avoid eye movements'
        event_df2.loc[start+768:end-1, "Event Description"] =  'Participant is imagining the following event: ' + translate_Event(labels1[i]) 
        event_df2.loc[start+2560:end-1, "Event Description"] = 'Participant is in inter-trial interval'   

    return [raw_file, raw_file2], [event_df, event_df2], "standard_1020", 50  
        















