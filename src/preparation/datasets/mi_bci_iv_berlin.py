import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
from datasets.datastructure import datastructure


def mi_bci_iv_berlin_read_file(file):
    columns = ['AF3', 'AF4', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'CFC7', 'CFC5', 'CFC3', 'CFC1', 
                'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 
                'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PO1', 'PO2', 'O1', 'O2']
    
    event_annotation1 = 'For calibration data: Arrows pointing left, right, or down were presented as visual cues on a computer screen. Cues were displayed for a period of 4s during which the subject was instructed to perform the cued motor imagery task. These periods were interleaved with 2s of blank screen and 2s with a fixation cross shown in the center of the screen. The fixation cross was superimposed on the cues, i.e. it was shown for 6s. These data sets are provided with complete marker information'
    event_annotation2 = 'For evaluation data: Motor imagery tasks were cued by soft acoustic stimuli (words left, right, and foot) for periods of varying length between 1.5 and 8 seconds. The end of the motor imagery period was indicated by the word stop. Intermitting periods had also a varying duration of 1.5 to 8s'
    usr_id = str(uuid.uuid1())
    is_1000 = "1000" in file
    is_calib = "calib" in file
    event_annotation = event_annotation1 if is_calib else event_annotation2
    mat = io.loadmat(file)
    data = mat['cnt']*0.1 # in microvolt
    nfo = mat['nfo']
    event1 = nfo[0][0][1][0][0][0]
    event2 = nfo[0][0][1][0][1][0]
    events=[event1, event2]
    for k in range(0, len(events)): 
        if(events[k] == 'left'):
            events[k] = 'left hand imagined movement'
        elif(events[k] == 'right'):
            events[k] = 'right hand imagined movement'
        elif(events[k] == 'foot'):
            events[k] = 'left foot or right foot imagined movement'
    raw_file, raw_event = datastructure(data, columns, file, usr_id,
                        session=1, 
                        rate=1000 if is_1000 else 100, 
                        filtering='Signals were band-pass filtered between 0.05 and 200 Hz and then digitized at 1000 Hz with 16 bit (0.1 uV) accuracy' + "downsampled at 100 Hz" if not is_1000 else "",
                        hardware='The recording was made using BrainAmp MR plus amplifiers and a Ag/AgCl electrode cap',
                        dataset="MI_BCI_IV_Berlin", section="MI", eeg_channels=59, event_annotation=event_annotation)
    
    if is_calib:
        mrk = mat['mrk']
        trials = mrk[0][0][0][0]
        trial_events = mrk[0][0][1][0]
        start = trials[0]
        for k in range(0, len(trials)-1):
            start = trials[k]
            end = trials[k + 1]
            h = trial_events[k]
            if(h == 1):
                this_event = events[0]
            elif(h == -1):
                this_event = events[1]
            raw_event.loc[start:end-1, ["Trial"]] = k + 1
            raw_event.loc[start:end-1, ['Event Description']] = 'Participant is imagining the following action: ' + this_event
            start = end
    return [raw_file], [raw_event], "standard_1005", 50

