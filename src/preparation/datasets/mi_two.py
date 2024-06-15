import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
from decimal import Decimal
from datasets.datastructure import datastructure


def mi_two_read_file(file):
    def translate_Event(i):
        if(i == 1):
            return 'right hand imagined movement'
        elif (i == 2):
            return 'left foot and right foot imagined movement'
        

    columns = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12', 
           'channel13', 'channel14', 'channel15']
    event_annotation = 'Participants had the task of performing sustained (5 seconds) kinaesthetic motor imagery (MI) of the right hand and of the feet each as instructed by the cue. Feedback was presented in form of a white coloured bar-graph. The length of the bar-graph reflected the amount of correct classifications over the last second'
    usr_id = str(uuid.uuid1())
    mat = io.loadmat(file)
    data = mat['data']
    raw_files, events_dfs = [], []
    for j in range(0,5): #iterate over runs, 5 runs for T 
        struct = data[0,j] 
        unzip_struct = struct[0,0] # get 5 elements of this struct:x, trial, ....
        x = unzip_struct[0] # this is x, 112128 * 15 double array 
        trial = unzip_struct[1][0] # this is trial, 1*20 double
        y = unzip_struct[2][0] # this is y, 1*20 double

        raw_file, event_df = datastructure(x, columns, file, usr_id, # We take only EEG for now
                    session=j, 
                    metadata='aged between 20 and 30 years, 8 naive to the task, and had no known medical or neurological diseases',
                    rate=512,
                    reference='reference electrode was mounted on the left mastoid and the ground electrode on the right mastoid',
                    hardware='EEG was measured with a biosignal amplifier and active Ag/AgCl electrodes (g.USBamp, g.LADYbird, Guger Technologies OG, Schiedlberg, Austria). Center electrodes at positions C3, Cz, and C4 and four additional electrodes around each center electrode with a distance of 2.5cm, 15 electrodes total',
                    dataset="MI_Two", section="MI", eeg_channels=15, event_annotation=event_annotation)


        previous = 0
        len_x = len(x)
        for k in range(0, len(trial)):
            event = y[k]
            translated_event = translate_Event(event)
            event_df.loc[previous:trial[k], "Trial"] = k
            event_df.loc[previous:trial[k], "Event Description"] = 'Participant imagines the following action: ' + translated_event 
            previous = trial[k]
        raw_files.append(raw_file)
        events_dfs.append(event_df)
    return raw_files, events_dfs, "ignore", 50
    


