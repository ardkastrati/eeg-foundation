import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
import mne
import os
from datasets.datastructure import datastructure

def mi_bci_iv_graz_b_read_file(file):
    def translate_Event(i):
        if(i == '276'):
            return 'eye open'
        if(i == '277'):
            return 'eye closed'
        if(i == '768'):
            return 'new trial'
        if(i == '769'):
            return 'left hand imagined movement'
        if(i == '770'):
            return 'right hand imagined movement'
        if(i == '781'):
            return 'bci feedback'
        if(i == '783'):
            return 'Unknown'
        if(i == '1023'):
            return 'rejected trial'
        if(i == '1077'):
            return 'eye movement horizontally'
        if(i == '1078'):
            return 'eye movement vertically'
        if(i == '1079'):
            return 'eye rotation'
        if(i == '1081'): 
            return 'eye blink'
        if(i == '32766'):
            return 'new run'

    columns = ['C3', 'Cz', 'C4']
    event_annotation = 'At the beginning of each session, a recording of approximately 5 minutes was performed to estimate the EOG influence. The recording was divided into 3 blocks: (1) two minutes with eyes open (looking at a fixation cross on the screen), (2) one minute with eyes closed, and (3) one minute with eye movements. Subjects were instructed with a text on the monitor to perform either eye blinking, rolling, up-down or left-right movements. Cue-based screening paradigm: each trial started with a fixation cross and an additional short acoustic warning tone (1kHz, 70ms). Some seconds later a visual cue (an arrow pointing either to the left or right, according to the requested class) was presented for 1.25 seconds. Afterwards the subjects had to imagine the corresponding hand movement over a period of 4 seconds. Each trial was followed by a short break of at least 1.5 seconds. A randomized time of up to 1 second was added to the break to avoid adaptation. Feedback: At the beginning of each trial (second 0) the feedback (a gray smiley) was centered on the screen. At second 2, a short warning beep (1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. Depending on the cue, the subjects were required to move the smiley towards the left or right side by imagining left or right hand movements, respectively. During the feedback period the smiley changed to green when moved in the correct direction, otherwise it became red.'
    usr_id = str(uuid.uuid1())

    raw = mne.io.read_raw_gdf(file)
    events, event_dict = mne.events_from_annotations(raw)
    new_event_dict = {v:k for k,v in event_dict.items()}
    raw_file, event_df = datastructure(raw.get_data()[:3, :].T*1e6, columns, file, usr_id, # We take only EEG for now
                        session=int(os.path.basename(file)[3:5]), 
                        rate=250, 
                        metadata='right-handed, had normal or corrected-to-normal vision',
                        reference='The electrode position Fz served as EEG ground and left mastoid as reference for EOG',
                        filtering='Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling frequency of 250 Hz. They were bandpass- filtered between 0.5 Hz and 100 Hz, and a notch filter at 50 Hz was enabled. Same for EOG',
                        hardware='Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5cm) were used to record the EEG',
                        dataset="BBCI_IV_Graz_b", section="MI", eeg_channels=3, event_annotation=event_annotation)
    run_counter = 0
    trial_counter = 0
    k = 0
    while k < len(events):
        if(k == 1 or k == 2 or k == 3 or k == 4 or k == 5 or k == 6): 
            run_counter += 1
        this_event = events[k]
        start = this_event[0]
        raw_event = this_event[2]
        if(k < len(events) - 1):
            this_event_next = events[k + 1]
            end = this_event_next[0]
        else:  
            end = len(event_df)
        if(start == end): 
            raw_event2 = this_event_next[2]
            code1 = new_event_dict.get(raw_event)
            code2 = new_event_dict.get(raw_event2)
            event1 = translate_Event(code1)
            event2 = translate_Event(code2)
            if(event1 == 'new run'):
                run_counter += 1 
            if(event1 == 'new trial'):
                trial_counter += 1 
            if(event2 == 'new run'):
                run_counter += 1
            if(event2 == 'new trial'):
                trial_counter += 1
            this_event_next_next = events[k+2]
            end = this_event_next_next[0]

            event_df.loc[start:end-1, ["Run"]] = run_counter
            event_df.loc[start:end-1, ["Trial"]] = trial_counter

            event_df.loc[start:end-1, ["Event Description"]] = f'Participant is performing the following action: ' + event2 + ", " +  event1
            k += 2
        else: 
            code1 = new_event_dict.get(raw_event)
            event1 = translate_Event(code1)
            if(event1 == 'Start of a new run'):
                run_counter += 1 
            if(event1 == 'Start of a trial'):
                trial_counter +=1

            event_df.loc[start:end-1, ["Run"]] = run_counter
            event_df.loc[start:end-1, ["Trial"]] = trial_counter
            event_df.loc[start:end-1, ["Event Description"]] = f'Participant is performing the following action: ' + event1
            k+=1
    # Split the raw files by run counter
    return [raw_file], [event_df], "standard_1020", 50