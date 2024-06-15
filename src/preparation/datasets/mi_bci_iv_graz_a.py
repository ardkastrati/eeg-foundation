import uuid
from decimal import Decimal
import mne
from datasets.datastructure import datastructure



def mi_bci_iv_graz_a_read_file(file):
    def translate_Event(i):
        if(i == 1):
            return 'Rejected trial'
        elif(i == 2):
            return 'eye movement'
        elif(i == 3):
            return 'eye open'
        elif(i == 4):
            return 'eye closed'
        elif(i == 5):
            return 'new run'
        elif(i == 6):
            return 'new trial'
        elif(i == 7):
            return 'left hand imagined movement'
        elif(i == 8):
            return 'right hand imagined movement'
        elif(i == 9):
            return 'left foot or right foot imagined movement'
        elif(i == 10):
            return 'tongue imagined movement'


    columns = ['FZ', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3',
            'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    event_annotation = 'At the beginning of each session, a recording of approximately 5 minutes was performed to estimate the EOG influence: (1) two minutes with eyes open (2) one minute with eyes closed, and (3) one minute with eye movements. At the beginning of a trial (t = 0 s), a fixation cross appeared on the black screen. After two seconds (t = 2 s), a cue in the form of an arrow pointing either to the left, right, down or up (corresponding to one of the four classes left hand, right hand, foot or tongue) appeared and stayed on the screen for 1.25s. This prompted the subjects to perform the desired motor imagery task. No feedback was provided. The subjects were ask to carry out the motor imagery task until the fixation cross disappeared from the screen at t = 6s.'
    usr_id = str(uuid.uuid1())
    raw = mne.io.read_raw_gdf(file)
    events, event_dict = mne.events_from_annotations(raw)
    raw_file, event_df = datastructure(raw.get_data()[:22, :].T*1e6, columns, file, usr_id, # We take only EEG for now
                        session=1, 
                        rate=250, 
                        reference='left mastoid serving as reference and the right mastoid as ground',
                        filtering='EEG signals sampled with 250 Hz, bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of the amplifier was set to 100 μV. An additional 50 Hz notch filter was enabled to suppress line noise. EOG signals also sampled with 250Hz, bandpass filtered between 0.5Hz and 100Hz (with the 50Hz notch filter enabled), and the sensitivity of the amplifier was set to 1mV',
                        hardware='Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5cm) were used to record the EEG',
                        dataset="MI_BCI_IV_Graz_a", section="MI", eeg_channels=22, event_annotation=event_annotation)
    run_counter = 0
    trial_counter = 0
    j = 0
    while j < len(events):
        this_event = events[j]
        start = this_event[0]
        raw_event = this_event[2]
        if(j < len(events) - 1):
            this_event_next = events[j + 1]
            end = this_event_next[0]
        else:  
            end = len(event_df)
        if(start == end): 
            raw_event2 = this_event_next[2]
            if(raw_event == 5):
                run_counter += 1 
            if(raw_event == 6):
                trial_counter += 1 
            if(raw_event2 == 5):
                run_counter += 1
            if(raw_event2 == 6):
                trial_counter += 1
            this_event_next_next = events[j+2]
            end = this_event_next_next[0]
            
            event_df.loc[start:end-1, ["Run"]] = run_counter
            event_df.loc[start:end-1, ["Trial"]] = trial_counter
            event_df.loc[start:end-1, ["Event Description"]] = f'Participant is performing the following action: ' + translate_Event(raw_event2) + ", " +  translate_Event(raw_event)
            j+=2
        else: 
            if(raw_event == 5):
                run_counter += 1 
            if(raw_event == 6):
                trial_counter +=1
            event_df.loc[start:end-1, ["Run"]] = run_counter
            event_df.loc[start:end-1, ["Trial"]] = trial_counter
            event_df.loc[start:end-1, ["Event Description"]] = f'Participant is performing the following action: ' + translate_Event(raw_event)
            j+=1
    # TODO: split the runs
    return [raw_file], [event_df], "standard_1020", 50





  
