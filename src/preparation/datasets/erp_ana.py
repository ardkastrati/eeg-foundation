import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
from datasets.datastructure import datastructure



def erp_ana_read_file(file):
    columns = ['FP1', 'FP2', 'F5', 'AFZ', 'F6', 'T7', 'CZ', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2', 'stimulation']

    usr_id = str(uuid.uuid1())
    mat = io.loadmat(file)
    data = mat['data']
    session = ''
    if 'session01' in file:
        session = 1
    elif 'session02' in file:
        session = 2
    elif 'session03' in file:
        session = 3
    elif 'session04' in file:
        session = 4
    elif 'session05' in file:
        session = 5
    elif 'session06' in file:
        session = 6
    elif 'session07' in file:
        session = 7
    elif 'session08' in file:
        session = 8

    raw_file, event_df = datastructure(data, columns, file, usr_id,
                        session=session, 
                        rate=, 
                        filtering=
                        hardware=,
                        dataset="ERP_ANA", section="ERP", eeg_channels=len(columns) - 1, event_annotation=)

    for h in range(1, len(event_df)):
        if(event_df.loc[h, 'stimulation'] == 33285):
            event_df.loc[h, 'Event Description'] = 'There is an event-related potential in participant. The participant is focusing on the target alien which turns from red to cyan and 5 non-target aliens flash which the participant is not focusing on'
        elif(event_df.loc[h, 'stimulation'] == 33286):
            event_df.loc[h, 'Event Description'] = 'There is no event-related potential in participant. The participant is focusing on the target alien which stays red and does not flash. 6 non-target aliens flash which the participant is not focusing on'
        else:
            event_df.loc[h, 'Event Description'] = event_df.loc[h-1, 'Event Description']

    start = 0
    trial_counter = 1
    non_target_counter = 0
    target_counter = 0
    for h in range(0, len(event_df)): 
        if(target_counter!=2 and event_df.loc[h, 'stimulation'] == 33285):
            target_counter += 1

        elif(non_target_counter!=10 and event_df.loc[h,'stimulation'] == 33286):
            non_target_counter += 1

        elif(target_counter == 2 and non_target_counter == 10 and df.loc[h, 'stimulation'] == 33286):
            part = (event_df.loc[start:h-1, :]).copy(deep = True)
            part.reset_index(drop = True, inplace = True)
            part = part.drop(columns=['stimulation'], axis = 1)
            start = h
            target_counter = 0
            non_target_counter = 1
            part.loc[0, 'Subject'] = 'Subject' + str(i)
            part.loc[0, 'User id'] = usr_id
            part.loc[0, 'Subject Metadata'] = 'Subjects with mean (sd) age 25.96 (4.46)'
            part.loc[0, 'trial'] = str(trial_counter)
            if(k == 1 or k == 2):
                part.loc[0, 'run'] = 'Run' + str(1)
            elif(k == 3 or k == 4):
                part.loc[0, 'run'] = 'Run' + str(2)
            part.loc[0, 'session'] = 'Session' + str(j)
            part.loc[0, 'Event Annotation'] = 'The interface of Brain Invaders is composed of 36 aliens (symbols) that flashes on the computer screen in 12 groups of six aliens. In the Brain Invaders P300 paradigm, a repetition is composed of 12 flashes (i.e., one for each group), of which two include the Target symbol (Target flashes) and 10 do not (non-Target flash). In the training session the Target alien was chosen randomly at each repetition through the use of a predefined randomised list. The task of the user was to focus on the Target alien. Once height repetitions of flashes were completed, a new Target was selected until the Target list was completed. Each session consisted in two runs, one in a Non-Adaptive (classical) and one in an Adaptive (calibration less) mode of operation. The order of the runs was randomised for each session. In both runs there was a Training (calibration) phase and an Online phase, always passed in this order. In the non-Adaptive run the data from the Training phase was used for calibrating the classifier used during the Online phase using the training-test version of the MDM algorithm. In the Adaptive session, the data from the training phase was not used at all, instead the classifier was initialised with generic class geometric means (the class grand average estimated on a database of subjects participating to a previous Brain Invaders experiment) and continuously adapted to the incoming data using the Riemannian method. Subjects were completely blind to the mode of operation and the two runs appeared to them identical'
            part.loc[0, 'Rate'] = 'Data were acquired with no digital filter applied and a sampling frequency of 512 samples per second'
            part.loc[0, 'Hardware'] = 'Stimulus was displayed on a ViewSonic (California, US) screen with length 22’’. EEG signals were acquired by means of a research-grade amplifier (g.USBamp, g.tec, Schiedlberg, Austria) and the g.GAMMAcap (g.tec, Schiedlberg, Austria) equipped with 16 wet Silver/Silver Chloride electrodes, placed according to the 10-20 international system. The amplifier was linked by USB connection to the PC where the data were acquired by means of the software OpenVibe. A visual P300 Brain-Computer Interface inspired by the famous vintage video game Space Invaders (Taito, Tokyo, Japan) was used in the experiment'
            part.loc[0, 'Reference'] = 'The reference was placed on the left earlobe and the ground at the FZ scalp location'
            part.loc[0, 'Source file'] = str(k) + '.mat'
            if(k == 1 or k == 2):
                part.loc[0, 'Experimental condition'] = 'Adaptive'
            elif(k == 3 or k == 4):
                part.loc[0, 'Experimental condition'] = 'Non-Adaptive'
            if(k == 1 or k == 3) :
                part.loc[0, 'training/online'] = 'training'
            elif(k == 2 or k == 4):
                part.loc[0, 'training/online'] = 'online'
            part.loc[0, 'Dataset'] = 'ERP_ANA'
            part.loc[0, 'Section'] = 'ERP'
            store_path = '/itet-stor/cxuan/deepeye_storage/foundation_prepared/ERP_ERP_ANA_Subject' + str(i) + '_' + usr_id + '_session' + str(j)  + '_run' + str(k) + '_trial' + str(trial_counter) + '.pkl'
            result = part.astype(pd.SparseDtype(str,fill_value=''))
            result.to_pickle(store_path) 
            print(store_path+' :done')
            trial_counter+=1

        elif(target_counter == 2 and non_target_counter == 10 and df.loc[h, 'stimulation'] == 33285):
            part = (df.loc[start:h-1, :]).copy(deep = True)
            part.reset_index(drop = True, inplace = True)
            part = part.drop(columns=['stimulation'], axis = 1)
            start = h
            target_counter == 1
            non_target_counter = 0
            part.loc[0, 'Subject'] = 'Subject' + str(i)
            part.loc[0, 'User id'] = usr_id
            part.loc[0, 'Subject Metadata'] = 'Subjects with mean (sd) age 25.96 (4.46)'
            part.loc[0, 'trial'] = str(trial_counter)
            if(k == 1 or k == 2):
                part.loc[0, 'run'] = 'Run' + str(1)
            elif(k == 3 or k == 4):
                part.loc[0, 'run'] = 'Run' + str(2)
            part.loc[0, 'session'] = 'Session' + str(j)
            part.loc[0, 'Event Annotation'] = 'The interface of Brain Invaders is composed of 36 aliens (symbols) that flashes on the computer screen in 12 groups of six aliens. In the Brain Invaders P300 paradigm, a repetition is composed of 12 flashes (i.e., one for each group), of which two include the Target symbol (Target flashes) and 10 do not (non-Target flash). In the training session the Target alien was chosen randomly at each repetition through the use of a predefined randomised list. The task of the user was to focus on the Target alien. Once height repetitions of flashes were completed, a new Target was selected until the Target list was completed. Each session consisted in two runs, one in a Non-Adaptive (classical) and one in an Adaptive (calibration less) mode of operation. The order of the runs was randomised for each session. In both runs there was a Training (calibration) phase and an Online phase, always passed in this order. In the non-Adaptive run the data from the Training phase was used for calibrating the classifier used during the Online phase using the training-test version of the MDM algorithm. In the Adaptive session, the data from the training phase was not used at all, instead the classifier was initialised with generic class geometric means (the class grand average estimated on a database of subjects participating to a previous Brain Invaders experiment) and continuously adapted to the incoming data using the Riemannian method. Subjects were completely blind to the mode of operation and the two runs appeared to them identical'
            part.loc[0, 'Rate'] = 'Data were acquired with no digital filter applied and a sampling frequency of 512 samples per second'
            part.loc[0, 'Hardware'] = 'Stimulus was displayed on a ViewSonic (California, US) screen with length 22’’. EEG signals were acquired by means of a research-grade amplifier (g.USBamp, g.tec, Schiedlberg, Austria) and the g.GAMMAcap (g.tec, Schiedlberg, Austria) equipped with 16 wet Silver/Silver Chloride electrodes, placed according to the 10-20 international system. The amplifier was linked by USB connection to the PC where the data were acquired by means of the software OpenVibe. A visual P300 Brain-Computer Interface inspired by the famous vintage video game Space Invaders (Taito, Tokyo, Japan) was used in the experiment'
            part.loc[0, 'Reference'] = 'The reference was placed on the left earlobe and the ground at the FZ scalp location'
            part.loc[0, 'Source file'] = str(k) + '.mat'
            if(k == 1 or k == 2):
                part.loc[0, 'Experimental condition'] = 'Adaptive'
            elif(k == 3 or k == 4):
                part.loc[0, 'Experimental condition'] = 'Non-Adaptive'
            if(k == 1 or k == 3) :
                part.loc[0, 'training/online'] = 'training'
            elif(k == 2 or k == 4):
                part.loc[0, 'training/online'] = 'online'
            part.loc[0, 'Dataset'] = 'ERP_ANA'
            part.loc[0, 'Section'] = 'ERP'
            store_path = '/itet-stor/cxuan/deepeye_storage/foundation_prepared/ERP_ERP_ANA_Subject' + str(i) + '_' + usr_id + '_session' + str(j)  + '_run' + str(k) + '_trial' + str(trial_counter) + '.pkl'
            result = part.astype(pd.SparseDtype(str,fill_value=''))
            result.to_pickle(store_path) 
            print(store_path+' :done')
            trial_counter+=1
            
    
    


    
