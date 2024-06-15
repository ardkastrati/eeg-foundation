import pandas as pd
import numpy as np 
import scipy.io as io
import uuid
import os
from datasets.datastructure import datastructure

def mi_gvh_read_file(file):
    columns_G = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'FC6', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
                'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                'CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PPO1h',
                'PPO2h', 'POz', 'EOG-R-Top', 'EOG-R-Side', 'EOG-R-Bottom', 'EOG-L-Top', 'EOG-L-Side', 'EOG-L-Bottom']

    columns_V = ['AFz', 'F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2',
                'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P3', 'P1', 'Pz', 'P2', 'P4', 'EOG-L-Top', 'EOG-R-Top', 
                'EOG-L-Side', 'EOG-R-Side', 'EOG-L-Bottom', 'EOG-R-Bottom']


    columns_H = ['FC3', 'FCz', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CPz', 'CP4', 'A2']


    def translate_Event(i):
        if(i == 503587):
            return 'right hand movement towards object'
        elif(i == 501794):
            return 'right hand movement palmar grasp'
        elif(i == 534562):
            return 'inter-trial interval'
        elif(i == 503588):
            return 'right hand movement towards object'
        elif(i == 501795):
            return 'right hand movement lateral grasp'
        elif(i == 534563):
            return 'inter-trial interval'
        elif(i == 768):
            return 'rest'
        elif(i == 769):
            return 'inter-trial interval'
        elif(i == 10):
            return 'eye movement vertically'
        elif(i == 11):
            return 'inter-trial interval'
        elif(i == 12):
            return 'eye movement horizontally'
        elif(i == 13):
            return 'inter-trial interval'
        elif(i == 14):
            return 'eye blink'
        elif(i == 15):
            return 'inter-trial interval'
        else:
            return 'unknown'

    
    
    columns = None
    if 'G' in str(os.path.basename(file)): 
        columns = columns_G

        mat = io.loadmat(file)
        usr_id = str(uuid.uuid1())
        data = mat['signal']
        data = np.transpose(data)
        raw_file, event_df = datastructure(data, columns, file, usr_id, # We take only EEG for now
                                    session="1", 
                                    metadata='Participants were able-bodied and right handed',
                                    rate=256,
                                    reference='Reference at right earlobe and ground at channel AFz',
                                    filtering='All signals were recorded using a sampling frequency of 256 Hz and prefiltered using an 8th order Chebyshev filter from 0.01 to 100 Hz. A notch filter at 50 Hz were used to suppress the power noise',
                                    hardware='Recordings using the gel-based recording system (g.tec USBamp/g.tec Ladybird system, g.tec medical engineering GmbH, Austria) were performed at the Institute of Neural Engineering at Graz University of Technology. EEG measured with 58 active electrodes positioned over frontal, central, and parietal areas. EOG recorded by using six additional electrodes positioned infra and superior orbital to the left and right eye and on the outer canthi',
                                    dataset="MI_GVH", section="MI", eeg_channels=len(columns), event_annotation='Participants were seated on a chair in front of a table and instructed to rest their right hand on a sensorized base position which was positioned in front of them. On the table, we placed an empty jar and a jar with a spoon stuck in it. Both objects were in a comfortable reaching distance equidistant to the study participants right hand. Participants were instructed to perform reach-and-grasp actions using their right hand towards the objects placed on the table. In case of the empty jar they grasped the objects using a palmar grasp. In case of the spoon, they were instructed to grasp the spoon with a lateral grasp. Though participants performed the tasks in a self-initiated manner, we instructed them to focus their gaze on the designated object for 2 seconds before initiating the reach-and-grasp action. Once they completed the grasp, they held the object for at least 1-2 seconds. We also recorded 3 minutes of rest at the start, after the second movement run (at half time) and at the end of the experiment, where participants were tasked to focus their gaze on a fixation point in the middle of the table. In addition, we recorded horizontal and vertical eye movements as well as blinks')        
        
        events = mat['events']
        codes = events['codes'][0][0]
        pos = events['positions'][0][0]    

        for j in range(0, len(pos)-1):
            start = pos[j][0]
            end = pos[j+1][0]
            event_df.loc[start:end-1, 'Trial'] = j+1
            code = codes[j][0]
            event_df.loc[start:end-1, 'Event Description'] = translate_Event(code)
        return [raw_file],  [event_df], "standard_1010", 50

    elif 'V' in str(os.path.basename(file)):
        columns = columns_V

        mat = io.loadmat(file)
        usr_id = str(uuid.uuid1())
        data = mat['signal']
        data = np.transpose(data)
        raw_file, event_df = datastructure(data, columns, file, usr_id, # We take only EEG for now
                                    session="1", 
                                    metadata='Participants were able-bodied and right handed',
                                    rate=256,
                                    reference='Reference at right earlobe and ground at channel AFz',
                                    filtering='All signals were recorded using a sampling frequency of 256 Hz and prefiltered using an 8th order Chebyshev filter from 0.01 to 100 Hz. A notch filter at 50 Hz were used to suppress the power noise',
                                    hardware='The recordings with the EEG-Versatile system (Bitbrain, Spain) were conducted in the office environment of Bitbrain (Zaragoza, Spain), guided by personnel of the Institute of Neural Engineering, Graz University of Technology. EEG measured by using 32 water-based electrodes positioned over frontal, central and parietal positions. EOG measured by six electrodes positioned infra and superior orbital and the outer canthi',
                                    dataset="MI_GVH", section="MI", eeg_channels=len(columns), event_annotation='Participants were seated on a chair in front of a table and instructed to rest their right hand on a sensorized base position which was positioned in front of them. On the table, we placed an empty jar and a jar with a spoon stuck in it. Both objects were in a comfortable reaching distance equidistant to the study participants right hand. Participants were instructed to perform reach-and-grasp actions using their right hand towards the objects placed on the table. In case of the empty jar they grasped the objects using a palmar grasp. In case of the spoon, they were instructed to grasp the spoon with a lateral grasp. Though participants performed the tasks in a self-initiated manner, we instructed them to focus their gaze on the designated object for 2 seconds before initiating the reach-and-grasp action. Once they completed the grasp, they held the object for at least 1-2 seconds. We also recorded 3 minutes of rest at the start, after the second movement run (at half time) and at the end of the experiment, where participants were tasked to focus their gaze on a fixation point in the middle of the table. In addition, we recorded horizontal and vertical eye movements as well as blinks')        
        
        events = mat['events']
        codes = events['codes'][0][0][0]
        pos = events['positions'][0][0][0]

        for j in range(0, len(pos)-1):
            start = pos[j]
            end = pos[j+1]
            event_df.loc[start:end-1, 'Trial'] = j+1
            event_df.loc[start:end-1, 'Trial'] = j+1
            code = codes[j]
            event_df.loc[start:end-1, 'Event Description'] = translate_Event(code)
 
        return [raw_file],  [event_df], "standard_1020", 50

    elif 'H' in str(os.path.basename(file)):
        columns = columns_H

        mat = io.loadmat(file)
        usr_id = str(uuid.uuid1())
        data = mat['signal']
        data = np.transpose(data)
        raw_file, event_df = datastructure(data, columns, file, usr_id, # We take only EEG for now
                                    session="1", 
                                    metadata='Participants were able-bodied and right handed',
                                    rate=256,
                                    reference='Reference and ground at left earlobe',
                                    filtering='All signals were recorded using a sampling frequency of 256 Hz and prefiltered using an 8th order Chebyshev filter from 0.01 to 100 Hz. A notch filter at 50 Hz were used to suppress the power noise',
                                    hardware= 'The recordings with the EEG-Hero headset (Bitbrain, Spain) were performed in the office environment of Bitbrain (Zaragoza, Spain), guided by personnel of the Institute of Neural Engineering, Graz University of Technology. EEG measured by 11 dry electrodes located over sensorimotor areas according to the international 10/20 system',
                                    dataset="MI_GVH", section="MI", eeg_channels=len(columns), event_annotation='Participants were seated on a chair in front of a table and instructed to rest their right hand on a sensorized base position which was positioned in front of them. On the table, we placed an empty jar and a jar with a spoon stuck in it. Both objects were in a comfortable reaching distance equidistant to the study participants right hand. Participants were instructed to perform reach-and-grasp actions using their right hand towards the objects placed on the table. In case of the empty jar they grasped the objects using a palmar grasp. In case of the spoon, they were instructed to grasp the spoon with a lateral grasp. Though participants performed the tasks in a self-initiated manner, we instructed them to focus their gaze on the designated object for 2 seconds before initiating the reach-and-grasp action. Once they completed the grasp, they held the object for at least 1-2 seconds. We also recorded 3 minutes of rest at the start, after the second movement run (at half time) and at the end of the experiment, where participants were tasked to focus their gaze on a fixation point in the middle of the table. In addition, we recorded horizontal and vertical eye movements as well as blinks')        
        events = mat['events']
        codes = events['codes'][0][0][0]
        pos = events['positions'][0][0][0]

        for j in range(0, len(pos)-1):
            start = pos[j]
            end = pos[j+1]
            event_df.loc[start:end-1, 'Trial'] = j+1
            code = codes[j]
            event_df.loc[start:end-1, 'Event Description'] = translate_Event(code)
    
        return [raw_file],  [event_df], "standard_1020", 50

    




