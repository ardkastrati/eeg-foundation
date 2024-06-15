import pandas as pd

def datastructure(data, columns, file, usr_id, metadata="Missing", timeseries="Missing", session="Missing", filtering="Missing", rate="Missing", hardware="Missing", reference="Missing", dataset="Missing", section="Missing", eeg_channels="Missing", event_annotation="Missing", trial="Missing", event_description="Missing"):
    df = pd.DataFrame(data = data, columns = columns)
    df['Source File'] = file
    df['User id'] = usr_id
    df['Subject Metadata'] = metadata
    df['Timeseries'] = timeseries
    df['Trial'] = trial
    df['Session'] = session
    df['Filtering'] = filtering
    df['Rate'] = rate
    df['Hardware'] = hardware
    df['Reference'] = reference
    df['Dataset'] = dataset
    df['Section'] = section
    df['Channels'] = eeg_channels
    df['Event Annotation'] = event_annotation
    df['Event Description'] = event_description
    df_channels = df[columns]
    df_events = df.drop(columns=columns)
    return df_channels, df_events