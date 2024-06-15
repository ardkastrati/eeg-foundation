import os
import numpy as np
import pandas as pd
import re
from itertools import groupby
from tqdm import tqdm
import json

class TaskCreator:
    def __init__(self, prepared_data_directory='./', 
                       save_task_directory="./", 
                       load_file_pattern='*', 
                       task_name='all.json', 
                       task_type='Classification',
                       verbose=False):

        self.prepared_data_directory = prepared_data_directory
        self.save_task_directory = save_task_directory
        self.load_file_pattern = re.compile(load_file_pattern)
        self.task_name = task_name
        self.task_type = task_type
        self.verbose = verbose
        
        self.start_time = None
        self.length_time = None
        self.start_channel = 1
        self.end_channel = 129
        self.on_blocks = None
        self.off_blocks = None
        self.filters = []
        self.ignore_events = []
        self.labels = []

        print("Task creator is initialized with: ")
        print("Directory to load prepared data: " + self.prepared_data_directory)
        print("Directory to save tasks: " + self.save_task_directory)
        print("Task name: " + self.task_name)
        print("Looking for file that match: " + load_file_pattern)


    def extract_data_at_events(self, extract_pattern, name_start_time, start_time, name_length_time, length_time):
        self.extract_pattern = extract_pattern
        self.start_time = start_time
        self.length_time = length_time

        print("Task creator is instructed to look for units that match structure: " + str(self.extract_pattern))
        print("Time dimension -- Cut start info: " + name_start_time)
        print("Time dimension -- Cut length info: " + name_length_time)

    def extract_data_continuously(self, name_length_time, length_time):
        self.length_time = length_time

        print("Task creator is instructed to extract data continuosly! ")
        print("Time dimension -- Cut length info: " + name_length_time)

    def ignoreEvent(self, name, f):
        self.ignore_events.append((name, f))
        print('Task creator is instructed to ignore the event: ' + name)

    def blocks(self, on_blocks, off_blocks):
        self.on_blocks = on_blocks
        self.off_blocks = off_blocks
        print("Blocks to be used are: " + str(on_blocks))
        print("Blocks to be ignored are: " + str(off_blocks))

    def addFilter(self, name, f):
        self.filters.append((name, f))
        print('Task creator is instructed to use filter: ' + name)

    def addLabel(self, name, f):
        self.labels.append((name, f))
        print('Task creator is instructed to use label: ' + name)

    def process_stream(self, stream):
        # load the pickle files
        if self.verbose: print(f"Stream is composed of {len(stream)} eye movement events.")
        stream_df, indices_in_order = self._read_stream(stream)
        if not indices_in_order:
            raise Exception(f"Indices are not in order in the sample: {stream_df}")
        events_df = self._get_event_info(stream_df)
        if self.verbose: print(events_df["type"])
        events_df = self._ignore_events(events_df)
        select = self._filter_blocks(events_df)
        select &= self._filter_pattern(events_df)
        select &= self._filter_events(events_df, stream_df)
        events_df = self._compute_labels(events_df)
        if hasattr(self, 'extract_pattern'):
            samples = self._extract_samples(stream_df, events_df, select)
        else:
            samples = self._extract_samples_continuously(stream_df, events_df, select)
        if self.verbose: print(f"Finished processing stream, and got {len(samples)} samples")
        return samples

    def process_subject(self, dataset_group, dataset, subject):
        print("Starting creating task...")
        all_units = sorted(os.listdir(self.prepared_data_directory))
        print("Length of all units: ", len(all_units))
        all_units = [unit for unit in all_units if self.load_file_pattern.match(unit) and f"{dataset_group}-{dataset}-{subject}" in unit]   
        print("Length of all units after pattern: ", len(all_units))
        all_streams = self._create_streams(all_units) # TODO
        print("Length of all streams: ", len(all_streams))
        all_samples = []
        for stream in all_streams:
            samples = self.process_stream(stream)
            all_samples.extend(samples)
        return all_samples

    def run_sequential(self, user_ids=None, split="", stream_pattern=None):
        print("Starting creating task...")
        all_units = sorted(os.listdir(self.prepared_data_directory))
        print("Length of all units: ", len(all_units))
        all_units = [unit for unit in all_units if self.load_file_pattern.match(unit)]
        print("Length of all units after pattern: ", len(all_units))
        all_streams = self._create_streams(all_units, stream_pattern)
        print("Length of all streams: ", len(all_streams))
        chosen_samples = []
        progress = tqdm(all_streams)
        for stream, user_id in progress:
            if user_ids is not None and user_id.lower() not in user_ids:
                continue
            samples = self.process_stream(stream)
            chosen_samples.extend(samples)
        # Save as JSON the chosen samples
        print(f"Finished creating task, and got {len(chosen_samples)} samples")
        if self.verbose: print(chosen_samples[:4])
        task = {self.task_name + "_" + split: chosen_samples, "task_type": self.task_type}
        json.dump(task, open(os.path.join(self.save_task_directory, self.task_name + "_" + split + ".json"), 'w'))

    def _get_event_info(self, sample_df):
        events = sample_df.copy() # sample_df[sample_df['type'] != sample_df['type'].shift()]
        events = events.reset_index(drop=False)
        events = events.apply(pd.Series.explode)
        events = events[events['type'] != events['type'].shift()]
        return events

    def get_relative_path(full_path, levels_to_remove=2):
        # Split the path into its components
        path_parts = full_path.split(os.sep)
        
        # Join the path parts starting from the specified level
        relative_path = os.sep.join(path_parts[levels_to_remove:])
        
        return relative_path

    def _read_stream(self, stream):
        sample_dfs = []
        for event in stream:
            next_df = pd.read_pickle(os.path.join(self.prepared_data_directory, event))
            next_df["picklename"] = str(get_relative_path(os.path.join(self.prepared_data_directory, event)))
            # if self.verbose: print("Reading", str(os.path.join(self.prepared_data_directory, event)))
            sample_dfs.append(next_df)
        sample_df = pd.concat(sample_dfs, axis=0)
        sample_df = self.fix_types(sample_df)
        indices_in_order = (sample_df.index.to_series().diff().fillna(1) == 1).all()
        return sample_df, indices_in_order

    def _create_streams(self, all_units, stream_pattern=None):
        # Regex pattern to extract the session part of the filename
        pattern = re.compile(r'(.+segment-)')
        # Grouping filenames using groupby
        grouped_units = [list(group) for key, group in groupby(all_units, key=lambda x: pattern.match(x).group(1))]
        if stream_pattern is not None:
            streams = []
            # If stream pattern is not None, then we need to find the stream that matches the pattern
            for group in grouped_units:
                for i in range(len(group) - len(stream_pattern)):
                    valid_stream = True
                    for j in range(len(stream_pattern)):
                        valid_stream = valid_stream and (stream_pattern[j] in group[i + j])
                    stream = [group[i + j] for j in range(len(stream_pattern))]
                    if valid_stream:
                        streams.append(stream)
        else:
            streams = grouped_units

        streams_with_user_id = []
        for stream in streams:
            user_ids = {filename.split('-')[2] for filename in stream}
            if len(user_ids) != 1:
                raise ValueError(f"Stream contains multiple user IDs: {user_ids}")
            user_id = user_ids.pop()  # Get the single user ID
            streams_with_user_id.append((stream, user_id))
        return streams_with_user_id

    def _filter_blocks(self, events):
        select = events['type'].apply(lambda x: True)
        if self.on_blocks is None:
            return select
        if self.verbose: print("Filtering the blocks: ")
        select = events["block"].isin(self.on_blocks)
        if self.verbose: print(list(zip(events["type"].tolist(), events["latency"].tolist(), select))[:100])
        return select

    def _ignore_events(self, events):
        if self.verbose: print("Ignoring the events:")
        ignore = events['type'].apply(lambda x: False)
        for name, f in self.ignore_events:
            if self.verbose: print("Applying: " + name)
            ignore |= f(events)
            if self.verbose: print(list(zip(events["type"].tolist(), events["latency"].tolist(), ignore))[:100])
        select = ignore.apply(lambda x: not x)
        return events.loc[select]

    def _filter_pattern(self, events):
        select = events['type'].apply(lambda x: True)
        if not hasattr(self, 'extract_pattern'):
            return select
        if self.verbose: print(f"Checking the pattern: {self.extract_pattern}")
        for i, event in enumerate(self.extract_pattern):
            select &= events['type'].shift(-i).isin(event)
        if self.verbose: print(list(zip(events["type"].tolist(), events["latency"].tolist(), select))[:100])
        return select

    def _filter_events(self, events, stream=None):
        if self.verbose: print("Filtering the events")
        select = events['type'].apply(lambda x: True)
        for name, f in self.filters:
            if self.verbose: print("Applying filter: " + name)
            select &= f(events, stream)
            if self.verbose: print(list(zip(events["type"].tolist(), events["latency"].tolist(), select))[:100])
        return select
    
    def _compute_labels(self, events):
        if self.verbose: print("Computing the label.")
        for name, f in self.labels:
            if self.verbose: print("Appending the next label: " + name)
            events[f"label_{name}"] = f(events)
            if self.verbose: print(events[f"label_{name}"])
        return events

    def _extract_samples_continuously(self, stream, events, select):
        if self.verbose: print("Extracting data continuously")
        all_samples = []
        # extract the useful data
        if self.verbose: print(stream)
        
        events['group'] = (select != select.shift()).cumsum()
        filtered_groups = events.loc[select].groupby('group')
        start = []
        for _, group in filtered_groups: # For each stream
            # rint(group)
            start.extend(np.arange(group.iloc[0]['latency'], group.iloc[-1]['endtime'] - 500 + 1, 500))

        length = 500
        if self.verbose: print(start)
        if self.verbose: print(length)
        for s in start:
            if s + length <= stream.index[-1]:
                window = stream.loc[int(s):int(s + length)]
                sample = {"input": window["picklename"].unique().tolist(), "output": None, "start": int(s), "length": int(length)}
                all_samples.append(sample)
            else:
                # Possible to be "padding" able. (in case we need later)
                if self.verbose: print("Stream is not enough long to get the required length for the sample, skipping.")
        return all_samples

    def _extract_samples(self, stream, events, select):
        if self.verbose: print("Extracting data from the interested events: ")
        all_samples = []
        # extract the useful data
        if self.verbose: print(stream)

        start = self.start_time(events).loc[select]
        length = events['type'].apply(lambda x: self.length_time).loc[select]
        labels = events.filter(like='label_').loc[select]
        if self.verbose: print(start)
        if self.verbose: print(length)
        if self.verbose: print(labels)
        for s, l, (label_idx, label) in zip(start, length, labels.iterrows()):
            if s + l <= stream.index[-1]:
                window = stream.loc[int(s):int(s + l)]
                sample = {"input": window["picklename"].unique().tolist(), "output": label.to_dict(), "start": int(s), "length": int(l)}
                all_samples.append(sample)
            else:
                # Possible to be "padding" able. (in case we need later)
                if self.verbose: print("Stream is not enough long to get the required length for the sample, skipping.")
        return all_samples

    def fix_types(self, sample_df):
        sample_df['sac_endpos_x'] = pd.to_numeric(sample_df['sac_endpos_x'], errors='coerce')
        sample_df['sac_startpos_x'] = pd.to_numeric(sample_df['sac_startpos_x'], errors='coerce')
        sample_df['sac_endpos_y'] = pd.to_numeric(sample_df['sac_endpos_y'], errors='coerce')
        sample_df['sac_startpos_y'] = pd.to_numeric(sample_df['sac_startpos_y'], errors='coerce')
        sample_df['fix_avgpos_x'] = pd.to_numeric(sample_df['fix_avgpos_x'], errors='coerce')
        sample_df['fix_avgpos_y'] = pd.to_numeric(sample_df['fix_avgpos_y'], errors='coerce')
        return sample_df