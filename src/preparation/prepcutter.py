import os
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import re
import logging
from logging.handlers import RotatingFileHandler
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pymatreader import read_mat
from preprocessing import create_raw, find_bad_by_prep, filter, zapline_clean, interpolate, average_reference
from datasets.mi_limb import mi_limb_read_file

class PrepCutter:
    def __init__(self, dataset, load_directory='./', save_directory='./', load_file_pattern='*', read_file=None, override=False, verbose=False, preprocess=True):
        self.dataset = dataset
        self.load_directory = load_directory
        self.save_directory = save_directory
        self.load_file_pattern = re.compile(load_file_pattern)
        self.read_file = read_file
        self.override = override
        self.verbose = verbose
        self.preprocess = preprocess
        print("PrepCutter initialized with:")
        print(f"Loading from: {self.load_directory}")
        print(f"Saving to: {self.save_directory}")
        self.progress_log_dir = os.path.join(self.save_directory, "progress_log")
        # Define the directory for logs using try-except (because parallelism)
        try:
            os.makedirs(self.progress_log_dir, exist_ok=True)
        except OSError as e:
            print(f"Directory {self.progress_log_dir} already exists, or error in creation: {e}")

    def process_dataset(self):
        for root, dirs, files in tqdm(os.walk(self.load_directory)):
            for file in files:
                if not self.load_file_pattern.match(file):
                    continue
                # Processing logic here
                print(f"Processing: {file}")
                if not self.override and self.is_processed(self.dataset, file):
                    continue
                print(root, file)
                # Read file
                raw_files, events, montage_kind, line_freq = self.read_file(os.path.join(root, file))
                # Assumption raw_files are a list of dataframes of EEG data (column names are the channel names), and events is a dataframe with "latency", "type", "description"
                for trial, (raw_file, event) in tqdm(enumerate(zip(raw_files, events))):
                    # ------ run_minimal_preprocessing
                    bad_channels = None
                    if self.preprocess:
                        print("Preprocessing")
                        raw = create_raw(raw_file, event)
                        bad_raw, bad_channels = find_bad_by_prep(raw, montage_kind, line_freq, matlab_strict=False)
                        bad_filtered_raw = filter(bad_raw, matlab_strict=False)
                        bad_filtered_zaplined_raw = zapline_clean(bad_filtered_raw)
                        bad_filtered_zaplined_interpolated_raw = interpolate(bad_filtered_zaplined_raw, matlab_strict=False)
                        final_raw = average_reference(bad_filtered_zaplined_interpolated_raw)
                    else:
                        print("Not preprocessing")
                        raw = create_raw(raw_file, event)
                        bad_channels = str([])
                        final_raw = raw

                    if bad_channels is None: # PrepPipeline failed
                        continue
                    # ------ create a dataframe with preprocessed data - TODO: reorder the channels for the final dataframe
                    my_recording_df = pd.DataFrame(data = final_raw.get_data().T*1e6, columns = final_raw.ch_names)

                    # ------ add bad channel info
                    my_recording_df["PrepPipeline"] = bad_channels

                    # ------ add the time dimension
                    my_recording_df['time'] = final_raw.times

                    # ------ add the session information
                    my_recording_df['file'] = file

                    # ------ add event information
                    my_recording_df = self._add_event_info(my_recording_df, event)
                    self._create_segments(my_recording_df, self.dataset, file, trial, root)
                self.mark_as_processed(self.dataset, file)

        return f"Completed {self.dataset}"

    def log_path(self, dataset, file):
        return os.path.join(self.progress_log_dir, f"{dataset}-{file}.log")

    def is_processed(self, dataset, file):
        return os.path.exists(self.log_path(dataset, file.rstrip('.mat')))

    def mark_as_processed(self, dataset, file):
        open(self.log_path(dataset, file.rstrip('.mat')), 'a').close()


    def _create_segments(self, my_recording_df, dataset, file, trial, root=None):
        combined_columns = my_recording_df[['Event Description', 'Trial']].apply(tuple, axis=1)
        segment_event = combined_columns != combined_columns.shift()
        # Get the indices of the change points
        change_indices = my_recording_df.index[segment_event].tolist()

        # Add the last index of the DataFrame to mark the end of the last segment
        change_indices.append(len(my_recording_df))

        # Create the segments based on the change indices
        segments = []
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            segment = self._compress_columns(my_recording_df.iloc[start_idx:end_idx])
            segments.append(segment)
        # Save each smaller dataframe as a CSV file
        for i, segment in enumerate(segments):
            # segment = segment.astype(pd.SparseDtype(str,fill_value=''))
            suffixes = ['.mat', '.gdf', '.edf']
            filename = file
            for suffix in suffixes:
                if file.endswith(suffix):
                    filename = file.rstrip(suffix) 
            if root is not None:
                if "test" in root:
                    filename = filename + "-test"
                elif "train" in root:
                    filename = filename + "-train"
            segment.to_csv(self.save_directory + f"{dataset}-{filename}-{trial}-segment-{str(i).zfill(9)}.csv") # Todo add subjects

    def _compress_columns(self, df):
        channels = df["Channels"].iloc[0]
        for col in df.columns[int(channels):]:
            df.loc[:, col] = df.loc[:, col].apply(lambda x: str(x))
            mask = (df[col] == df[col].shift(1))
            df.loc[:, col] = df.loc[:, col].where(~mask, '')
        return df

    def _add_event_info(self, my_recording_df, event):
        my_recording_df = pd.concat([my_recording_df, event], axis=1)
        return my_recording_df


if __name__ == "__main__":
    prepcutter = PrepCutter(dataset="MI_Limb",
                            load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_Limb/",
                            save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                            load_file_pattern=r'.*\.mat$',
                            read_file=mi_limb_read_file,
                            override=False,
                            verbose=False, 
                            preprocess=False)
    prepcutter.process_dataset()