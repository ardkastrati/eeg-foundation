import os
import sys
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


def filter_index(
    index_paths,
    path_prefix,
    min_duration,
    max_duration,
    discard_sr,
    discard_datasets,
):
    """
    Filter the json `data_dir` index based on the specified conditions.
    """
    filtered_index = []
    index_lens = [0] * len(index_paths)
    index_sizes = [0] * len(index_paths)
    print("Filtering data dir", file=sys.stderr)

    # iterate over each data_dir (paths to json indices of the data),
    # i.e. we have one index for .edf and one for .pkl data
    nr_files = 0
    for index_nr, index_path in enumerate(index_paths):

        with open(index_path, "r") as file:
            index = json.load(file)
            nr_files += len(index)

        index_size = 0

        for i, index_element in enumerate(index):

            if i % 10_000 == 0:
                print(("=" * 10) + f"filtered {i} files" + ("=" * 10), file=sys.stderr)

            if (
                index_element["duration"] >= min_duration
                and index_element["duration"] <= max_duration
                and index_element["sr"] not in discard_sr
                and index_element["Dataset"] not in discard_datasets
            ):
                # the edf index comes with relative, the csv index with absolute paths
                if index_element["path"].endswith(".edf"):
                    index_element["path"] = path_prefix + index_element["path"]
                filtered_index.append(index_element)
                index_lens[index_nr] += 1
                index_sizes[index_nr] += os.path.getsize(index_element["path"]) / (
                    1024**3
                )

    print(nr_files, "files found in total", file=sys.stderr)
    print(index_lens, "files selected per index", file=sys.stderr)
    print(index_sizes, "GB of data selected per index", file=sys.stderr)
    print(len(filtered_index), "files selected in total", file=sys.stderr)
    return filtered_index, index_lens, index_sizes


def create_raw(
    data,
    ch_names1,
    sr,
    ch_names2=None,
):
    """
    Create a raw MNE object from EEG data.

    Parameters:
        data (pandas.DataFrame): The EEG data.
        ch_names1 (list): The channel names for the EEG data.
        sr (float): The sampling rate of the EEG data.
        ch_names2 (list, optional): The channel names for the MNE object. If not provided, it defaults to ch_names1.

    Returns:
        raw (mne.io.RawArray): The raw MNE object.
    """
    if ch_names2 == None:
        ch_names2 = ch_names1
    ch_types = ["eeg" for _ in range(len(ch_names1))]
    info = mne.create_info(ch_names2, ch_types=ch_types, sfreq=sr)
    eeg_data = (
        np.array(data[ch_names1].T, dtype="float") / 1_000_000
    )  # in Volt # TODO not sure if each dataset is in uv
    raw = mne.io.RawArray(eeg_data, info)
    return raw


def avg_channel(raw):
    """
    Applies average referencing to the given raw data.

    Parameters:
        raw (Raw): The raw data to be processed.

    Returns:
        Raw: The raw data with average referencing applied.
    """
    avg = raw.copy().add_reference_channels(ref_channels="AVG_REF")
    avg = avg.set_eeg_reference(ref_channels="average")
    return avg


# Add subject ID information for each path
def get_subject_id(filepath):
    if filepath.endswith("pkl"):
        # Regular expression to match the UUID in the middle of the file path
        match = re.search(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", filepath
        )
        if match:
            return match.group(0)
        return None
    elif filepath.endswith("edf"):
        parts = filepath.split("/")
        subject_id = parts[
            2
        ]  # The subject ID is the third element after splitting by '/'
        return subject_id
    else:
        assert False, f"invaliv file format for file {filepath}"


class load_path_data:
    def __init__(self, min_duration, max_duration):
        self.min_duration = min_duration
        self.max_duration = max_duration
        logger = logging.getLogger("pyprep")
        logger.setLevel(logging.ERROR)
        mne.set_log_level("WARNING")

    def get_duration(self, index_element):
        # if index_element["duration"] >= self.max_duration:
        #     dur = np.inf
        # else:
        #     dur = int(index_element["duration"])
        # return dur
        return int(index_element["duration"])

    def __call__(self, index_element):
        """
        Loads data from the given index element's path and returns a dictionary containing channel data.

        Args:
            index_element (dict): The index element containing information about the data file.

        Returns:
            dict: A dictionary containing channel data, where the keys are channel names and the values are the flattened data arrays.

        Raises:
            AssertionError: If the path is invalid.

        """
        if index_element["path"].endswith(".edf"):
            # For EDF: all channels are good at the moment
            eeg_data = mne.io.read_raw_edf(
                index_element["path"],
                include=index_element["channels"],
                preload=True,
            )

        # note: some files end in ".pkl", others only in "pkl"
        elif index_element["path"].endswith("pkl"):
            # Load DataFrame from pickle
            with open(index_element["path"], "rb") as file:
                df = pd.read_pickle(file)
                # Create a mne.Raw to be compatible with the coming processing steps
                eeg_data = create_raw(
                    data=df,
                    ch_names1=index_element["good_channels"],
                    sr=index_element["sr"],
                )

        else:
            assert False, "Invalid path"

        # Add average reference
        eeg_data = avg_channel(eeg_data)

        # Datastructure to access data for each channel
        channel_data_dict = {}

        # Note: channel_data_dict also includes the AVG_REF channel
        for channel in eeg_data.ch_names:
            idx = eeg_data.ch_names.index(channel)
            data, times = eeg_data[idx, :]
            # Flatten the data to 1D if required
            channel_data_dict[channel] = data.flatten()

        return channel_data_dict


class LocalLoader:
    def __init__(
        self,
        min_duration=1,
        patch_size=16,
        max_nr_patches=8_500,
        win_shifts=[0.25, 0.5, 1, 2, 4, 8],
        win_shift_factor=0.25,
        num_threads=1,
        base_stor_dir="/scratch/mae",
    ):

        self.min_duration = min_duration
        self.patch_size = patch_size
        self.max_nr_patches = max_nr_patches
        self.win_shifts = win_shifts
        self.win_shift_factor = win_shift_factor

        self.num_threads = num_threads
        self.base_stor_dir = base_stor_dir

        # Create the STORDIR, i.e. the location on the local node where the EEG data will be stored.
        if not os.path.exists(self.base_stor_dir):
            os.makedirs(self.base_stor_dir)
        elif not os.access(self.base_stor_dir, os.W_OK):
            print(
                f"The directory {self.base_stor_dir} is not writable. Please check the permissions."
            )

    # Copied from ByTrialDistributedSampler
    def get_nr_y_patches(self, win_size, sr):
        return int((sr // 2 * win_size + 1) // self.patch_size)

    def get_nr_x_patches(self, win_size, dur):
        win_shift = win_size * self.win_shift_factor
        x_datapoints_per_second = 1 / win_shift
        x_datapoints = dur * x_datapoints_per_second + 1
        return int(x_datapoints // self.patch_size)

    def get_nr_patches(self, win_size, sr, dur):
        return self.get_nr_y_patches(win_size=win_size, sr=sr) * (
            self.get_nr_x_patches(win_size=win_size, dur=dur)
        )

    def get_max_nr_patches(self, sr, dur):
        suitable_win_sizes = self.get_suitable_win_sizes(sr, dur)
        return max(
            [self.get_nr_patches(win_size, sr, dur) for win_size in suitable_win_sizes]
        )

    def get_suitable_win_sizes(self, sr, dur):
        return [
            win_shift
            for win_shift in self.win_shifts
            if self.get_nr_y_patches(win_shift, sr) >= 1
            and self.get_nr_x_patches(win_shift, dur) >= 1
        ]

    def get_suitable_win_size(self, sr, dur):
        win_sizes = self.get_suitable_win_sizes(sr, dur)
        return None if not win_sizes else win_sizes[0]

    def get_max_y_patches(self, sr, dur):
        max_win_shift = max(self.get_suitable_win_sizes(sr, dur))
        return self.get_nr_y_patches(win_size=max_win_shift, sr=sr)

    def load(self, index_chunk, thread_id):

        num_files_in_subdir = 20_000  # Number of files to store in each subdirectory. (make a method argument)

        subdirs = {}  # List of subdirectories holding the spectrograms.
        print("Storing spectros to (STORDIR): " + self.base_stor_dir, file=sys.stderr)

        p_loader = load_path_data(
            min_duration=self.min_duration, max_duration=self.max_duration
        )
        trial_index = {}  # Dict of paths to the saved signals & metadata.
        channel_set = set()

        num_trials = 0
        num_files = 0

        print("Starting to save the trials locally", file=sys.stderr)
        print(f"Nr. trials on {thread_id}:", len(index_chunk), file=sys.stderr)

        for num_processed_elements, index_element in enumerate(index_chunk):

            sr = index_element["sr"]
            dur = index_element["duration"]

            if dur < self.min_duration or dur > self.max_duration:
                continue

            channel_data_dict = p_loader(index_element)

            trial_info = {
                "num_channels": len(channel_data_dict),
                "channels": [],
                "paths": [],
                "sr": sr,
                "dur": dur,
                "ref": index_element["ref"],
                "Dataset": index_element["Dataset"],
                "SubjectID": index_element["SubjectID"],
            }

            for channel, channel_signal in channel_data_dict.items():
                channel_set.add(channel)

                # Convert to u_volt (micro-volt)
                channel_signal = channel_signal * 1_000_000

                # Find maximum duration which does not nuke CUDA memory
                patches = self.get_max_nr_patches(sr, dur)
                if patches > self.max_nr_patches:
                    # Split channel_signal 1-d tensor into chunks of max_dur duration
                    # Did some arithmetic to get the formula right, it's just nr_x_patches * nr_y_patches
                    #  rearranged for duration, and we iterate over the full win_shift space
                    max_durs = [
                        int(
                            (
                                (self.patch_size**2) * self.max_nr_patches
                                - sr * win_shift / 2
                                - 1
                            )
                            / (
                                sr / self.win_shift_factor / 2
                                + 1 / self.win_shift_factor / win_shift
                            )
                        )
                        for win_shift in (
                            self.win_shifts if sr >= 120 else self.win_shifts[1:]
                        )
                    ]
                    max_dur = min(max_durs)
                    jump = int(sr * max_dur)
                    signal_chunks = [
                        channel_signal[i : i + jump]
                        for i in range(0, len(channel_signal), jump)
                    ]
                else:
                    signal_chunks = [channel_signal]

                for signal in signal_chunks:

                    # Determine which subdirectory to use.
                    # Calculate index for the file within its subdirectory.
                    subdir_index = num_files // num_files_in_subdir

                    if num_files % num_files_in_subdir == 0:

                        subdirs[subdir_index] = os.path.join(
                            self.base_stor_dir, f"{thread_id}_{subdir_index}"
                        )
                        os.makedirs(subdirs[subdir_index], exist_ok=True)
                        print(
                            f"Created new temporary directory {subdirs[subdir_index]}",
                            file=sys.stderr,
                        )
                        print(
                            f"Current progress: {num_processed_elements}/{len(index_chunk)}...",
                            file=sys.stderr,
                        )

                        # In order to not lose progress: store the current state of the trial_index
                        path_to_trial_index = os.path.join(
                            self.base_stor_dir, f"data_index_{thread_id}.txt"
                        )
                        with open(path_to_trial_index, "w") as file:
                            json.dump(trial_index, file)

                    file_name = (
                        "signal" + "_" + str(thread_id) + "_" + str(num_files) + ".npy"
                    )  # Create the filename.
                    save_path = os.path.join(subdirs[subdir_index], file_name)

                    np.save(save_path, signal)  # Save the signal as a numpy file.
                    num_files += 1

                    # Store the path to the signal and some metadata.
                    trial_info["channels"].append(channel)
                    trial_info["paths"].append(save_path)

            # == stored data for all channels to different files

            trial_index[num_trials] = trial_info
            num_trials += 1
            # == stored everything for this index_element (trial) ==

        # == stored all index_elements for this index_chunk ==

        print(f"Saved {num_trials} trials on process {thread_id}", file=sys.stderr)

        # Store the trial_index
        path_to_trial_index = os.path.join(
            self.base_stor_dir, f"data_index_{thread_id}.txt"
        )
        with open(path_to_trial_index, "w") as file:
            json.dump(trial_index, file)

        return path_to_trial_index, channel_set, num_trials

    def run(self, index, chunk_duration):
        # split the index file into num_threads many groups
        chunk_size = ceil(len(index) / self.num_threads)
        print("chunk_size", chunk_size, file=sys.stderr)
        index_chunks = [
            index[i : i + chunk_size] for i in range(0, len(index), chunk_size)
        ]

        chunk_paths = []

        with tqdm(total=self.num_threads, desc="Overall Progress") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []

                for idx, chunk in enumerate(index_chunks):
                    futures.append(
                        executor.submit(
                            self.load,
                            chunk,
                            chunk_duration,
                            idx,
                        )
                    )

                for future in as_completed(futures):
                    chunk_path_to_spg_paths = future.result()
                    chunk_paths.append(chunk_path_to_spg_paths)
                    print(chunk_path_to_spg_paths)  # Log the completion
                    pbar.update(1)

        return chunk_paths
        return chunk_paths
