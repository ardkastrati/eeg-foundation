from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from math import ceil
import os
import sys
import json

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


def filter_index(
    index_paths,
    path_prefix,
    min_duration,
    max_duration,
    select_sr,
    select_ref,
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
                index_element["sr"] not in select_sr
                # and edf_file["ref"] in select_ref
                and index_element["duration"] >= min_duration
                and index_element["duration"] <= max_duration
                and (
                    "Dataset" not in index_element  # escapes tueg index file
                    or index_element["Dataset"] not in discard_datasets
                )
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


class load_path_data:
    def __init__(self):
        logger = logging.getLogger("pyprep")
        logger.setLevel(logging.ERROR)
        mne.set_log_level("WARNING")

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
                    # TODO: only include the good_channels here (once the pre-processing is finished)
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
        num_threads=1,
        base_stor_dir="/scratch/mae",
    ):
        self.base_stor_dir = base_stor_dir
        self.num_threads = num_threads

        # Create the STORDIR, i.e. the location on the local node where the spectrograms will be stored.
        if not os.path.exists(self.base_stor_dir):
            os.makedirs(self.base_stor_dir)
        elif not os.access(self.base_stor_dir, os.W_OK):
            print(
                f"The directory {self.base_stor_dir} is not writable. Please check the permissions."
            )

    def load(self, index_chunk, chunk_duration, thread_id):

        num_files_in_subdir = 20_000  # Number of files to store in each subdirectory. (make a method argument)

        # Create a temporary directory within the storage directory.
        # parent = tempfile.TemporaryDirectory(dir=self.base_stor_dir)
        subdirs = {}  # List of subdirectories holding the spectrograms.
        print("Storing spectros to (STORDIR): " + self.base_stor_dir, file=sys.stderr)

        p_loader = load_path_data()
        signal_chunks_index = {}  # Dict of paths to the saved signals & metadata.
        channel_set = set()

        i = 0

        print("Starting to save the spectrograms locally", file=sys.stderr)
        print(f"len index_chunk on {thread_id}:", len(index_chunk), file=sys.stderr)

        for count_processed_elements, index_element in enumerate(index_chunk):
            channel_data_dict = p_loader(index_element)
            sr = index_element["sr"]  # sampling rate

            for channel, signal in channel_data_dict.items():

                # Convert to u_volt (micro-volt)
                signal = signal * 1_000_000

                # Divide signals into chunks of 5s and store them into a list
                chunks = []
                # chunk_duration = 4  # duration of each crop in seconds
                if (
                    len(signal) >= sr * chunk_duration
                ):  # Only proceed if signal is at least one chunk long
                    # Calculate number of full chunks
                    num_chunks = int(len(signal) // (sr * chunk_duration))
                    for j in range(num_chunks):
                        start_idx = int(j * sr * chunk_duration)
                        end_idx = int(start_idx + sr * chunk_duration)
                        chunk = signal[start_idx:end_idx]
                        chunks.append(chunk)

                # Store each chunk to self.STORDIR
                for signal_chunk in chunks:

                    # Determine which subdirectory to use.
                    # Calculate index for the file within its subdirectory.
                    subdir_index = i // num_files_in_subdir

                    if i % num_files_in_subdir == 0:
                        # Make a new temporary directory
                        # subdirs[subdir_index] = tempfile.mkdtemp(dir=parent.name)
                        subdirs[subdir_index] = os.path.join(
                            self.base_stor_dir, f"{thread_id}_{subdir_index}"
                        )
                        os.makedirs(subdirs[subdir_index], exist_ok=True)
                        print(
                            f"Created new temporary directory {subdirs[subdir_index]}",
                            file=sys.stderr,
                        )
                        print(
                            f"Current progress: {count_processed_elements}/{len(index_chunk)}...",
                            file=sys.stderr,
                        )

                    file_name = (
                        "signal" + "_" + str(thread_id) + "_" + str(i) + ".npy"
                    )  # Create the filename.
                    save_path = os.path.join(subdirs[subdir_index], file_name)

                    np.save(save_path, signal_chunk)  # Save the signal as a numpy file.
                    # Store the path to the signal and some metadata.
                    signal_chunks_index[i] = {
                        "path": save_path,
                        "ref": index_element["ref"],
                        "sr": index_element["sr"],
                        "duration": index_element["duration"],
                        "channel": channel,
                    }

                    i += 1

                channel_set.add(channel)
                chunks.clear()  # Clear the list to free memory
                # == stored all chunks for this channel ==

            channel_data_dict.clear()  # Clear the dictionary to free memory
            # == stored all channels for this index_element ==

        # == stored all index_elements for this index_chunk ==

        print(f"Saved {i} signal chunks on process {thread_id}", file=sys.stderr)

        # Store the list of paths to the spectrograms.
        path_to_signal_chunks_index = os.path.join(
            self.base_stor_dir, f"data_index_{thread_id}.txt"
        )
        with open(path_to_signal_chunks_index, "w") as file:
            json.dump(signal_chunks_index, file)

        return path_to_signal_chunks_index, channel_set

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
