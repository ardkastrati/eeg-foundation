from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

from tqdm import tqdm
from src.data.transforms import (
    crop_spectrogram,
    load_path_data,
    load_channel_data,
    fft_256,
    custom_fft,
    standardize,
)
import torch
import mne
import lightning
import matplotlib.pyplot as plt
import json
import numpy as np
import tempfile
import os
import time
from socket import gethostname
from torch.nn.functional import interpolate
import sys


class LocalLoader:
    def __init__(
        self,
        num_threads,
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

        # Create a temporary directory within the storage directory.
        parent_path = os.path.join(self.base_stor_dir, f"parent_{thread_id}")
        os.makedirs(parent_path, exist_ok=True)
        parent = tempfile.TemporaryDirectory(dir=parent_path)
        subdirs = {}  # List of subdirectories holding the spectrograms.
        print("Storing spectros to (STORDIR): " + parent.name, file=sys.stderr)

        p_loader = load_path_data()
        spg_paths = {}  # List of paths to the saved spectrograms.

        i = 0

        print("Starting to save the spectrograms locally", file=sys.stderr)
        print(f"len index_chunk on {thread_id}:", len(index_chunk), file=sys.stderr)

        for index_element in index_chunk:
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
                    subdir_index = i // 10_000

                    if i % 10_000 == 0:
                        # Make a new temporary directory
                        subdirs[subdir_index] = tempfile.mkdtemp(dir=parent.name)
                        print(
                            f"Created new temporary directory {subdirs[subdir_index]}",
                            file=sys.stderr,
                        )

                    file_name = "signal_" + str(i) + ".npy"  # Create the filename.
                    save_path = os.path.join(subdirs[subdir_index], file_name)

                    np.save(
                        save_path, signal_chunk
                    )  # Save the spectrogram as a numpy file.
                    spg_paths[i] = save_path  # Store the path in the index.

                    i += 1

                # == stored all chunks for this channel ==
            # == stored all channels for this index_element ==
        # == stored all index_elements for this index_chunk ==

        print(f"Saved {i} spectrograms on thread {thread_id}", file=sys.stderr)

        # Store the list of paths to the spectrograms.
        path_to_spg_paths = os.path.join(parent_path, "data_index")
        with open(path_to_spg_paths, "w") as file:
            json.dump(spg_paths, file)

        return path_to_spg_paths

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


def load_and_save_spgs(
    raw_paths,
    STORDIR="/scratch/mae",
    TMPDIR="/home/maxihuber/eeg-foundation/tmp_dir",
    window_size=1,
    window_shift=0.125,
    target_size=(64, 64),
    chunk_duration=4,
):
    pass
    # print("Using temporary directory (TMPDIR): " + TMPDIR)
    # print("Storing spectros to (STORDIR): " + STORDIR)

    # # Callable. Given an element of raw_paths index, loads and returns all specified channels from the EEG file.
    # p_loader = load_path_data()

    # # Prepare the FFT (STFT) transformation with GPU acceleration (per channel, I think). Might be able to do all channels in parallel?
    # # fft = fft_256(window_size=window_size, window_shift=window_shift, sr=??, cuda=True)

    # # Crop the spectrogram to the target size.
    # # Maybe needs more cropping information because we removed the precropping when loading per channel.
    # crop = crop_spectrogram(target_size=target_size)
    # std = standardize()

    # # Create the STORDIR, i.e. the location on the local node where the spectrograms will be stored.
    # if not os.path.exists(STORDIR):
    #     os.makedirs(STORDIR)
    # elif not os.access(STORDIR, os.W_OK):
    #     print(f"The directory {STORDIR} is not writable. Please check the permissions.")

    # # Create a temporary directory within the storage directory.
    # parent = tempfile.TemporaryDirectory(dir=STORDIR)

    # subdirs = {}  # List of subdirectories holding the spectrograms.

    # spg_paths = {}  # List of paths to the saved spectrograms.

    # i = 0
    # print("Starting to save the spectrograms locally")
    # print(len(raw_paths))
    # for index_element in raw_paths:
    #     print("=" * 100, file=sys.stderr)

    #     # print(index_element["path"])

    #     # Load data for this file_path
    #     channel_data_dict = p_loader(index_element)
    #     sr = index_element["sr"]  # sample rate

    # fft = custom_fft(
    #     window_seconds=1,
    #     window_shift=0.0625,
    #     sr=sr,
    #     cuda=True,
    # )

    #     for channel, signal in channel_data_dict.items():

    #         # Convert to u_volt (micro-volt)
    #         # not sure, if everything is in volt
    #         # (i.e. if this transform is always necessary)
    #         signal = signal * 1_000_000

    #         # Convert to tensor
    #         signal = torch.from_numpy(signal)

    #         # Maybe this squeezing is only necessary for EDF, find out when it breaks
    #         signal = signal.squeeze(0)

    #         # Divide signals into chunks of 5s and store them into a list
    #         chunks = []
    #         # chunk_duration = 4  # duration of each crop in seconds
    #         if (
    #             len(signal) >= sr * chunk_duration
    #         ):  # Only proceed if signal is at least one chunk long
    #             # Calculate number of full chunks
    #             num_chunks = int(len(signal) // (sr * chunk_duration))
    #             for j in range(num_chunks):
    #                 start_idx = int(j * sr * chunk_duration)
    #                 end_idx = int(start_idx + sr * chunk_duration)
    #                 chunk = signal[start_idx:end_idx]
    #                 chunks.append(chunk)

    #         # For each chunk, transform it into a spectrogram using FFT and save it to disk.
    #         for signal_chunk in chunks:

    #             # Determine which subdirectory to use.
    #             # Calculate index for the file within its subdirectory.
    #             subdir_index = i // 10_000
    #             index = i % 10_000

    #             if i % 10_000 == 0:
    #                 # Make a new temporary directory
    #                 subdirs[subdir_index] = tempfile.mkdtemp(dir=parent.name)
    #                 print(f"Created new temporary directory {subdirs[subdir_index]}")

    #             signal_chunk = signal_chunk.to("cuda")  # Transfer the data to GPU.

    #             # Could do interpolation signal to higher frequency here (we don't atm)

    #             # STFT: returns spectrogram in DB (Decibel) scale
    #             spg = fft(signal_chunk)  # Compute the spectrogram using FFT.

    #             # Crop spectrogram to target_size
    #             spg = crop(spg)

    #             # Normalize cropped spectrogram (for model input)
    #             spg = std(spg)

    #             # Transfer the spectrogram back to CPU and convert to numpy array.
    #             spg = spg.cpu().numpy()

    #             file_name = "spg" + str(index) + ".npy"  # Create the filename.
    #             save_path = os.path.join(subdirs[subdir_index], file_name)

    #             np.save(save_path, spg)  # Save the spectrogram as a numpy file.
    #             spg_paths[i] = save_path  # Store the path in the index.

    #             i += 1

    # print("Saved the spectrograms locally")

    # # Store the list of paths to the spectrograms.
    # path_to_spg_paths = os.path.join(parent.name, "data_index")
    # with open(path_to_spg_paths, "w") as file:
    #     json.dump(spg_paths, file)

    # # Store the path to the list of paths for access by other processes.
    # with open(os.path.join(TMPDIR, f"index_path_{gethostname()}.txt"), "w") as file:
    #     file.write(path_to_spg_paths)

    # return spg_paths, parent, subdirs  # Return the data index and directory info.
