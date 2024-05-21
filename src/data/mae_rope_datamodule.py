import os
import sys
import time
import glob
import shutil
import psutil

import json
import numpy as np
import statistics
from socket import gethostname
from pympler import asizeof

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import wandb

from src.data.mae_rope_dataset import PathDataset
from src.data.mae_rope_sampler import DurationBasedSampler
from src.data.mae_rope_distributedsampler import DurationBasedDistributedSampler
from src.data.transforms import custom_fft, crop_and_normalize_spg


class TrainDataModule(LightningDataModule):
    def __init__(
        self,
        # Dataset
        train_val_split=[0.95, 0.05],
        # Dataloader
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        # Data Loading
        stor_dir="/scratch/mae",
        # TODO: Data Filtering (sr, dur, channel, etc.)
        # Data Preprocessing
        window_size=1,
        window_shift=1 / 16,
    ):
        super().__init__()

        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.stor_dir = stor_dir

        self.window_size = window_size
        self.window_shift = window_shift

        self.save_hyperparameters(logger=False)

    # == Setup ==========================================================================================================================

    def setup(self, stage=None):

        paths_to_data_index = glob.glob(os.path.join(self.stor_dir, "data_index_*.txt"))
        print("[setup] Collecting data from these files:", paths_to_data_index)

        signal_index = {}
        num_datapoints = 0
        data_seconds_from_source = 0

        for path_to_data_index in paths_to_data_index:
            with open(path_to_data_index, "r") as index_file:
                # Read information about a subset of the files
                chunks_index = json.load(index_file)
                for _, chunk_dict in chunks_index.items():
                    signal_index[num_datapoints] = chunk_dict
                    data_seconds_from_source += chunk_dict["duration"]
                    num_datapoints += 1

        print(f"[setup] We have {num_datapoints} many signals.")
        print(f"[setup] This is {data_seconds_from_source} seconds.")

        """
        signal_index is of the following shape:
        {
            0: {"path": path/to/signal0.npy, "sr": ..., "duration": ..., "channel": ..., "SubjectID": ...},
            1: {"path": path/to/signal1.npy, "sr": ..., "duration": ..., "channel": ..., "SubjectID": ...},
            ...
            num_datapoints-1: {"path": path/to/signaln-1.npy, "sr": ..., "duration": ..., "channel": ..., "SubjectID": ...},

        """

        # Sampling rates and corresponding max durations
        max_durations = {
            100: 1000,
            128: 1000,
            160: 1000,
            200: 1000,
            250: 1000,
            256: 1000,
            400: 1000,
            500: 1000,
            512: 1000,
            563: 1000,
            565: 1000,
            567: 1000,
            717: 1000,
            1000: 1000,
        }

        # subject_id -> channel -> [(path/to/signal, sr, dur, time_used)]
        # time_used is how many seconds of this signal we have already used within a batch
        id_to_channel_to_signals = {}

        for idx, index_element in signal_index.items():
            subject_id = index_element["SubjectID"]
            channel = index_element["channel"]
            if subject_id not in id_to_channel_to_signals:
                id_to_channel_to_signals[subject_id] = {channel: [[idx, index_element]]}
            else:
                if channel not in id_to_channel_to_signals[subject_id]:
                    id_to_channel_to_signals[subject_id][channel] = [
                        [idx, index_element]
                    ]
                else:
                    id_to_channel_to_signals[subject_id][channel].append(
                        [idx, index_element]
                    )

        for subject_id, channel_to_signals in id_to_channel_to_signals.items():
            channel_to_duration = {
                channel: sum([signal[1]["duration"] for signal in signals])
                for channel, signals in channel_to_signals.items()
            }
            print(f"[setup] {subject_id}:", channel_to_duration)

        entire_dataset = PathDataset(signal_index)

        print("Nr. of datapoints:", len(entire_dataset), file=sys.stderr)

        train_size = int(self.train_val_split[0] * len(entire_dataset))
        val_size = len(entire_dataset) - train_size

        print("TRAINSIZE IS:::::::::::::::::::::", train_size)
        print("VALSIZE IS:::::::::::::::::::::::", val_size)

        # Instantiate datasets for training and validation
        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(
        #     entire_dataset, [train_size, val_size]
        # )
        self.train_dataset = entire_dataset

        self.train_sampler = DurationBasedDistributedSampler(
            self.train_dataset, id_to_channel_to_signals, max_durations
        )
        # self.val_sampler = DurationBasedDistributedSampler(
        #     self.val_dataset, id_to_channel_to_signals, max_durations
        # )

    # == Collate Functions ===============================================================================================================

    def custom_collate_fn(self, batch):
        # Batch: list of (signal, sr, chn) tuples

        # Fourier transform => spectrograms
        fft = custom_fft(
            window_seconds=1,
            window_shift=1 / 16,
            sr=batch[0][1],
            cuda=False,
        )
        spgs = []
        chn_list = []

        for signal, sr, chn in batch:
            spgs.append(fft(signal))
            chn_list.append(chn)

        # Cropping to target dimensions -> this can be done more efficiently
        # Currently, we just crop/pad each spectrogram to median_length (time-axis)
        #  (note: the freq-axis is the same for each spg, as they are from the same subject & dataset)
        spg_lengths = [spg.shape[1] for spg in spgs]
        median_length = int(statistics.median(spg_lengths))

        # Crop/pad and normalize spectrograms
        spgs = [crop_and_normalize_spg(spg, median_length) for spg in spgs]

        batch = torch.stack(spgs)
        batch.unsqueeze_(1)

        # batch.shape: B, 1, H, W

        return {"batch": batch, "chn_list": chn_list}

    def probe_max_patches(self, batch):
        return {"batch": torch.randn(1, 1, 48, 45_952), "chn_list": [None]}

    # == Data Loaders ===================================================================================================================

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.custom_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            batch_sampler=self.train_sampler,
        )

    def val_dataloader(self):
        pass
        # return DataLoader(
        #     self.val_dataset,
        #     collate_fn=self.custom_collate_fn,
        #     num_workers=self.num_workers,
        #     pin_memory=self.pin_memory,
        #     prefetch_factor=self.prefetch_factor,
        #     batch_sampler=self.sampler,
        # )
