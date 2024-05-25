import os
import random
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
from natsort import natsorted

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import wandb

from src.data.mae_rope_dataset import PathDataset, TrialDataset, ChannelDataset
from src.data.mae_rope_distributedsampler import ByTrialDistributedSampler
from src.data.transforms import (
    crop_spg,
    custom_fft,
    crop_and_normalize_spg,
    normalize_spg,
)


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
        data_index_pattern="data_index_*.txt",
        # Network
        patch_size=16,
        max_nr_patches=4_000,
        win_shifts=[0.25, 0.5, 1, 2, 4, 8],
        win_shift_factor=0.25,
    ):
        super().__init__()

        self.train_val_split = train_val_split

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.stor_dir = stor_dir
        self.data_index_pattern = data_index_pattern

        self.patch_size = patch_size
        self.max_nr_patches = max_nr_patches
        self.win_shifts = win_shifts
        self.win_shift_factor = win_shift_factor

        self.save_hyperparameters(logger=False)

    # == Setup ==========================================================================================================================

    def setup(self, stage=None):

        paths_to_data_index = natsorted(
            glob.glob(os.path.join(self.stor_dir, self.data_index_pattern))
        )
        print("[setup] Collecting data from these files:", paths_to_data_index)

        # full_trial_index = {}
        full_channel_index = {}
        num_trials = 0
        num_signals = 0
        data_seconds = 0

        for path_to_data_index in paths_to_data_index:
            with open(path_to_data_index, "r") as index_file:
                # Read information about a subset of the files
                trial_index = json.load(index_file)
                for _, trial_info in trial_index.items():
                    # full_trial_index[num_trials] = trial_info
                    for chn, path in zip(trial_info["channels"], trial_info["paths"]):
                        full_channel_index[num_signals] = {
                            "path": path,
                            "channel": chn,
                            "sr": trial_info["sr"],
                            "dur": trial_info["dur"],
                            "trial_idx": num_trials,
                            "SubjectID": trial_info["SubjectID"],
                        }
                        num_signals += 1
                        data_seconds += trial_info["dur"]
                    num_trials += 1

        print(f"[setup] We have data from {num_trials} trials.")
        print(f"[setup] This is {int(data_seconds)} seconds (single-channel).")

        print(
            "[setup] Truncated trial index:",
            {
                channel_idx: full_channel_index[channel_idx]
                for channel_idx in full_channel_index
                if channel_idx < 5
            },
            file=sys.stderr,
        )

        full_dataset = ChannelDataset(full_channel_index)
        # full_dataset = TrialDataset(full_trial_index)

        self.train_size = int(self.train_val_split[0] * len(full_dataset))
        self.val_size = len(full_dataset) - self.train_size

        print(f"[setup] Train size: {self.train_size}")
        print(f"[setup] Val size: {self.val_size}")

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [self.train_size, self.val_size]
        )

        self.train_sampler = ByTrialDistributedSampler(
            mode="train",
            full_dataset=full_dataset,
            subset_indices=self.train_dataset.indices,
            patch_size=self.patch_size,
            max_nr_patches=self.max_nr_patches - 500,
            win_shifts=self.win_shifts,
            win_shift_factor=self.win_shift_factor,
            shuffle=True,
            seed=0,
        )

        self.val_sampler = ByTrialDistributedSampler(
            mode="val",
            full_dataset=full_dataset,
            subset_indices=self.val_dataset.indices,
            patch_size=self.patch_size,
            max_nr_patches=self.max_nr_patches - 500,
            win_shifts=self.win_shifts,
            win_shift_factor=self.win_shift_factor,
            shuffle=False,
            seed=0,
        )

    # == Collate Functions ===============================================================================================================

    def snake_collate_fn(self, batch):

        batch_len = len(batch)
        print("[snake_collate_fn] Batch size:", batch_len, file=sys.stderr)

        srs = [sample["sr"] for sample in batch]
        assert all(
            sr == srs[0] for sr in srs
        ), f"[snake_collate_fn] Differing sampling rates within batch, srs={srs}"

        sr = srs[0]
        min_dur = min([sample["dur"] for sample in batch])

        # Randomly sample a win_size for the STFT
        valid_win_shifts = [
            win_shift
            for win_shift in self.win_shifts
            if self.train_sampler.get_nr_y_patches(win_shift, sr) >= 1
            and self.train_sampler.get_nr_x_patches(win_shift, min_dur) >= 1
        ]
        assert (
            len(valid_win_shifts) > 0
        ), "[snake_collate_fn] No valid valid_win_shifts found"

        # print("[snake_collate_fn] Sampling set:", valid_win_shifts, file=sys.stderr)
        win_size = random.choice(valid_win_shifts)

        fft = custom_fft(
            window_seconds=win_size,
            window_shift=win_size * self.win_shift_factor,
            sr=sr,
            cuda=False,
        )

        spgs = []  # Spectrograms of this batch
        chns = []  # List to store channel tensors for each patch
        spgs_w = set()  # Widths of the spectrograms

        H, W = 0, 0

        for i, sample in enumerate(batch):

            signal = sample["signal"]
            channel = self.get_generic_channel_name(sample["channel"])

            # STFT
            spg = fft(signal)
            spg = crop_spg(spg)

            H_new, W_new = spg.shape[0], spg.shape[1]
            h_new, w_new = H_new // self.patch_size, W_new // self.patch_size

            # Create a tensor filled with the current channel value
            channel_tensor = torch.full((h_new, w_new), channel, dtype=torch.float32)

            spgs.append(spg)
            chns.append(channel_tensor)
            spgs_w.add(w_new)

            H = H_new
            W += W_new

        total_patches = H * W // (self.patch_size**2)

        assert (
            total_patches <= self.max_nr_patches
        ), f"Total patches: {total_patches}, Max patches: {self.max_nr_patches}"

        h, w = H // self.patch_size, W // self.patch_size

        # Now, given the H, W information, we can create a tensor of desired shape (B, H_new, W_new)

        # First, randomly sample a valid batch size
        batch_size = random.choice(
            [
                batch_size
                for batch_size in range(1, batch_len + 1)
                if w % batch_size == 0
            ]
        )

        spgs_rows = [[]] * batch_size
        chns_rows = [[]] * batch_size
        means_rows = [[]] * batch_size

        row_size = W // batch_size
        cur_row = 0
        cur_W = 0

        for spg, chn in zip(spgs, chns):
            if cur_W + spg.shape[1] <= row_size:
                # Can use full spg
                spg, mean = normalize_spg(spg)
                spgs_rows[cur_row].append(spg)
                chns_rows[cur_row].append(chn)
                means_rows[cur_row].append(mean)
                cur_W += spg.shape[1]
                if cur_W == row_size:
                    cur_W = 0
                    cur_row += 1
            else:
                # Need to split this spg to multiple rows
                #  in a while loop
                spg_W = spg.shape[1]
                while spg_W > 0:
                    take_W = min(spg_W, row_size - cur_W)
                    spg_chunk = spg[:, :take_W]
                    chn_chunk = chn[:, :take_W]
                    spg_chunk, mean_chunk = normalize_spg(spg_chunk)
                    spgs_rows[cur_row].append(spg_chunk)
                    chns_rows[cur_row].append(chn_chunk)
                    means_rows[cur_row].append(mean_chunk)
                    spg_W -= take_W
                    cur_W += take_W
                    cur_W %= row_size
                    if cur_W == 0:
                        cur_row += 1

        # Concatenate rows
        final_batch = torch.stack([torch.cat(row, dim=1) for row in spgs_rows])
        channels = torch.stack([torch.cat(row, dim=1) for row in chns_rows])

        assert (
            final_batch.shape == channels.shape
        ), f"Batch shape: {final_batch.shape}, Channels shape: {channels.shape}"
        assert all(
            [final_batch[i].shape[1] == row_size for i in range(batch_size)]
        ), f"Batch shape: {final_batch.shape} is not ({H}, {W})"

        final_batch.unsqueeze_(1)
        channels.unsqueeze_(1)

        # Flatten the channels tensor, the batch will be automatically by the network
        channels = channels.flatten(2)

        # Send the constructed batch to the network
        return {
            "batch": final_batch,
            "channels": channels,
            "means": means_rows,
            "win_size": win_size,
        }

    def multi_trial_batch_collafe_fn(self, batch):

        print("[custom_collate_fn] Batch size:", len(batch), file=sys.stderr)

        # Batch: list of (signals, channels, win_size, trial_info["sr"], trial_info["duration"]) tuples
        win_size = batch[0][2]
        sr = batch[0][3]
        print("[custom_collate_fn] Window size:", win_size, file=sys.stderr)
        print("[custom_collate_fn] Sampling rate:", sr, file=sys.stderr)

        spgs = {}
        total_dur = 0

        # Transform signals to spectrograms
        for signals, chn_list, _, _, dur in batch:
            # Fourier transform => spectrograms
            fft = custom_fft(
                window_seconds=win_size,
                window_shift=win_size / 4,
                sr=sr,
                cuda=False,
            )

            for chn, signal in zip(chn_list, signals):
                chn = chn.lower()
                spg = fft(signal)
                spg = crop_spg(spg)
                spg = normalize_spg(spg)
                if chn not in spgs:
                    spgs[chn] = [spg]
                else:
                    spgs[chn].append(spg)

            total_dur += dur

        channel_to_len = {chn: sum([spg.shape[1] for spg in spgs[chn]]) for chn in spgs}
        max_len = max(max(channel_to_len.values()), self.patch_size)

        # concatenate along time axis, padding up to max_len
        spgs_cat_pad_by_channel = []
        channels = []
        for chn, spectros in spgs.items():

            # Concatenate signals from the same channel along the time dimension
            concat_spectros = torch.cat(spectros, dim=1)

            # Determine the amount of padding needed
            padding_length = max_len - concat_spectros.shape[1]
            if padding_length > 0:
                # Apply zero-padding
                padding = torch.zeros((concat_spectros.shape[0], padding_length))
                padded_spectros = torch.cat((concat_spectros, padding), dim=1)
            else:
                # No padding needed if concat_spectros is already of length max_len or more
                padded_spectros = concat_spectros

            spgs_cat_pad_by_channel.append(padded_spectros)
            channels.append(chn)

        batch = torch.stack(spgs_cat_pad_by_channel)
        batch.unsqueeze_(1)

        print(
            "[custom_collate_fn] Batch shape:",
            batch.shape,
            "(B, C, H, W)",
            file=sys.stderr,
        )

        return {
            "batch": batch,
            "chn_list": channels,
            "win_size": win_size,
        }

    def single_trial_batch_collate_fn(self, batch):

        print("[custom_collate_fn] Batch size:", len(batch), file=sys.stderr)

        # Batch: list of (signals, channels, win_size, trial_info["sr"], trial_info["duration"]) tuples
        win_size = batch[0][2]
        sr = batch[0][3]
        print("[custom_collate_fn] Window size:", win_size, file=sys.stderr)
        print("[custom_collate_fn] Sampling rate:", sr, file=sys.stderr)

        spgs = {}
        total_dur = 0

        # Transform signals to spectrograms
        for signals, chn_list, _, _, dur in batch:
            # Fourier transform => spectrograms
            fft = custom_fft(
                window_seconds=win_size,
                window_shift=win_size / 4,
                sr=sr,
                cuda=False,
            )

            for chn, signal in zip(chn_list, signals):
                chn = chn.lower()
                spg = fft(signal)
                spg = crop_spg(spg)
                spg = normalize_spg(spg)
                if chn not in spgs:
                    spgs[chn] = [spg]
                else:
                    spgs[chn].append(spg)

            total_dur += dur

        print({chn: len(spgs[chn]) for chn in spgs}, file=sys.stderr)

        # Concatenate signals of the same channel
        spgs_cat_by_channel = []
        channels = []
        for chn, spectros in spgs.items():
            # spectros is a list of torch tensors
            # need to concatenate these along the time axis
            # TODO: add empty patches in between or something
            # spgs_cat_by_channel.append(torch.cat(spectros, dim=1))
            spgs_cat_by_channel.append(spectros[0])
            channels.append(chn)

        batch = torch.stack(spgs_cat_by_channel)
        batch.unsqueeze_(1)
        print(
            "[custom_collate_fn] Batch shape:",
            batch.shape,
            "(B, C, H, W)",
            file=sys.stderr,
        )

        return {
            "batch": batch,
            "chn_list": channels,
            "win_size": win_size,
            "sr": sr,
            "dur": total_dur,
        }

    def probe_max_patches(self, batch):
        return {
            "batch": torch.randn(1, 1, 48, 45_952),
            "chn_list": [None],
            "win_size": 1,
        }

    # == Data Loaders ===================================================================================================================

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.snake_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            collate_fn=self.snake_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    # == Helpers ========================================================================================================================
    def get_generic_channel_name(self, channel_name):
        channel_name = channel_name.lower()
        # Remove "eeg " prefix if present
        if channel_name.startswith("eeg "):
            channel_name = channel_name[4:]
        # Simplify names with a dash and check if it ends with "-"
        if "-" in channel_name:
            if channel_name.endswith("-"):
                return "None"
            return channel_name.split("-")[0]
        return channel_name
