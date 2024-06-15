import pickle
import sys

from matplotlib import pyplot as plt
import mne
import pandas as pd
import numpy as np

import mne
import json
import pandas as pd
import scipy
import scipy.signal
import random
import numpy as np
import time
import os
import csv
import torch
from torch.utils.data import Dataset

import torchaudio
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
import multiprocessing
from torchvision import transforms
import torch.nn.functional as F


from pyprep.prep_pipeline import PrepPipeline

import logging


class crop_spectrogram:
    def __init__(self, target_size=(64, 64)):
        self.target_height = target_size[0]
        self.target_width = target_size[1]

    def __call__(self, spectrogram):
        # Crop or pad the height
        height_pad = self.target_height - (
            spectrogram.shape[0] - 4
        )  # Adjust for the vertical crop starting at index 4

        if height_pad > 0:
            # If padding is needed (spectrogram is shorter than target height)
            pad = torch.nn.ZeroPad2d(
                (0, 0, 4, height_pad)
            )  # Pad the top by 4 rows, bottom by height_pad rows
            spectrogram = pad(spectrogram)
            # Adjust crop to new padding
            spectrogram = spectrogram[: self.target_height, :]
        else:
            # If cropping is needed (or just slicing without padding)
            spectrogram = spectrogram[4 : 4 + self.target_height, :]

        # Crop or pad the width
        width_pad = self.target_width - spectrogram.shape[1]

        if width_pad > 0:
            # Pad the width if necessary
            pad = torch.nn.ZeroPad2d(
                (0, width_pad, 0, 0)
            )  # Left padding is 0, right padding is width_pad
            spectrogram = pad(spectrogram)
        elif width_pad < 0:
            # Crop the width if necessary
            spectrogram = spectrogram[:, : self.target_width]

        return spectrogram


class standardize:
    def __call__(self, spectrogram):
        # normalize on per sample basis
        spectrogram = (spectrogram - torch.mean(spectrogram)) / (
            torch.std(spectrogram) * 2
        )
        return spectrogram


class fft_256:

    # compute spetrogram from channel data
    def __init__(self, window_size=4.0, window_shift=0.25, sr=256, cuda=False):

        super().__init__()
        self.fft256 = torchaudio.transforms.Spectrogram(
            n_fft=int(4.0 * 256),
            win_length=int(window_size * sr),
            hop_length=int(window_shift * sr),
            normalized=True,
        )
        if cuda:
            self.fft256 = self.fft256.to("cuda")

    # perform SFFT on data
    def __call__(self, data):

        spectrogram = self.fft256(data)

        # convert to decibel
        spectrogram = 20 * torch.log10(spectrogram)

        # normalize on per sample basis
        spectrogram = (spectrogram - torch.mean(spectrogram)) / (
            torch.std(spectrogram) * 2
        )

        return spectrogram


# class custom_fft:
#     """
#     FFT transform that takes in a window size and shift
#     and computes the spectrogram using the torchaudio library.

#     The output is converted to decibel scale, and normalized to have zero mean and unit variance.
#     """

#     def __init__(self, window_seconds, window_shift, sr, cuda=False):
#         super().__init__()
#         win_length = int(sr * window_seconds)
#         hop_length = int(sr / 16)
#         # print("win_length", win_length)
#         # print("hop_length", hop_length)
#         self.fft = torchaudio.transforms.Spectrogram(
#             n_fft=win_length,
#             win_length=win_length,
#             hop_length=hop_length,
#             normalized=True,
#         )
#         if cuda:
#             self.fft = self.fft.to("cuda")

#     def __call__(self, data):
#         """
#         Apply short-time Fourier transform (STFT) to the input data.

#         Args:
#             data (torch.Tensor): The input data.

#         Returns:
#             torch.Tensor: The transformed data.
#         """
#         spg = self.fft(data)

#         # # Maxim: inserted this, but not sure if needed
#         # spectrogram = torch.abs(spectrogram)

#         # # convert to decibel, avoid log(0)
#         # spectrogram = 20 * torch.log10(spectrogram + 1e-10)

#         return spg


class custom_fft:
    """
    FFT transform that takes in a window size and shift
    and computes the spectrogram using the torchaudio library.

    The output is converted to decibel scale, and normalized to have zero mean and unit variance.
    """

    def __init__(self, window_seconds, window_shift, sr, cuda=False):
        super().__init__()
        win_length = int(sr * window_seconds)
        hop_length = int(sr * window_shift)
        self.fft = torchaudio.transforms.Spectrogram(
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            normalized=True,
        )
        if cuda:
            self.fft = self.fft.to("cuda")

    def __call__(self, data):
        spg = self.fft(data)
        spg = spg**2
        return spg


def crop_spg(spg, patch_size):
    # Crop spg to the nearest multiple of 16 in both dimensions
    # Integer division and multiplication to find the nearest multiple
    new_height = (spg.shape[0] // patch_size) * patch_size
    new_width = (spg.shape[1] // patch_size) * patch_size
    spg = spg[:new_height, :new_width]  # Crop both dimensions
    return spg


def crop_and_normalize_spg(spg, time_steps):

    # Calculate the nearest multiple of 16 for height and width
    new_height = (spg.shape[0] // 16) * 16
    spg = spg[:new_height, :]

    new_width = (time_steps // 16) * 16
    # Calculate the padding needed for width
    width_diff = new_width - spg.shape[1]
    if width_diff > 0:
        spg = normalize_spg(spg)
        # Pad to the nearest multiple of 16
        padding = (0, width_diff, 0, 0)  # (left, right, top, bottom)
        spg = F.pad(spg, padding, "constant", 0)
    else:
        # Crop the width to the nearest multiple of 16
        spg = spg[:, :new_width]
        spg = normalize_spg(spg)

    return spg


def normalize_spg(spg):
    # Get frequency bin-wise means, stds
    freq_means = spg[:, :].mean(dim=1, keepdim=True)
    freq_stds = spg[:, :].std(dim=1, keepdim=True)
    # Divide each frequency bin by its mean
    normalized_spg = spg[:, :] / (freq_means + 1e-8)
    # Transform to decibel-scale
    db_spg = 10 * torch.log10(normalized_spg + 1e-8)
    db_means = 10 * torch.log10(freq_means + 1e-8)
    db_stds = 10 * torch.log10(freq_stds + 1e-8)
    return db_spg, db_means, db_stds


def normalize_spg_OLD(spg):
    # Divide spectrogram by frequency bin-wise means
    freq_means = spg[:, :].mean(dim=1, keepdim=True)
    freq_stds = spg[:, :].std(dim=1, keepdim=True)
    # Divide each frequency bin by its mean
    normalized_spg = (spg[:, :] - freq_means) / freq_stds
    # Transform to decibel-scale
    db_spg = 10 * torch.log10(normalized_spg)
    db_means = 10 * torch.log10(freq_means)
    db_stds = 10 * torch.log10(freq_stds)
    return db_spg, db_means, db_stds


def plot_spg(spg, sr, dur):
    plt.pcolormesh(spg, shading="auto", cmap="RdBu")
    plt.ylabel("Frequency Bins")
    plt.xlabel("Steps")
    plt.title(f"Spectrogram: [sr={sr}, dur={dur}]")
    plt.colorbar(label="")
    plt.show()


def create_raw(
    data,
    ch_names1,
    sr,
    ch_names2=None,
):
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
    avg = raw.copy().add_reference_channels(ref_channels="AVG_REF")
    avg = avg.set_eeg_reference(ref_channels="average")
    return avg


class load_path_data:
    def __init__(self):
        logger = logging.getLogger("pyprep")
        logger.setLevel(logging.ERROR)
        mne.set_log_level("WARNING")

    def __call__(self, index_element):

        if index_element["path"].endswith(".edf"):
            # For EDF: all channels are good at the moment
            eeg_data = mne.io.read_raw_edf(
                index_element["path"],
                include=index_element["channels"],
                preload=True,
            )

        elif index_element["path"].endswith("pkl"):
            # Load DataFrame from pickle
            with open(index_element["path"], "rb") as file:
                df = pd.read_pickle(file)
                # Create a mne.Raw to be compatible with the coming processing steps
                eeg_data = create_raw(
                    data=df,
                    # TODO: only include the good_channels here
                    ch_names1=index_element["channels"],
                    sr=index_element["sr"],
                )

        else:
            assert False, "Invalid path"

        # Add average reference
        eeg_data = avg_channel(eeg_data)

        # Datastructure to access data for each channel
        channel_data_dict = {}

        # Note: includes also AVG_REF channel
        for channel in eeg_data.ch_names:
            idx = eeg_data.ch_names.index(channel)
            data, times = eeg_data[idx, :]
            # Flatten the data to 1D if required
            channel_data_dict[channel] = data.flatten()

        return channel_data_dict


def load_channel_data():
    pass
