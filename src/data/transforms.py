import sys

import mne
import json
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


class crop_spectrogram:

    def __init__(self, target_size=(128, 1024)):

        self.target_height = target_size[0]
        self.target_width = target_size[1]

    def __call__(self, spectrogram):

        # crop vertically

        spectrogram = spectrogram[4 : 4 + self.target_height, :]

        # crop&pad horizontally (padding shouldn't be necessary because of our sample selection but just in case)

        width_pad = self.target_width - spectrogram.shape[1]

        if width_pad > 0:

            pad = torch.nn.ZeroPad2d((0, width_pad, 0, 0))
            spectrogram = pad(spectrogram)
        elif width_pad < 0:
            spectrogram = spectrogram[:, 0 : self.target_width]

        # add 1 'image channel'

        return spectrogram


class fft_256:

    # compute spetrogram from channel data
    def __init__(self, window_size=4.0, window_shift=0.25, cuda=False):

        super().__init__()
        self.fft256 = torchaudio.transforms.Spectrogram(
            n_fft=int(4.0 * 256),
            win_length=int(window_size * 256),
            hop_length=int(window_shift * 256),
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


class load_channel_data:

    def __init__(self, precrop=True, crop_idx=[60, 316]):

        self.crop_idx = crop_idx
        self.precrop = precrop

    def __call__(self, data):

        mne.set_log_level("WARNING")

        # takes as input a tuple of path and string and returns the channaldata cropped to the timeframe [1min:5min15seconds]
        path = data["path"]
        chn = data["chn"]

        # include = chn -> only loads the data for the channel we want the sample from.
        edf_data = mne.io.read_raw_edf(path, include=chn, preload=True)

        data = edf_data.get_data()

        sr = int(edf_data.info["sfreq"])

        channel_data = edf_data[chn][0]

        # convert to u_volt
        channel_data = channel_data * 1000000

        channel_data = torch.from_numpy(channel_data)

        channel_data = channel_data.squeeze(0)
        # cropping, removing first minute

        if self.precrop:
            channel_data = channel_data[sr * self.crop_idx[0] : sr * self.crop_idx[1]]

        # interpolate to 256sr
        if sr == 250:

            new_length = (len(channel_data) // 250) * 256
            channel_data = (
                interpolate(
                    channel_data.unsqueeze(0).unsqueeze(0), new_length, mode="nearest"
                )
                .squeeze(0)
                .squeeze(0)
            )

        return channel_data
