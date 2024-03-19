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
import matplotlib.pyplot as plt
import torchaudio
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
import multiprocessing
from torchvision import transforms


class EDFDataset(Dataset):

    def __init__(
        self,
        data_dir,
        window_size=2.0,
        window_shift=0.125,
        debug=False,
        target_size=(64, 2048),
        min_duration=316,
        select_sr=[250, 256],
        select_ref=["AR"],
        random_sample=False,
        fixed_sample=True,
        use_cache=False,
        interpolate_250to256=True,
    ):

        print("=====================")
        print(data_dir)
        # load dataset
        with open(data_dir, "r") as file:
            data = json.load(file)

        self.data = data
        self.index = []

        # parameters for fft
        self.window_shift = window_shift
        self.window_size = window_size
        self.target_size = target_size

        self.fft_256 = torchaudio.transforms.Spectrogram(
            n_fft=int(window_size * 256),
            win_length=int(window_size * 256),
            hop_length=int(window_shift * 256),
            normalized=True,
        )

        self.img_height = self.target_size[0]
        self.target_width = self.target_size[1]

        self.crop = CropSpectrogram(target_size=target_size)
        self.transform = transforms.Compose([self.fft_256, self.crop])
        self.interpolate_250to256 = interpolate_250to256
        self.select_sr = select_sr

        self.path_prefix = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"

        self.load_times = []
        self.transform_times = []

        self.shmpath = "/dev/shm/spec/"

    def load_channel_data(self, path, chn, pre_crop=True):

        # takes as input an index and returns a tensor containing the channel_data associated with that index
        # open_start = time.time()
        full_path = self.path_prefix + path
        # include = chn -> only loads the data for the channel we want the sample from.
        edf_data = mne.io.read_raw_edf(full_path, include=chn, preload=False)

        # open_end = time.time()
        # open_time = open_end - open_start
        data = edf_data.get_data()
        # print("opening took" + str(open_time))

        sr = int(edf_data.info["sfreq"])

        channel_data = edf_data[chn][0]
        # convert to myvolt
        channel_data = channel_data * 1000000

        channel_data = torch.from_numpy(channel_data)

        channel_data = channel_data.squeeze(0)
        # cropping, removing first minute
        if pre_crop:

            target_length = self.window_shift * self.target_width * sr

            channel_data = channel_data[sr : int(sr + target_length)]

        # interpolate to 256sr
        if self.interpolate_250to256 and sr == 250:

            new_length = (len(channel_data) // 250) * 256
            channel_data = (
                interpolate(
                    channel_data.unsqueeze(0).unsqueeze(0), new_length, mode="nearest"
                )
                .squeeze(0)
                .squeeze(0)
            )

        return channel_data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        path, lbl = self.data[idx]

        raw = mne.io.read_raw_edf(self.path_prefix + path, preload=False)
        channels = raw.ch_names
        spgs = []

        for chn in channels:

            if "EEG" not in chn:

                continue

            channel_data = self.load_channel_data(path, chn)

            # apply transforms

            spectrogram = self.transform(channel_data)

            # convert to DB

            spectrogram = 20 * torch.log10(spectrogram)

            # normalize on per sample basis
            spectrogram = (spectrogram - torch.mean(spectrogram)) / (
                torch.std(spectrogram) * 2
            )
            spgs.append(spectrogram)

        while len(spgs) < 1:
            spgs.append(torch.zeros(1, 64, 2048))

        return (spgs, lbl)


class CropSpectrogram(object):

    def __init__(self, target_size):

        self.target_height = target_size[0]
        self.target_width = target_size[1]

    def __call__(self, spectrogram):

        # crop vertically
        spectrogram = spectrogram[4:68, :]

        # crop&pad horizontally (padding shouldn't be necessary because of our sample selection but just in case)

        width_pad = self.target_width - spectrogram.shape[1]

        if width_pad > 0:
            pad = torch.nn.ZeroPad2d((0, width_pad, 0, 0))
            spectrogram = pad(spectrogram)
        elif width_pad < 0:
            spectrogram = spectrogram[:, 0 : self.target_width]
        # add 1 'image channel'
        return spectrogram.unsqueeze(0)


if __name__ == "__main__":

    dset = EDFFinetuneDataset(data_dir="/home/schepasc/tuab_label_paths")
    print(dset[0][0][0].shape)
