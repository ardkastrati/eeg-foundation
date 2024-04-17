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
import wandb

# Dataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset


class EDFDataset(Dataset):
    """
    Not used anymore I think!
    """

    def __init__(
        self,
        data_dir,  #
        window_size=4.0,
        window_shift=0.25,
        debug=False,
        target_size=(128, 1024),
        min_duration=316,
        select_sr=[250, 256],
        select_ref=["AR"],
        random_sample=False,
        fixed_sample=True,
        use_cache=False,
        interpolate_250to256=True,
    ):

        print("Initialize EDF Dataset\n=====================")
        # print("data dir: ", data_dir)

        # load dataset
        with open(data_dir, "r") as file:
            data = json.load(file)
        self.data = data

        # Build index for pathnames to the EEG data
        self.index = (
            []
        )  # list of tuples, each containing three elements (File Path, Channel Name, Reference)
        self.build_channel_index(
            min_duration=min_duration, select_sr=select_sr, select_ref=select_ref
        )
        # print("print head(self.index)")
        # print(self.index[:5])

        # parameters for fft
        self.window_shift = window_shift
        self.window_size = window_size
        self.target_size = target_size

        # print("store fourier transform into self.fft_256")
        # the object stored into self.fft_256 can be used to transform
        # audio signals into their spectrogram representations, which
        # are visual representations of the spectrum of frequencies in the
        # signal as they vary with time
        # https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html
        self.fft_256 = torchaudio.transforms.Spectrogram(
            n_fft=int(window_size * 256),
            win_length=int(window_size * 256),
            hop_length=int(window_shift * 256),
            normalized=True,
        )

        # print("crop spectogram")
        self.crop = CropSpectrogram(target_size=target_size)

        # print("compose fft and crop")
        # Transforms are common image transformations. They can be chained together using Compose.
        # https://pytorch.org/vision/0.9/transforms.html?highlight=compose#compositions-of-transforms
        self.transform = transforms.Compose([self.fft_256, self.crop])

        # Initialize other attributes for EDF dataset
        self.img_height = self.target_size[0]
        self.target_width = self.target_size[1]
        self.interpolate_250to256 = interpolate_250to256
        self.select_sr = select_sr
        self.path_prefix = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"  # prefix to where the pathnames in self.data_dir are located
        self.load_times = []
        self.transform_times = []
        self.shmpath = "/dev/shm/spec/"

        # Set the log level to 'WARNING' to see only warnings and errors when reading EDF files
        mne.set_log_level("WARNING")

    def get_times(self):

        return np.mean(self.load_times), np.mean(self.transform_times)

    def build_channel_index(self, min_duration=0.0, select_sr=[256], select_ref=["AR"]):
        """
        Read the json dictionary and generate an index.

        It can be filtered by
        - minimum duration (in seconds)
        - sampling frequency
        - reference (?)

        Could be put into utils, also used in EDFDataModule...
        """

        # iterate over the JSON directory
        for edf_file in self.data:
            channel_names = edf_file["channels"]
            path = edf_file["path"]
            sampling_rate = edf_file["sr"]
            ref = edf_file["ref"]
            duration = edf_file["duration"]
            # check whether this JSON entry satisfies the specified conditions
            # if it does, add it to the index structure
            if (
                sampling_rate in select_sr
                and ref in select_ref
                and duration >= min_duration
                and duration <= 1260
            ):
                for chn in channel_names:
                    self.index.append((path, chn, ref))

        print("len of created data_dir index: ", len(self.index))

    def load_channel_data(self, path, chn, pre_crop=True):
        """
        Takes as input an index and returns a tensor containing the channel_data associated with that index
        """

        full_path = self.path_prefix + path

        # Read EEG data of channel chn from path
        # Preload=False: don't load data into memory immediately (instead, data will be read as needed)
        # print("Fetching EDF Data in EDFDataset")
        time_start = time.time()
        edf_data = mne.io.read_raw_edf(full_path, include=chn, preload=False)
        time_end = time.time()

        read_edf_time = time_end - time_start
        # self.log({"edf_open_time": read_edf_time})

        # data = edf_data.get_data()
        # print("opening took" + str(open_time))

        sr = int(edf_data.info["sfreq"])  # fetch sampling rate of channel data

        # Retrieve data for the specified channel chn from the EDF file
        #   edf_data[chn]: index into the raw data object, only retrieving data for the chn channel
        #   edf_data[chn][0]: data for the channel (as a NumPy array)
        #   edf_data[chn][1]: timestamps array for the channel (as a NumPy array)
        time_start = time.time()
        channel_data = edf_data[chn][0]
        time_end = time.time()

        channel_load_time = time_end - time_start
        # self.log({"channel_load_time": channel_load_time})

        # convert the signal from volts to microvolts
        # this is apparently a common practice in EEG processing (necessary because EEG signals are typically very small in amplitude)
        channel_data = channel_data * 1000000
        channel_data = torch.from_numpy(
            channel_data
        )  # turn data from NumPy to PyTorch tensor
        channel_data = channel_data.squeeze(0)  # dunno

        # Option to remove first minute from the beginning of each EEG channel's data (noisy)
        if pre_crop:
            target_length = self.window_shift * self.target_width * sr
            channel_data = channel_data[sr : int(sr + target_length)]

        # Interpolate to 256 (sr)
        # Adjust the sampling rate of EEG channel data from 250 Hz to 256 Hz
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
        """
        Expected to return the size of the dataset
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        Supports fetching a data sample for a given key (`idx`).
        Returns its spectogram as a torch tensor

        DataLoader by default constructs an index sampler that yields integral indices.
        This is also true for this EDFDataset implementation.
        """

        path, chn, ref = self.index[
            idx
        ]  # fetch index entry (File Path, Channel Name, Reference)
        channel_data = self.load_channel_data(path, chn)  # reads data from .edf file

        # apply transforms
        spectrogram = self.transform(channel_data)

        # converting the spectrogram values to a logarithmic scale, specifically to decibels (dB)
        spectrogram = 20 * torch.log10(spectrogram)

        # normalize on per sample basis
        spectrogram = (spectrogram - torch.mean(spectrogram)) / (
            torch.std(spectrogram) * 2
        )

        # transform_time = transform_end_time - transform_start_time

        return spectrogram


class CropSpectrogram(object):
    """
    A transformation class to crop and pad spectrograms to a specified target size.

    Parameters:
    - target_size (tuple of int): The target height and width (H, W) for the cropped spectrogram.
    """

    def __init__(self, target_size):

        self.target_height = target_size[0]
        self.target_width = target_size[1]

    def __call__(self, spectrogram):

        # crop vertically
        spectrogram = spectrogram[4:68, :]

        # either crop or pad horizontally to ensure the resulting width matches self.target_width
        #  (padding shouldn't be necessary because of our sample selection but just in case)
        width_pad = self.target_width - spectrogram.shape[1]

        if width_pad > 0:  # horizontal padding
            pad = torch.nn.ZeroPad2d((0, width_pad, 0, 0))
            spectrogram = pad(spectrogram)
        elif width_pad < 0:  # horizontal crop
            spectrogram = spectrogram[:, 0 : self.target_width]

        # additional dimension is added using unsqueeze(0) to simulate a single-channel image
        return spectrogram.unsqueeze(0)


# Don't know what this does
class EDFCacheDataset(Dataset):

    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


# Don't know what this does
class EDFDiscDataset(Dataset):

    def __init__(self, file_index):
        self.file_index = file_index

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        path = self.file_index[idx]

        spectro = np.load(path)
        spectro = torch.from_numpy(spectro)

        return spectro.unsqueeze(0)


def plotSpectro(spectrogram):
    # Convert the tensor to numpy for plotting
    spectrogram_np = spectrogram.numpy()

    # Assuming the spectrogram is a single-channel (time-frequency representation),
    # you can select the first channel if it has an extra dimension
    if spectrogram_np.ndim > 2:
        spectrogram_np = spectrogram_np[0]

    plt.figure(figsize=(10, 4))
    # Plotting the spectrogram. The aspect ratio 'auto' adjusts the axis to fit the data
    plt.imshow(spectrogram_np, aspect="auto", origin="lower")
    plt.colorbar(label="Intensity (dB)")
    plt.ylabel("Frequency Bin")
    plt.xlabel("Time Frame")
    plt.title("Spectrogram")
    print("saving spectrogram")
    plt.savefig("src/data/spectro.png", bbox_inches="tight")


if __name__ == "__main__":

    print(f"Running program: {__file__}")

    # ...
    multiprocessing.set_start_method("spawn")

    # Initialize dataset holding EEG data (in the .edf format)
    data_dir = "/home/maxihuber/eeg-foundation/src/data/000_json"
    dset = EDFDataset(data_dir, select_sr=[250, 256], min_duration=316)

    # Fetch spectogram data contained in file that is referenced by the first entry in data_dir (idx = 0)
    spectro = dset[0]

    print("printing spectrogram")
    print(spectro)
    print("====================")

    # Plot spectogram
    print("plotting spectrogram")
    plotSpectro(spectro)
    print("====================")

    # load, trs = dset.get_times()
    # print("load time average:" + str(load))
    # print("transformtime average: " + str(trs))
