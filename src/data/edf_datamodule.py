import shutil
import sys
import torch
import os
import json

import wandb
import src.data.edf_loader_torch as custom_data
from collections.abc import Sequence
import psutil

import numpy as np
from src.data.transforms import crop_spectrogram, load_channel_data, fft_256
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import src.utils.serialize as serialize
import tracemalloc
import src.data.saving as save
import tempfile
import os
from socket import gethostname
import time
from pympler import asizeof

import glob

from src.data.transforms import (
    crop_spectrogram,
    load_path_data,
    load_channel_data,
    fft_256,
    custom_fft,
    standardize,
)


class SimpleDataset(Dataset):

    def __init__(self, paths, sampling_rates, target_size):
        self.paths = paths
        self.sampling_rates = sampling_rates
        self.ffts = {}  # we have different STFTs for different sampling rates
        self.crop = crop_spectrogram(target_size=target_size)
        self.std = standardize()

    def __getitem__(self, idx):
        signal_path = self.paths[idx]
        signal_sr = self.sampling_rates[idx]
        signal_chunk = np.load(signal_path)
        signal_chunk = torch.from_numpy(signal_chunk)
        # print("signal_chunk.shape", signal_chunk.shape, file=sys.stderr)
        # print("__getitem__.shape", spg.shape)

        if signal_sr not in self.ffts:
            self.ffts[signal_sr] = custom_fft(
                window_seconds=1,
                window_shift=0.0625,
                sr=signal_sr,
                cuda=False,
            )

        # == apply transforms to raw signal (on CPU) ==
        # Applies STFT, returns spectrogram in DB (Decibel) scale
        spg = self.ffts[signal_sr](signal_chunk)  # Compute the spectrogram using FFT.
        # print("spg.shape", spg.shape, file=sys.stderr)
        # Crop spectrogram to target_size
        spg = self.crop(spg)
        # print("cropped spg.shape", spg.shape, file=sys.stderr)
        # Normalize cropped spectrogram (for model input)
        spg = self.std(spg)
        # print("std spg.shape", spg.shape, file=sys.stderr)
        spg.unsqueeze_(0)

        return spg

    def __len__(self):
        return len(self.paths)


# https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule
class EDFDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir=[
            "/home/maxihuber/eeg-foundation/src/data/000_json"
        ],  # base path of json file containing
        batch_size: int = 64,  # size of the batches of data to be used during training
        num_workers: int = 1,  # nr of subprocesses to use for data loading
        pin_memory: bool = False,  # pre-load the data into CUDA pinned memory before transferring it to the GPU (if True)
        window_size=4.0,  # size of the window (in seconds) to segment the continuous EEG data into
        window_shift=0.25,  # amount by which the window is shifted for each segment (determines the overlap between consecutive windows of data)
        chunk_duration=4,  # duration of each chunk of EEG data (in seconds)
        min_duration=1000,  # minimum duration of EEG recordings (in seconds)
        max_duration=1200,  # maximum duration of EEG recordings (in seconds)
        select_sr=[
            250,
            256,
        ],  # list of sampling rates to select from the data (in Hz = 1/s)
        select_ref=[
            "AR"
        ],  # list of reference types (e.g., 'AR' for average reference) to select from the data
        discard_datasets=[],  # list of datasets to discard
        interpolate_250to256=True,  # enables interpolation of data from 250 Hz to 256 Hz
        train_val_split=[
            0.95,
            0.05,
        ],  # proportion of data to be used for training and validation
        STORDIR="/dev/shm/mae",
        path_prefix="/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf",
        target_size=[
            64,
            2048,
        ],  # target size of the data after processing, in [channels, samples] (?)
        stor_mode="NONE",  # ?
        prefetch_factor: int = None,
        runs_dir: str = None,
    ) -> None:

        # Initialize attributes

        print("Initialize EDF DataModule\n=====================")
        # print("data dir: ", data_dir)

        super().__init__()

        # self.run_dir = f"{runs_dir}/{os.environ['SLURM_ARRAY_JOB_ID']}"
        self.run_dir = f"{runs_dir}/{955197}"
        self.TMPDIR = f"{self.run_dir}/tmp"
        os.makedirs(self.TMPDIR, exist_ok=True)

        self.STORDIR = STORDIR  # holds the EEG signals, which were pre-loaded into node-local memory (or disk in case of STORDIR="/scratch/...")
        self.stor_mode = stor_mode
        self.data_dir = data_dir
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters(logger=False)
        self.train_val_split = train_val_split
        self.hostname = gethostname()

        # Specifics on how to load the spectrograms
        self.window_size = window_size
        self.window_shift = window_shift
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.chunk_duration = chunk_duration
        self.select_sr = select_sr
        self.select_ref = select_ref
        self.discard_datasets = discard_datasets
        self.interpolate_250to256 = interpolate_250to256
        self.path_prefix = path_prefix  # can this be moved into the config?
        self.index = []
        self.file_index = []
        self.target_size = target_size
        self.prefetch_factor = prefetch_factor

        # If set to True will call prepare_data() on LOCAL_RANK=0 for every node. If set to False will only call from NODE_RANK=0, LOCAL_RANK=0.
        # We want it to be true because we want to load the spectrograms into memory on every node.
        self.prepare_data_per_node = True

    def filter_data_dir(self, min_duration, max_duration, select_sr, select_ref):
        #     """
        #     Filter the json `data_dir` index based on the specified conditions.
        #     """
        #     filtered_index = []
        #     print("Filtering data dir", file=sys.stderr)

        #     # iterate over each data_dir (paths to json indices of the data),
        #     # i.e. we have one index for .edf and one for .pkl data
        #     for data_dir in self.data_dir:

        #         with open(data_dir, "r") as file:
        #             data_index = json.load(file)

        #         for index_element in data_index:
        #             print("=" * 100, file=sys.stderr)

        #             if (
        #                 index_element["sr"] in select_sr
        #                 # and edf_file["ref"] in select_ref
        #                 and index_element["duration"] >= min_duration
        #                 and index_element["duration"] <= max_duration
        #             ):
        #                 # the edf index comes with relative, the csv index with absolute paths
        #                 if index_element["path"].endswith(".edf"):
        #                     index_element["path"] = self.path_prefix + index_element["path"]
        #                 filtered_index.append(index_element)

        #     print(len(filtered_index), "files found in total", file=sys.stderr)
        #     return filtered_index
        pass

    def build_channel_index(
        self,
        min_duration=0,
        max_duration=1200,
        select_sr=[256],
        select_ref=["AR"],
    ):
        # """
        # Read the json dictionary (`data_dir`) and generate an index.

        # It can be filtered by
        # - minimum duration (in seconds)
        # - maximum duration (in seconds)
        # - sampling frequency (in Hz)
        # - EEG reference

        # Could be put into utils, also used in EDFDataModule...
        # """
        # d_index = []

        # with open(self.data_dir, "r") as file:
        #     data = json.load(file)

        # # print(data)
        # # print("select_sr", select_sr)
        # # print("select_ref", select_ref)
        # # print("min_duration", min_duration)
        # # print("max_duration", max_duration)

        # # iterate over the JSON directory
        # for edf_file in data:
        #     channel_names = edf_file["channels"]
        #     path = edf_file["path"]
        #     sampling_rate = edf_file["sr"]
        #     ref = edf_file["ref"]
        #     duration = edf_file["duration"]
        #     # check whether this JSON entry satisfies the specified conditions
        #     # if it does, add it to the index structure
        #     # print("sampling_rate in select_sr", sampling_rate in select_sr)
        #     # print("ref in select_ref", ref in select_ref)
        #     # print("duration >= min_duration", duration >= min_duration)
        #     # print("duration <= max_duration", duration <= max_duration)

        #     if (
        #         sampling_rate in select_sr
        #         and ref in select_ref
        #         and duration >= min_duration
        #         and duration <= max_duration
        #     ):
        #         for chn in channel_names:
        #             d_index.append(
        #                 {
        #                     "path": self.path_prefix + path,
        #                     "chn": chn,
        #                     "ref": ref,
        #                     "sr": sampling_rate,
        #                     "duration": duration,
        #                 }
        #             )

        # print("len of created data_dir index: ", len(d_index))

        # return d_index
        pass

    # # See https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
    def prepare_data(self) -> None:
        #     """
        #     This method is called by the node with LOCAL_RANK=0 once, before the training process starts.
        #     It is used to perform any data downloading or preprocessing tasks that are independent of the
        #     cross-validation folds or the distributed setup.

        #     The current implementation loads all data into node local memory.

        #     With LightningDataModule.prepare_data_per_node=True, this method is executed on every node with LOCAL_RANK=0.

        #     ATTENTION: When using multiple GPUs to train, this needs the newest lightning dev-build (2.3.0.dev). [quote Pascal]
        #     There is a barrier for all processes here, and in stable release lightning, it will timeout after 30 min, which is usually not enough to store all the spectrograms.
        #     """
        #     print(f"Preparing data on {self.hostname}", file=sys.stderr)
        #     print("RAM memory % used:", psutil.virtual_memory()[2], file=sys.stderr)
        #     print(
        #         "RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000, file=sys.stderr
        #     )
        #     print(
        #         "RAM Total (GB):", psutil.virtual_memory()[0] / 1000000000, file=sys.stderr
        #     )

        #     start_time = time.time()

        #     # Filter the data_dir json file based on the specified conditions.
        #     raw_paths = self.filter_data_dir(
        #         min_duration=self.min_duration,
        #         max_duration=self.max_duration,
        #         select_sr=self.select_sr,
        #         select_ref=self.select_ref,
        #     )

        #     local_loader = save.LocalLoader(
        #         num_threads=7,
        #         base_stor_dir=self.STORDIR,
        #     )

        #     # chunk_paths is a list containing paths to json files, which in turn are of the form
        #     # {0: path_to/signal0.npy, 1: path_to/signal1.npy, ...}
        #     chunk_paths = local_loader.run(raw_paths, self.chunk_duration)
        #     print(chunk_paths, file=sys.stderr)

        #     # we store this list into a json file in the TMPDIR,
        #     # so that each process can access it in the setup method afterwards
        #     with open(
        #         os.path.join(self.TMPDIR, f"index_path_{gethostname()}.json"), "w"
        #     ) as file:
        #         json.dump(chunk_paths, file)

        #     # # save the spectrograms. note that these attributes are not shared by the processes on other GPUs, that's why an index is saved in the /tmp directory.
        #     # # Also returning the temporary directory objects, so that they don't close themselfes. Can close them in the teardown section (for example at beginning of the testing loop)
        #     # self.spg_paths, self.parent, self.subdir = save.load_and_save_spgs(
        #     #     raw_paths=raw_paths,
        #     #     STORDIR=self.STORDIR,
        #     #     TMPDIR=self.TMPDIR,
        #     #     window_size=self.window_size,
        #     #     window_shift=self.window_shift,
        #     #     target_size=self.target_size,
        #     #     chunk_duration=self.chunk_duration,
        #     # )
        #     # # TODO: I think we don't need to store the self.spg_paths attribute because we never use it again
        #     # print(f"Stored {len(self.spg_paths)} spectrograms!")

        #     end_time = time.time()

        #     # Logging
        #     self.data_preparation_time = end_time - start_time
        #     with open(
        #         f"{self.run_dir}/metrics/data_preparation_time_{os.environ['SLURM_PROCID']}_{self.hostname}.txt",
        #         "w",
        #     ) as file:
        #         file.write(str(self.data_preparation_time))

        #     wandb.log({"data_preparation_time": self.data_preparation_time}, step=0)

        #     print(f"Prepared data on {self.hostname}")
        #     print("RAM memory % used:", psutil.virtual_memory()[2])
        #     print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
        #     print("RAM Total (GB):", psutil.virtual_memory()[0] / 1000000000)
        pass

    # See https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
    def setup(self, stage=None) -> None:
        """
        This method is called after prepare_data, but before train_dataloader, val_dataloader, and test_dataloader.
        It is used to perform any dataset-specific setup that depends on the cross-validation folds or the distributed setup.

        Every node will call this method and initialize the datasets.
        """

        print("Before dataset creation", file=sys.stderr)

        # == Fetch paths to data on local memory/disk ==

        # self.pointer_file_paths = sorted(
        #     glob.glob(
        #         os.path.join(
        #             f"/itet-stor/maxihuber/net_scratch/runs/{os.environ['SLURM_ARRAY_JOB_ID']}/tmp",
        #             f"index_path_{gethostname()}_*.txt",
        #         )
        #     )
        # )
        self.pointer_file_paths = sorted(
            glob.glob(
                os.path.join(
                    f"/itet-stor/maxihuber/net_scratch/runs/{955197}/tmp",
                    f"index_path_{gethostname()}_*.txt",
                )
            )
        )
        print(
            "Collecting data from these files:",
            self.pointer_file_paths,
            file=sys.stderr,
        )

        paths = {}
        sampling_rates = {}
        num_datapoints = 0

        for pointer_file_path in self.pointer_file_paths:

            with open(pointer_file_path, "r") as pointer_file:
                path_to_data_index = pointer_file.read()

                with open(path_to_data_index, "r") as index_file:
                    chunks_index = json.load(index_file)

                    for _, chunk_dict in chunks_index.items():

                        # paths.append(chunk_dict["path"])
                        # sampling_rates.append(chunk_dict["sr"])
                        paths[num_datapoints] = chunk_dict["path"]
                        sampling_rates[num_datapoints] = chunk_dict["sr"]
                        num_datapoints += 1

                    # print(
                    #     "num_datapoints so far:",
                    #     num_datapoints,
                    #     file=sys.stderr,
                    # )
                    # paths_size = asizeof.asizeof(paths)
                    # print(
                    #     f"Total size of the paths including elements: {paths_size} bytes",
                    #     file=sys.stderr,
                    # )
                    # sampling_rates_size = asizeof.asizeof(sampling_rates)
                    # print(
                    #     f"Total size of the sampling rates including elements: {sampling_rates_size} bytes",
                    #     file=sys.stderr,
                    # )
                    # print(
                    #     f"RAM memory % used on {gethostname()}:",
                    #     psutil.virtual_memory()[2],
                    #     file=sys.stderr,
                    # )
                    # print(
                    #     "RAM Used (GB):",
                    #     psutil.virtual_memory()[3] / 1_000_000_000,
                    #     file=sys.stderr,
                    # )

        # == Initialize Datasets ==
        entire_dataset = SimpleDataset(
            paths=paths,
            sampling_rates=sampling_rates,
            target_size=self.target_size,
        )
        print("After dataset creation", file=sys.stderr)

        # for i in range(len(entire_dataset)):
        #     spg = entire_dataset[i]
        #     if spg.size() != torch.Size([1, 64, 64]):
        #         print(f"Issue at index {i}: {spg.size()}")

        print("Nr. of datapoints:", len(entire_dataset), file=sys.stderr)

        train_size = int(self.train_val_split[0] * len(entire_dataset))
        val_size = len(entire_dataset) - train_size

        print("TRAINSIZE IS:::::::::::::::::::::", train_size)
        print("VALSIZE IS:::::::::::::::::::::::", val_size)

        # Instantiate datasets for training and validation
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            entire_dataset, [train_size, val_size]
        )

        print("After LightningDataModule.setup", file=sys.stderr)
        print("RAM memory % used:", psutil.virtual_memory()[2], file=sys.stderr)
        print(
            "RAM Used (GB):",
            psutil.virtual_memory()[3] / 1_000_000_000,
            file=sys.stderr,
        )

        # assert False, "break after setup"

    def train_dataloader(self):

        # Handle variable lengths (x-axis) of spectrograms
        def custom_collate_fn(batch):
            return torch.stack(batch)

        return DataLoader(
            self.train_dataset,
            shuffle=True,  # apparently, it is not advised to set shuffle=True because Lightning will do it for us in distributed setting, TODO: look into it later
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        # QUESTION: why are we using val_dataset for testing

        pass

    def predict_dataloader(self):
        pass

    # def teardown(self, stage):
    #     """Clean up subdirectories and files in self.STORDIR after training/validation/testing.

    #     :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
    #     """
    #     print(f"Teardown called for stage: {stage}", sys.stderr)
    #     if os.path.exists(self.STORDIR):
    #         print(
    #             f"Removing all files and subdirectories in: {self.STORDIR}", sys.stderr
    #         )
    #         shutil.rmtree(self.STORDIR)

    # def state_dict(self):
    #     """Called when saving a checkpoint. Implement to generate and save the datamodule state.

    #     :return: A dictionary containing the datamodule state that you want to save.
    #     """
    #     return {}

    # def load_state_dict(self, state_dict) -> None:
    #     """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
    #     `state_dict()`.

    #     :param state_dict: The datamodule state returned by `self.state_dict()`.
    #     """
    #     pass
