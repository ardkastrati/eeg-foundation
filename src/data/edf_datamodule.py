import torch
import os
import json
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

class SimpleDataset(Dataset):
    
    def __init__(self, spg_paths):

        self.spg_paths = spg_paths
    
    def __getitem__(self, idx):

        spg = np.load(self.spg_paths[idx])
        spg = torch.from_numpy(spg)
        spg = spg.unsqueeze(0)
        return spg

    def __len__(self):
# https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule    return len(self.spg_paths)


class EDFDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir="/home/maxihuber/eeg-foundation/src/data/000_json",  # base path of json file containing
        batch_size: int = 64,  # size of the batches of data to be used during training
        num_workers: int = 1,  # nr of subprocesses to use for data loading
        pin_memory: bool = False,  # pre-load the data into CUDA pinned memory before transferring it to the GPU (if True)
        window_size=4.0,  # size of the window (in seconds) to segment the continuous EEG data into
        window_shift=0.25,  # amount by which the window is shifted for each segment (determines the overlap between consecutive windows of data)
        min_duration=1000,  # minimum duration of EEG recordings (in seconds)
        max_duration=1200,  # maximum duration of EEG recordings (in seconds)
        select_sr=[
            250,
            256,
        ],  # list of sampling rates to select from the data (in Hz = 1/s)
        select_ref=[
            "AR"
        ],  # list of reference types (e.g., 'AR' for average reference) to select from the data
        interpolate_250to256=True,  # enables interpolation of data from 250 Hz to 256 Hz
        train_val_split=[
            0.95,
            0.05,
        ],  # proportion of data to be used for training and validation
        TMPDIR="",  # temporary directory for storing intermediate data (e.g. for caching)
        STORDIR = "/dev/shm/mae",
        target_size=[
            64,
            2048,
        ],  # target size of the data after processing, in [channels, samples] (?)
        stor_mode="NONE",  # ?
    ) -> None:

        # Initialize attributes

        print("Initialize EDF DataModule\n=====================")
        # print("data dir: ", data_dir)

        super().__init__()
        self.TMPDIR = TMPDIR
        self.stor_mode = stor_mode
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters(logger=False)
        self.train_val_split = train_val_split

        # Specifics on how to load the spectrograms
        self.window_size = window_size
        self.window_shift = window_shift
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.select_sr = select_sr
        self.select_ref = select_ref
        self.interpolate_250to256 = interpolate_250to256
        self.path_prefix = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"  # can this be moved into the config?
        self.index = []
        self.file_index = []
        self.target_size = target_size

        self.STORDIR = STORDIR        # Build index for pathnames to the EEG data
        self.build_channel_index(
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            select_ref=self.select_ref,
            select_sr=self.select_sr,
        )

    def build_channel_index(
        self, min_duration=0, max_duration=1200, select_sr=[256], select_ref=["AR"]
    ):
        """
        Read the json dictionary and generate an index.

        It can be filtered by
        - minimum duration (in seconds)
        - maximum duration (in seconds)
        - sampling frequency (in Hz)
        - EEG reference

        Could be put into utils, also used in EDFDataModule...
        """
        d_index = []
        
        with open(self.data_dir, "r") as file:
            data = json.load(file)

        # iterate over the JSON directory
        for edf_file in data:
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
                and duration <= max_duration
            ):
                for chn in channel_names:
                    d_index.append({'path' : self.path_prefix+path, 'chn' : chn})

        print("len of created data_dir index: ", len(d_index))

    # This is where I will have the most impact
    # See https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
    def prepare_data(self) -> None:
        
        """
        ATTENTION: When using multiple GPU to train, needs the newest lightning dev-build. There is a barrier for all processes here, and in stable release lightning, 
        it will timeout after 30 min, which is usually not enough to store all the spectrograms.
        Here you can store data on the local machine. 
        """
        print("preparing data")

        #build an index, you can select which files you want to include with parameters.
        raw_paths = self.build_channel_index(max_duration=self.max_duration, min_duration=self.min_duration, select_sr=self.select_sr, select_ref=self.select_ref)


        #save the spectrograms. note that these attributes are not shared by the processes on other GPUs, that's why an index is saved in the /tmp directory.
        #Also returning the temporary directory objects, so that they don't close themselfes. Can close them in the teardown section (for example at beginning of the testing loop)
        self.spg_paths, self.parent, self.subdir = save.load_and_save_spgs(raw_paths, STORDIR = self.STORDIR,TMPDIR = self.TMPDIR)
        

    def setup(self, stage= None) -> None: 


        print("before dataset creation")
        print('RAM memory % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000) 
          
        #entire_dataset = CacheDataset(self.cache, num_workers=self.num_workers)
        """
        if self.stor_mode == "LOAD" :
            entire_dataset = custom_data.EDFDataset(
                self.data_dir,
                target_size=self.target_size,
                window_size=self.window_size,
                window_shift=self.window_shift,
                min_duration=self.min_duration,
                select_sr = self.select_sr,
                select_ref = self.select_ref,
                interpolate_250to256 = self.interpolate_250to256
                )
        
        """
        #access the index in the tmpdir, so each process has it.
        with open(os.path.join(self.TMPDIR, "index_path.txt"), 'r') as file:
            index_path = file.read()
        #load the index and change the keys back to integers, since they get converted to strings on saving.
        with open(index_path, 'r') as file:
            paths = json.load(file)
            paths = {int(key): value for key, value in paths.items()}
        
        
        entire_dataset = SimpleDataset(paths)

        train_size = int(self.train_val_split[0] * len(entire_dataset))
        val_size = len(entire_dataset) - train_size

        print("TRAINSIZE IS:::::::::::::::::::::", train_size)

        # Instantiate datasets for training and validation
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            entire_dataset, [train_size, val_size]
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        # QUESTION: why are we using val_dataset for testing

        
        pass

    def predict_dataloader(self):
        pass

    def teardown(self, stage) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

        #can optionally delete the temporary directory here after the training is done.
        
        pass

    def state_dict(self):
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
