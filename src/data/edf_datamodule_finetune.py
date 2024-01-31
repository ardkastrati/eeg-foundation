import torch
import os
import json
import src.data.edf_loader_finetune as custom_data
from collections.abc import Sequence
import psutil
from monai.data import CacheDataset
import time

import numpy as np
from src.data.monai_transforms import crop_spectrogram, load_channel_data, fft_256
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from monai.transforms import Compose
import src.utils.serialize as serialize
import tracemalloc

def collate_samples(batch):

       
        out_samples = []
        out_labels = []
        for sample in batch:
            out_samples.append(sample[0])
            out_labels.append(sample[1])

        
        out_labels = torch.tensor(out_labels)

        return out_samples, out_labels

    
class EDFDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir = "/home/schepasc/eeg-foundation/src/data/000_json",
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,

        window_size = 2.0,
        window_shift = 0.125,
        min_duration = 316,
        max_duration = 1200,
        select_sr = [250, 256],
        select_ref = ['AR'],
        interpolate_250to256 = True,
        train_val_split = [0.95, 0.05],
        TMPDIR = "",
        target_size = [64, 2048],
        stor_mode = "NONE"
    ) -> None: 
        
        super().__init__()
        self.TMPDIR = TMPDIR
        self.stor_mode = stor_mode
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters(logger=False)
        self.train_val_split = train_val_split
        # specifics on how to load the spectrograms
        self.window_size = window_size
        self.window_shift = window_shift
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.select_sr = select_sr
        self.select_ref = select_ref
        self.interpolate_250to256 = interpolate_250to256
        self.path_prefix = "/itet-stor/schepasc/deepeye_storage/foundation/tueg/edf" 
        self.index = []
        self.file_index = []
        self.target_size = target_size


       

        




    
    
    def prepare_data(self) -> None:
        
        
        
        pass

        

    def setup(self, stage= None) -> None: 


        
        
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
        
        
        

        train_size = int(self.train_val_split[0] * len(entire_dataset))
        val_size = len(entire_dataset) - train_size

        print("TRAINSIZE IS:::::::::::::::::::::", train_size)
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(entire_dataset, [train_size, val_size])
    
        

    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            shuffle = True,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            collate_fn = collate_samples
        )
      
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            collate_fn = collate_samples,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        ) 


    def teardown(self, stage) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
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



