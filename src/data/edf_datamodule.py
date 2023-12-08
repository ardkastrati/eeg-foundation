import torch
import os

import src.data.edf_loader as custom_data


from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

class EDFDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir = "/home/schepasc/eeg-foundation/src/data/debug_json",
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,

        window_size = 4.0,
        overlap = 0.25,
        min_duration = 1000,
        specific_sr = 256,
        random_sample = False,
        fixed_sample = True,
        train_val_split = [0.9, 0.1]
    ) -> None: 
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters(logger=False)
        self.train_val_split = train_val_split
        # specifics on how to load the spectrograms
        self.window_size = window_size
        self.overlap = overlap
        self.min_duration = min_duration
        self.specific_sr = specific_sr
        self.random_sample = random_sample
        self.fixed_sample = fixed_sample

        # data transformations
        self.transforms = transforms.ToTensor()

    
    def prepare_data(self) -> None:

        pass

    def setup(self, stage= None) -> None: 

        
        entire_dataset = custom_data.EDFDataset(
            self.data_dir,
            window_size=self.window_size,
            overlap=self.overlap,
            random_sample=self.random_sample,
            fixed_sample=self.fixed_sample,
            min_duration=self.min_duration,
            )
        
        #entire_dataset = custom_data.one_image_dataset()
        train_size = int(self.train_val_split[0] * len(entire_dataset))
        print("TRAINSIZE IS:::::::::::::::::::::", train_size)
        
        val_size = len(entire_dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(entire_dataset, [train_size, val_size])
    
    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
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


