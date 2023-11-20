import torch
import os

import src.data.foundation_loader as custom_data


from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

class EDFTESTDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "/itet-stor/schepasc/deepeye_storage/tueg/edf/000",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None: 
        
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.ToTensor()

    
    def prepare_data(self) -> None:

        pass

    def setup(self) -> None: 

        
        entire_dataset = custom_data.EDFTESTDataset(data_dir)
        train_size = int(0.8 * len(entire_dataset))
        val_size = len(entire_dataset) - train_size

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
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


    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


