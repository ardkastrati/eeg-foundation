import src.models.mae_original as mae

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class MAEModule(LightningModule):

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
        ) -> None:
            """Initialize a `MNISTLitModule`.

            :param net: The model to train.
            :param optimizer: The optimizer to use for training.
            :param scheduler: The learning rate scheduler to use for training.
            """
            super().__init__()

            # this line allows to access init params with 'self.hparams' attribute
            # also ensures init params will be stored in ckpt
            self.save_hyperparameters(logger=False)
            
            self.net = net
            

    def forward(self, x):
          
          loss, _, _,_ = self.net(x)
          return loss

    def training_step(self, batch, batch_idx):
          b, h, w = batch.shape
          batch = batch.reshape(b,1,1024,128)
          
          
          batch.to("cuda")
          return self.forward(batch)
    
    
    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr = 0.001)

