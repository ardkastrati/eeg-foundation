import os
import pickle
import src.models.mae_original as mae
import wandb
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

import timm.optim.optim_factory as optim_factory

# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightningmodule
# A LightningModule organizes your PyTorch code into 6 sections:
#  - Initialization (__init__ and setup()).
#  - Train Loop (training_step())
#  - Validation Loop (validation_step())
#  - Test Loop (test_step())
#  - Prediction Loop (predict_step())
#  - Optimizers and LR Schedulers (configure_optimizers())


class MAEModule(LightningModule):
    """
    Organizing structure that facilitates the training process
    """

    def __init__(
        self,
        net: torch.nn.Module,  # core neural net that will be trained
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        img_log_frq: 1000,
        learning_rate=0.0002,
        mask_ratio=0.5,
    ) -> None:

        # Initialize attributes
        print("Initialize MAEModule\n=====================")

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.img_log_frq = img_log_frq
        self.net = net
        self.epoch_start_time = 0
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.epoch_train_times = []

    def forward(self, x):
        """ """
        # executes a forward pass of the neural network,
        # i.e. MaskedAutoencoderViT.forward(x) in our case
        loss, _, mask, _ = self.net(x, mask_ratio=self.mask_ratio)
        return loss

    def training_step(self, batch, batch_idx):
        """ """

        loss = self.forward(batch)

        # log the loss on each training step
        # wandb.log(
        #     {"train_loss": loss.item(), "trainer/global_step": self.trainer.global_step}
        # )
        self.log("train_loss", loss.item())
        self.log("trainer/global_step", self.trainer.global_step)

        # step-wise update
        epoch = self.current_epoch
        step = self.trainer.global_step

        # plot the inputs and outputs of the model every xyz epochs/batches
        if step % self.img_log_frq == 0:
            self.visualize_plots(batch, local_tag=epoch, log_tag="train")

        return loss

    def validation_step(self, batch, batch_idx):
        """"""
        loss = self.forward(batch)
        self.log("val_loss", loss.item(), on_epoch=True)
        epoch = self.current_epoch
        step = self.trainer.global_step
        # plot the inputs and outputs of the model every xyz epochs/batches
        scaled_frq = self.img_log_frq * 0.5
        if step % scaled_frq == 0:
            self.visualize_plots(batch, local_tag=epoch, log_tag="val")

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        param_groups = optim_factory.add_weight_decay(self.net, 0.0001)
        optimizer = torch.optim.AdamW(param_groups, lr=0.0002, betas=(0.9, 0.95))
        return optimizer

    def plot_and_save(
        self, save_dir, image, save_local=False, save_log=False, log_tag=""
    ):
        # print("in plot_and_save")
        # print(save_log)

        plt.pcolormesh(image, shading="auto", cmap="viridis")
        plt.ylabel("Frequency Bins")
        plt.xlabel("steps")
        plt.title("Spectrogram")
        plt.colorbar(label="")

        if save_log:
            self.logger.experiment.log({log_tag: [wandb.Image(plt)]})
        plt.clf()

    def visualize_plots(self, batch, local_tag, log_tag):

        # print("in visualize_plots")

        with torch.no_grad():
            # get image matrix (in image has format 1, 128, 1024), just first image of the batch
            val_image2 = batch[0][0]
            val_image2 = val_image2.cpu()
            val_image2 = val_image2.numpy()

            save_log = False
            if self.logger is not None:
                # print("using WANDB logger!")
                save_log = True

            self.plot_and_save(
                f"in_spectrogram{local_tag}",
                val_image2,
                save_log=save_log,
                log_tag=f"{log_tag}_input_plot",
            )

            # returns prediction image in size (b, 128, 1024)
            loss_rec, pred, mask, _ = self.net.forward(
                batch, mask_ratio=self.mask_ratio
            )

            mask = mask[0, :]

            # save masked image

            in_patch = self.net.patchify(batch[:1, :, :, :])
            in_patch = in_patch[0, :, :]

            # apply mask
            for i in range(512):
                in_patch[i] = in_patch[i] * (1 - mask[i])

            # reassemble
            in_patch = in_patch[np.newaxis, :, :]
            in_patch = self.net.unpatchify(in_patch)

            in_patch = in_patch.cpu()
            in_patch = in_patch.numpy()
            in_patch = np.squeeze(in_patch)

            self.plot_and_save(
                f"masked_spectrogram{local_tag}",
                in_patch,
                save_log=save_log,
                log_tag=f"{log_tag}_in_masked",
            )

            # save predicted image
            pred_patches = pred
            pred = self.net.unpatchify(pred)
            pred = pred[0, :, :]
            pred = pred.cpu()

            pred = np.squeeze(pred, axis=0)

            self.plot_and_save(
                f"predicted_image{local_tag}",
                pred,
                save_log=save_log,
                log_tag=f"{log_tag}_predicted",
            )

            # overlay predicted patches over masked patches in original image

            pred_patches = pred_patches[0, :, :]

            for i in range(512):
                pred_patches[i] = pred_patches[i] * mask[i]

            pred_patches = pred_patches[np.newaxis, :, :]
            pred_patches = self.net.unpatchify(pred_patches)

            pred_patches = pred_patches.cpu()
            pred_patches = pred_patches.numpy()
            pred_patches = np.squeeze(pred_patches)

            overlay = pred_patches + in_patch

            self.plot_and_save(
                f"predicted_image_overlay{local_tag}",
                overlay,
                save_log=save_log,
                log_tag=f"{log_tag}_overlay_pred",
            )
