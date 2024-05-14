import src.models.mae_original as mae
import wandb
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import matplotlib.pyplot as plt
import numpy as np

import timm.optim.optim_factory as optim_factory
from socket import gethostname

# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightningmodule
# A LightningModule organizes your PyTorch code into 6 sections:
#  - Initialization (__init__ and setup())
#  - Train Loop (training_step())
#  - Validation Loop (validation_step())
#  - Test Loop (test_step())
#  - Prediction Loop (predict_step())
#  - Optimizers and LR Schedulers (configure_optimizers())


class MAEModule(LightningModule):
    """
    Organizing structure that facilitates the training process.
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
        max_epochs=None,
    ) -> None:

        # Initialize attributes
        print("Initialize MAEModule\n=====================")

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False, ignore=["net"])
        self.save_hyperparameters(logger=False)
        self.img_log_frq = img_log_frq
        self.net = net
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.hostname = gethostname()
        self.max_epochs = max_epochs

        # Initialize metrics logging datastructures
        self.epoch_train_times = [[] for _ in range(self.max_epochs)]
        self.epoch_train_throughputs = [[] for _ in range(self.max_epochs)]
        self.train_losses = [[] for _ in range(self.max_epochs)]
        self.val_losses = [[] for _ in range(self.max_epochs)]
        self.vis_train_plots = [[] for _ in range(self.max_epochs)]
        self.vis_val_plots = [[] for _ in range(self.max_epochs)]

    def forward(self, x):
        """
        Executes a forward pass of the neural network.

        Args:
            x: Input tensor.

        Returns:
            loss: The calculated loss.
        """
        loss, pred, mask, _ = self.net(x, mask_ratio=self.mask_ratio)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step on the given batch of data.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The loss value computed during the training step.
        """

        # Forward pass of the model
        loss = self.forward(batch)

        # Logging
        epoch = self.current_epoch
        step = self.trainer.global_step

        # self.train_losses[epoch].append((step, epoch, batch_idx, loss.detach()))
        ############################
        if step % 100 == 0:
            wandb.log(
                {"train_loss": loss.detach()},
                step=self.trainer.global_step,
            )
        if step % 200 == 0:
            self.visualize_plots(batch, local_tag=epoch, log_tag="train")
        ############################

        # self.log("train_loss", loss, on_step=True, on_epoch=True)

        # plot the inputs and outputs of the model every xyz epochs/batches
        #     self.visualize_plots(batch, local_tag=epoch, log_tag="train")

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step for the model.

        Args:
            batch: The input batch for validation.
            batch_idx: The index of the current batch.

        Returns:
            The loss value for the validation step.
        """
        loss = self.forward(batch)

        epoch = self.current_epoch
        step = self.trainer.global_step
        # self.val_losses[epoch].append((step, epoch, batch_idx, loss.detach()))
        # wandb.log(
        #     {"val_loss": loss.detach()},
        #     step=self.trainer.global_step,
        # )
        # Log validation loss for checkpointing: TODO rank_zero_only ?
        self.log("val_loss", loss, rank_zero_only=True)
        ############################
        if step % 50 == 0:
            wandb.log(
                {"val_loss": loss.detach()},
                step=self.trainer.global_step,
            )
        if step % 100 == 0:
            self.visualize_plots(batch, local_tag=epoch, log_tag="val")
        ############################
        # self.log("val_loss", loss.item(), on_epoch=True)
        # plot the inputs and outputs of the model every xyz epochs/batches
        # scaled_frq = self.img_log_frq * 0.5
        # if step % scaled_frq == 0:
        #     self.visualize_plots(batch, local_tag=epoch, log_tag="val")

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The configured optimizer.
        """
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
        plt.title(f"Spectrogram: {log_tag}")
        plt.colorbar(label="")

        # fig, ax = (
        #     plt.subplots()
        # )  # Use plt.subplots() to create new figure and axes objects
        # c = ax.pcolormesh(image, shading="auto", cmap="viridis")
        # ax.set_ylabel("Frequency Bins")
        # ax.set_xlabel("Steps")
        # ax.set_title("Spectrogram")
        # fig.colorbar(c, ax=ax, label="")

        if save_log:
            # self.logger.experiment.log({log_tag: [wandb.Image(plt)]})
            wandb.log(
                {log_tag: [wandb.Image(plt)]},
                step=self.trainer.global_step,
            )
        plt.clf()
        # self.vis_train_plots[self.current_epoch].append(
        #     {
        #         "data": {log_tag: [wandb.Image(plt)]},
        #         "epoch": self.trainer.current_epoch,
        #     }
        # )

    def visualize_plots(self, batch, local_tag, log_tag):

        # print("in visualize_plots")

        with torch.no_grad():
            # get image matrix (in image has format 1, 128, 1024), just first image of the batch
            val_image2 = batch[0][0]
            val_image2 = val_image2.cpu()
            val_image2 = val_image2.numpy()

            save_log = True
            # if self.logger is not None:
            #     # print("using WANDB logger!")
            #     save_log = True

            self.plot_and_save(
                f"in_spectrogram{local_tag}",
                val_image2,
                save_log=save_log,
                log_tag=f"{log_tag}_input_plot",
            )

            # returns prediction image in size (b, 128, 1024)
            # print("batch shape", batch.shape)
            loss_rec, pred, mask, _ = self.net.forward(
                batch, mask_ratio=self.mask_ratio
            )
            # print("pred shape", pred.shape)
            # print("mask shape", mask.shape)

            mask = mask[0, :]

            # save masked image

            in_patch = self.net.patchify(batch[:1, :, :, :])
            # print("in_patch.shape 1:", in_patch.shape)
            in_patch = in_patch[0, :, :]
            # print("in_patch.shape 2:", in_patch.shape)

            # apply mask
            for i in range(in_patch.shape[0] - 1):
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
            # print("pred_patches.shape:", pred_patches.shape)

            for i in range(pred_patches.shape[0] - 1):
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
