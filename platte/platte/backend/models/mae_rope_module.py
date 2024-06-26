import sys
from matplotlib import pyplot as plt
import numpy as np
import wandb
import torch
import torch.nn as nn
from lightning import LightningModule
import timm.optim.optim_factory as optim_factory


class MAEModuleRoPE(LightningModule):
    def __init__(
        self,
        net: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        learning_rate=0.0002,
        log_frq_lr=1000,
        train_log_frq_loss=100,
        train_log_frq_imgs=200,
        val_log_frq_loss=50,
        val_log_frq_imgs=100,
    ):
        super().__init__()

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learning_rate = learning_rate

        self.log_frq_lr = log_frq_lr
        self.train_log_frq_loss = train_log_frq_loss
        self.train_log_frq_imgs = train_log_frq_imgs
        self.val_log_frq_loss = val_log_frq_loss
        self.val_log_frq_imgs = val_log_frq_imgs

        self.save_hyperparameters(logger=False, ignore=["net"])

    # == Training, Validation, Testing ==================================================================================================

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        train_loss, flattened_pred, masked_indices = self(batch)

        # print("[training step] train_loss:", train_loss.detach())
        # self.log("train_loss", train_loss, rank_zero_only=False, sync_dist=True)

        # Logging
        if self.trainer.global_step % self.train_log_frq_loss == 0:
            wandb.log(
                {"train_loss": train_loss.detach()},
                step=self.trainer.global_step,
            )
        if self.trainer.global_step % self.log_frq_lr == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            wandb.log(
                {"learning_rate": current_lr},
                step=self.trainer.global_step,
            )

        # Plotting
        if self.trainer.global_step % self.train_log_frq_imgs == 0:
            self.plot_spgs(batch, log_tag="train")

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, flattened_pred, masked_indices = self(batch)

        # For checkpointing: log the validation loss with self.log
        self.log(
            "val_loss",
            val_loss,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch["batch"].shape[0],
        )

        # Logging
        if self.trainer.global_step % self.val_log_frq_loss == 0:
            wandb.log(
                {"val_loss": val_loss.detach()},
                step=self.trainer.global_step,
            )

        # Plotting
        if self.trainer.global_step % self.val_log_frq_imgs == 0:
            self.plot_spgs(batch, log_tag="val")

        return val_loss

    def test_step(self, batch, batch_idx):
        pass

    # == Checkpoints ====================================================================================================================

    # None atm

    # == Helpers ========================================================================================================================

    def configure_optimizers(self):
        param_groups = optim_factory.add_weight_decay(self.net, 0.0001)
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.learning_rate, betas=(0.9, 0.95)
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=1,
            threshold=0.05,
            threshold_mode="rel",
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }

    def plot_and_log(self, image, title):

        plt.pcolormesh(image, shading="auto", cmap="viridis")
        plt.ylabel("Frequency Bins")
        plt.xlabel("Steps")
        plt.title(title)
        plt.colorbar(label="")

        wandb.log(
            {title: [wandb.Image(plt)]},
            step=self.trainer.global_step,
        )
        plt.clf()
        plt.close()

    def plot_spgs(self, batch, log_tag):

        with torch.no_grad():

            B, C, H, W = batch["batch"].shape
            # print(f"[plot_spgs] batch.shape: {batch['batch'].shape}")

            # == Input Image ==

            sample_input_image = batch["batch"][0][0].cpu().numpy()

            self.plot_and_log(
                image=sample_input_image,
                title=f"{log_tag}_input",
            )

            # == Masked Input Image ==

            _, flattened_pred, mask = self.net.forward(batch)

            flattened_batch = self.net.patchify(batch["batch"], B, H, W)
            # print(f"[plot_spgs] flattened_batch.shape: {flattened_batch.shape}")
            B, N, D = flattened_batch.shape

            # Apply the mask
            masked_flattened_batch = flattened_batch.masked_fill_(mask.unsqueeze(-1), 0)
            # print(f"[plot_spgs] masked_flattened_batch.shape: {flattened_batch.shape}")

            masked_batch = self.net.unpatchify(masked_flattened_batch, B, H, W)
            # print(
            #     "[plot_spgs] masked_batch.shape:", masked_batch.shape, file=sys.stderr
            # )

            masked_sample_input_image = masked_batch[0][0].cpu().numpy()

            self.plot_and_log(
                image=masked_sample_input_image,
                title=f"{log_tag}_masked_input",
            )

            # == Predicted Output Image ==

            pred = self.net.unpatchify(flattened_pred, B, H, W)

            sample_output_image = pred[0][0].cpu().numpy()

            self.plot_and_log(
                image=sample_output_image,
                title=f"{log_tag}_predicted",
            )
