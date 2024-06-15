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


class MLP(torch.nn.Module):
    def __init__(self, emb_dim):
        super(MLP, self).__init__()

        # Define layers
        self.fc1 = torch.nn.Linear(emb_dim, emb_dim // 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(emb_dim // 2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class MAEModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        img_log_frq: 1000,
        learning_rate=0.0002,
        mask_ratio=0.5,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.img_log_frq = img_log_frq
        self.net = net
        self.epoch_start_time = 0
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio

    def forward(self, x):

        predictions = self.net.forward_finetune(x)

        return predictions

    def training_step(self, batch, batch_idx):

        x, y = batch
        y = y.float()
        y = y.cpu()

        predictions = self.forward(x)
        lossfn = torch.nn.BCELoss()
        print("predictions are")
        print(predictions)
        loss = lossfn(predictions, y)
        print("loss is")
        print(loss.item())
        self.log("train_loss", loss.item(), on_epoch=True)

        return loss

    def on_train_epoch_end(self):

        pass

    def validation_step(self, batch, batch_idx):

        x, y = batch

        y = y.float()
        y = y.cpu()
        pred = self.forward(x)

        lossfn = torch.nn.BCELoss()

        loss = lossfn(pred, y)
        print(loss.item())
        self.log("val_loss", loss.item(), on_epoch=True)

    def configure_optimizers(self):
        # keeping it simple
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return optimizer

    def visualize_plots(self, batch, local_tag, log_tag):

        print("Visualizing the Validation Plots!")
        with torch.no_grad():
            # get image matrix (in image has format 1, 128, 1024), just first image of the batch
            val_image2 = batch[0][0]
            val_image2 = val_image2.cpu()
            val_image2 = val_image2.numpy()

            save_log = False
            if self.logger is not None:
                print("using WANDB logger!")
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
