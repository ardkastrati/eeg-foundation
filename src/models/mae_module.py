import src.models.mae_original as mae
import wandb
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import matplotlib.pyplot as plt
import numpy as np
class MAEModule(LightningModule):

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            img_log_frq : 100
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
            self.img_log_frq = img_log_frq
            self.net = net
            

    def forward(self, x):
          
          loss, _, mask,_ = self.net(x, mask_ratio = 0.5)
          
          return loss
    
    def plot_and_save(self,save_dir, image, save_local = True, save_log = False, log_tag =""):

        plt.pcolormesh(image, shading='auto', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('steps')
        plt.title('Spectrogram in Decibels')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        if save_local: 
            plt.savefig(save_dir)   
            print("SAVED AS", save_dir)
        if save_log :
            self.logger.experiment.log({log_tag: [wandb.Image(plt)]})
        plt.clf()

    def training_step(self, batch, batch_idx):
        
        batch.to("cuda")
        loss = self.forward(batch)
        
        self.log("train_loss", loss, on_epoch=True)
        
        self.net.eval()

        epoch = self.current_epoch
        if (epoch % self.img_log_frq == 0):

            #plot the inputs and outputs of the model every xyz epochs/batches

            self.visualize_plots(batch, local_tag=epoch, log_tag="train")


        self.net.train()

        return loss
    def validation_step(self, batch, batch_idx):
         
        self.net.eval()

        loss = self.forward(batch)
        self.log("val_loss", loss, on_epoch=True)
        epoch = self.current_epoch
        #plot the inputs and outputs of the model every xyz epochs/batches
        if (epoch % self.img_log_frq == 0):

            self.visualize_plots(batch, local_tag=epoch, log_tag="val")

        self.net.train()
        
       
        



    def configure_optimizers(self):
        #keeping it simple
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0002)
        return optimizer

    def visualize_plots(self, batch, local_tag, log_tag):

        with torch.no_grad():
                #get image matrix (in image has format 1, 128, 1024), just first image of the batch
                val_image2 = batch[0][0]
                val_image2 = val_image2.cpu()
                val_image2 = val_image2.numpy()
                
                save_log = False
                if self.logger is not None :
                     print("using WANDB logger!")
                     save_log = True
                
                self.plot_and_save(f"in_spectrogram{local_tag}", val_image2, 
                                   save_log = save_log, log_tag= f"{log_tag}_input_plot")
            
                #returns prediction image in size (b, 128, 1024)
                loss_rec, pred, mask, _ = self.net.forward(batch, mask_ratio = 0.5)
                
                mask = mask[0,:]
                

                #save masked image
                
                in_patch = self.net.patchify(batch[:1, :, : ,:])
                in_patch = in_patch[0, :, :]
                
                #apply mask
                for i in range (512):
                    in_patch[i] = in_patch[i] * (1 -mask[i])
                
                #reassemble
                in_patch = in_patch[np.newaxis, :,:]
                in_patch = self.net.unpatchify(in_patch)

                in_patch = in_patch.cpu()
                in_patch = in_patch.numpy()
                in_patch = np.squeeze(in_patch)
               
                
                self.plot_and_save(f"masked_spectrogram{local_tag}", in_patch, 
                                   save_log=save_log, log_tag=f"{log_tag}_in_masked")
                
                    
                #save predicted image
                pred_patches = pred
                pred = self.net.unpatchify(pred)
                pred = pred[0, :, :]
                pred = pred.cpu()
                pred = pred.detach().numpy()
                
                pred = np.squeeze(pred, axis = 0)
                
                self.plot_and_save(f"predicted_image{local_tag}", pred, 
                                   save_log = save_log, log_tag =f"{log_tag}_predicted")    
                
                

                #overlay predicted patches over masked patches in original image

                pred_patches = pred_patches[0, :, :]
                
                for i in range (512):
                    pred_patches[i] = pred_patches[i] * mask[i]

                pred_patches = pred_patches[np.newaxis, :,:]
                pred_patches = self.net.unpatchify(pred_patches)

                pred_patches = pred_patches.cpu()
                pred_patches = pred_patches.numpy()
                pred_patches = np.squeeze(pred_patches)

                overlay = pred_patches + in_patch

                self.plot_and_save(f"predicted_image_overlay{local_tag}", overlay, 
                                   save_log=save_log, log_tag =f"{log_tag}_overlay_pred")
