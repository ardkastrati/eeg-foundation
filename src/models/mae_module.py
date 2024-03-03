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
class MAEModule(LightningModule):

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            img_log_frq : 1000,
            learning_rate = 0.0002,
            mask_ratio = 0.5
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
          
          loss, _, mask,_ = self.net(x, mask_ratio = self.mask_ratio)
          
          return loss
    
    def plot_and_save(self,save_dir, image, save_local = False, save_log = False, log_tag =""):

        plt.pcolormesh(image, shading='auto', cmap='viridis')
        plt.ylabel('Frequency Bins')
        plt.xlabel('steps')
        plt.title('Spectrogram')
        plt.colorbar(label='')
        
            
        if save_log :
            self.logger.experiment.log({log_tag: [wandb.Image(plt)]})
        plt.clf()

    def training_step(self, batch, batch_idx):
        
        
        loss = self.forward(batch)
        
        self.log("train_loss", loss.item(), on_epoch=True)
        
        

        epoch = self.current_epoch
        step = self.trainer.global_step
        
        if ( step % self.img_log_frq == 0):

            #plot the inputs and outputs of the model every xyz epochs/batches

            self.visualize_plots(batch, local_tag=epoch, log_tag="train")


        

        return loss
    def on_train_epoch_end(self): 
        
        end_time = time.time()
        self.log("Time per Epoch", end_time - self.epoch_start_time)
        self.epoch_start_time = time.time()
        
        

    def validation_step(self, batch, batch_idx):
         
       
        
        
        loss = self.forward(batch)
        self.log("val_loss", loss.item(), on_epoch=True)
        epoch = self.current_epoch
        step = self.trainer.global_step
        #plot the inputs and outputs of the model every xyz epochs/batches
        scaled_frq  = self.img_log_frq * 0.5
        if (step % scaled_frq == 0):
            
            self.visualize_plots(batch, local_tag=epoch, log_tag="val")
        
        
   
       
        



    def configure_optimizers(self):
        param_groups = optim_factory.add_weight_decay(self.net, 0.0001)
        optimizer = torch.optim.AdamW(param_groups, lr=0.0002, betas=(0.9, 0.95))
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
                loss_rec, pred, mask, _ = self.net.forward(batch, mask_ratio = self.mask_ratio)
                
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
