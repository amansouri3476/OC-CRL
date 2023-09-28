from pytorch_lightning.callbacks import Callback
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance


class VisualizationLoggerCallback(Callback):
    

    def __init__(self, visualize_every=100, n_samples=3, **kwargs):
        
        self.visualize_every = visualize_every
        self.n_samples = n_samples
    
    def on_fit_start(self, trainer, pl_module):
        
        self.datamodule = trainer.datamodule
        datamodule_hparams = trainer.datamodule.hparams
        self.model = pl_module.model
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % (self.visualize_every+1) == 0:
            
            renormalize = self.datamodule.renormalize()
            pl_module.eval()
            with torch.no_grad():
                # should get 3 sample images and their prediction
                # generate different subplots for each sample
                # log each subplot to wandb
                # remember to log masks as well

                # Predict.
                if type(batch["images"]) is list: # inertia balls dataset
                    batch = batch["images"][0]
                    batch = batch[:self.n_samples]
                else: # clevr dataset which doesn't return a tuple
                    batch = batch["images"][:self.n_samples]
                    
                recon_combined, recons, masks, slots, attention_scores, slots_init = self.model(batch)

                recon_combined = recon_combined.permute(0,2,3,1).detach().cpu().numpy()
                recons = recons.detach().cpu().numpy()
                images = renormalize(batch.permute(0,2,3,1))
                recon_combined = renormalize(recon_combined)
                recons = renormalize(recons)
                masks = masks.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]

                for idx in range(self.n_samples):

                    image = images[idx].cpu().numpy()
                    recon_combined_ = recon_combined[idx]
                    recons_ = recons[idx]
                    masks_ = masks[idx]
                    # Visualize.
                    num_slots = len(masks_)
                    fig, ax = plt.subplots(2, num_slots + 2, figsize=(15, 4))
                    ax[0,0].imshow(image)
                    ax[0,0].set_title('Image')
#                     ax[0,1].imshow(recon_combined_)
                    ax[0,1].imshow((recon_combined_ * 255).astype(np.uint8), vmin=0, vmax=255)
                    ax[0,1].set_title('Recon.')
                    ax[1,0].imshow(masks_.sum(axis=0), vmin=0, vmax=1) # masks_ is a numpy array, hence the use of axis instead of dim
#                     ax[1,0].imshow((np.round(masks_.sum(axis=0))))
                    ax[1,0].set_title('Masks sum')
                    ax[1,1].imshow(masks_.sum(axis=0), vmin=0, vmax=1)
#                     ax[1,1].imshow((np.round(masks_.sum(axis=0))))
                    ax[1,1].set_title('Masks sum')
                    for i in range(num_slots):
                        temp = recons_[i] * masks_[i] + (1 - masks_[i])
                        ax[0, i + 2].imshow(temp, vmin=0, vmax=1)
#                         ax[0, i + 2].imshow((temp * 255).astype(np.uint8))
                        ax[0, i + 2].set_title('Slot %s' % str(i + 1))
                        ax[1, i + 2].imshow(masks_[i], vmin=0, vmax=1)
                        ax[1, i + 2].set_title('Slot %s mask' % str(i + 1))
                    for i in range(ax.shape[1]):
                        ax[0, i].grid(False)
                        ax[0, i].axis('off')
                        ax[1, i].grid(False)
                        ax[1, i].axis('off')


                    wandb.log({f"Reconstruction and Masks for sample {idx}": fig})

            
