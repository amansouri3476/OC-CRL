from typing import Any, Dict, List, Callable, Union, TypeVar, Tuple, Container
import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import wandb

import time

class CLEVRSlotAttentionAE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """
        A LightningModule organizes your PyTorch code into 5 sections:
            - Setup for all computations (init).
            - Train loop (training_step)
            - Validation loop (validation_step)
            - Test loop (test_step)
            - Optimizers (configure_optimizers)
        Read the docs:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
        """
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # slot_attention_autoencoder
        self.model = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)
        
        
    def forward(self, batch):
        return self.model(batch)

    def loss(self, reconstructions, images):
        # `reconstructions` and `images` both have a shape of: [batch_size, num_channels, width, height]
        return F.mse_loss(reconstructions, images, reduction="mean")
#         return F.mse_loss(reconstructions, images, reduction="sum")

    def training_step(self, batch, batch_idx):

        training_step_outputs = self.common_step(batch, batch_idx, stage="train")
    
        return training_step_outputs
    
    
    def validation_step(self, batch, batch_idx):
        
        validation_step_outputs = self.common_step(batch, batch_idx, stage="val")
        
        return validation_step_outputs
    
    
    def test_step(self, batch, batch_idx):
        
        test_step_outputs = self.common_step(batch, batch_idx, stage="test")
        
        return test_step_outputs

    def common_step(self, batch, batch_idx, stage="train"):
        """
        batch is a dictionary that contains the following keys:
        ['image']
        
        - 'image': CLEVR dataset images, shape: [batch_size, num_channels, width, height]
        """    
        
        # Get the images in the batch, pass them through AE, get the reconstructions, compute the loss, repeat
        images = batch["images"] # [batch_size, num_channels, width, height]
        recon_combined, recons, masks, slots = self(images)
            
        loss = self.loss(recon_combined, images)

        self.log(f"{stage}_loss", loss.item())
        
        return {"loss": loss}
    
    
    def configure_optimizers(self):
        
#         optimizer_grouped_parameters = [{"params": self.parameters()}]
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                   , self.parameters()
                                                                  )
            
        # for pytorch scheduler objects, we should use utils.instantiate()
        if self.hparams.scheduler_config.scheduler['_target_'].startswith("torch.optim"):
            scheduler = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer)

        # for transformer function calls, we should use utils.call()
        elif self.hparams.scheduler_config.scheduler['_target_'].startswith("transformers"):
            scheduler = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer)
        
        else:
            raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
            
        scheduler_dict = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                , resolve=True)
        scheduler_dict["scheduler"] = scheduler

        return [optimizer], [scheduler_dict]