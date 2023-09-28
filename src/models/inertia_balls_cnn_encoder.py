from typing import Any, Dict, List, Callable, Union, TypeVar, Tuple, Container
from itertools import permutations, product
import torch
from torch import nn
from torch.nn import functional as F
from zmq import device
import src
from src.utils.additional_loggers import get_img_rec_table_data
from src.models.contrastive_pl import Contrastive
from src.utils.general import init_weights
from src.utils.lp_solver import lp_solver_pulp, lp_solver_cvxpy
from src.utils.segmentation_metrics import adjusted_rand_index
from src.utils.graphic_utils import draw_scene
import wandb
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
import numpy as np
import scipy
from scipy import optimize
from pytorch_lightning.utilities import rank_zero_only


class Inert(Contrastive):
    """
    This model needs an encoder, that is initialized in the parent class. In addition to an 
    encoder, we should use another module that translates slots to representations ready to be 
    matched (by the mechanisms, not colors or CoM, etc.), both of these modules weights can be
    frozen or trained. Based on matching assignments, latents z will be reordered to get aligend
    with offset mechanisms 'b'. Then m_z1 will be calculated, and z2 will also be computed through
    the learnable module. Then through learning, we hope that the representation z can now be
    disentangled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(logger=False)
        # projection to latents. This module projects slots to the latent space of interest. These 
        # projections will be the targets of disentanglement.
        
        self.z_dim = self.hparams["z_dim"]
        self.n_balls = self.hparams["n_balls"]
        self.encoder = hydra.utils.instantiate(self.hparams.encoder)
        if not self.hparams.separate_projection_head:
            self.slot_projection_to_z = torch.nn.Sequential(*[hydra.utils.instantiate(layer_config) for _, layer_config in self.hparams.projection_to_z.items()])
            self.model = torch.nn.Sequential(self.encoder, self.slot_projection_to_z)
        else:
            self.model = self.encoder
            self.slot_projection_to_z = nn.ModuleList()
            for i in range(self.z_dim * self.n_balls):
                projection_head = torch.nn.Sequential(*[hydra.utils.instantiate(layer_config) for _, layer_config in self.hparams.projection_to_z.items()])
                self.slot_projection_to_z.append(projection_head)

        
        if self.hparams["projection_to_z_ckpt_path"] is not None:
            ckpt_path = self.hparams["projection_to_z_ckpt_path"]
            self.slot_projection_to_z = torch.load(ckpt_path)

        # freeze the parameters of the module if needed
        if self.hparams.projection_to_z_freeze:
            for param in self.slot_projection_to_z.parameters():
                param.requires_grad = False

        if kwargs.get("additional_logger"):
            self.additional_logger = hydra.utils.instantiate(self.hparams.additional_logger)
        else:
            self.additional_logger = None

        # self.sparsity_degree = self.hparams.get("sparsity_degree", self.hparams["n_balls"])
        self.known_mechanism = self.hparams.get("known_mechanism", True)


    def _load_messages(self):
        print(f"MSE\nBaseline: {self.baseline}")

    def loss(self, m_z1, z2):
        return F.mse_loss(m_z1, z2, reduction="mean")

    def forward(self, batch):
        # batch: batch of images -> [batch_size, num_channels, width, height]
        # projection to target dimension
        if not self.hparams.separate_projection_head:
            z_pred = self.model(batch)
        else:
            latent_projected = []
            for i in range(self.z_dim * self.n_balls):
                projection_property = self.slot_projection_to_z[i](self.model(batch))
                latent_projected.append(projection_property)
            latent_projected = torch.cat(latent_projected, dim=-1)
            latent_projected.to(batch.device)
            z_pred = latent_projected.clone()

        return z_pred


    def training_step(self, batch, batch_idx):

        # with torch.no_grad():
        #     for name, param in self.slot_projection_to_z.named_parameters():
        #         if len(param.size()) > 1: # excluding bias terms
        #             u,s,vh = torch.linalg.svd(param.data, full_matrices=False)
        #             eps = torch.tensor([0.01], device=param.device)
        #             param.data = u @ torch.diag(torch.maximum(s, eps)) @ vh
        #             # print(f"\n-----HI:{torch.maximum(s, eps)}\n\nname:{name}\n\nparam:{param.data}")

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
        ['latents', 'images', 'matrices', 'colors']
        
        - 'latents': (Tuple) (z1, z2), latents for two consecutive timesteps, each having a dimension of [batch_size, n_balls*z_dim]
        - 'images': (Tuple) (x1, x2), images corresponding to latents (z1,z2) for two consecutive timesteps, each having
        a dimension of [batch_size, num_channels, width, height]
        - 'segmentation_masks': (Tuple) (segmentation_masks1, segmentation_masks2), segmentation masks for each ball and in two consecutive timesteps, each having
        a dimension of [batch_size, n_balls, width, height, 1]. Required for ARI computation to evaluate object discovery.
        - 'matrices': (Tuple) (A,b), matrices that determine the mechanisms (i.e. offsets)
        - 'mechanism_permutation': (Tensor), matrix of shape [batch_size, sparsity_degree] that specifies which mechanism should be applied to which ball
        - 'colors': (Tensor), tensor containing the normalized (by 255) colors of balls
            [batch_size, n_balls, 3]
        """

        (z1, z2), (x1, x2), (segmentation_masks1, segmentation_masks2), (A, b), mechanism_permutation, (coordinates_1, coordinates_2), (colors_1, colors_2) = batch["latents"], batch["images"], batch["segmentation_masks"], batch["matrices"], batch["mechanism_permutation"], batch["coordinates"], batch["colors"]

        bs = x1.shape[0]
        device = x1.device
        img_width = x1.shape[-2]
        img_height = x1.shape[-1]

        z_pred_1 = self(x1)
        z_pred_2 = self(x2)

        # note that for cnn encoder the b will be received as [n_balls, z_dim]
        C = 0.5
        b_true = b.clone() # [batch_size, 1, z_dim]
        if self.known_mechanism:
            b = b_true.clone() # [batch_size, n_balls, z_dim]
        else:
            if self.hparams["signed_change"]:
                b = (torch.sign(b_true) * C).to(device) # [bs, n_balls, z_dim]
            else:
                b_base = (torch.abs(b_true)>0.) * C # [bs, n_balls, self.z_dim]
                b = b_base.to(device) # [bs, n_balls, self.z_dim]

        
        if self.trainer.global_step % 50 == 0:
            print(f"b:\n{b[:15]}")
            print(f"z1:\n{z1[:15]}")
            print(f"z2:\n{z2[:15]}")
            print(f"z_pred_1:\n{z_pred_1[:15]}")
            print(f"z_pred_2:\n{z_pred_2[:15]}")
            
        m_z1 = z_pred_1 + b
        latent_loss = F.mse_loss(m_z1, z_pred_2, reduction="mean")
        n_balls = self.hparams["n_balls"]
        
        self.log(f"{stage}_latent_loss", latent_loss.item())

        z_true = z1
        z_pred = z_pred_1

        # b: [batch_size, n_balls (n_mech), z_dim]
        baseline_loss = ((torch.norm(b, p=2, dim=-1).sum(dim=-1))**2).mean()
        self.log(f"{stage}_baseline_loss", baseline_loss.item())
        
        loss = latent_loss
        self.log(f"{stage}_loss", loss.item())

        if stage == "train":
            return {"loss": loss}
        else:
            # if batch_idx % 50 == 0:
                # print(f"z_true:\n{z_true[:40].clone().detach().cpu().numpy()}")
                # print(f"pred_z:\n{-100. * z_pred[:40].clone().detach().cpu().numpy() / 20.}")
            # return {"loss": loss, "true_z": z_true, "pred_z": -100. * z_pred.detach() / 20., "ratio":pixel_to_latent_space_norm_ratio}
            return {"loss": loss, "true_z": z_true, "pred_z": z_pred.detach()}

    def background_slot(self, attention_scores, n_bg_slots=1):
        # attention_scores: [batch_size, num_slots, height * width]
        n_slots = attention_scores.shape[1]
        device = attention_scores.device
        # There can be more than 1 background slot, so we have to find the n_slots-n_balls masks with the least std and 
        # don't let them to be selected by the LP solution
        background_masks_sorted_asc, background_masks_sorted_index_asc = torch.sort(torch.std(attention_scores, dim=-1), -1, descending=False)
        background_masks = []
        for i in range(n_bg_slots):
            background_slot_indices = background_masks_sorted_index_asc[:, i]
            background_slot_mask = torch.nn.functional.one_hot(background_slot_indices, num_classes=n_slots).bool() # [bs, n_slots]
            background_masks.append(background_slot_mask)

        return background_masks
