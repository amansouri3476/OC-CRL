from typing import Any, Dict, List, Callable, Union, TypeVar, Tuple, Container
from itertools import permutations
import torch
from torch import nn
from torch.nn import functional as F
from src.models.contrastive_pl import Contrastive
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
    encoder, we should load another module that translates slots to representations ready to be 
    matched, both of these modules weights will be frozen. A third module transforms slot 
    representations to z latents (different from the one used for matching). Based on matching 
    assignments, latents z will be reordered to get aligend with offset mechanisms 'b'. Then m_z1 
    will be calculated, and z2 will also be computed through the learnable module. Then through 
    learning, we hope that the representation z can now be disentangled. This is measured in 
    validation loops
    """
    
    def __init__(self, n_latents, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # projection to matching targets. This module projects slots to be matched with some feature
        # of the objects in the scene.
        self.slot_projection_to_match_target = torch.nn.Sequential(*[hydra.utils.instantiate(layer_config) for _, layer_config in self.hparams.projection_to_match_target.items()])
        if self.hparams["projection_to_match_target_ckpt_path"] is not None:
            ckpt_path = self.hparams["projection_to_match_target_ckpt_path"]
            self.slot_projection_to_match_target = torch.load(ckpt_path)

        # freeze the parameters of the module if needed
        if self.hparams.projection_to_match_target_freeze:
            for param in self.slot_projection_to_match_target.parameters():
                param.requires_grad = False


        # projection to latents. This module projects slots to the latent space of interest. These 
        # projections will be the targets of disentanglement.
        if self.hparams.encoder.slot_size <= 8: # Then the mlp config won't work, because it should be 
                                        # divisible by 2, 3 times
            self.slot_projection_to_z = torch.nn.Sequential(
                torch.nn.Linear(self.hparams.encoder.slot_size, self.hparams.encoder.slot_size, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hparams.encoder.slot_size, self.hparams.encoder.slot_size, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hparams.encoder.slot_size, self.hparams.z_dim, bias=False),
                torch.nn.ReLU(),
            )
        else:
            self.slot_projection_to_z = torch.nn.Sequential(*[hydra.utils.instantiate(layer_config) for _, layer_config in self.hparams.projection_to_z.items()])
        
        if self.hparams["projection_to_z_ckpt_path"] is not None:
            ckpt_path = self.hparams["projection_to_z_ckpt_path"]
            self.slot_projection_to_z = torch.load(ckpt_path)


        # freeze the parameters of the module if needed
        if self.hparams.projection_to_z_freeze:
            for param in self.slot_projection_to_z.parameters():
                param.requires_grad = False

    def _load_messages(self):
        print(f"MSE\nBaseline: {self.baseline}")

    def loss(self, m_z1, z2):
        return F.mse_loss(m_z1, z2, reduction="mean")

    def forward(self, batch):

        # batch: batch of images -> [batch_size, num_channels, width, height]
        recon_combined, recons, masks, slots = self.model(batch) # slots: [batch_size, num_slots, slot_size]

        # projection to target dimension
        slots_projected = self.slot_projection_to_match_target(slots) # slots_projected: [batch_size, num_slots, target_dim]

        return slots, slots_projected, recon_combined, recons, masks


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
        ['latents', 'images', 'matrices', 'colors']
        
        - 'latents': (Tuple) (z1, z2), latents for two consecutive timesteps, each having a dimension of [batch_size, n_balls*2]
        - 'images': (Tuple) (x1, x2), images corresponding to latents (z1,z2) for two consecutive timesteps, each having
        a dimension of [batch_size, num_channels, width, height]
        - 'matrices': (Tuple) (A,b), matrices that determine the mechanisms (i.e. offsets)
        - 'colors': (Tensor), tensor containing the normalized (by 255) colors of balls
            [batch_size, n_balls, 3]
        """    

        (z1, z2), (x1, x2), (A, b), colors = batch["latents"], batch["images"], batch["matrices"], batch["colors"]

        # 1. pass x1 to the encoder, and get slots.
        # 2. pass slots to projection module to be matched (along with targets, be it CoM or colors)
        # 3. get the assignments and reorder slots
        # 4. pass reordered slots to the second projection module to obtain latents z
        # 5. apply the mechanism (adding offset) only to that many slots that correspond to some mech.
        # 6. apply steps 1-4 for x2 to obtain object_z2 (or use the indices from step 3)
        # 7. compute the mse loss between m_z1 and z2, and return the result.

        # 1,2
        slots, slots_projected, recon_combined, recons, masks = self(x1)

        # 3
        # TODO: This should be decided in config, whether to use colors, or something else.
        # targets = colors.float()
        targets = z1.view(z1.size(0), -1, 2).float() # [batch_size, n_balls, 2]

        # slots_projected: [batch_size, num_slots, target_dim]
        # targets: [batch_size, n_balls, 3 or 2]
        hungarian_loss, indices = self.compute_loss_and_reorder(slots_projected, targets)
        # hungarian_loss: scalar
        # indices: [batch_size, min(num_slots, n_balls), 2]

        # reorder slots. Note that slots that do not correspond to objects are dropped in this operation
        # slots: [batch_size, num_slots, slot_size]
        
        # indices[:, :, 0] is just a sorted range, indices[:, :, 1] gives the id of the object 
        # corresponding to each slot.
        indices = indices[:, :, 1].unsqueeze(-1)
        permutations = indices.repeat(1, 1, slots.size(-1))
        object_slots_reordered = torch.gather(slots, 1, permutations) # [batch_size, n_balls, slot_size]
        
        # 4
        objects_z1 = self.slot_projection_to_z(object_slots_reordered) # [batch_size, n_balls, z_dim]

        # 5; b: [batch_size, n_balls*2]
        m_objects_z1 = objects_z1 + b.squeeze().view(b.size(0), -1, 2)

        # 6 TODO: either use the indices from step 3, or redo computations from step 1
        slots, slots_projected, recon_combined, recons, masks = self(x2)

        # TODO: This should be decided in config, whether to use colors, or something else.
        # targets = colors.float()
        targets = z2.view(z2.size(0), -1, 2).float() # [batch_size, n_balls, 2]
        hungarian_loss, indices = self.compute_loss_and_reorder(slots_projected, targets)
        indices = indices[:, :, 1].unsqueeze(-1)
        permutations = indices.repeat(1, 1, slots.size(-1))
        object_slots_reordered = torch.gather(slots, 1, permutations) # [batch_size, n_balls, slot_size]
        objects_z2 = self.slot_projection_to_z(object_slots_reordered) # [batch_size, n_balls, z_dim]

        # 7
        bs = m_objects_z1.size(0)
        loss = self.loss(m_objects_z1.view(bs,-1), objects_z2.view(bs,-1))

        self.log(f"{stage}_loss", loss.item())
        
        if stage is "train":
            return {"loss": loss}
        else:
            return {"loss": loss, "true_z": z1, "pred_z": objects_z1.view(bs,-1)}


    def compute_loss_and_reorder(self, slot_projections, targets):
        """Huber loss for sets, matching elements with the Hungarian algorithm.
        This loss is used as reconstruction loss in the paper 'Deep Set Prediction
        Networks' https://arxiv.org/abs/1906.06565, see Eq. 2. For each element in the
        batches we wish to compute min_{pi} ||y_i - x_{pi(i)}||^2 where pi is a
        permutation of the set elements. We first compute the pairwise distances
        between each point in both sets and then match the elements using the scipy
        implementation of the Hungarian algorithm. This is applied for every set in
        the two batches. Note that if the number of points does not match, some of the
        elements will not be matched. As distance function we use the Huber loss.
        Args:
            x: Batch of sets of size [batch_size, n_points_1, dim_points]. Each set in the
            batch contains n_points many points, each represented as a vector of
            dimension dim_points.
            y: Batch of sets of size [batch_size, n_points_2, dim_points].
        Returns:
            Average distance between all sets in the two batches.
        """

        y = targets
        x = slot_projections

        # [batch_size, n_points_2, n_points_1]
        pairwise_cost = torch.nn.HuberLoss(reduction="none")(
            y.unsqueeze(dim=-2), x.unsqueeze(dim=-3)).mean(dim=-1).squeeze()

        # [batch_size, 2, min(n_points_1, n_points_2)]
        # Each element in the batch has results in a tensor of pairwise costs
        # Each tensor of pairwise costs has an optimal assignment, and this 
        # assignment is returned in terms of a list for rows and columns corresponding
        # to the optimal assignment.
        pairwise_cost_detached = pairwise_cost.detach().clone().cpu()
        indices = torch.tensor(
            list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_detached)), device=x.device)

        # [batch_size, min(n_points_1, n_points_2), 2]
        transposed_indices = torch.permute(indices, dims=(0, 2, 1))

        # VERY IMPORTANT NOTE & TODO: the following line works only if row indices returned by scipy optimize are SORTED!
        # o.w. it will break. Then the indices should be sorted in the order of row indices for it 
        # to work.
        actual_costs = torch.gather(pairwise_cost, 2, transposed_indices[:,:,1].unsqueeze(-1)).float()
        
        return torch.mean(torch.sum(actual_costs, dim=1)), transposed_indices