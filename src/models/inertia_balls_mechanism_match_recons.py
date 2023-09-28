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
        self.z_dim = self.hparams["z_dim"]

        # projection to latents. This module projects slots to the latent space of interest. These 
        # projections will be the targets of disentanglement.
        if not self.hparams.separate_projection_head:
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
        else:
            self.slot_projection_to_z = nn.ModuleList()
            for i in range(self.z_dim):
                # projection head is supposed to take slot_size dimensional vectors
                # to scalars to be concatenated and construct z.
                projection_head = torch.nn.Sequential(*[hydra.utils.instantiate(layer_config) for _, layer_config in self.hparams.projection_to_z.items()])
                self.slot_projection_to_z.append(projection_head)

        if kwargs.get("additional_logger"):
            self.additional_logger = hydra.utils.instantiate(self.hparams.additional_logger)
        else:
            self.additional_logger = None

        if kwargs.get("w_latent_loss", None) is not None: # because kwargs.get() value might be zero, but in that case we don't want
                                                          # the if clause to be executed!
            self.w_latent_loss = self.hparams.w_latent_loss
        else:
            self.w_latent_loss = 1.0
        if kwargs.get("w_recons_loss", None) is not None:
            self.w_recons_loss = self.hparams.w_recons_loss
        else:
            self.w_recons_loss = 1.0
        if kwargs.get("w_similarity_loss", None) is not None:
            self.w_similarity_loss = self.hparams.w_similarity_loss
        else:
            self.w_similarity_loss = 1.0

        # loss warmup parameters
        if kwargs.get("wait_steps", None) is not None:
            self.wait_steps = self.hparams.wait_steps
        else:
            self.wait_steps = 0
        if kwargs.get("linear_steps", None) is not None:
            self.linear_steps = self.hparams.linear_steps
        else:
            self.linear_steps = 1.0
        
        self.w_latent_loss_scheduled = 0.0

        if kwargs.get("ball_matching", None) is not None:
            self.matching = self.hparams.ball_matching
        if kwargs.get("latent_matching", None) is not None:
            self.latent_matching = self.hparams.latent_matching

        self.visualization_method = self.hparams.get("visualization_method", "2D")
        self.sparsity_degree = self.hparams.get("sparsity_degree", self.hparams["n_balls"])
        self.known_mechanism = self.hparams.get("known_mechanism", True)
        self.cvxpylayer = lp_solver_cvxpy(n_mechanisms=self.sparsity_degree, num_slots=self.hparams["encoder"]["num_slots"])


    def _load_messages(self):
        print(f"MSE\nBaseline: {self.baseline}")

    def loss(self, m_z1, z2):
        return F.mse_loss(m_z1, z2, reduction="mean")

    def forward(self, batch, num_slots=None, num_iterations=None, slots_init=None):

        # `num_slots` and `num_iterations` keywords let us try the model with a different number of slots and iterations
        # in each pass
        num_slots = num_slots if num_slots is not None else None
        num_iterations = num_iterations if num_iterations is not None else None

        # batch: batch of images -> [batch_size, num_channels, width, height]
        recon_combined, recons, masks, slots, attention_scores, slots_init = self.model(batch, num_slots=num_slots, num_iterations=num_iterations, slots_init=slots_init) # slots: [batch_size, num_slots, slot_size]

        # projection to target dimension
        if not self.hparams.separate_projection_head:
            slots_projected = self.slot_projection_to_z(slots) # slots_projected: [batch_size, num_slots, z_dim]
        else:
            slots_projected = []
            for i in range(self.z_dim):
                # slots has shape: [batch_size, num_slots, slot_size]
                projection_property = self.slot_projection_to_z[i](slots) # [batch_size, num_slots, 1]
                slots_projected.append(projection_property)
            slots_projected = torch.cat(slots_projected, dim=-1) # [batch_size, num_slots, z_dim]
            slots_projected.to(slots.device)

        return slots, slots_projected, recon_combined, recons, masks, attention_scores, slots_init


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

        import time
        t0 = time.time()
        (z1, z2), (x1, x2), (segmentation_masks1, segmentation_masks2), (A, b), mechanism_permutation, (coordinates_1, coordinates_2), (colors_1, colors_2) = batch["latents"], batch["images"], batch["segmentation_masks"], batch["matrices"], batch["mechanism_permutation"], batch["coordinates"], batch["colors"]
        # mask = b[:, 0] != 0
        # b[mask] = b[mask] * 3.
        # mask = b[:, 1] != 0
        # b[mask] = b[mask] * 3.
        # mask = b[:, 2] != 0
        # b[mask] = b[mask] * 3.
        # mask = b[:, 3] != 0
        # b[mask] = b[mask] * 3.
        # mask = b[:, 4] != 0
        # b[mask] = b[mask] * 3.
        # mask = b[:, 5] != 0
        # b[mask] = b[mask] * 3.
        # TODO: this is only for the wrong dataset
        # temporary code
        # if b.shape[1] >= 3:
        #     b[:, 2] = b[:, 2]/5
        #     if b.shape[1] >= 4:
        #         b[:, 3] = b[:, 3]/4
        # temporary code
        # if b.shape[1] < self.z_dim:
        #     true_z_dim = b.shape[1]
        #     self.true_z_dim = true_z_dim
        #     # make them the same dim so the summations are fine
        #     b = torch.cat((b, torch.zeros(b.shape[0],self.z_dim-b.shape[1]).to(b.device)), dim=-1).to(b.device)
        bs = x1.shape[0]
        device = x1.device
        img_width = x1.shape[-2]
        img_height = x1.shape[-1]
        # 1. pass x1,x2 to the encoder, and get slots, then project slots to the latent dimension z
        #    to obtain object_z1,object_z2 (note that some of the entries doesn't necessarily 
        #    correspond to objects as num_slots > n_objects)
        # 2. construct the cost matrix based on mechanisms being applied to projected slots and
        #    and the projections at 't+1', i.e. || mech(object_z1) - object_z2 ||^2
        # 3. get the assignments and reorder slots
        # 4. apply the mechanism (adding offset) only to that many slots that correspond to some mech.
        #    Alternatively sum the cost matrix over the minimum assignment. Return the resulting loss
        #    to be optimized.

        # 1
        # attention_scores: [batch_size, num_slots, height * width]
        # slots: [batch_size, num_slots, slot_size]
        # slots_projected: [batch_size, num_slots, z_dim] slots, slots_projected, recon_combined, recons, masks, attention_scores
        # masks: [batch_size, num_slots, width, height, 1]
        # t0 = time.perf_counter()
        slots_1, slots_projected_1, recon_combined_1, recons_1, masks_1, attention_scores_1, slots_init_1 = self(x1)
        slots_2, slots_projected_2, recon_combined_2, recons_2, masks_2, attention_scores_2, slots_init_2 = self(x2, slots_init=slots_init_1)
        # print(f"model fwd: {time.perf_counter()-t0}")
        n_slots = slots_1.size(1)

        # TODO: earlier experiments don't work now
        # if self.z_dim != 4: # if some constraint is satisfied
        #     # b: [batch_size, sparsity_degree*z_dim] -> [batch_size, sparsity_degree, z_dim] (sparsity_degree <= n_balls)
        #     b = b.squeeze().view(b.size(0), -1, self.z_dim)
        #     n_mechanisms = b.size(1) # equal to the number of balls or the sparsity degree if it is less than n_balls
        # else:
        # get b_true and b_hypothetical
        # b_true should be [batch_size, 1, z_dim] where only one entry at dim=-1 is non-zero
        # b should be [batch_size, z_dim * 2, z_dim], i.e. each hypothetical b has z_dim * 2 mechanisms that change
        # z_dim dimensions of the latent, but they change that via a one-hot mechanism
        # C = 0.05
        C = 0.1
        b_true = b.unsqueeze(1).clone() # [batch_size, 1, z_dim]
        if self.hparams["known_mechanism"]:
            b = b_true # [batch_size, 1, z_dim]
        else:
            if self.hparams["known_action"]: # we know what property is changing but don't know by how much
                if self.hparams["signed_change"]:
                    # b_base = torch.cat([(torch.abs(b_true)>0.), -1.0*(torch.abs(b_true)>0.)], dim=1) * C # [bs, 2, self.z_dim]
                    # b = b_base.to(device) # [bs, 2, self.z_dim]
                    b = (torch.sign(b_true) * C).to(device) # [bs, 1, z_dim]
                    # b[..., 2] = b[..., 2]
                    # b[..., 3] = b[..., 3]
                else:
                    b_base = (torch.abs(b_true)>0.) * C # [bs, 1, self.z_dim]
                    b = b_base.to(device) # [bs, 1, self.z_dim]
            else:
                if self.hparams["signed_change"]:
                    b_base = torch.cat([torch.eye(self.z_dim),-1.0*torch.eye(self.z_dim)],dim=0) * C # mechanisms are 'signed' (+/- C)
                    b = torch.tile(b_base.unsqueeze(0), (bs, 1, 1)).to(device) # [bs, self.z_dim * 2, self.z_dim]
                else:
                    b_base = torch.eye(self.z_dim) * C # mechanisms are NOT 'signed' (+C)
                    b = torch.tile(b_base.unsqueeze(0), (bs, 1, 1)).to(device) # [bs, self.z_dim, self.z_dim]
        n_mechanisms = b.size(1)

        n_balls = self.hparams["n_balls"]

        # 2, 3, 4
        # [batch_size, num_slots, num_mechanisms, z_dim]
        slots_projected_1_expanded = slots_projected_1.unsqueeze(2).repeat(1,1,n_mechanisms,1)
        slots_projected_2_expanded = slots_projected_2.unsqueeze(2).repeat(1,1,n_mechanisms,1)

        # all possible mechanisms applied to all possible slot projections
        # [batch_size, num_slots, num_mechanisms, z_dim]
        m_slots_projected_1_expanded = slots_projected_1_expanded + b.unsqueeze(1)

        if self.matching: # and stage == "train":
            if self.latent_matching == "constrained_lp":
                matching_loss, indices_1, indices_2 = self.compute_loss_and_reorder_constrained_lp(
                    slots_1, slots_2, slots_projected_1, slots_projected_2, attention_scores_1, attention_scores_2
            , m_slots_projected_1_expanded, slots_projected_2_expanded
                )
            elif self.latent_matching == "lin_sum_assignment":
                # [batch_size, num_mechanisms, num_slots, z_dim]
                slots_projected_1_expanded = slots_projected_1_expanded.permute(0,2,1,3)
                slots_projected_2_expanded = slots_projected_2_expanded.permute(0,2,1,3)
                m_slots_projected_1_expanded = m_slots_projected_1_expanded.permute(0,2,1,3)

                # Since we have initialized slots at t,t+1 the same way, 'most likely' the slots will be in the same order
                # , so we don't try to align them.

                matching_loss, indices_1, indices_2 = self.compute_loss_and_reorder_lin_sum_assignment(
                    slots_projected_1, attention_scores_1, m_slots_projected_1_expanded
                    , slots_projected_2_expanded
                )
            elif self.latent_matching == "argmin":
                # indices_1,2: [batch_size, 1(=sparsity degree), z_dim]
                # selected_mechanism_idx: [batch_size, 1]
                # matching_loss: scalar
                if self.hparams["known_action"]:
                    mask = (torch.abs(b)>0.).unsqueeze(2).repeat(1, 1, n_slots**2, 1) # [batch_size, n_mechanisms, n_slots^2, z_dim]
                else:
                    mask = None
                if self.hparams["double_matching"]:
                    matching_loss, indices_1, indices_2, selected_mechanism_idx = self.compute_loss_and_reorder_argmin_double_matching(
                        m_slots_projected_1_expanded, slots_projected_2_expanded, attention_scores_1, attention_scores_2
                        , changed_property_idx=mask
                    )
                else:
                    # [batch_size, num_mechanisms, num_slots, z_dim]
                    slots_projected_1_expanded = slots_projected_1_expanded.permute(0,2,1,3)
                    slots_projected_2_expanded = slots_projected_2_expanded.permute(0,2,1,3)
                    m_slots_projected_1_expanded = m_slots_projected_1_expanded.permute(0,2,1,3)
                    if self.hparams["known_action"]:
                        mask = (torch.abs(b)>0.).unsqueeze(2).repeat(1, 1, n_slots, 1) # [batch_size, n_mechanisms, n_slots, z_dim]
                    else:
                        mask = None
                    matching_loss, indices_1, indices_2, selected_mechanism_idx = self.compute_loss_and_reorder_argmin(
                        m_slots_projected_1_expanded, slots_projected_2_expanded, attention_scores_1, attention_scores_2
                        , changed_property_idx=mask
                    )
                    if self.trainer.global_step%20 == 0:
                        
                        # finding the slot corresponding to the changed ball. we need to find that slot both at t, t+1
                        segmentation_masks1_ = ((segmentation_masks1 > 0.)*1.).float() # [batch_size, n_balls+1, width, height, 1]
                        segmentation_masks2_ = ((segmentation_masks2 > 0.)*1.).float() # [batch_size, n_balls+1, width, height, 1]

                        # [batch_size, n_slots, n_slots]
                        # diff_1 = ((segmentation_masks1_.reshape(bs, n_slots, -1)[:, :, None] * masks_1.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)
                        # _, segmentation_slot_mapping_1 = torch.max(diff_1, dim=-1) # [batch_size, n_slots=n_segmentation_masks]
                        diff_1 = ((segmentation_masks1_.reshape(bs, n_slots, -1)[:, :, None] * masks_1.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)/(torch.norm(segmentation_masks1_.reshape(bs, n_slots, -1), p=2, dim=-1)[:, :, None] * torch.norm(masks_1.reshape(bs, n_slots, -1), p=2, dim=-1)[:, None, :])
                        indices = torch.tensor(
                        np.array(
                            list(
                                map(scipy.optimize.linear_sum_assignment, -diff_1.clone().detach().cpu().numpy()))
                                ),
                                device=segmentation_masks1.device
                        )
                        segmentation_slot_mapping_1 = indices[:, 1, :] # [batch_size, n_slots=n_segmentation_masks]
                        # [batch_size, 1]
                        ball_slot_idx_1 = torch.gather(segmentation_slot_mapping_1, -1, 1+mechanism_permutation.to(device)) # there is a +1 because we should skip the correspondence with the background mask

                        # diff_2 = ((segmentation_masks2_.reshape(bs, n_slots, -1)[:, :, None] * masks_2.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)
                        # _, segmentation_slot_mapping_2 = torch.max(diff_2, dim=-1) # [batch_size, n_slots=n_segmentation_masks]
                        diff_2 = ((segmentation_masks2_.reshape(bs, n_slots, -1)[:, :, None] * masks_2.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)/(torch.norm(segmentation_masks2_.reshape(bs, n_slots, -1), p=2, dim=-1)[:, :, None] * torch.norm(masks_2.reshape(bs, n_slots, -1), p=2, dim=-1)[:, None, :])
                        indices = torch.tensor(
                        np.array(
                            list(
                                map(scipy.optimize.linear_sum_assignment, -diff_2.clone().detach().cpu().numpy()))
                                ),
                                device=segmentation_masks2.device
                        )
                        segmentation_slot_mapping_2 = indices[:, 1, :] # [batch_size, n_slots=n_segmentation_masks]

                if self.hparams["known_mechanism"]:
                    if self.hparams["signed_change"]:
                        selected_mechanism_idx = (b.squeeze() != 0.).nonzero()[:, 1] + self.z_dim * (-torch.sign(torch.gather(b.squeeze(), 1, (b.squeeze()!=0).nonzero()[:,1].reshape(-1,1)))>0).reshape(-1)
                    else:
                        selected_mechanism_idx = (b.squeeze() != 0.).nonzero()[:, 1]
                else:
                    # retrieving true b
                    b = b_true.clone()
                    if self.hparams["known_action"]:
                        if self.hparams["signed_change"]:
                            selected_mechanism_idx = self.z_dim * selected_mechanism_idx + (torch.abs(b_true)>0.).squeeze().nonzero()[:, 1].reshape(-1,1)
                        else:
                            selected_mechanism_idx = (torch.abs(b_true)>0.).squeeze().nonzero()[:, 1]
            else:
                raise Exception("The provided latent matching procedure is not supported.")
            
            latent_loss = matching_loss
        else:
            # we know what ball and what property has changed. Only for debugging purposes

            # What we do here relies on segmentation_masks corresponding to the same 
            # objects at t,t+1, i.e., if at t segmentation_masks_1[0:2] correspond to
            # a cone, cylinder, and a cube, whatever change happens at t+1, the masks
            # should follow their corresponding changed object. The reason is that we
            # are using changed object index as a supervision signal, and it will not
            # work if the segmentation masks aren't coherent at t,t+1 (and of course
            # they should be coherent with the object numbering used when assigning 
            # the offsets.)

            # finding the slot corresponding to the changed ball. we need to find that slot both at t, t+1
            segmentation_masks1_ = ((segmentation_masks1 > 0.)*1.).float() # [batch_size, n_balls+1, width, height, 1]
            segmentation_masks2_ = ((segmentation_masks2 > 0.)*1.).float() # [batch_size, n_balls+1, width, height, 1]

            # [batch_size, n_slots, n_slots]
            diff_1 = ((segmentation_masks1_.reshape(bs, n_slots, -1)[:, :, None] * masks_1.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)
            _, segmentation_slot_mapping_1 = torch.max(diff_1, dim=-1) # [batch_size, n_slots=n_segmentation_masks]
            # [batch_size, 1]
            ball_slot_idx_1 = torch.gather(segmentation_slot_mapping_1, -1, 1+mechanism_permutation.to(device)) # there is a +1 because we should skip the correspondence with the background mask

            diff_2 = ((segmentation_masks2_.reshape(bs, n_slots, -1)[:, :, None] * masks_2.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)
            _, segmentation_slot_mapping_2 = torch.max(diff_2, dim=-1) # [batch_size, n_slots=n_segmentation_masks]
            # [batch_size, 1]
            ball_slot_idx_2 = torch.gather(segmentation_slot_mapping_2, -1, 1+mechanism_permutation.to(device)) # there is a +1 because we should skip the correspondence with the background mask

            # using the indices obtained, find the changed slot at t,t+1
            i1 = ball_slot_idx_1.unsqueeze(-1).repeat(1, 1, self.z_dim) # [batch_size, 1, z_dim]
            changed_ball_slot_projected_1 = torch.gather(slots_projected_1, 1, i1) # [batch_size, 1, z_dim]
            i2 = ball_slot_idx_2.unsqueeze(-1).repeat(1, 1, self.z_dim) # [batch_size, 1, z_dim]
            changed_ball_slot_projected_2 = torch.gather(slots_projected_2, 1, i2) # [batch_size, 1, z_dim]

            # picking the changed property from the changed slot projections and computing the loss

            # b: [batch_size, 1, z_dim] comes from known_action and unsigned offsets (unknown), ex: [0., 0., 0.1, 0.]
            mask = (torch.abs(b.squeeze())>0.) # boolean mask [batch_size, z_dim]
            m_changed_ball_slot_projected_1 = (changed_ball_slot_projected_1.squeeze()+b.squeeze())[mask] # [batch_size]
            cost = torch.nn.MSELoss(reduction="none")(m_changed_ball_slot_projected_1, changed_ball_slot_projected_2.squeeze()[mask]) # [batch_size]
            std_ = 0.5
            reg_lambda = 0.0
            variance_regularizer = 0.5*(((torch.std(changed_ball_slot_projected_1.view(-1, self.z_dim), dim=0)-std_*torch.ones((1, self.z_dim), device=device))**2).sum()+((torch.std(changed_ball_slot_projected_2.view(-1, self.z_dim), dim=0)-std_*torch.ones((1, self.z_dim), device=device))**2).sum())
            cost = cost + variance_regularizer * reg_lambda
            # computing the average of the ratio ||x2-x1||^2/||z2-z1||^2
            pixel_space_norm_2 = torch.linalg.norm(x2.permute(0,2,3,1)-x1.permute(0,2,3,1), dim=-1).sum(-1).sum(-1) # [batch_size]
            latent_space_norm_2 = torch.nn.MSELoss(reduction="none")(changed_ball_slot_projected_2.squeeze()[mask], changed_ball_slot_projected_1.squeeze()[mask]) # [batch_size]
            pixel_to_latent_space_norm_ratio = pixel_space_norm_2/latent_space_norm_2
            if self.trainer.global_step % 10 == 0:
                print(f"b:{b[:5]}")
                print(f"changed_ball_slot_projected_1:\n{changed_ball_slot_projected_1[:5]}")
                print(f"changed_ball_slot_projected_2:\n{changed_ball_slot_projected_2[:5]}")
                print(f"cost:\n{cost[:5]}")

            latent_loss = cost.mean()
            indices_1 = i1.clone()
            indices_2 = i2.clone()

            if self.hparams["known_mechanism"]:
                if self.z_dim > 1:
                    if self.hparams["signed_change"]:
                        # selected_mechanism_idx = self.z_dim * selected_mechanism_idx + (torch.abs(b_true)>0.).squeeze().nonzero()[:, 1].reshape(-1,1)
                        selected_mechanism_idx = (b.squeeze() != 0.).nonzero()[:, 1] + self.z_dim * (-torch.sign(torch.gather(b.squeeze(), 1, (b.squeeze()!=0).nonzero()[:,1].reshape(-1,1)))>0).reshape(-1)
                    else:
                        selected_mechanism_idx = (b.squeeze() != 0.).nonzero()[:, 1]
                else:
                    selected_mechanism_idx = torch.zeros(b.shape[0])
            else:
                # retrieving true b
                b = b_true.clone()
                if self.hparams["known_action"]:
                    if self.hparams["signed_change"]:
                        # selected_mechanism_idx = self.z_dim * selected_mechanism_idx + (torch.abs(b_true)>0.).squeeze().nonzero()[:, 1].reshape(-1,1)
                        selected_mechanism_idx = (b.squeeze() != 0.).nonzero()[:, 1] + self.z_dim * (-torch.sign(torch.gather(b.squeeze(), 1, (b.squeeze()!=0).nonzero()[:,1].reshape(-1,1)))>0).reshape(-1)
                    else:
                        try:
                            # selected_mechanism_idx = (torch.abs(b_true)>0.).squeeze().nonzero()[:, 1]
                            selected_mechanism_idx = (b.squeeze() != 0.).nonzero()[:, 1]
                        except: # for the case where z_dim=1
                            selected_mechanism_idx = (torch.abs(b_true)>0.).squeeze().nonzero()[:, 0]


        # [batch_size, n_mechanisms, z_dim]
        permutations_1 = indices_1
        permutations_2 = indices_2
        object_zhat_reordered_1 = torch.gather(slots_projected_1, 1, permutations_1) # [batch_size, n_balls, z_dim]
        object_zhat_reordered_2 = torch.gather(slots_projected_2, 1, permutations_2) # [batch_size, n_balls, z_dim]


        self.log(f"{stage}_latent_loss", latent_loss.item())

        # for mcc computation there are two methods when there are sparse changes:
        if self.hparams["use_all_balls_mcc"]:
            # finding the slot corresponding to the changed ball. we need to find that slot both at t, t+1
            segmentation_masks1_ = ((segmentation_masks1 > 0.)*1.).float() # [batch_size, n_balls+1, width, height, 1]
            segmentation_masks2_ = ((segmentation_masks2 > 0.)*1.).float() # [batch_size, n_balls+1, width, height, 1]

            # [batch_size, n_slots, n_slots]
            diff_1 = ((segmentation_masks1_.reshape(bs, n_slots, -1)[:, :, None] * masks_1.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)
            _, segmentation_slot_mapping_1 = torch.max(diff_1, dim=-1) # [batch_size, n_slots=n_segmentation_masks]
            # diff_1 = ((segmentation_masks1_.reshape(bs, n_slots, -1)[:, :, None] * masks_1.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)/(torch.norm(segmentation_masks1_.reshape(bs, n_slots, -1), p=2, dim=-1)[:, :, None] * torch.norm(masks_1.reshape(bs, n_slots, -1), p=2, dim=-1)[:, None, :])
            # indices = torch.tensor(
            # np.array(
            #     list(
            #         map(scipy.optimize.linear_sum_assignment, -diff_1.clone().detach().cpu().numpy()))
            #         ),
            #         device=segmentation_masks1.device
            # )
            # segmentation_slot_mapping_1 = indices[:, 1, :] # [batch_size, n_slots=n_segmentation_masks]
            objects_projected_slots_1 = torch.gather(slots_projected_1, 1, segmentation_slot_mapping_1[:, 1:].unsqueeze(-1).repeat(1, 1, self.z_dim))

            # background_slot_indices_one_hot_mask_1 = self.background_slot(attention_scores_1) # [batch_size, n_slots]
            # objects_projected_slots_1 = slots_projected_1[~background_slot_indices_one_hot_mask_1[0]].reshape(bs,n_balls,-1) # [batch_size, n_balls, z_dim]
            objects_projected_slots_1_flattened = objects_projected_slots_1.view(bs, -1)

            z_true = z1
            z_pred = objects_projected_slots_1_flattened

        else:
            # 2. find the indices of the balls that have changed and only pass those to be compared with
            # z_true
            if self.sparsity_degree < n_balls:
                z_true_mask = torch.abs(z2 - z1) > 0.
                z_true_mask = (torch.abs(z2.view(-1, n_balls, self.z_dim)-z1.view(-1, n_balls, self.z_dim)) > 0).sum(-1) > 0.
                z_true_1 = z1.view(bs, -1, self.z_dim)[z_true_mask].view(bs,-1)
                z_true_2 = z2.view(bs, -1, self.z_dim)[z_true_mask].view(bs,-1)
                # temporary code
                # z_true_mask = (torch.abs(z2.view(-1, n_balls, true_z_dim)-z1.view(-1, n_balls, true_z_dim)) > 0).sum(-1) > 0.
                # z_true_1 = z1.view(bs, -1, true_z_dim)[z_true_mask].view(bs,-1)
                # z_true_2 = z2.view(bs, -1, true_z_dim)[z_true_mask].view(bs,-1)

            else:
                z_true_1 = z1
                z_true_2 = z2
    
            z_true = z_true_2
            z_pred = object_zhat_reordered_2.view(bs,-1)

        if self.hparams["disentangle_z_dim"] < self.z_dim:
            disentangle_z_dim = self.hparams["disentangle_z_dim"]
            # z_true = z_true.reshape(bs, -1, self.z_dim)[...,:disentangle_z_dim].reshape(bs, -1)
            # z_pred = z_pred.reshape(bs, -1, self.z_dim)[...,:disentangle_z_dim].reshape(bs, -1)
            
            target_property_indices = self.hparams["target_property_indices"]
            z_true = z_true.reshape(bs, -1, self.z_dim)[..., target_property_indices].reshape(bs, -1)
            z_pred = z_pred.reshape(bs, -1, self.z_dim)[..., target_property_indices].reshape(bs, -1)
            # temporary code
            # z_true = z_true.reshape(bs, -1, true_z_dim)[..., target_property_indices].reshape(bs, -1)
            # z_pred = z_pred.reshape(bs, -1, self.z_dim)[..., target_property_indices].reshape(bs, -1)


        if self.hparams.get("pair_recons", False): # for colour disentanglement we should reconstruct both t,t+1
            reconstruction_losses = F.mse_loss(recon_combined_1, x1, reduction="mean") + F.mse_loss(recon_combined_2, x2, reduction="mean")
        else:
            reconstruction_losses = F.mse_loss(recon_combined_1, x1, reduction="mean") # + F.mse_loss(recon_combined_2, x2, reduction="mean")
        self.log(f"{stage}_reconstruction_loss", reconstruction_losses.item())

        # b: [batch_size, n_balls (n_mech), z_dim]
        baseline_loss = ((torch.norm(b, p=2, dim=-1).sum(dim=-1))**2).mean()
        self.log(f"{stage}_baseline_loss", baseline_loss.item())

        if self.z_dim == 1:
            # indices_1,2 were [batch_size, n_mechanisms, 1] before the following
            indices_1 = indices_1.repeat(1, 1, 2)
            indices_2 = indices_2.repeat(1, 1, 2)
            indices_1[:,:,0] = torch.arange(n_mechanisms)
            indices_2[:,:,0] = torch.arange(n_mechanisms)
        elif self.z_dim == 2:
            # indices: [batch_size, n_mechanisms, 2]
            indices_1[:,:,0] = torch.arange(n_mechanisms)
            indices_2[:,:,0] = torch.arange(n_mechanisms)
        elif self.z_dim >= 3:
            # indices: [batch_size, n_mechanisms, 2]
            indices_1 = indices_1[:, :, :2]
            indices_2 = indices_2[:, :, :2]
            indices_1[:,:,0] = 0 # there is only one mechanism
            indices_2[:,:,0] = 0

        # adjusting w_latent_loss according to the warmup schedule during training
        if stage == "train":
            if self.trainer.global_step < self.wait_steps:
                self.w_latent_loss_scheduled = 0.0
            elif self.wait_steps <= self.trainer.global_step < self.wait_steps + self.linear_steps:
                self.w_latent_loss_scheduled = self.w_latent_loss * (self.trainer.global_step - self.wait_steps) / self.linear_steps
            else:
                self.w_latent_loss_scheduled = self.w_latent_loss
            self.log("w_latent_loss", self.w_latent_loss_scheduled)
        
        # if the encoder is frozen, the reconstruction cannot be affected in anyway, so it shouldn't be computed and be backpropagated through
        # there would also be no weight associated since it is only one loss
        if self.hparams.encoder_freeze:
            loss = self.w_latent_loss_scheduled * latent_loss
        else:
            loss = self.w_latent_loss_scheduled * latent_loss + self.w_recons_loss * reconstruction_losses # + self.w_similarity_loss * slot_similarity_loss # + 0.01 * ((1-sigma_1_prod) ** 2)
        # loss = reconstruction_losses

        self.log(f"{stage}_loss", loss.item())

        # computing ARI scores (assumes n_balls+1 segmentation masks and predictions, so background is included as well)
        # compute ARI only on validation and test dataset as it slows down the pipiline
        # if stage != "train":
        segmentation_masks1_ = segmentation_masks1.squeeze().reshape(bs, n_balls+1, -1).permute(0, 2, 1)
        segmentation_masks1_one_hot = F.one_hot(torch.argmax(segmentation_masks1_, -1))

        # TODO: This is not correct, it's not the case that there are exactly this number of background masks!
        # the rest can be duplicates for the already found balls

        # prediction masks are of shape [batch_size, num_slots, width, height, 1], so if num_slots > n_balls+1, then we should
        # choose n_mechanisms+1 of the masks as predictions to compute the ARI. Our way is to compute the std of the masks, and
        # select the top n_balls with highest std as masks corresponding to balls, and the mask with lowest std for the background.
        masks1_sorted_desc, masks_1_sorted_index_desc = torch.sort(torch.std(masks_1.reshape(bs, n_slots, -1), dim=-1), -1, descending=True)
        masks_balls_index = masks_1_sorted_index_desc[:, :n_balls]
        masks1_sorted_asc, masks_1_sorted_index_asc = torch.sort(torch.std(masks_1.reshape(bs, n_slots, -1), dim=-1), -1, descending=False)
        masks_background_index = masks_1_sorted_index_asc[:, 0].unsqueeze(-1)
        ball_n_background_idx = torch.cat((masks_balls_index, masks_background_index), dim=-1).unsqueeze(-1).repeat(1, 1, img_width*img_height)

        # [bs, n_balls+1, width * height]
        pred_segmentation_masks1 = torch.gather(masks_1.reshape(bs, n_slots, -1), 1, ball_n_background_idx)
        # pred_segmentation_masks1 = pred_segmentation_masks1.reshape(bs, n_balls+1, img_width, img_height, -1)
        pred_segmentation_masks1 = pred_segmentation_masks1.permute(0, 2, 1) # [bs, width * height, n_balls+1]
        # pred_segmentation_masks1 = pred_segmentation_masks1.squeeze().reshape(bs, n_balls+1, -1).permute(0, 2, 1)
        pred_segmentation_masks1_one_hot = F.one_hot(torch.argmax(pred_segmentation_masks1, -1))

        ari = adjusted_rand_index(segmentation_masks1_one_hot, pred_segmentation_masks1_one_hot).mean()
        self.log(f"{stage}_ARI", ari)

        # TODO: might require detaching and cloning tensors before sending them out to logger
        self.eval()
        with torch.no_grad():
            if self.additional_logger:
                train = True if stage == "train" else False
                self.log_reconstructed_samples(
                    batch_1=x1,
                    z1=z1,
                    coordinates_1=coordinates_1,
                    attention_masks_1=attention_scores_1,
                    recon_combined_1=recon_combined_1, recons_1=recons_1, masks_1=masks_1,
                    batch_2=x2,
                    z2=z2,
                    coordinates_2=coordinates_2,
                    attention_masks_2=attention_scores_2,
                    recon_combined_2=recon_combined_2, recons_2=recons_2, masks_2=masks_2,
                    b=b,
                    mechanism_permutation=mechanism_permutation,
                    selected_mechanism_idx=selected_mechanism_idx,
                    colors_1=colors_1,
                    colors_2=colors_2,
                    indices_1=indices_1,
                    indices_2=indices_2,
                    # pairwise_cost=pairwise_cost,
                    pairwise_cost=None,
                    num_slots=n_slots,
                    table_name=f"{stage}/Masks_and_Reconstructions_{self.global_step}",
                    # num_samples_to_log=3,
                    train=train,
                )
        self.train()

        if stage == "train":
            return {"loss": loss}
        else:
            return {"loss": loss, "true_z": z_true, "pred_z": z_pred.detach()}


    def compute_loss_and_reorder_constrained_lp(
        self, slots_1, slots_2, slots_projected_1, slots_projected_2, attention_scores_1, attention_scores_2
        , m_slots_projected_1_expanded, slots_projected_2_expanded
    ):
        """
        m_slots_projected_1_expanded: all possible mechanisms applied to all possible slot projections
            [batch_size, num_slots, num_mechanisms, z_dim]
        
        slots_projected_2_expanded:
            [batch_size, num_slots, num_mechanisms, z_dim]
        """
        
        bs = m_slots_projected_1_expanded.shape[0]
        n_slots = m_slots_projected_1_expanded.shape[1]
        n_mechanisms = m_slots_projected_1_expanded.shape[2]
        z_dim = m_slots_projected_1_expanded.shape[-1]
        device = slots_1.device
        
        # We need to manipulate m_slots_projected_1_expanded and slots_projected_2_expanded so they align and result in 
        # the computation of all possible combinations of m_k(s_t^i) and s_{t+1}^j.

        # [batch_size, n_mechanisms, n_slots^2, z_dim]
        m_slots_projected_1_expanded = m_slots_projected_1_expanded.permute(0,2,1,3).repeat(1, 1, 1, n_slots).reshape(bs, n_mechanisms, -1, z_dim)
        slots_projected_2_expanded = slots_projected_2_expanded.permute(0,2,1,3).repeat(1, 1, n_slots, 1)


        # find the duplicate slots (their indices) based on projected slots
        # duplicate_threshold = 0.998
        # slots_1_dot_prod = torch.bmm(slots_projected_1, slots_projected_1.transpose(2, 1))
        # norms_1 = torch.norm(slots_projected_1, p=2, dim=-1) # [batch_size, num_slots]
        # product_of_norms_1 = torch.bmm(norms_1.unsqueeze(-1), norms_1.unsqueeze(-1).transpose(2,1))
        # slots_1_xcorr_normalized = slots_1_dot_prod/product_of_norms_1
        # slots_1_xcorr_detached_normalized = slots_1_xcorr_normalized.clone().detach().cpu().numpy()
        # duplicate_slots_1_mask = slots_1_xcorr_normalized > duplicate_threshold # [batch_size, num_slots, num_slots]

        # slots_2_dot_prod = torch.bmm(slots_projected_2, slots_projected_2.transpose(2, 1))
        # norms_2 = torch.norm(slots_projected_2, p=2, dim=-1) # [batch_size, num_slots]
        # product_of_norms_2 = torch.bmm(norms_2.unsqueeze(-1), norms_2.unsqueeze(-1).transpose(2,1))
        # slots_2_xcorr_normalized = slots_2_dot_prod/product_of_norms_2
        # slots_2_xcorr_detached_normalized = slots_2_xcorr_normalized.clone().detach().cpu().numpy()
        # duplicate_slots_2_mask = slots_2_xcorr_normalized > duplicate_threshold # [batch_size, num_slots, num_slots]

        # # create the mask of indices in the [n_mechanism by (n_slots ** 2)] cost matrix that have to
        # # be set to a high number to prevent a duplicate slot from being selected when solving the LP.
        # duplicate_mask_1 = torch.triu(duplicate_slots_1_mask).sum(-2)>1
        # duplicate_mask_1 = duplicate_mask_1.unsqueeze(1).repeat(1,n_slots,1).permute(0, 2, 1)
        # duplicate_mask_1 = duplicate_mask_1.unsqueeze(1).repeat(1,n_mechanisms,1,1).view(bs, n_mechanisms, -1)

        # duplicate_mask_2 = torch.triu(duplicate_slots_2_mask).sum(-2)>1
        # duplicate_mask_2 = duplicate_mask_2.unsqueeze(1).repeat(1,n_mechanisms,n_slots)
        
        # find the duplicate slots (their indices) based on the slots themselves
        # duplicate_threshold = 0.9
        # slots_1_dot_prod = torch.bmm(slots_1, slots_1.transpose(2, 1))
        # norms_1 = torch.norm(slots_1, p=2, dim=-1) # [batch_size, num_slots]
        # product_of_norms_1 = torch.bmm(norms_1.unsqueeze(-1), norms_1.unsqueeze(-1).transpose(2,1))
        # slots_1_xcorr_normalized = slots_1_dot_prod/product_of_norms_1
        # slots_1_xcorr_detached_normalized = slots_1_xcorr_normalized.clone().detach().cpu().numpy()
        # duplicate_slots_1_mask = slots_1_xcorr_normalized > duplicate_threshold # [batch_size, num_slots, num_slots]

        # slots_2_dot_prod = torch.bmm(slots_2, slots_2.transpose(2, 1))
        # norms_2 = torch.norm(slots_2, p=2, dim=-1) # [batch_size, num_slots]
        # product_of_norms_2 = torch.bmm(norms_2.unsqueeze(-1), norms_2.unsqueeze(-1).transpose(2,1))
        # slots_2_xcorr_normalized = slots_2_dot_prod/product_of_norms_2
        # slots_2_xcorr_detached_normalized = slots_2_xcorr_normalized.clone().detach().cpu().numpy()
        # duplicate_slots_2_mask = slots_2_xcorr_normalized > duplicate_threshold # [batch_size, num_slots, num_slots]

        # # create the mask of indices in the [n_mechanism by (n_slots ** 2)] cost matrix that have to
        # # be set to a high number to prevent a duplicate slot from being selected when solving the LP.
        # duplicate_mask_1 = torch.triu(duplicate_slots_1_mask).sum(-2)>1
        # duplicate_mask_1 = duplicate_mask_1.unsqueeze(1).repeat(1,n_slots,1).permute(0, 2, 1)
        # duplicate_mask_1 = duplicate_mask_1.unsqueeze(1).repeat(1,n_mechanisms,1,1).view(bs, n_mechanisms, -1)

        # duplicate_mask_2 = torch.triu(duplicate_slots_2_mask).sum(-2)>1
        # duplicate_mask_2 = duplicate_mask_2.unsqueeze(1).repeat(1,n_mechanisms,n_slots)

        # create the mask of indices in the [n_mechanism by (n_slots ** 2)] cost matrix that have to
        # be set to a high number to prevent background slots from being selected when solving the LP.

        pairwise_cost = torch.nn.HuberLoss(reduction="none")(m_slots_projected_1_expanded, slots_projected_2_expanded).mean(dim=-1)

        # it is modified to manipulate the matching, but for actual computations of the loss, we need the differentiable unaltered version
        # of the costs
        pairwise_cost_temp = pairwise_cost.clone()

        inf_ = 500.0 # also the solution of lp is very sensitive to this value

        # ABSOLUTELY CRITICAL FOR THE SOLVER TO WORK PROPERLY! (float and double issue with precisions)
        v, i = torch.min(pairwise_cost_temp, dim=-1, keepdim=True)
        pairwise_cost_temp /= v # .min()

        # pairwise_cost_temp[duplicate_mask_1] = inf_ # v.max() # v.repeat(1, 1, n_slots ** 2)[duplicate_mask_1]
        # pairwise_cost_temp[duplicate_mask_2] = inf_ # v.max() # v.repeat(1, 1, n_slots ** 2)[duplicate_mask_2]

        # if background slots should be removed from the matching procedure
        if self.hparams["rm_background_in_matching"]:
            background_masks_1 = self.background_slot(attention_scores_1)
            background_slot_mask_1 = torch.zeros_like(background_masks_1[0]).bool().to(device)
            for bg_mask in background_masks_1:
                background_slot_mask_1 = torch.logical_or(background_slot_mask_1, bg_mask)
            
            background_slot_mask_1 = background_slot_mask_1.unsqueeze(1)
            background_slot_mask_1 = background_slot_mask_1.repeat(1,n_slots,1).permute(0, 2, 1).repeat(1,n_mechanisms,1).view(bs, n_mechanisms, -1)

            background_masks_2 = self.background_slot(attention_scores_2)
            background_slot_mask_2 = torch.zeros_like(background_masks_2[0]).bool().to(device)
            for bg_mask in background_masks_2:
                background_slot_mask_2 = torch.logical_or(background_slot_mask_2, bg_mask)
            
            background_slot_mask_2 = background_slot_mask_2.unsqueeze(1)
            background_slot_mask_2 = background_slot_mask_2.repeat(1,n_mechanisms,n_slots).view(bs, n_mechanisms, -1)

            pairwise_cost_temp[background_slot_mask_1] = inf_
            pairwise_cost_temp[background_slot_mask_2] = inf_

        # The old way in which only one background is selected.
        # background_slot_indices_1 = torch.argmin(torch.std(attention_scores_1, dim=-1), dim=-1, keepdim=True)
        # background_slot_indices_2 = torch.argmin(torch.std(attention_scores_2, dim=-1), dim=-1, keepdim=True)
        # background_slot_mask_1 = torch.nn.functional.one_hot(background_slot_indices_1, num_classes=n_slots).bool() # [bs, n_slots]
        # background_slot_mask_1 = background_slot_mask_1.repeat(1,n_slots,1).permute(0, 2, 1).repeat(1,n_mechanisms,1).view(bs, n_mechanisms, -1)
        # background_slot_mask_2 = torch.nn.functional.one_hot(background_slot_indices_2, num_classes=n_slots).bool()
        # background_slot_mask_2 = background_slot_mask_2.repeat(1,n_mechanisms,n_slots).view(bs, n_mechanisms, -1)


        pairwise_cost_temp[pairwise_cost_temp > inf_] = inf_
        # solving the batch of lps
        # solution (assignment weights): [batch_size, n_mechanisms, num_slots ** 2]
        solution = self.cvxpylayer(pairwise_cost_temp.clone().detach(), solver_args={'max_iters':10000})[0]
        w = solution.clone()
        w[w < 0.0] = 0.0
        m = torch.distributions.Categorical(w)
        bs = solution.shape[0]
        indices = m.sample()
        indices = torch.tensor(indices.clone().detach(), device=pairwise_cost.device) # drops any gradients, we don't need them
        indices_1 = torch.div(indices, n_slots, rounding_mode='floor')
        indices_2 = indices%n_slots

        # making sure that no slot is selected for two mechanisms when sampled from the categorical distribution given by the solution of lps
        mask = torch.ones(indices.size())
        for i in range(bs):
                _, counts_1 = torch.unique(indices_1[i], sorted=True, dim=-1, return_counts=True)
                _, counts_2 = torch.unique(indices_2[i], sorted=True, dim=-1, return_counts=True)
                if len(counts_1) < n_mechanisms or len(counts_2) < n_mechanisms:
                    mask[i, :] = 0.
                resample_mask = ~ (mask > 0.)

        counter = 0
        while resample_mask.any() and counter <= 10:
            counter += 1
            indices_ = m.sample()
            indices[resample_mask] = indices_[resample_mask]
            indices_1 = torch.div(indices, n_slots, rounding_mode='floor')
            indices_2 = indices%n_slots
            mask = torch.ones(indices.size())
            for i in range(bs):
                _, counts_1 = torch.unique(indices_1[i], sorted=True, dim=-1, return_counts=True)
                _, counts_2 = torch.unique(indices_2[i], sorted=True, dim=-1, return_counts=True)
                if len(counts_1) < n_mechanisms or len(counts_2) < n_mechanisms:
                    mask[i, :] = 0.
            resample_mask = ~ (mask > 0.)

        indices_1 = indices_1.unsqueeze(-1).repeat(1, 1, z_dim)
        indices_2 = indices_2.unsqueeze(-1).repeat(1, 1, z_dim)

        # [batch_size, n_mechanisms]
        actual_costs = torch.gather(pairwise_cost, 2, indices.unsqueeze(-1)).float()
        # hungarian_loss: scalar
        hungarian_loss = torch.mean(torch.sum(actual_costs, dim=1))

        return hungarian_loss, indices_1, indices_2


    def compute_loss_and_reorder_lin_sum_assignment(
        self, slots_projected_1, attention_scores_1
        , m_slots_projected_1_expanded, slots_projected_2_expanded):

        """
        Args:
        m_slots_projected_1_expanded: all possible mechanisms applied to all possible slot projections
            [batch_size, n_mechanisms, n_slots, z_dim]
        
        slots_projected_2_expanded:
            [batch_size, n_mechanisms, n_slots, z_dim]
        """

        predictions = m_slots_projected_1_expanded
        targets = slots_projected_2_expanded
        device = predictions.device
        n_slots = m_slots_projected_1_expanded.shape[2]
        n_mechanisms = m_slots_projected_1_expanded.shape[1]
        device = attention_scores_1.device

        # find the duplicate slots
        duplicate_threshold = 0.998
        slots_1_dot_prod = torch.bmm(slots_projected_1, slots_projected_1.transpose(2, 1))
        norms_1 = torch.norm(slots_projected_1, p=2, dim=-1) # [batch_size, num_slots]
        product_of_norms_1 = torch.bmm(norms_1.unsqueeze(-1), norms_1.unsqueeze(-1).transpose(2,1))
        slots_1_xcorr_normalized = slots_1_dot_prod/product_of_norms_1
        slots_1_xcorr_detached_normalized = slots_1_xcorr_normalized.clone().detach().cpu().numpy()
        duplicate_slots_1_mask = slots_1_xcorr_normalized > duplicate_threshold # [batch_size, num_slots, num_slots]

        # create the mask of indices in the [n_mechanism x n_slots] cost matrix that have to
        # be set to a high number to prevent a duplicate slot from being selected when solving the LP.
        duplicate_mask_1 = torch.triu(duplicate_slots_1_mask).sum(-2)>1
        duplicate_mask_1 = duplicate_mask_1.unsqueeze(1).repeat(1,n_mechanisms,1)
        
        # create the mask of indices in the [n_mechanism x n_slots] cost matrix that have to
        # be set to a high number to prevent background slots from being selected when solving the LP.
        
        # ------------
        # TODO: This can be done for both t,t+1 for instance in the constrained lp setting
        # There can be more than 1 background slot, so we have to find the n_slots-n_balls masks with the least std and 
        # don't let them to be selected by the LP solution
        background_masks1_sorted_asc, background_masks_1_sorted_index_asc = torch.sort(torch.std(attention_scores_1, dim=-1), -1, descending=False)
        background_masks = []
        # TODO: This is not correct, it's not the case that there are exactly this number of background masks!
        # the rest can be duplicates for the already found balls
        for i in range(n_slots-n_mechanisms):
            background_slot_indices = background_masks_1_sorted_index_asc[:, i]
            background_slot_mask = torch.nn.functional.one_hot(background_slot_indices, num_classes=n_slots).bool() # [bs, n_slots]
            background_masks.append(background_slot_mask)

        background_slot_mask_1 = torch.zeros_like(background_masks[0]).bool().to(device)
        for bg_mask in background_masks:
            background_slot_mask_1 = torch.logical_or(background_slot_mask_1, bg_mask)
        
        background_slot_mask_1 = background_slot_mask_1.unsqueeze(1)
        background_slot_mask_1 = background_slot_mask_1.repeat(1,n_mechanisms,1)
        # ------------
        # background_slot_indices_1 = torch.argmin(torch.std(attention_scores_1, dim=-1), dim=-1, keepdim=True)
        # background_slot_mask_1 = torch.nn.functional.one_hot(background_slot_indices_1, num_classes=n_slots).bool()
        # background_slot_mask_1 = background_slot_mask_1.repeat(1,n_mechanisms,1)
        
        # TOOD: This loss can be a MSE loss instead of Huber loss
        # [batch_size, n_mechanisms, n_slots]
        pairwise_cost = torch.nn.HuberLoss(reduction="none")(predictions, targets).mean(dim=-1)

        # Each element in the batch has results in a tensor of pairwise costs
        # Each tensor of pairwise costs has an optimal assignment, and this 
        # assignment is returned in terms of a list for rows and columns corresponding
        # to the optimal assignment.
        pairwise_cost_detached = pairwise_cost.clone().detach().cpu()

        inf_ = 500.0
        # pairwise_cost_detached[duplicate_mask_1] = inf_
        pairwise_cost_detached[background_slot_mask_1] = inf_
        pairwise_cost_detached[pairwise_cost_detached > inf_] = inf_

        # [batch_size, 2, n_mechanisms]; 2 is because there's a list specifying the row of optimal assignments, and
        # one list specifying the column of optimal assignments. example:
        # indices[0]: tensor([[0, 1, 2, 3], [3, 0, 4, 1]], device='cuda:0')
        indices = torch.tensor(
            np.array(
                list(
                    map(scipy.optimize.linear_sum_assignment, pairwise_cost_detached))
                    ),
                    device=device
            )
            
        # [batch_size, n_mechanisms, 2]
        transposed_indices = torch.permute(indices, dims=(0, 2, 1))
        # [batch_size, n_mechanisms]
        actual_costs = torch.gather(pairwise_cost, 2, transposed_indices[:,:,1].unsqueeze(-1)).float()
        
        # [batch_size, n_mechanisms, z_dim]
        indices_1 = transposed_indices[:,:,1].unsqueeze(-1).repeat(1, 1, self.z_dim)
        indices_2 = transposed_indices[:,:,1].unsqueeze(-1).repeat(1, 1, self.z_dim)
        # hungarian_loss: scalar
        hungarian_loss = torch.mean(torch.sum(actual_costs, dim=1))

        return hungarian_loss, indices_1, indices_2


    def compute_loss_and_reorder_argmin_double_matching(
        self, m_slots_projected_1_expanded, slots_projected_2_expanded
        , attention_scores_1=None, attention_scores_2=None, changed_property_idx=None
    ):
        """
        m_slots_projected_1_expanded: all possible mechanisms applied to all possible slot projections
            [batch_size, num_slots, num_mechanisms, z_dim]
        
        slots_projected_2_expanded:
            [batch_size, num_slots, num_mechanisms, z_dim]

        changed_property_idx:
            [batch_size, num_mechanisms, num_slots^2, z_dim]
        """
        
        bs = m_slots_projected_1_expanded.shape[0]
        n_slots = m_slots_projected_1_expanded.shape[1]
        n_mechanisms = m_slots_projected_1_expanded.shape[2]
        z_dim = m_slots_projected_1_expanded.shape[-1]
        device = m_slots_projected_1_expanded.device
        
        # We need to manipulate m_slots_projected_1_expanded and slots_projected_2_expanded so they align and result in 
        # the computation of all possible combinations of m_k(s_t^i) and s_{t+1}^j.

        # [batch_size, n_mechanisms, n_slots^2, z_dim]
        m_slots_projected_1_expanded = m_slots_projected_1_expanded.permute(0,2,1,3).repeat(1, 1, 1, n_slots).reshape(bs, n_mechanisms, -1, z_dim)
        slots_projected_2_expanded = slots_projected_2_expanded.permute(0,2,1,3).repeat(1, 1, n_slots, 1)

        if self.hparams["known_action"]:
            # [batch_size, num_slots^2]
            pairwise_cost = torch.nn.HuberLoss(reduction="none")(m_slots_projected_1_expanded[changed_property_idx], slots_projected_2_expanded[changed_property_idx]).view(bs, -1)
        else:
            pairwise_cost = torch.nn.HuberLoss(reduction="none")(m_slots_projected_1_expanded[changed_property_idx], slots_projected_2_expanded[changed_property_idx]).mean(dim=-1)

        # if background slots should be removed from the matching procedure
        if self.hparams["rm_background_in_matching"]:
            background_masks_1 = self.background_slot(attention_scores_1)
            background_slot_mask_1 = torch.zeros_like(background_masks_1[0]).bool().to(device)
            for bg_mask in background_masks_1:
                background_slot_mask_1 = torch.logical_or(background_slot_mask_1, bg_mask) # [batch_size, num_slots]

            background_slot_mask_1 = background_slot_mask_1.unsqueeze(1) # [batch_size, 1, num_slost]
            background_slot_mask_1 = background_slot_mask_1.repeat(1,n_slots,1).permute(0, 2, 1).repeat(1,n_mechanisms,1).view(bs, n_mechanisms, -1) # [batch_size, 1, num_slost^2]

            background_masks_2 = self.background_slot(attention_scores_2)
            background_slot_mask_2 = torch.zeros_like(background_masks_2[0]).bool().to(device)
            for bg_mask in background_masks_2:
                background_slot_mask_2 = torch.logical_or(background_slot_mask_2, bg_mask)
            
            background_slot_mask_2 = background_slot_mask_2.unsqueeze(1)
            background_slot_mask_2 = background_slot_mask_2.repeat(1,n_mechanisms,n_slots).view(bs, n_mechanisms, -1) # [batch_size, 1, num_slost^2]

            inf_ = 10.0
            # print(f"3:background_slot_mask_1\n{background_slot_mask_1[:2]}")
            # print(f"4:background_slot_mask_2\n{background_slot_mask_2[:2]}")
            pairwise_cost[background_slot_mask_1.squeeze()] = inf_
            pairwise_cost[background_slot_mask_2.squeeze()] = inf_

        min_cost, min_idx = torch.min(pairwise_cost, dim=-1, keepdim=True) # [batch_size, 1]

        mechanism_idx = torch.div(min_idx, n_slots ** 2, rounding_mode='floor')
        slot_pair_idx = torch.fmod(min_idx, n_slots ** 2)

        slot_pair_idx = torch.tensor(slot_pair_idx.clone().detach(), device=device) # drops any gradients, we don't need them
        indices_1 = torch.div(slot_pair_idx, n_slots, rounding_mode='floor')
        indices_2 = slot_pair_idx%n_slots

        indices_1 = indices_1.unsqueeze(-1).repeat(1, 1, self.z_dim)
        indices_2 = indices_2.unsqueeze(-1).repeat(1, 1, self.z_dim)

        # matching_loss: scalar
        matching_loss = torch.mean(min_cost)

        return matching_loss, indices_1, indices_2, mechanism_idx
        
    def compute_loss_and_reorder_argmin(
        self, m_slots_projected_1_expanded, slots_projected_2_expanded
        , attention_scores_1=None, attention_scores_2=None, changed_property_idx=None
    ):
        """
        Args:
        m_slots_projected_1_expanded: all possible mechanisms applied to all possible slot projections
            [batch_size, n_mechanisms, n_slots, z_dim]
        
        slots_projected_2_expanded:
            [batch_size, n_mechanisms, n_slots, z_dim]

        changed_property_idx:
            [batch_size, n_mechanism, n_slots]
        """

        bs = m_slots_projected_1_expanded.shape[0]
        n_slots = m_slots_projected_1_expanded.shape[2]
        n_mechanisms = m_slots_projected_1_expanded.shape[1]
        z_dim = m_slots_projected_1_expanded.shape[-1]
        device = m_slots_projected_1_expanded.device
        
        if self.hparams["known_action"]:
            # [batch_size, n_mechanisms, n_slots]
            # pairwise_cost = torch.nn.HuberLoss(reduction="none")(m_slots_projected_1_expanded[changed_property_idx].reshape(bs, n_mechanisms, n_slots)
            pairwise_cost = torch.nn.MSELoss(reduction="none")(m_slots_projected_1_expanded[changed_property_idx].view(bs, n_mechanisms, n_slots)
            , slots_projected_2_expanded[changed_property_idx].view(bs, n_mechanisms, n_slots))
        else:
            # pairwise_cost = torch.nn.HuberLoss(reduction="none")(m_slots_projected_1_expanded, slots_projected_2_expanded).mean(dim=-1)
            pairwise_cost = torch.nn.MSELoss(reduction="none")(m_slots_projected_1_expanded, slots_projected_2_expanded).mean(dim=-1)

        # if background slots should be removed from the matching procedure
        if self.hparams["rm_background_in_matching"]:
            background_masks_1 = self.background_slot(attention_scores_1)
            background_slot_mask_1 = torch.zeros_like(background_masks_1[0]).bool().to(device)
            for bg_mask in background_masks_1:
                background_slot_mask_1 = torch.logical_or(background_slot_mask_1, bg_mask)
            
            background_slot_mask_1 = background_slot_mask_1.unsqueeze(1)
            background_slot_mask_1 = background_slot_mask_1.repeat(1,n_mechanisms,1) # [batch_size, n_mechanisms, n_slots]

            
            inf_ = 10.0
            pairwise_cost[background_slot_mask_1] = inf_

        min_cost, min_idx = torch.min(pairwise_cost.view(bs, -1), dim=-1, keepdim=True) # [batch_size, 1]

        mechanism_idx = torch.div(min_idx, n_slots, rounding_mode='floor')
        slot_pair_idx = torch.fmod(min_idx, n_slots) # [batch_size, 1]
        slot_pair_idx = torch.tensor(slot_pair_idx.clone().detach(), device=device) # drops any gradients, we don't need them
        indices_1 = slot_pair_idx
        indices_2 = slot_pair_idx

        indices_1 = indices_1.unsqueeze(-1).repeat(1, 1, self.z_dim)
        indices_2 = indices_2.unsqueeze(-1).repeat(1, 1, self.z_dim)

        # matching_loss: scalar
        matching_loss = torch.mean(min_cost)

        return matching_loss, indices_1, indices_2, mechanism_idx
        
    def compute_loss_and_reorder(self, predictions, targets, duplicate_mask_1, duplicate_mask_2):
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
        
            x: Batch of sets of size [batch_size, n_mechanisms, n_slots^2, dim_points]. (dim_points=2)
            y: Batch of sets of size [batch_size, n_mechanisms, n_slots^2, dim_points].
            duplicate_mask_1: A boolean mask of positions that should be set to a high number so
            no duplicate slot is picked at time t.
                [batch_size, n_mechanisms, n_slots^2]
            duplicate_mask_2: A boolean mask of positions that should be set to a high number so
            no duplicate slot is picked at time t+1.
                [batch_size, n_mechanisms, n_slots^2]
        Returns:
            Minimum distance between all sets in the two batches.
        """

        x = predictions
        y = targets
        device = x.device

        # TOOD: This loss can be a MSE loss instead of Huber loss
        # [batch_size, n_mechanisms, n_slots^2]
        pairwise_cost = torch.nn.HuberLoss(reduction="none")(x, y).mean(dim=-1)

        # replacing the indices that should not be selected by a high value
        pairwise_cost[duplicate_mask_1] = 100.0
        pairwise_cost[duplicate_mask_2] = 100.0


        # Each element in the batch has results in a tensor of pairwise costs
        # Each tensor of pairwise costs has an optimal assignment, and this 
        # assignment is returned in terms of a list for rows and columns corresponding
        # to the optimal assignment.
        pairwise_cost_detached = pairwise_cost.clone().detach().cpu()

        # now solve the LP
        indices = np.array(
        list(
            map(lp_solver, pairwise_cost_detached)),
            dtype=int
            )

        indices = torch.tensor(indices, device=device)

        # [batch_size, n_mechanisms, 2]
        transposed_indices = torch.permute(indices, dims=(0, 2, 1))
        # transposed_indices = torch.permute(indices, dims=(0, 2, 1))
        transposed_indices = indices

        # [batch_size, n_mechanisms]
        actual_costs = torch.gather(pairwise_cost, 2, transposed_indices[:,:,1].unsqueeze(-1)).float()

        # TODO: We shouldn't remove the one with lower loss, but we don't know that, we can remove those that aren't the min
        # but let's forget that for now, just remove the duplicates.

        return torch.mean(torch.sum(actual_costs, dim=1)), transposed_indices, pairwise_cost


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


    def slots_cosine_dist(self, slots):
        # slots: [num_slots, slot_size]
        num_slots = slots.size(0)
        dot_products = [sum(slots[i] * slots[j])/(torch.norm(slots[i], 2)*torch.norm(slots[j], 2)) for i,j in product(range(num_slots),range(num_slots))]
        diag_products = [sum(slots[i] * slots[i])/(torch.norm(slots[i], 2)**2) for i in range(num_slots)]
        non_diag = sum(dot_products) - sum(diag_products)
        return non_diag


    @rank_zero_only
    def log_reconstructed_samples(
        self, batch_1, z1, coordinates_1, attention_masks_1,
        recon_combined_1, recons_1, masks_1
        , batch_2, z2, coordinates_2, attention_masks_2
        , recon_combined_2, recons_2, masks_2
        , b
        , mechanism_permutation
        , selected_mechanism_idx
        , colors_1
        , colors_2
        , indices_1
        , indices_2
        , pairwise_cost
        , num_slots
        , table_name
        # , num_samples_to_log
        , train
        , additional_to_log={}
    ):
        logger = self.additional_logger
        num_samples_to_log = logger.num_samples_to_log(trainer=self.trainer, train=train)

        if num_samples_to_log <= 0:
            return

        if self.visualization_method == "clevr":
            figures = self.get_image_reconstructions_sparse_clevr(batch_1, z1, coordinates_1, attention_masks_1
                                                , recon_combined_1, recons_1, masks_1
                                                , batch_2, z2, coordinates_2, attention_masks_2
                                                , recon_combined_2, recons_2, masks_2
                                                , b, mechanism_permutation
                                                , selected_mechanism_idx
                                                , colors_1, colors_2, indices_1, indices_2
                                                , num_slots
                                                , num_samples_to_log)
        else:
            figures = self.get_image_reconstructions(batch_1, z1, coordinates_1, attention_masks_1
                                                    , recon_combined_1, recons_1, masks_1
                                                    , batch_2, z2, coordinates_2, attention_masks_2
                                                    , recon_combined_2, recons_2, masks_2
                                                    , b, mechanism_permutation
                                                    , selected_mechanism_idx
                                                    , colors_1, indices_1, indices_2
                                                    , pairwise_cost
                                                    , num_slots
                                                    , num_samples_to_log)

        columns, data = get_img_rec_table_data(
            imgs=figures,
            step=self.trainer.global_step,
            num_samples_to_log=num_samples_to_log,
        )

        logger.log_table(table_name=table_name, train=train, columns=columns, row_list=data)


    def get_image_reconstructions(self, batch_1
                                    , z1
                                    , coordinates_1
                                    , attention_masks_1
                                    , recon_combined_1, recons_1, masks_1
                                    , batch_2
                                    , z2
                                    , coordinates_2
                                    , attention_masks_2
                                    , recon_combined_2, recons_2, masks_2
                                    , b
                                    , mechanism_permutation
                                    , selected_mechanism_idx
                                    , colors
                                    , indices_1
                                    , indices_2
                                    , pairwise_cost
                                    , num_slots
                                    , num_samples_to_log):

        import matplotlib.pyplot as plt
        plt.cla()
        plt.close('all')

        # cost_dot_prod = torch.bmm(pairwise_cost.transpose(2, 1), pairwise_cost)
        # norms = torch.norm(pairwise_cost, p=2, dim=-2) # [batch_size, num_slots]
        # product_of_norms = torch.bmm(norms.unsqueeze(-1), norms.unsqueeze(-1).transpose(2,1))
        # cost_xcorr_detached_normalized = (cost_dot_prod/product_of_norms).clone().detach().cpu().numpy()
        # xcorr_threshold = 0.995
        # pairwise_cost_detached = pairwise_cost.clone().detach().cpu().numpy()

        right = 10
        left = 10
        top = 10
        bottom = 10
        resolution = batch_1.shape[-1]
        # if self.z_dim != 4:
        #     side = 4
        #     underline_spacing = 3
        # else:
        #     side = 7
        #     underline_spacing = 7
            
        side = 7
        underline_spacing = 7

        def add_margin(img, top, right, bottom, left, color):
            width, height, n_channels = img.shape
            new_width = width + right + left
            new_height = height + top + bottom
            new_image = np.ones((new_width, new_height, n_channels))
            new_image[top:new_height-bottom, left:new_width-right, :] = img
            return new_image

        def check_bounds(ball_xy, side):
            ball_x = int(resolution*ball_xy[0])
            ball_y = resolution - int(resolution*ball_xy[1])
            if ball_x+side >= resolution or ball_x-side <= 0 or ball_y+side >= resolution:
                return True
            return False
        
        def draw_sqaure_border(img, side, center_x, center_y, color):
            last_y = np.min([img.shape[0]-1, center_y+side])
            last_x = np.min([img.shape[1]-1, center_x+side+1])
            img[center_y-side:last_y, center_x-side] = color
            img[center_y-side:last_y, np.min([center_x+side, last_x])] = color

            img[center_y-side, center_x-side:last_x] = color
            img[last_y, center_x-side:last_x] = color
            return img

        def _clamp(array):
            array[array > 1.0] = 1.0
            array[array < 0.0] = 0.0
            return array

        renormalize = self.trainer.datamodule.renormalize()

        # indices: [batch_size, sparsity_degree (<=n_balls), 2]
        indices_1 = indices_1.clone().detach().cpu().numpy()
        indices_2 = indices_2.clone().detach().cpu().numpy()
        recon_combined_1 = renormalize(recon_combined_1.permute(0,2,3,1).detach().cpu().numpy())
        recon_combined_2 = renormalize(recon_combined_2.permute(0,2,3,1).detach().cpu().numpy())
        recons_1 = renormalize(recons_1.detach().cpu().numpy())
        recons_2 = renormalize(recons_2.detach().cpu().numpy())
        masks_1 = masks_1.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]
        masks_2 = masks_2.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]

        # when sparsity is present, we should only incorporate those balls that have changed.
        z_changed_mask = torch.any(torch.abs(z2.view(z1.shape[0], -1, self.hparams["z_dim"]) - z1.view(z1.shape[0], -1, self.hparams["z_dim"])) > 0., dim=-1) # [batch_size, n_balls]
        # z_changed_mask = torch.abs(z2 - z1) > 0. # [batch_size, n_balls * z_dim]
        z1_ = z1.view(z1.shape[0], -1, self.hparams["z_dim"])[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]
        z2_ = z2.view(z2.shape[0], -1, self.hparams["z_dim"])[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]
        # coordinates_1_ = coordinates_1.view(z_changed_mask.shape[0], -1, 2)[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]
        coordinates_1_ = coordinates_1.view(z_changed_mask.shape[0], -1, 2) # [batch_size, n_balls, 2]

        sparsity_degree = indices_1.shape[1]
        figs = []

        for idx in range(num_samples_to_log):

            
            image = renormalize(batch_1[idx].permute(1,2,0).clone().detach().cpu().numpy())
            mask = image < 0.95
            image_t1 = renormalize(batch_2[idx].permute(1,2,0).clone().detach().cpu().numpy())
            # if the offsets change positions, we can have a shaded version of the image at time t to appear as a background to the input image at t+1.
            # however, if only colour is changed, we should only show the original image at t+1 (because positions haven't changed and the two will collide)
            if self.z_dim == 2:
                image_t1_mask = image_t1.copy()
                image_t1_mask[mask] = image_t1[mask] * 0.8
            else:
                image_t1_mask = image_t1

            b_ = b[idx].clone().detach().cpu().numpy() # [sparsity_degree (<=n_balls), z_dim]
            recon_combined_1_ = recon_combined_1[idx]
            recon_combined_2_ = recon_combined_2[idx]
            recons_1_ = recons_1[idx]
            recons_2_ = recons_2[idx]
            masks_1_ = masks_1[idx]
            masks_2_ = masks_2[idx]

            # [sparsity_degree (<=n_balls), 2]
            _indices_1 = indices_1[idx].copy()
            _indices_2 = indices_2[idx].copy()

            indices_1_ = _indices_1[:, [1, 0]] # [sparsity_degree (<=n_balls), 2], but now [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_1_ = indices_1_[np.argsort(indices_1_[:, 0])] # it's now sorted in ascending order of slot ids.
            indices_2_ = _indices_2[:, [1, 0]] # [sparsity_degree (<=n_balls), 2], but [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_2_ = indices_2_[np.argsort(indices_2_[:, 0])] # it's now sorted in ascending order of slot ids.
            colors_ = colors[idx].clone().detach().cpu().numpy()
            colors_mask = z_changed_mask[idx].clone().detach().cpu().numpy()
            mechanism_permutation_ = mechanism_permutation[idx].clone().detach().cpu().numpy()
            selected_mechanism_idx_ = selected_mechanism_idx[idx].clone().detach().cpu().numpy()
            num_slots_ = num_slots
            
            fig, ax = plt.subplots(9, num_slots_ + 1, figsize=(36, 36))

            quiver_scale = 1.0 # 1.0 # 0.4
            quiver_width = 0.005
            # t
            ax[0,0].imshow(_clamp(image))
            ax[0,0].set_title("Input Image at 't'")
            if self.z_dim == 4:
                offset_dim = np.array(np.abs(b_)>0).nonzero()[1].squeeze() # offset_dim: [0,1,2,3]
                if offset_dim == 0:
                    # ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], np.linalg.norm(b_, 2, axis=1), scale=0.4)
                    ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], color=colors_[mechanism_permutation_], scale=quiver_scale)
                    # ax[5,0].quiver(np.concatenate((coordinates_1_[idx][mechanism_permutation_,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((coordinates_1_[idx][mechanism_permutation_,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((b_[:,0], np.array([0.0, 0.0]))), np.concatenate((b_[:,1]
                    # , np.array([0.0, 0.0]))), color=colors_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
                if offset_dim == 1:
                    # ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], np.linalg.norm(b_, 2, axis=1), scale=0.4)
                    ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], color=colors_[mechanism_permutation_], scale=quiver_scale)
                    # ax[5,0].quiver(np.concatenate((coordinates_1_[idx][mechanism_permutation_,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((coordinates_1_[idx][mechanism_permutation_,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((b_[:,0], np.array([0.0, 0.0]))), np.concatenate((b_[:,1]
                    # , np.array([0.0, 0.0]))), color=colors_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
                if offset_dim == 2:
                    # sparsity_degree <= n_balls
                    ax[1,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_[mechanism_permutation_])
                    ax[1,0].set_xticks(np.arange(sparsity_degree))
                    ax[1,0].set_xlabel("ball id")
                    ax[1,0].set_ylabel("hue offset")
                    # ax[5,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_[mechanism_permutation_])
                    # ax[5,0].set_xlabel("ball id")
                    # ax[5,0].set_ylabel("hue offset")
                    # ax[5,0].set_xticks(np.arange(sparsity_degree))
                if offset_dim == 3:
                    # draw the previous shape with its colour and the new shape with the new colour at the left and right
                    # of the image
                    prev_z = z1_[idx].clone()
                    prev_z[:2] = torch.tensor([0.25, 0.5], device=prev_z.device)
                    prev_colour = (np.array([0.8, 0.8, 0.8]) * 255.).astype(int)
                    next_z = z2_[idx].clone()
                    next_z[:2] = torch.tensor([0.75, 0.5], device=next_z.device)
                    next_colour = (255. * colors_[mechanism_permutation_].squeeze()).astype(int)
                    im = draw_scene(torch.stack([prev_z,next_z], dim=0), colours=[prev_colour,next_colour])
                    ax[1,0].imshow(im)
                    # ax[5,0].imshow(im)

                if selected_mechanism_idx_ == 0:
                    ax[5,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), 0.2, 0.0, color=colors_[mechanism_permutation_], scale=quiver_scale)
                if selected_mechanism_idx_ == 1:
                    ax[5,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), 0.0, 0.2, color=colors_[mechanism_permutation_], scale=quiver_scale)
                if selected_mechanism_idx_ == 4:
                    ax[5,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), -0.2, 0.0, color=colors_[mechanism_permutation_], scale=quiver_scale)
                if selected_mechanism_idx_ == 5:
                    ax[5,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), 0.0, -0.2, color=colors_[mechanism_permutation_], scale=quiver_scale)
                if selected_mechanism_idx_ == 2:
                    heights = [z1_[idx][2].clone().detach().cpu().numpy(),z1_[idx][2].clone().detach().cpu().numpy()+0.1]
                    ax[5,0].bar(np.arange(2), height=heights, color=colors_[mechanism_permutation_])
                if selected_mechanism_idx_ == 6:
                    heights = [z1_[idx][2].clone().detach().cpu().numpy(),z1_[idx][2].clone().detach().cpu().numpy()-0.1]
                    ax[5,0].bar(np.arange(2), height=heights, color=colors_[mechanism_permutation_])
                if selected_mechanism_idx_ == 3:
                    prev_z = z1_[idx].clone()
                    prev_z[:2] = torch.tensor([0.25, 0.5], device=prev_z.device)
                    prev_colour = (np.array([0.8, 0.8, 0.8]) * 255.).astype(int)
                    next_z = z2_[idx].clone()
                    next_z[:2] = torch.tensor([0.75, 0.5], device=next_z.device)
                    next_colour = (255. * colors_[mechanism_permutation_].squeeze()).astype(int)
                    next_z[3] = 3 if next_z[3] == 3 else prev_z[3] + 1
                    im = draw_scene(torch.stack([prev_z,next_z], dim=0), colours=[prev_colour,next_colour])
                    ax[5,0].imshow(im)
                if selected_mechanism_idx_ == 7:
                    prev_z = z1_[idx].clone()
                    prev_z[:2] = torch.tensor([0.25, 0.5], device=prev_z.device)
                    prev_colour = (np.array([0.8, 0.8, 0.8]) * 255.).astype(int)
                    next_z = z2_[idx].clone()
                    next_z[:2] = torch.tensor([0.75, 0.5], device=next_z.device)
                    next_colour = (255. * colors_[mechanism_permutation_].squeeze()).astype(int)
                    next_z[3] = 0 if next_z[3] == 0 else prev_z[3] - 1
                    im = draw_scene(torch.stack([prev_z,next_z], dim=0), colours=[prev_colour,next_colour])
                    ax[5,0].imshow(im)



            elif self.z_dim == 2:
                quiver_scale = 2.5 # 1.0 # 0.4
                quiver_width = 0.005
                # ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], np.linalg.norm(b_, 2, axis=1), scale=0.4)
                ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], color=colors_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
                # ax[1,0].quiver(z1[idx][:,0], z1[idx][:,1], b_[:,0], b_[:,1], color=colors_, scale=0.4)

                # ax[5,0].quiver(np.concatenate((coordinates_1_[idx].view(-1,2)[:,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                # , np.concatenate((coordinates_1_[idx].view(-1,2)[:,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                # , np.concatenate((b_[mechanism_permutation_,0], np.array([0.0, 0.0]))), np.concatenate((b_[mechanism_permutation_,1]
                # , np.array([0.0, 0.0]))), color=colors_[colors_mask], scale=quiver_scale) #, width=quiver_width)
                ax[5,0].quiver(np.concatenate((coordinates_1_[idx][mechanism_permutation_,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                , np.concatenate((coordinates_1_[idx][mechanism_permutation_,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                , np.concatenate((b_[:,0], np.array([0.0, 0.0]))), np.concatenate((b_[:,1]
                , np.array([0.0, 0.0]))), color=colors_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
            elif self.z_dim == 1:
                # sparsity_degree <= n_balls
                ax[1,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_[mechanism_permutation_])
                ax[1,0].set_xticks(np.arange(sparsity_degree))
                ax[1,0].set_xlabel("ball id")
                ax[1,0].set_ylabel("hue offset")
                ax[5,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_[mechanism_permutation_])
                ax[1,0].set_xticks(np.arange(sparsity_degree))
                ax[5,0].set_xlabel("ball id")
                ax[5,0].set_ylabel("hue offset")

            ax[1,0].set_title(f'Offsets (known mechanism:{self.known_mechanism})')
            ax[5,0].set_title(f'Offsets (known mechanism:{self.known_mechanism})')
            
            ax[2,0].imshow(_clamp(masks_1_.sum(axis=0)), vmin=0, vmax=1)
            ax[2,0].set_title("Decoder Masks sum 't'")
            ax[3,0].imshow((recon_combined_1_ * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[3,0].set_title("Reconstruction 't'")

            # t+1
            ax[4,0].imshow(image_t1_mask) 
            ax[4,0].set_title("Input Image at 't+1'")
            # ax[3,0].imshow(image_t1_mask) 
            # ax[3,0].set_title("Input Image at 't+1'")
            ax[6,0].imshow(_clamp(masks_2_.sum(axis=0)), vmin=0, vmax=1)
            ax[6,0].set_title("Decoder Masks sum 't+1'")
            ax[7,0].imshow((recon_combined_2_ * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[7,0].set_title("Reconstruction 't+1'")
            
            for slot_id in range(num_slots_):

                # ---------- step t ---------- #
                temp = image.copy()
                # For each slot, find the corresponding ball and its color
                # Then recolor the reconstruction with the color
                if slot_id in indices_1_[:, 0]:
                    ball_id = indices_1_[(indices_1_[:, 0]==slot_id).nonzero(), 1].squeeze()
                    if isinstance(ball_id, np.ndarray) and ball_id.size > 1: # if a slot has been matched to more than one mechanism.
                        ball_id = mechanism_permutation_[ball_id[0]]
                    else:
                        ball_id = mechanism_permutation_[ball_id]
                    ball_center_xy = coordinates_1[idx].view(-1,2)[ball_id].clone().detach().cpu().numpy()
                    # ball_center_xy = coordinates_1[idx].view(-1,2)[ball_id*2:ball_id*2+2][0].clone().detach().cpu().numpy()
                    ball_x = int(resolution*ball_center_xy[0])
                    ball_y = resolution - int(resolution*ball_center_xy[1])
                    if check_bounds(ball_center_xy, side=side):                
                        temp = add_margin(temp, top, right, bottom, left, (1.0, 1.0, 1.0))
                        ball_x += left
                        ball_y += top
                else:
                    ball_id = None

                # recolor if needed:
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    temp[ball_y+underline_spacing, ball_x-3:ball_x+3,:] = c
                else:
                    temp = image.copy()
                ax[0, slot_id + 1].imshow(_clamp(temp))
                ax[0, slot_id + 1].set_title(f'Slot {slot_id}')

                # Attention Masks
                attn = attention_masks_1[idx].reshape(-1, resolution, resolution)[slot_id].clone().detach().cpu().numpy()
                ax[1, slot_id + 1].imshow(_clamp(attn))
                ax[1, slot_id + 1].set_title(f"Attention mask 't'")

                # Decoder Masks
                ax[2, slot_id + 1].imshow(_clamp(masks_1_[slot_id]), vmin=0, vmax=1)
                ax[2, slot_id + 1].set_title(f"Slot {slot_id} Decoder Masks 't'")

                # Reconstruction Per Slot
                rec = recons_1_[slot_id] * masks_1_[slot_id] + (1 - masks_1_[slot_id])
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    if check_bounds(ball_center_xy, side=side):                
                        rec = add_margin(rec, top, right, bottom, left, (1.0, 1.0, 1.0))
                        # ball_x += left
                        # ball_y += top
                    rec = draw_sqaure_border(rec, side-1, ball_x, ball_y, c)
                else:
                    pass
                ax[3, slot_id + 1].imshow(_clamp(rec), vmin=0, vmax=1)
                ax[3, slot_id + 1].set_title(f"Slot {slot_id} Recons 't'")

                # ---------- step t+1 ---------- #

                temp_2 = image_t1.copy()
                # For each slot, find the corresponding ball and its color
                # Then recolor the reconstruction with the color
                if slot_id in indices_2_[:, 0]:
                    ball_id = indices_2_[(indices_2_[:, 0]==slot_id).nonzero(), 1].squeeze()
                    if isinstance(ball_id, np.ndarray) and ball_id.size > 1: # if a slot has been matched to more than one mechanism.
                        ball_id = mechanism_permutation_[ball_id[0]]
                    else:
                        ball_id = mechanism_permutation_[ball_id]
                    ball_center_xy = coordinates_2[idx].view(-1,2)[ball_id].clone().detach().cpu().numpy()
                    # ball_center_xy = coordinates_2[idx].view(-1,2)[ball_id*2:ball_id*2+2][0].clone().detach().cpu().numpy()
                    ball_x = int(resolution*ball_center_xy[0])
                    ball_y = resolution - int(resolution*ball_center_xy[1])
                    if check_bounds(ball_center_xy, side=side):
                        temp_2 = add_margin(temp_2, top, right, bottom, left, (1.0, 1.0, 1.0))
                        ball_x += left
                        ball_y += top
                            
                else:
                    ball_id = None

                # recolor if needed:
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    temp_2[np.min([ball_y+underline_spacing, temp_2.shape[0]-1]), ball_x-3:ball_x+3,:] = c
                else:
                    temp_2 = image_t1.copy()
                ax[4, slot_id + 1].imshow(_clamp(temp_2))
                ax[4, slot_id + 1].set_title(f'Slot {slot_id}')

                # Attention Masks
                attn = attention_masks_2[idx].reshape(-1, resolution, resolution)[slot_id].clone().detach().cpu().numpy()
                ax[5, slot_id + 1].imshow(_clamp(attn))
                ax[5, slot_id + 1].set_title(f"Attention mask 't+1'")

                # Decoder Masks
                ax[6, slot_id + 1].imshow(_clamp(masks_2_[slot_id]), vmin=0, vmax=1)
                ax[6, slot_id + 1].set_title(f"Slot {slot_id} Decoder Masks 't+1'")

                # Reconstruction Per Slot
                rec = recons_2_[slot_id] * masks_2_[slot_id] + (1 - masks_2_[slot_id])
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    if check_bounds(ball_center_xy, side=side):                
                        rec = add_margin(rec, top, right, bottom, left, (1.0, 1.0, 1.0))
                        # ball_x += left
                        # ball_y += top
                    rec = draw_sqaure_border(rec, side-1, ball_x, ball_y, c)
                else:
                    pass
                ax[7, slot_id + 1].imshow(_clamp(rec), vmin=0, vmax=1)
                ax[7, slot_id + 1].set_title(f"Slot {slot_id} Recons 't+1'")

            # pairwise_cost_ = pairwise_cost_detached[idx]
            # plt.matshow(pairwise_cost_, cmap=plt.cm.Blues)
            # for i in range(pairwise_cost_.shape[0]):
            #     for j in range(pairwise_cost_.shape[1]):
            #         c = format(pairwise_cost_[i,j]*100, ".2f")
            #         plt.text(j, i, str(c), va='center', ha='center')
            #         if j == indices[idx][i, 1]:
            #             plt.text(j, i, "____", va='center', ha='center', color=colors_[indices[idx][i, 0]])
            # plt.matshow(cost_xcorr_detached_normalized[idx]>xcorr_threshold)
            # plt.matshow(cost_xcorr_detached_normalized[idx])

            for i, j in product(range(ax.shape[0]), range(ax.shape[1])):
                ax[i, j].grid(False)
                ax[i, j].axis('off')

            figs.append(fig)

        return figs
    
    def get_image_reconstructions_sparse_clevr(self, batch_1
                                    , z1
                                    , coordinates_1
                                    , attention_masks_1
                                    , recon_combined_1, recons_1, masks_1
                                    , batch_2
                                    , z2
                                    , coordinates_2
                                    , attention_masks_2
                                    , recon_combined_2, recons_2, masks_2
                                    , b
                                    , mechanism_permutation
                                    , selected_mechanism_idx
                                    , colors_1
                                    , colors_2
                                    , indices_1
                                    , indices_2
                                    , num_slots
                                    , num_samples_to_log):

        import matplotlib.pyplot as plt
        plt.cla()
        plt.close('all')

        def _clamp(array):
            array[array > 1.0] = 1.0
            array[array < 0.0] = 0.0
            return array

        renormalize = self.trainer.datamodule.renormalize()

        # indices: [batch_size, sparsity_degree (<=n_balls), 2]
        indices_1 = indices_1.clone().detach().cpu().numpy()
        indices_2 = indices_2.clone().detach().cpu().numpy()
        recon_combined_1 = renormalize(recon_combined_1.permute(0,2,3,1).detach().cpu().numpy())
        recon_combined_2 = renormalize(recon_combined_2.permute(0,2,3,1).detach().cpu().numpy())
        recons_1 = renormalize(recons_1.detach().cpu().numpy())
        recons_2 = renormalize(recons_2.detach().cpu().numpy())
        masks_1 = masks_1.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]
        masks_2 = masks_2.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]

        # when sparsity is present, we should only incorporate those balls that have changed.
        z_changed_mask = torch.any(torch.abs(z2.view(z1.shape[0], -1, self.hparams["z_dim"]) - z1.view(z1.shape[0], -1, self.hparams["z_dim"])) > 0., dim=-1) # [batch_size, n_balls]
        # z_changed_mask = torch.abs(z2 - z1) > 0. # [batch_size, n_balls * z_dim]
        z1_ = z1.view(z1.shape[0], -1, self.hparams["z_dim"])[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]
        z2_ = z2.view(z2.shape[0], -1, self.hparams["z_dim"])[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]

        # temporary code
        # z_changed_mask = torch.any(torch.abs(z2.view(z1.shape[0], -1, self.true_z_dim) - z1.view(z1.shape[0], -1, self.true_z_dim)) > 0., dim=-1) # [batch_size, n_balls]
        # # z_changed_mask = torch.abs(z2 - z1) > 0. # [batch_size, n_balls * z_dim]
        # z1_ = z1.view(z1.shape[0], -1, self.true_z_dim)[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]
        # z2_ = z2.view(z2.shape[0], -1, self.true_z_dim)[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]

        # coordinates_1_ = coordinates_1.view(z_changed_mask.shape[0], -1, 2)[z_changed_mask].view(z_changed_mask.shape[0], -1) # [batch_size, sparsity_degree * z_dim]
        coordinates_dim = 3 if "z" in self.trainer.datamodule.hparams["dataset"]["properties_list"] else 2
        coordinates_1_ = coordinates_1.view(z_changed_mask.shape[0], -1, coordinates_dim) # [batch_size, n_balls, 2 or 3]

        sparsity_degree = indices_1.shape[1]
        figs = []

        for idx in range(num_samples_to_log):

            image = renormalize(batch_1[idx].permute(1,2,0).clone().detach().cpu().numpy())
            mask = (image > image.mean() - 0.05)
            image_t1 = renormalize(batch_2[idx].permute(1,2,0).clone().detach().cpu().numpy())
            # if the offsets change positions, we can have a shaded version of the image at time t to appear as a background to the input image at t+1.
            # however, if only colour is changed, we should only show the original image at t+1 (because positions haven't changed and the two will collide)
            # if self.z_dim == 2: # TODO: incorrect, z_dim isn't important, it's the property list that determines this
            #     image_t1_mask = image_t1.copy()
            #     image_t1_mask[mask] = image_t1[mask] * 0.8
            # else:
            #     image_t1_mask = image_t1
            image_t1_mask = image_t1

            b_ = b[idx].clone().detach().cpu().numpy() # [sparsity_degree (<=n_balls), z_dim]
            recon_combined_1_ = recon_combined_1[idx]
            recon_combined_2_ = recon_combined_2[idx]
            recons_1_ = recons_1[idx]
            recons_2_ = recons_2[idx]
            masks_1_ = masks_1[idx]
            masks_2_ = masks_2[idx]

            # [sparsity_degree (<=n_balls), 2]
            _indices_1 = indices_1[idx].copy()
            _indices_2 = indices_2[idx].copy()

            indices_1_ = _indices_1[:, [1, 0]] # [sparsity_degree (<=n_balls), 2], but now [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_1_ = indices_1_[np.argsort(indices_1_[:, 0])] # it's now sorted in ascending order of slot ids.
            indices_2_ = _indices_2[:, [1, 0]] # [sparsity_degree (<=n_balls), 2], but [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_2_ = indices_2_[np.argsort(indices_2_[:, 0])] # it's now sorted in ascending order of slot ids.
            colors_1_ = colors_1[idx].clone().detach().cpu().numpy()
            colors_2_ = colors_2[idx].clone().detach().cpu().numpy()
            colors_mask = z_changed_mask[idx].clone().detach().cpu().numpy()
            mechanism_permutation_ = mechanism_permutation[idx].clone().detach().cpu().numpy()
            selected_mechanism_idx_ = selected_mechanism_idx[idx].clone().detach().cpu().numpy()
            num_slots_ = num_slots

            fig, ax = plt.subplots(9, num_slots_ + 1, figsize=(36, 36))

            quiver_scale = 1.0 # 1.0 # 0.4
            quiver_width = 0.005
            # t
            ax[0,0].imshow(_clamp(image))
            ax[0,0].set_title("Input Image at 't'")
            if self.z_dim >= 3:
                offset_dim = np.array(np.abs(b_)>0).nonzero()[1].squeeze() # offset_dim: a number in [0,1,2,3]
                if offset_dim == 0:
                    # ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], np.linalg.norm(b_, 2, axis=1), scale=0.4)
                    ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], color=colors_1_[mechanism_permutation_], scale=quiver_scale)
                    # ax[5,0].quiver(np.concatenate((coordinates_1_[idx][mechanism_permutation_,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((coordinates_1_[idx][mechanism_permutation_,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((b_[:,0], np.array([0.0, 0.0]))), np.concatenate((b_[:,1]
                    # , np.array([0.0, 0.0]))), color=colors_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
                if offset_dim == 1:
                    # ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], np.linalg.norm(b_, 2, axis=1), scale=0.4)
                    ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], color=colors_1_[mechanism_permutation_], scale=quiver_scale)
                    # ax[5,0].quiver(np.concatenate((coordinates_1_[idx][mechanism_permutation_,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((coordinates_1_[idx][mechanism_permutation_,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                    # , np.concatenate((b_[:,0], np.array([0.0, 0.0]))), np.concatenate((b_[:,1]
                    # , np.array([0.0, 0.0]))), color=colors_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
                if offset_dim == 2:
                    # sparsity_degree <= n_balls
                    ax[1,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_1_[mechanism_permutation_])
                    ax[1,0].set_xticks(np.arange(sparsity_degree))
                    ax[1,0].set_xlabel("ball id")
                    ax[1,0].set_ylabel("hue offset")
                    # ax[5,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_[mechanism_permutation_])
                    # ax[5,0].set_xlabel("ball id")
                    # ax[5,0].set_ylabel("hue offset")
                    # ax[5,0].set_xticks(np.arange(sparsity_degree))
                if offset_dim == 3:
                    # draw nothing
                    pass

            elif self.z_dim == 2:
                quiver_scale = 2.5 # 1.0 # 0.4
                quiver_width = 0.005
                # ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], np.linalg.norm(b_, 2, axis=1), scale=0.4)
                ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], color=colors_1_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
                # ax[1,0].quiver(z1[idx][:,0], z1[idx][:,1], b_[:,0], b_[:,1], color=colors_, scale=0.4)

                # ax[5,0].quiver(np.concatenate((coordinates_1_[idx].view(-1,2)[:,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                # , np.concatenate((coordinates_1_[idx].view(-1,2)[:,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                # , np.concatenate((b_[mechanism_permutation_,0], np.array([0.0, 0.0]))), np.concatenate((b_[mechanism_permutation_,1]
                # , np.array([0.0, 0.0]))), color=colors_[colors_mask], scale=quiver_scale) #, width=quiver_width)
                ax[5,0].quiver(np.concatenate((coordinates_1_[idx][mechanism_permutation_,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                , np.concatenate((coordinates_1_[idx][mechanism_permutation_,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                , np.concatenate((b_[:,0], np.array([0.0, 0.0]))), np.concatenate((b_[:,1]
                , np.array([0.0, 0.0]))), color=colors_1_[mechanism_permutation_], scale=quiver_scale) #, width=quiver_width)
            elif self.z_dim == 1:
                # sparsity_degree <= n_balls
                ax[1,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_1_[mechanism_permutation_])
                ax[1,0].set_xticks(np.arange(sparsity_degree))
                ax[1,0].set_xlabel("ball id")
                ax[1,0].set_ylabel("hue offset")
                ax[5,0].bar(np.arange(sparsity_degree), height=b_.squeeze(), color=colors_1_[mechanism_permutation_])
                ax[1,0].set_xticks(np.arange(sparsity_degree))
                ax[5,0].set_xlabel("ball id")
                ax[5,0].set_ylabel("hue offset")

            ax[1,0].set_title(f'Offsets (known mechanism:{self.known_mechanism})')
            ax[5,0].set_title(f'Offsets (known mechanism:{self.known_mechanism})')
            
            ax[2,0].imshow(_clamp(masks_1_.sum(axis=0)), vmin=0, vmax=1)
            ax[2,0].set_title("Decoder Masks sum 't'")
            # ax[3,0].imshow((recon_combined_1_ * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[3,0].imshow(_clamp(recon_combined_1_), vmin=0, vmax=1)
            ax[3,0].set_title("Reconstruction 't'")

            # t+1
            ax[4,0].imshow(image_t1_mask) 
            ax[4,0].set_title("Input Image at 't+1'")
            # ax[3,0].imshow(image_t1_mask) 
            # ax[3,0].set_title("Input Image at 't+1'")
            ax[6,0].imshow(_clamp(masks_2_.sum(axis=0)), vmin=0, vmax=1)
            ax[6,0].set_title("Decoder Masks sum 't+1'")
            # ax[7,0].imshow((recon_combined_2_ * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[7,0].imshow(_clamp(recon_combined_2_), vmin=0, vmax=1)
            ax[7,0].set_title("Reconstruction 't+1'")
            
            for slot_id in range(num_slots_):

                # ---------- step t ---------- #
                temp = image.copy()
                # For each slot, find the corresponding ball and its color
                # Then recolor the reconstruction with the color
                if slot_id in indices_1_[:, 0]:
                    ball_id = indices_1_[(indices_1_[:, 0]==slot_id).nonzero(), 1].squeeze()
                    if isinstance(ball_id, np.ndarray) and ball_id.size > 1: # if a slot has been matched to more than one mechanism.
                        ball_id = mechanism_permutation_[ball_id[0]]
                    else:
                        ball_id = mechanism_permutation_[ball_id]
                    # ball_center_xy = coordinates_1[idx].view(-1,2)[ball_id].clone().detach().cpu().numpy()
                    # # ball_center_xy = coordinates_1[idx].view(-1,2)[ball_id*2:ball_id*2+2][0].clone().detach().cpu().numpy()
                    # ball_x = int(64*ball_center_xy[0])
                    # ball_y = 64 - int(64*ball_center_xy[1])
                    # if check_bounds(ball_center_xy, side=side):                
                    #     temp = add_margin(temp, top, right, bottom, left, (1.0, 1.0, 1.0))
                    #     ball_x += left
                    #     ball_y += top
                else:
                    ball_id = None

                # # recolor if needed:
                if ball_id is not None:
                    c = colors_1_[ball_id, :]
                    # temp[ball_y+underline_spacing, ball_x-3:ball_x+3,:] = c
                    temp[60-2:60+2, 32-5:32+5,:3] = c
                else:
                    temp = image.copy()

                # temp = image.copy()
                ax[0, slot_id + 1].imshow(_clamp(temp))
                ax[0, slot_id + 1].set_title(f'Slot {slot_id}')

                # Attention Masks
                attn = attention_masks_1[idx].reshape(-1, 64, 64)[slot_id].clone().detach().cpu().numpy()
                ax[1, slot_id + 1].imshow(_clamp(attn))
                ax[1, slot_id + 1].set_title(f"Attention mask 't'")

                # Decoder Masks
                ax[2, slot_id + 1].imshow(_clamp(masks_1_[slot_id]), vmin=0, vmax=1)
                ax[2, slot_id + 1].set_title(f"Slot {slot_id} Decoder Masks 't'")

                # Reconstruction Per Slot
                rec = recons_1_[slot_id] * masks_1_[slot_id] + (1 - masks_1_[slot_id])
                # if ball_id is not None:
                #     c = colors_[ball_id, :]
                #     if check_bounds(ball_center_xy, side=side):                
                #         rec = add_margin(rec, top, right, bottom, left, (1.0, 1.0, 1.0))
                #         # ball_x += left
                #         # ball_y += top
                #     rec = draw_sqaure_border(rec, side-1, ball_x, ball_y, c)
                # else:
                #     pass
                ax[3, slot_id + 1].imshow(_clamp(rec), vmin=0, vmax=1)
                ax[3, slot_id + 1].set_title(f"Slot {slot_id} Recons 't'")

                # ---------- step t+1 ---------- #

                temp_2 = image_t1.copy()
                i = 100

                # For each slot, find the corresponding ball and its color
                # Then recolor the reconstruction with the color
                if slot_id in indices_2_[:, 0]:
                    ball_id = indices_2_[(indices_2_[:, 0]==slot_id).nonzero(), 1].squeeze()
                    if isinstance(ball_id, np.ndarray) and ball_id.size > 1: # if a slot has been matched to more than one mechanism.
                        ball_id = mechanism_permutation_[ball_id[0]]
                    else:
                        ball_id = mechanism_permutation_[ball_id]
                    # ball_center_xy = coordinates_2[idx].view(-1,2)[ball_id].clone().detach().cpu().numpy()
                    # # ball_center_xy = coordinates_2[idx].view(-1,2)[ball_id*2:ball_id*2+2][0].clone().detach().cpu().numpy()
                    # ball_x = int(64*ball_center_xy[0])
                    # ball_y = 64 - int(64*ball_center_xy[1])
                    # if check_bounds(ball_center_xy, side=side):
                    #     temp_2 = add_margin(temp_2, top, right, bottom, left, (1.0, 1.0, 1.0))
                    #     ball_x += left
                    #     ball_y += top
                            
                else:
                    ball_id = None

                # recolor if needed:
                if ball_id is not None:
                    c = colors_2_[ball_id, :]
                    # temp_2[np.min([ball_y+underline_spacing, temp_2.shape[0]-1]), ball_x-3:ball_x+3,:] = c
                    temp_2[60-2:60+2, 32-5:32+5,:3] = c
                else:
                    temp_2 = image_t1.copy()
                # temp_2 = image_t1.copy()
                ax[4, slot_id + 1].imshow(_clamp(temp_2))
                ax[4, slot_id + 1].set_title(f'Slot {slot_id}')

                # Attention Masks
                attn = attention_masks_2[idx].reshape(-1, 64, 64)[slot_id].clone().detach().cpu().numpy()
                ax[5, slot_id + 1].imshow(_clamp(attn))
                ax[5, slot_id + 1].set_title(f"Attention mask 't+1'")

                # Decoder Masks
                ax[6, slot_id + 1].imshow(_clamp(masks_2_[slot_id]), vmin=0, vmax=1)
                ax[6, slot_id + 1].set_title(f"Slot {slot_id} Decoder Masks 't+1'")

                # Reconstruction Per Slot
                rec = recons_2_[slot_id] * masks_2_[slot_id] + (1 - masks_2_[slot_id])
                # if ball_id is not None:
                #     c = colors_[ball_id, :]
                #     if check_bounds(ball_center_xy, side=side):                
                #         rec = add_margin(rec, top, right, bottom, left, (1.0, 1.0, 1.0))
                #         # ball_x += left
                #         # ball_y += top
                #     rec = draw_sqaure_border(rec, side-1, ball_x, ball_y, c)
                # else:
                #     pass
                ax[7, slot_id + 1].imshow(_clamp(rec), vmin=0, vmax=1)
                ax[7, slot_id + 1].set_title(f"Slot {slot_id} Recons 't+1'")

            for i, j in product(range(ax.shape[0]), range(ax.shape[1])):
                ax[i, j].grid(False)
                ax[i, j].axis('off')

            figs.append(fig)

        return figs
