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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(logger=False)

        # This model is intended for ablation study of the case in which there is no latent loss. Hence there is
        # no projection layer to the target latent space. Slots will have the same dimensionality as our target
        # latents, and will be compared with them directly for disentanglement measurements (again, no projection)

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

        self.cvxpylayer = lp_solver_cvxpy(n_mechanisms=self.hparams["n_balls"], num_slots=self.hparams["encoder"]["num_slots"])


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

        # there is no projection to target dimension, slots have the same dimension as target latents
        slots_projected = slots # self.slot_projection_to_z(slots) # slots_projected: [batch_size, num_slots, z_dim]

        return slots, slots_projected, recon_combined, recons, masks, attention_scores, slots_init


    def training_step(self, batch, batch_idx):

        # with torch.no_grad():
        #     for name, param in self.slot_projection_to_z.named_parameters():
        #         if len(param.size()) > 1: # excluding bias terms
        #             u,s,vh = torch.linalg.svd(param.data, full_matrices=False)
        #             eps = torch.tensor([0.01], device=param.device)
        #             param.data = u @ torch.diag(torch.maximum(s, eps)) @ vh
        #             # print(f"\n-----:{torch.maximum(s, eps)}\n\nname:{name}\n\nparam:{param.data}")

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

        import time

        (z1, z2), (x1, x2), (A, b), colors = batch["latents"], batch["images"], batch["matrices"], batch["colors"]

        # 1. pass x1,x2 to the encoder, and get slots, which ARE OF THE SAME DIMENSION as latent dimension z,
        #    to obtain object_z1,object_z2 (note that some of the entries doesn't necessarily 
        #    correspond to objects as num_slots > n_objects)
        # 2. construct the cost matrix based on mechanisms being applied to projected slots and
        #    and the projections at 't+1', i.e. || mech(object_z1) - object_z2 ||^2
        # 3. get the assignments and reorder slots
        # 4. apply the mechanism (adding offset) only to that many slots that correspond to some mech.
        #    Alternatively sum the cost matrix over the minimum assignment. Return the resulting loss
        #    to be optimized.

        # 1
        # slots: [batch_size, num_slots, slot_size]
        # slots_projected: [batch_size, num_slots, z_dim] slots, slots_projected, recon_combined, recons, masks, attention_scores
        t0 = time.perf_counter()
        slots_1, slots_projected_1, recon_combined_1, recons_1, masks_1, attention_scores_1, slots_init_1 = self(x1)
        slots_2, slots_projected_2, recon_combined_2, recons_2, masks_2, attention_scores_2, slots_init_2 = self(x2, slots_init=slots_init_1)
        n_slots = slots_1.size(1)

        # # [batch_size, n_slots, n_slots]
        # all_dot_products = torch.bmm(slots_1, slots_1.permute(0,2,1))
        # # [batch_size, n_slots]
        # all_norms = torch.norm(slots_1, p=2, dim=-1)
        # # [batch_size, n_slots, n_slots]
        # normalized_prods = all_dot_products/all_norms.unsqueeze(1)
        # diag_prods = torch.diagonal(normalized_prods, dim1=-1, dim2=-2) # same slot products cosine distance should be removed
        # # squared to avoid negative values pushing the loss to -inf
        # slot_similarity_loss = ((normalized_prods.sum(-1).sum(-1) - diag_prods.sum(-1)).mean()) ** 2 
        # self.log(f"{stage}_slot_similarity_loss", slot_similarity_loss.item())

        # [batch_size, 2]
        # _, sigma_1, _ = torch.linalg.svd(slots_projected_1, full_matrices=True) # set to True to allow backprop
        # _, sigma_2, _ = torch.linalg.svd(slots_projected_2, full_matrices=True)
        # # print(f"\n-----------------\nSVD:{sigma_1}\n\n")
        # eps = 1e-6
        # sigma_1_prod = sigma_1.prod(dim=1).mean() + eps
        # sigma_1_prod = sigma_1.min(dim=1)[0].mean() + eps
        # sigma_2_prod = sigma_2.prod(dim=1).mean() + eps


        # b: [batch_size, n_balls*2] -> [batch_size, n_balls, 2]
        b = b.squeeze().view(b.size(0), -1, 2)
        bs = b.size(0)
        n_mechanisms = b.size(1) # equal to the number of balls

        # 2, 3, 4
        # [batch_size, num_slots, num_mechanisms, z_dim]
        slots_projected_1_expanded = slots_projected_1.unsqueeze(2).repeat(1,1,n_mechanisms,1)
        slots_projected_2_expanded = slots_projected_2.unsqueeze(2).repeat(1,1,n_mechanisms,1)

        # all possible mechanisms applied to all possible slot projections
        # [batch_size, num_slots, num_mechanisms, z_dim]
        m_slots_projected_1_expanded = slots_projected_1_expanded + b.unsqueeze(1)

        # We need to manipulate m_slots_projected_1_expanded and slots_projected_2_expanded so they align and result in 
        # the computation of all possible combinations of m_k(s_t^i) and s_{t+1}^j.

        # [batch_size, n_mechanisms, n_slots^2, 2]
        m_slots_projected_1_expanded = m_slots_projected_1_expanded.permute(0,2,1,3).repeat(1, 1, 1, n_slots).reshape(bs, n_mechanisms, -1, 2)
        slots_projected_2_expanded = slots_projected_2_expanded.permute(0,2,1,3).repeat(1, 1, n_slots, 1)


        # find the duplicates (their indices)
        duplicate_threshold = 0.998
        slots_1_dot_prod = torch.bmm(slots_projected_1, slots_projected_1.transpose(2, 1))
        norms_1 = torch.norm(slots_projected_1, p=2, dim=-1) # [batch_size, num_slots]
        product_of_norms_1 = torch.bmm(norms_1.unsqueeze(-1), norms_1.unsqueeze(-1).transpose(2,1))
        slots_1_xcorr_normalized = slots_1_dot_prod/product_of_norms_1
        slots_1_xcorr_detached_normalized = slots_1_xcorr_normalized.clone().detach().cpu().numpy()
        duplicate_slots_1_mask = slots_1_xcorr_normalized > duplicate_threshold # [batch_size, num_slots, num_slots]

        slots_2_dot_prod = torch.bmm(slots_projected_2, slots_projected_2.transpose(2, 1))
        norms_2 = torch.norm(slots_projected_2, p=2, dim=-1) # [batch_size, num_slots]
        product_of_norms_2 = torch.bmm(norms_2.unsqueeze(-1), norms_2.unsqueeze(-1).transpose(2,1))
        slots_2_xcorr_normalized = slots_2_dot_prod/product_of_norms_2
        slots_2_xcorr_detached_normalized = slots_2_xcorr_normalized.clone().detach().cpu().numpy()
        duplicate_slots_2_mask = slots_2_xcorr_normalized > duplicate_threshold # [batch_size, num_slots, num_slots]

        # create the mask of indices in the [n_mechanism by (n_slots ** 2)] cost matrix that have to
        # be set to a high number to prevent a duplicate slot from being selected when solving the LP.
        duplicate_mask_1 = torch.triu(duplicate_slots_1_mask).sum(-2)>1
        duplicate_mask_1 = duplicate_mask_1.unsqueeze(1).repeat(1,n_slots,1).permute(0, 2, 1)
        duplicate_mask_1 = duplicate_mask_1.unsqueeze(1).repeat(1,n_mechanisms,1,1).view(bs, n_mechanisms, -1)

        duplicate_mask_2 = torch.triu(duplicate_slots_2_mask).sum(-2)>1
        duplicate_mask_2 = duplicate_mask_2.unsqueeze(1).repeat(1,n_mechanisms,n_slots)

        # create the mask of indices in the [n_mechanism by (n_slots ** 2)] cost matrix that have to
        # be set to a high number to prevent background slots from being selected when solving the LP.
        background_slot_indices_1 = torch.argmin(torch.std(attention_scores_1, dim=-1), dim=-1, keepdim=True)
        background_slot_indices_2 = torch.argmin(torch.std(attention_scores_2, dim=-1), dim=-1, keepdim=True)
        background_slot_mask_1 = torch.nn.functional.one_hot(background_slot_indices_1, num_classes=n_slots).bool()
        background_slot_mask_1 = background_slot_mask_1.repeat(1,n_slots,1).permute(0, 2, 1).repeat(1,n_mechanisms,1).view(bs, n_mechanisms, -1)
        background_slot_mask_2 = torch.nn.functional.one_hot(background_slot_indices_2, num_classes=n_slots).bool()
        background_slot_mask_2 = background_slot_mask_2.repeat(1,n_mechanisms,n_slots).view(bs, n_mechanisms, -1)

        pairwise_cost = torch.nn.HuberLoss(reduction="none")(m_slots_projected_1_expanded, slots_projected_2_expanded).mean(dim=-1)

        # it is modified to manipulate the matching, but for actual computations of the loss, we need the differentiable unaltered version
        # of the costs
        pairwise_cost_temp = pairwise_cost.clone()

        # not necessary anymore
        v, i = torch.min(pairwise_cost_temp, dim=-2, keepdim=True)
        mask_col = pairwise_cost_temp > v.repeat(1, n_mechanisms, 1)
        v, i = torch.min(pairwise_cost_temp, dim=-1, keepdim=True)
        mask_row = pairwise_cost_temp > v.repeat(1, 1, n_slots ** 2)

        # ABSOLUTELY CRITICAL FOR THE SOLVER TO WORK PROPERLY! (float and double issue with precisions)
        pairwise_cost_temp /= v # .min()

        # not required anymore
        # pairwise_cost_temp[torch.logical_and(mask_col, mask_row)] = 5.0
        # replacing the indices that should not be selected by a high value
        inf_ = 500.0 # also the solution of lp is very sensitive to this value

        v, i = torch.max(pairwise_cost_temp, dim=-1, keepdim=True)
        # pairwise_cost_temp[torch.logical_and(mask_col, mask_row)] = ub
        pairwise_cost_temp[duplicate_mask_1] = inf_ # v.max() # v.repeat(1, 1, n_slots ** 2)[duplicate_mask_1]
        pairwise_cost_temp[duplicate_mask_2] = inf_ # v.max() # v.repeat(1, 1, n_slots ** 2)[duplicate_mask_2]
        pairwise_cost_temp[background_slot_mask_1] = inf_
        pairwise_cost_temp[background_slot_mask_2] = inf_

        # avoiding unbounded issues in the solver
        pairwise_cost_temp[pairwise_cost_temp > inf_] = inf_

        # solving the batch of lps
        # solution (assignment weights): [batch_size, n_mechanisms, num_slots ** 2]
        import time
        t0 = time.perf_counter()
        solution = self.cvxpylayer(pairwise_cost_temp.clone().detach())[0]
        print(time.perf_counter()-t0)
        w = solution.clone()
        t0 = time.perf_counter()

        # satisfying the simplex constraint for the distribution passed to torch.distributions. since the solver approximates, there can
        # be rare deviations from the constraint of 0 < w < 1. We address that here below.
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
        while resample_mask.any() and counter <= bs:
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

        indices_1 = indices_1.unsqueeze(-1).repeat(1, 1, 2)
        indices_2 = indices_2.unsqueeze(-1).repeat(1, 1, 2)

        permutations_1 = indices_1
        permutations_2 = indices_2
        object_zhat_reordered_1 = torch.gather(slots_projected_1, 1, permutations_1) # [batch_size, n_balls, z_dim]
        object_zhat_reordered_2 = torch.gather(slots_projected_2, 1, permutations_2) # [batch_size, n_balls, z_dim]

        # [batch_size, n_mechanisms]
        actual_costs = torch.gather(pairwise_cost, 2, indices.unsqueeze(-1)).float()
        # hungarian_loss: scalar
        hungarian_loss = torch.mean(torch.sum(actual_costs, dim=1))
        self.log(f"{stage}_matching_loss", hungarian_loss.item())

        reconstruction_losses = F.mse_loss(recon_combined_1, x1, reduction="mean") # + F.mse_loss(recon_combined_2, x2, reduction="mean")
        self.log(f"{stage}_reconstruction_loss", reconstruction_losses.item())

        # b: [batch_size, n_balls (n_mech), 2]
        baseline_loss = ((torch.norm(b, p=2, dim=-1).sum(dim=-1))**2).mean()
        self.log(f"{stage}_baseline_loss", baseline_loss.item())

        # indices: [batch_size, n_mechanisms, 2]
        indices = indices.unsqueeze(-1).repeat(1, 1, 2)
        indices[:,:,0] = torch.arange(n_mechanisms)
        indices_1[:,:,0] = torch.arange(n_mechanisms)
        indices_2[:,:,0] = torch.arange(n_mechanisms)
        
        # since this is an ablation, we only use reconstruction loss to guide slots, and we will see if any disentanglement result
        # can be achieved without the guidance of any latent loss
        loss =  self.w_recons_loss * reconstruction_losses # + self.w_latent_loss * hungarian_loss + self.w_similarity_loss * slot_similarity_loss # + 0.01 * ((1-sigma_1_prod) ** 2)

        self.log(f"{stage}_loss", loss.item())

        # TODO: might require detaching and cloning tensors before sending them out to logger
        self.eval()
        with torch.no_grad():
            if self.additional_logger:
                indices_1[:,:,0] = torch.arange(n_mechanisms)
                indices_2[:,:,0] = torch.arange(n_mechanisms)
                train = True if stage == "train" else False
                self.log_reconstructed_samples(
                    batch_1=x1,
                    z1=z1,
                    attention_masks_1=attention_scores_1,
                    recon_combined_1=recon_combined_1, recons_1=recons_1, masks_1=masks_1,
                    batch_2=x2,
                    z2=z2,
                    attention_masks_2=attention_scores_2,
                    recon_combined_2=recon_combined_2, recons_2=recons_2, masks_2=masks_2,
                    b=b,
                    colors=colors,
                    indices_1=indices_1,
                    indices_2=indices_2,
                    pairwise_cost=pairwise_cost,
                    num_slots=n_slots,
                    table_name=f"{stage}/Masks_and_Reconstructions_{self.global_step}",
                    # num_samples_to_log=3,
                    train=train,
                )
        self.train()

        if stage == "train":
            return {"loss": loss}
        else:
            # print(f"\n\nz1:{z1}\n\n\noffsets:\n{b.view(bs,-1)}\n\ndim_ratio:{object_zhat_reordered_1.view(bs,-1)[:,0]/object_zhat_reordered_1.view(bs,-1)[:,1]}\n\nobject_zhat_reordered_1:{object_zhat_reordered_1.view(bs,-1)}\n\nobject_zhat_reordered_2:{object_zhat_reordered_2.view(bs,-1)}\n\nResidual:\n{object_zhat_reordered_2.view(bs,-1)-object_zhat_reordered_1.view(bs,-1)-b.view(bs,-1)}")
            return {"loss": loss, "true_z": z1, "pred_z": object_zhat_reordered_1.view(bs,-1).detach()}


    def slots_cosine_dist(self, slots):
        # slots: [num_slots, slot_size]
        num_slots = slots.size(0)
        dot_products = [sum(slots[i] * slots[j])/(torch.norm(slots[i], 2)*torch.norm(slots[j], 2)) for i,j in product(range(num_slots),range(num_slots))]
        diag_products = [sum(slots[i] * slots[i])/(torch.norm(slots[i], 2)**2) for i in range(num_slots)]
        non_diag = sum(dot_products) - sum(diag_products)
        return non_diag


    @rank_zero_only
    def log_reconstructed_samples(
        self, batch_1, z1, attention_masks_1,
        recon_combined_1, recons_1, masks_1
        , batch_2, z2, attention_masks_2
        , recon_combined_2, recons_2, masks_2
        , b
        , colors
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

        figures = self.get_image_reconstructions(batch_1, z1, attention_masks_1
                                                , recon_combined_1, recons_1, masks_1
                                                , batch_2, z2, attention_masks_2
                                                , recon_combined_2, recons_2, masks_2
                                                , b
                                                , colors, indices_1, indices_2
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
                                    , attention_masks_1
                                    , recon_combined_1, recons_1, masks_1
                                    , batch_2
                                    , z2
                                    , attention_masks_2
                                    , recon_combined_2, recons_2, masks_2
                                    , b
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
        side = 4

        def add_margin(img, top, right, bottom, left, color):
            width, height, n_channels = img.shape
            new_width = width + right + left
            new_height = height + top + bottom
            new_image = np.ones((new_width, new_height, n_channels))
            new_image[top:new_height-bottom, left:new_width-right, :] = img
            return new_image

        def check_bounds(ball_xy, side):
            ball_x = int(64*ball_xy[0])
            ball_y = 64 - int(64*ball_xy[1])
            if ball_x+side >= 64 or ball_x-side <= 0 or ball_y+side >= 64:
                return True
            return False
        
        def draw_sqaure_border(img, side, center_x, center_y, color):
            last_y = np.min([img.shape[0]-1, center_y+side])
            last_x = np.min([img.shape[1]-1, center_x+side+1])
            img[center_y-side:last_y, center_x-side] = color
            img[center_y-side:last_y, center_x+side] = color
            img[center_y-side, center_x-side:last_x] = color
            img[last_y, center_x-side:last_x] = color
            return img

        def _clamp(array):
            array[array > 1.0] = 1.0
            array[array < 0.0] = 0.0
            return array

        renormalize = self.trainer.datamodule.renormalize()

        # indices: [batch_size, n_mechanism (n_balls), 2]
        indices_1 = indices_1.clone().detach().cpu().numpy()
        indices_2 = indices_2.clone().detach().cpu().numpy()
        recon_combined_1 = renormalize(recon_combined_1.permute(0,2,3,1).detach().cpu().numpy())
        recon_combined_2 = renormalize(recon_combined_2.permute(0,2,3,1).detach().cpu().numpy())
        recons_1 = renormalize(recons_1.detach().cpu().numpy())
        recons_2 = renormalize(recons_2.detach().cpu().numpy())
        masks_1 = masks_1.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]
        masks_2 = masks_2.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]

        figs = []

        for idx in range(num_samples_to_log):

            
            image = renormalize(batch_1[idx].permute(1,2,0).clone().detach().cpu().numpy())
            mask = image < 0.95
            image_t1 = renormalize(batch_2[idx].permute(1,2,0).clone().detach().cpu().numpy())
            image_t1_mask = image_t1.copy()
            image_t1_mask[mask] = image_t1[mask] * 0.8

            b_ = b[idx].clone().detach().cpu().numpy()
            recon_combined_1_ = recon_combined_1[idx]
            recon_combined_2_ = recon_combined_2[idx]
            recons_1_ = recons_1[idx]
            recons_2_ = recons_2[idx]
            masks_1_ = masks_1[idx]
            masks_2_ = masks_2[idx]

            # [n_mechanism (n_balls), 2]
            _indices_1 = indices_1[idx].copy()
            _indices_2 = indices_2[idx].copy()

            indices_1_ = _indices_1[:, [1, 0]] # [n_balls, 2], but now [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_1_ = indices_1_[np.argsort(indices_1_[:, 0])] # it's now sorted in ascending order of slot ids.
            indices_2_ = _indices_2[:, [1, 0]] # [n_balls, 2], but [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_2_ = indices_2_[np.argsort(indices_2_[:, 0])] # it's now sorted in ascending order of slot ids.
            colors_ = colors[idx].clone().detach().cpu().numpy()
            num_slots_ = num_slots
            
            fig, ax = plt.subplots(8, num_slots_ + 1, figsize=(24, 24))

            # t
            ax[0,0].imshow(_clamp(image))
            ax[0,0].set_title("Input Image at 't'")
            quiver_scale = 1.0 # 0.4
            quiver_width = 0.005
            # ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], np.linalg.norm(b_, 2, axis=1), scale=0.4)
            ax[1,0].quiver(np.zeros(b_.shape[0]), np.zeros(b_.shape[0]), b_[:,0], b_[:,1], color=colors_, scale=quiver_scale) #, width=quiver_width)
            # ax[1,0].quiver(z1[idx][:,0], z1[idx][:,1], b_[:,0], b_[:,1], color=colors_, scale=0.4)
            ax[1,0].set_title('Offsets')
            ax[5,0].quiver(np.concatenate((z1[idx].view(-1,2)[:,0].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                , np.concatenate((z1[idx].view(-1,2)[:,1].clone().detach().cpu().numpy(), np.array([0.0, 1.0])))
                , np.concatenate((b_[:,0], np.array([0.0, 0.0]))), np.concatenate((b_[:,1]
                , np.array([0.0, 0.0]))), color=colors_, scale=quiver_scale) #, width=quiver_width)
            ax[5,0].set_title('Offsets')
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
                        ball_id = ball_id[0]
                    ball_center_xy = z1[idx][ball_id*2:ball_id*2+2].clone().detach().cpu().numpy()
                    ball_x = int(64*ball_center_xy[0])
                    ball_y = 64 - int(64*ball_center_xy[1])
                    if check_bounds(ball_center_xy, side=side):                
                        temp = add_margin(temp, top, right, bottom, left, (1.0, 1.0, 1.0))
                        ball_x += left
                        ball_y += top
                else:
                    ball_id = None

                # recolor if needed:
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    temp[ball_y+3, ball_x-3:ball_x+3,:] = c
                else:
                    temp = image.copy()
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
                        ball_id = ball_id[0]
                    ball_center_xy = z2[idx][ball_id*2:ball_id*2+2].clone().detach().cpu().numpy()
                    ball_x = int(64*ball_center_xy[0])
                    ball_y = 64 - int(64*ball_center_xy[1])
                    if check_bounds(ball_center_xy, side=side):
                        temp_2 = add_margin(temp_2, top, right, bottom, left, (1.0, 1.0, 1.0))
                        ball_x += left
                        ball_y += top
                            
                else:
                    ball_id = None

                # recolor if needed:
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    temp_2[ball_y+3, ball_x-3:ball_x+3,:] = c
                else:
                    temp_2 = image_t1.copy()
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
