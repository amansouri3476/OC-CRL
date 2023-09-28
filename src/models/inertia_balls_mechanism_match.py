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
        self.save_hyperparameters(logger=False)

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

        # self.slot_projection_to_z.apply(init_weights)

        # ################## Having a ReLU breaks everything
        # self.slot_projection_to_z = torch.nn.Linear(self.hparams.encoder.slot_size, 2, bias=False)
        # ##################

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


    def _load_messages(self):
        print(f"MSE\nBaseline: {self.baseline}")

    def loss(self, m_z1, z2):
        return F.mse_loss(m_z1, z2, reduction="mean")


    def forward(self, batch, num_slots=None, num_iterations=None, slots_init=None):

        # `num_slots` and `num_iterations` keywords let us try the model with a different number of slots and iterations
        # in each pass
        num_slots = num_slots if num_slots is not None else None
        num_iterations = num_iterations if num_iterations is not None else None

        slots_init = slots_init if slots_init is not None else None

        # batch: batch of images -> [batch_size, num_channels, width, height]
        slots, attention_scores, slots_init = self.model(batch, num_slots=num_slots, num_iterations=num_iterations, slots_init=slots_init) # slots: [batch_size, num_slots, slot_size]

        # projection to target dimension
        slots_projected = self.slot_projection_to_z(slots) # slots_projected: [batch_size, num_slots, z_dim]
        # print(f"\n\n---------forward---------\n\nslots:{slots[0:15]}\nprojection:{slots_projected[0:15]}")

        # with torch.no_grad():
        #     for name, param in self.slot_projection_to_z.named_parameters():
        #         if len(param.size()) > 1: # excluding bias terms
        #             u,s,vh = torch.linalg.svd(param.data, full_matrices=False)
        #             print(f"\n-----weight recons SVD-----:\n{s}")

            # _, s_slots, _ = torch.svd(slots)
            # _, s_proj, _ = torch.svd(slots_projected)
            # print(f"SVD slots:\n{s_slots}\n\nSVD proj:\n{s_proj}")

        return slots, slots_projected, attention_scores, slots_init


    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            for name, param in self.slot_projection_to_z.named_parameters():
                if len(param.size()) > 1: # excluding bias terms
                    u,s,vh = torch.linalg.svd(param.data, full_matrices=False)
                    eps = torch.tensor([0.01], device=param.device)
                    param.data = u @ torch.diag(torch.maximum(s, eps)) @ vh
                    # print(f"\n-----HI:{torch.maximum(s, eps)}\n\nname:{name}\n\nparam:{param.data}")

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
        # slots: [batch_size, num_slots, slot_size]
        # slots_projected: [batch_size, num_slots, z_dim]

        slots_1, slots_projected_1, attention_scores_1, slots_init_1 = self(x1)
        slots_2, slots_projected_2, attention_scores_2, slots_init_2 = self(x2, slots_init=slots_init_1)
        n_slots = slots_1.size(1)

        # Loss for similarity of slot representations

        # [batch_size, n_slots, n_slots]
        all_dot_products = torch.bmm(slots_1, slots_1.permute(0,2,1))
        # [batch_size, n_slots]
        all_norms = torch.norm(slots_1, p=2, dim=-1)
        # [batch_size, n_slots, n_slots]
        normalized_prods = all_dot_products/all_norms.unsqueeze(1)
        diag_prods = torch.diagonal(normalized_prods, dim1=-1, dim2=-2) # same slot products cosine distance should be removed
        # squared to avoid negative values pushing the loss to -inf
        slot_similarity_loss = ((normalized_prods.sum(-1).sum(-1) - diag_prods.sum(-1)).mean()) ** 2 
        self.log(f"{stage}_slot_similarity_loss", slot_similarity_loss.item())

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

        # [batch_size, num_mechanisms, num_slots, z_dim]
        slots_projected_1_expanded = slots_projected_1_expanded.permute(0,2,1,3)
        slots_projected_2_expanded = slots_projected_2_expanded.permute(0,2,1,3)
        m_slots_projected_1_expanded = m_slots_projected_1_expanded.permute(0,2,1,3)

        # Since we have initialized slots at t,t+1 the same way, 'most likely' the slots will be in the same order
        # , so we don't try to align them.

        hungarian_loss, indices = self.compute_loss_and_reorder(m_slots_projected_1_expanded, slots_projected_2_expanded)
        # hungarian_loss: scalar
        # indices: [batch_size, n_mechanisms, 2]

        # reorder slots projections. Note that slots projections that do not correspond to objects 
        # are dropped in this operation.

        # [batch_size, n_mechanism (n_balls), 2], slot id corresponding to each mechanism. Since both t and t+1 
        # are initialized with the same values, we gather projections with the same permutations because the 
        # order of slots at t, t+1 is (or should be) the same.
        permutations = indices[:, :, 1].unsqueeze(-1).repeat(1, 1, 2)
        
        object_zhat_reordered_1 = torch.gather(slots_projected_1, 1, permutations) # [batch_size, n_balls, z_dim]
        object_zhat_reordered_2 = torch.gather(slots_projected_2, 1, permutations) # [batch_size, n_balls, z_dim]
                
        loss =  hungarian_loss + slot_similarity_loss # + 0.01 * ((1-sigma_1_prod) ** 2)

        self.log(f"{stage}_loss", loss.item())

        # TODO: might require detaching and cloning tensors before sending them out to logger
        self.eval()
        with torch.no_grad():
            if self.additional_logger:
                train = True if stage is "train" else False
                self.log_reconstructed_samples(
                    batch_1=x1,
                    z1=z1,
                    attention_masks_1=attention_scores_1,
                    batch_2=x2,
                    z2=z2,
                    attention_masks_2=attention_scores_2,
                    colors=colors,
                    indices=indices,
                    num_slots=n_slots,
                    table_name=f"{stage}/Attention_masks_{self.global_step}",
                    train=train,
                )
        self.train()
        
        if stage is "train":
            return {"loss": loss}
        else:
            return {"loss": loss, "true_z": z1, "pred_z": object_zhat_reordered_1.view(bs,-1)}


    def compute_loss_and_reorder(self, predictions, targets):
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
            x: Batch of sets of size [batch_size, n_mechanisms, n_slots, dim_points]. (dim_points=2)
            y: Batch of sets of size [batch_size, n_mechanisms, n_slots, dim_points].
        Returns:
            Minimum distance between all sets in the two batches.
        """

        x = predictions
        y = targets
        device = x.device

        # TOOD: This loss can be a MSE loss instead of Huber loss
        # [batch_size, n_mechanisms, n_slots]
        pairwise_cost = torch.nn.HuberLoss(reduction="none")(x, y).mean(dim=-1)

        # Each element in the batch has results in a tensor of pairwise costs
        # Each tensor of pairwise costs has an optimal assignment, and this 
        # assignment is returned in terms of a list for rows and columns corresponding
        # to the optimal assignment.
        pairwise_cost_detached = pairwise_cost.clone().detach().cpu()
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
        
        return torch.mean(torch.sum(actual_costs, dim=1)), transposed_indices


    def slots_cosine_dist(self, slots):
        # slots: [num_slots, slot_size]
        num_slots = slots.size(0)
        dot_products = [sum(slots[i] * slots[j])/(torch.norm(slots[i], 2)*torch.norm(slots[j], 2)) for i,j in product(range(num_slots),range(num_slots))]
        diag_products = [sum(slots[i] * slots[i])/(torch.norm(slots[i], 2)**2) for i in range(num_slots)]
        non_diag = sum(dot_products) - sum(diag_products)
        return non_diag


    @rank_zero_only
    def log_reconstructed_samples(
        self, batch_1, z1, attention_masks_1
        , batch_2, z2, attention_masks_2
        , colors
        , indices
        , num_slots
        , table_name
        , train
        , additional_to_log={}
    ):
        logger = self.additional_logger
        num_samples_to_log = logger.num_samples_to_log(trainer=self.trainer, train=train)

        if num_samples_to_log <= 0:
            return


        figures = self.get_image_reconstructions(batch_1, z1, attention_masks_1
                                                , batch_2, z2, attention_masks_2
                                                , colors, indices
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
                                    , batch_2
                                    , z2
                                    , attention_masks_2
                                    , colors
                                    , indices
                                    , num_slots
                                    , num_samples_to_log):

        import matplotlib.pyplot as plt
        renormalize = self.trainer.datamodule.renormalize()

        # indices: [batch_size, n_mechanism (n_balls), 2]
        indices = indices.clone().detach().cpu().numpy()

        figs = []

        for idx in range(num_samples_to_log):

            
            image = renormalize(batch_1[idx].permute(1,2,0).clone().detach().cpu().numpy())
            mask = image < 0.95
            image_t1 = renormalize(batch_2[idx].permute(1,2,0).clone().detach().cpu().numpy())
            image_t1_mask = image_t1.copy()
            image_t1_mask[mask] = image_t1[mask] * 0.8

            # [n_mechanism (n_balls), 2]
            indices_1 = indices[idx].copy()
            indices_2 = indices[idx].copy()

            indices_1_ = indices_1[:, [1, 0]] # [n_balls, 2], but now [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_1_ = indices_1_[np.argsort(indices_1_[:, 0])] # it's now sorted in ascending order of slot ids.
            indices_2_ = indices_2[:, [1, 0]] # [n_balls, 2], but [:, 0] gives slot ids, and [:, 1] gives ball ids
            indices_2_ = indices_2_[np.argsort(indices_2_[:, 0])] # it's now sorted in ascending order of slot ids.
            colors_ = colors[idx].clone().detach().cpu().numpy()
            num_slots_ = num_slots
            
            fig, ax = plt.subplots(4, num_slots_ + 1, figsize=(15, 12))
            ax[0,0].imshow(image)
            ax[0,0].set_title('Image')
            ax[1,0].imshow(image) 
            ax[1,0].set_title('Image')
            ax[2,0].imshow(image_t1_mask) 
            ax[2,0].set_title("Image 't+1'")
            ax[3,0].imshow(image_t1_mask) 
            ax[3,0].set_title("Image 't+1'")
            
            for slot_id in range(num_slots_):

                # ---------- step t ---------- #
                temp = image.copy()
                # For each slot, find the corresponding ball and its color
                # Then recolor the reconstruction with the color
                if slot_id in indices_1_[:, 0]:
                    ball_id = indices_1_[(indices_1_[:, 0]==slot_id).nonzero(), 1].squeeze()
                    ball_center_xy = z1[idx][ball_id*2:ball_id*2+2].clone().detach().cpu().numpy()
                else:
                    ball_id = None

                # recolor if needed:
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    ball_x = int(64*ball_center_xy[0])
                    ball_y = 64 - int(64*ball_center_xy[1])
                    temp[ball_y+3, ball_x-3:ball_x+3,:] = c
                else:
                    temp = image.copy()
                ax[0, slot_id + 1].imshow(temp)
                ax[0, slot_id + 1].set_title(f'Slot {slot_id}')

                # ---------- step t+1 ---------- #

                temp_2 = image_t1.copy()
                # For each slot, find the corresponding ball and its color
                # Then recolor the reconstruction with the color
                if slot_id in indices_2_[:, 0]:
                    ball_id = indices_2_[(indices_2_[:, 0]==slot_id).nonzero(), 1].squeeze()
                    ball_center_xy = z2[idx][ball_id*2:ball_id*2+2].clone().detach().cpu().numpy()
                else:
                    ball_id = None

                # recolor if needed:
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    ball_x = int(64*ball_center_xy[0])
                    ball_y = 64 - int(64*ball_center_xy[1])
                    temp_2[ball_y+3, ball_x-3:ball_x+3,:] = c
                else:
                    temp_2 = image_t1.copy()
                ax[2, slot_id + 1].imshow(temp_2)
                ax[2, slot_id + 1].set_title(f'Slot {slot_id}')

                
                attn = attention_masks_1[idx].reshape(-1, 64, 64)[slot_id].clone().detach().cpu().numpy()
                ax[1, slot_id + 1].imshow(attn)
                ax[1, slot_id + 1].set_title(f"Attention mask 't'")
                attn = attention_masks_2[idx].reshape(-1, 64, 64)[slot_id].clone().detach().cpu().numpy()
                ax[3, slot_id + 1].imshow(attn)
                ax[3, slot_id + 1].set_title(f"Attention mask 't+1'")


            for i, j in product(range(ax.shape[0]), range(ax.shape[1])):
                ax[i, j].grid(False)
                ax[i, j].axis('off')

            figs.append(fig)

        return figs
