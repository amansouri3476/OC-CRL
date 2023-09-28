from http.client import ImproperConnectionState
from typing import Any, Dict, List, Callable, Union, TypeVar, Tuple, Container
import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import wandb
import torch
import numpy as np
import scipy
from scipy import optimize
from pytorch_lightning.utilities import rank_zero_only
import src.utils.additional_loggers
import warnings
warnings.filterwarnings("ignore")

class InertiaBallsSlotAttentionAESet(pl.LightningModule):
    """
    This lightning module will have a slot-attention autoencoder, and an mlp to project slot repr to colors, 
    and a reorder module.
    it will take input images, mappes them to slots, slots get mapped to 3 dimensional vectors representing colors.
    The set of slot-colors will be passed to the reorder module which computes the assignment based on predicted 
    slot-color sets and the ground truth set of colors. Then the reordered slot-color set will be passed to mse error
    module to compute the actual loss. This loss will update the parameters of mlp projection (from slots to colors).
    Slot attention AE weights will be frozen during training.
    """

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
        self.slot_autoencoder = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)
        if self.hparams["encoder_ckpt_path"] is not None:
            ckpt_path = self.hparams["encoder_ckpt_path"]
            self.slot_autoencoder = torch.load(ckpt_path)

        # freeze the parameters of slot_attention autoencoder if needed
        if self.hparams.encoder_freeze:
            for param in self.slot_autoencoder.parameters():
                param.requires_grad = False

        # projection
        self.slot_projection_mlp = torch.nn.Sequential(*[hydra.utils.instantiate(layer_config) for _, layer_config in self.hparams.projection_module.items()])


        if kwargs.get("additional_logger"):
            self.additional_logger = hydra.utils.instantiate(self.hparams.additional_logger)
        else:
            self.additional_logger = None
        

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

        actual_costs = torch.gather(pairwise_cost, 2, transposed_indices[:,:,1].unsqueeze(-1)).float()
        
        return torch.mean(torch.sum(actual_costs, dim=1)), transposed_indices

        
    def reorder(self, x, indices):
        """
        Reorder input tensor x based on the matching.

        Args:
            x: [batch_size, n_points, dim_points]
            indices: []
        """
        pass

        
    def forward(self, batch):

        # batch: batch of images -> [batch_size, num_channels, width, height]
        recon_combined, recons, masks, slots = self.slot_autoencoder(batch) # slots: [batch_size, num_slots, slot_size]

        # projection to target dimension
        slots_projected = self.slot_projection_mlp(slots) # slots_projected: [batch_size, num_slots, target_dim]

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
        
        # Get the images in the batch, pass them through AE, get the reconstructions, compute the loss, repeat
        images = x1 # [batch_size, num_channels, width, height]

        slots, slots_projected, recon_combined, recons, masks = self(images)

        # TODO: This should be decided in config, whether to use colors, or something else.
        targets = colors.float()
        # targets = z1.view(z1.size(0), -1, 2).float() # [batch_size, n_balls, 2]

        # slots_projected: [batch_size, num_slots, target_dim]
        # targets: [batch_size, n_balls, 3 or 2]
        loss, indices = self.compute_loss_and_reorder(slots_projected, targets)
        # loss: scalar
        # indices: [batch_size, min(num_slots, n_balls), 2]


        # if slot_attention weights are not frozen, then their share of the loss should be computed and added
        if not self.hparams.encoder_freeze:
            loss = loss + F.mse_loss(recon_combined, images, reduction="mean")

        self.log(f"{stage}_loss", loss.item())

        # TODO: might require detaching and cloning tensors before sending them out to logger
        self.eval()
        with torch.no_grad():
            if self.additional_logger:
                train = True if stage is "train" else False
                self.log_reconstructed_samples(
                    batch=images,
                    recon_combined=recon_combined,
                    recons=recons,
                    masks=masks,
                    colors=colors,
                    indices=indices,
                    table_name=f"{stage}/rec_{self.global_step}",
                    train=train,
                )
        self.train()
        
        return {"loss": loss}
    
    
    def configure_optimizers(self):
        
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                   , params
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


    @rank_zero_only
    def log_reconstructed_samples(
        self, batch, recon_combined, recons, masks, colors, indices, table_name, train, additional_to_log={}
    ):
        logger = self.additional_logger
        num_samples_to_log = logger.num_samples_to_log(trainer=self.trainer, train=train)

        if num_samples_to_log <= 0:
            return


        figures = self.get_image_reconstructions(batch, recon_combined, recons, masks, colors, indices, num_samples_to_log)

        columns, data = slot_based_disentanglement.utils.additional_loggers.get_img_rec_table_data(
            imgs=figures,
            step=self.trainer.global_step,
            num_samples_to_log=num_samples_to_log,
        )

        logger.log_table(table_name=table_name, train=train, columns=columns, row_list=data)


    def get_image_reconstructions(self, batch, recon_combined, recons, masks, colors, indices, num_samples_to_log):

        # indices: [batch_size, min(num_slots, n_balls), 2]
        # if n_balls < num_slots, then indices[0, 2, 1] gives the id of the matching slot to ball 2 (for 
        # element 0 in the batch)

        import matplotlib.pyplot as plt
        renormalize = self.trainer.datamodule.renormalize()

        recon_combined = recon_combined.permute(0,2,3,1).detach().cpu().numpy()
        recons = recons.detach().cpu().numpy()
        images = renormalize(batch.permute(0,2,3,1))
        recon_combined = renormalize(recon_combined)
        recons = renormalize(recons)
        masks = masks.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]

        colors = colors.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()
        figs = []

        for idx in range(num_samples_to_log):

            image = images[idx].cpu().numpy()
            recon_combined_ = recon_combined[idx]
            recons_ = recons[idx]
            masks_ = masks[idx]
            
            
            indices_ = indices[idx] # [min(num_slots, n_balls), 2]
            indices_ = indices_[:, [1, 0]] # [n_balls, 2], but [:, 0] gives slot ids, and [:, 1] gives ball ids
            # indices_ = indices_[indices_[:, 0].sort()[1]] # it's now sorted in ascending order of slot ids.
            indices_ = indices_[np.argsort(indices_[:, 0])] # it's now sorted in ascending order of slot ids.

            colors_ = colors[idx] # [n_balls, 3]
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

                # For each slot, find the corresponding ball and its color
                # Then recolor the reconstruction with the color
                
                if i in indices_[:, 0]:
                    ball_id = indices_[(indices_[:, 0]==i).nonzero(), 1].squeeze()
                else:
                    ball_id = None

                temp = recons_[i] * masks_[i] + (1 - masks_[i])

                # recolor if needed:
                if ball_id is not None:
                    c = colors_[ball_id, :]
                    ball_pixels = (temp.mean(-1) <= 0.95)
                    temp[ball_pixels] = c

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


            figs.append(fig)

        return figs