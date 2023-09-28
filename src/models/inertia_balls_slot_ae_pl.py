from typing import Any, Dict, List, Callable, Union, TypeVar, Tuple, Container
import numpy as np
from itertools import permutations, product
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import hydra
from omegaconf import OmegaConf
import wandb
import src
from src.utils.additional_loggers import get_img_rec_table_data
from src.utils.general import init_weights
from src.utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
from src.utils.segmentation_metrics import adjusted_rand_index
from sklearn import random_projection
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
# import time

class InertiaBallsSlotAttentionAE(pl.LightningModule):
    """
    This class implements the vanilla slot attention autoencoder for our environment.
    It is only trained for reconstructing the input images, thus slots are not projected
    to any lower dimension latent space.
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
        # it also allows for access to params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # slot_attention_autoencoder
        self.model = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)
        if self.hparams["encoder_ckpt_path"] is not None:
            ckpt_path = self.hparams["encoder_ckpt_path"]
            # only load the weights, i.e. HPs should be overwritten from the passed config
            # b/c maybe the ckpt has num_slots=7, but we want to test it w/ num_slots=12
            # NOTE: NEVER DO self.model = self.model.load_state_dict(...), raises _IncompatibleKey error
            self.model.load_state_dict(torch.load(ckpt_path))

        # freeze the parameters of encoder if needed
        if self.hparams.encoder_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if kwargs.get("additional_logger"):
            self.additional_logger = hydra.utils.instantiate(self.hparams.additional_logger)
        else:
            self.additional_logger = None

        self.compute_mcc = kwargs.get("compute_mcc", False)
        
        
    def forward(self, batch, num_slots=None, num_iterations=None, slots_init=None):

        # `num_slots` and `num_iterations` keywords let us try the model with a different number of slots and iterations
        # in each pass
        num_slots = num_slots if num_slots is not None else None
        num_iterations = num_iterations if num_iterations is not None else None

        # batch: batch of images -> [batch_size, num_channels, width, height]
        recon_combined, recons, masks, slots, attention_scores, slots_init = self.model(batch, num_slots=num_slots, num_iterations=num_iterations, slots_init=slots_init) # slots: [batch_size, num_slots, slot_size]

        return slots, recon_combined, recons, masks, attention_scores, slots_init

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
        ['latents', 'images', 'matrices', 'colors']
        
        - 'latents': (Tuple) (z1, z2), latents for two consecutive timesteps, each having a dimension of [batch_size, n_balls*2]
        - 'images': (Tuple) (x1, x2), images corresponding to latents (z1,z2) for two consecutive timesteps, each having
        a dimension of [batch_size, num_channels, width, height]
        - 'matrices': (Tuple) (A,b), matrices that determine the mechanisms (i.e. offsets)
        """    

        (z1, z2), (x1, x2), (segmentation_masks1, segmentation_masks2), (coordinates_1, coordinates_2), (colors_1, colors_2) = batch["latents"], batch["images"], batch["segmentation_masks"], batch["coordinates"], batch["colors"]

        img_width = x1.shape[-2]
        img_height = x1.shape[-1]
        # Get the images in the batch, pass them through AE, get the reconstructions, compute the loss, repeat
        images = x1 # [batch_size, num_channels, width, height]
        slots_1, recon_combined_1, recons_1, masks_1, attention_scores_1, slots_init_1 = self(x1)
        slots_2, recon_combined_2, recons_2, masks_2, attention_scores_2, slots_init_2 = self(x2, slots_init=slots_init_1)

        n_slots = slots_1.size(1)
        slot_dim = slots_1.shape[-1]

        bs = z1.size(0)
        n_balls = self.hparams["n_balls"]

        loss = self.loss(recon_combined_1, x1) + self.loss(recon_combined_2, x2)
        
        self.log(f"{stage}_loss", loss.item())

        # computing ARI scores (assumes n_balls+1 segmentation masks and predictions, so background is included as well)
        # compute ARI only on validation and test dataset as it slows down the pipiline
        # if stage != "train":
        segmentation_masks1_ = segmentation_masks1.squeeze().reshape(bs, n_balls+1, -1).permute(0, 2, 1)
        segmentation_masks1_one_hot = F.one_hot(torch.argmax(segmentation_masks1_, -1))

        # TODO: remove background slots, and pass the rest for mcc computation
        # Important: The following assumes n_slots=n_balls+1 so removing the background
        # and passing the rest would be fine, but more generally, we should sort the slots
        # based on the std of the attention scores, remove duplicate ones, and pick the top
        # n_ball stds as the predicted segmentation masks of the balls. THIS IS NOT WHAT
        # HAPPENS BELOW. IT HAS TO BE IMPLEMENTED LATER.
        # TODO: Note that the following only works when n_slots=n_balls+1
        # prediction masks are of shape [batch_size, num_slots, width, height, 1], so if num_slots > n_balls+1, then we should
        # choose n_balls+1 of the masks as predictions to compute the ARI. Our way is to compute the std of the masks, and
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
        pred_segmentation_masks1 = pred_segmentation_masks1.squeeze().reshape(bs, n_balls+1, -1).permute(0, 2, 1)
        pred_segmentation_masks1_one_hot = F.one_hot(torch.argmax(pred_segmentation_masks1, -1))

        # pred_segmentation_masks1 = masks.squeeze().reshape(bs, n_balls+1, -1).permute(0, 2, 1)
        # pred_segmentation_masks1_one_hot = F.one_hot(torch.argmax(pred_segmentation_masks1, -1))

        ari = adjusted_rand_index(segmentation_masks1_one_hot, pred_segmentation_masks1_one_hot).mean()
        self.log(f"{stage}_ARI", ari)

        # finding the slot corresponding to the changed ball. we need to find that slot both at t, t+1
        segmentation_masks1_ = (segmentation_masks1 > 0.)*1. # [batch_size, n_balls+1, width, height, 1]
        segmentation_masks2_ = (segmentation_masks2 > 0.)*1. # [batch_size, n_balls+1, width, height, 1]

        # [batch_size, n_slots, n_slots]
        diff_1 = ((segmentation_masks1_.reshape(bs, n_slots, -1)[:, :, None] * masks_1.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)
        # segmentation_slot_mapping_1[:, i]=j means that i-th segmentation mask correspond to j-th slot
        _, segmentation_slot_mapping_1 = torch.max(diff_1, dim=-1) # [batch_size, n_slots=n_segmentation_masks]

        diff_2 = ((segmentation_masks2_.reshape(bs, n_slots, -1)[:, :, None] * masks_2.reshape(bs, n_slots, -1)[:, None, :])**2).sum(-1)
        _, segmentation_slot_mapping_2 = torch.max(diff_2, dim=-1) # [batch_size, n_slots=n_segmentation_masks]

        # slot_object_mapped[0] will always be the background, slot_object_mapped[1] will always correspond to object 1, and so on
        slot_object_mapped = torch.gather(slots_1, 1, segmentation_slot_mapping_1.unsqueeze(-1).repeat(1,1,slot_dim))

        # TODO: might require detaching and cloning tensors before sending them out to logger
        self.eval()
        with torch.no_grad():
            if self.additional_logger is not None:
                train = True if stage == "train" else False
                self.log_reconstructed_samples(
                    batch_1=x1,
                    attention_masks_1=attention_scores_1,
                    recon_combined_1=recon_combined_1, recons_1=recons_1, masks_1=masks_1,
                    colors=colors_1,
                    num_slots=n_slots,
                    table_name=f"{stage}/Masks_and_Reconstructions_{self.global_step}",
                    # num_samples_to_log=3,
                    train=train,
                )
        self.train()

        if stage == "train":
            return {"loss": loss}
        else:
            # TODO: remove background slots, and pass the rest for mcc computation
            # Important: The following assumes n_slots=n_balls+1 so removing the background
            # and passing the rest would be fine, but more generally, we should sort the slots
            # based on the std of the attention scores, remove duplicate ones, and pick the top
            # n_ball stds as the predicted segmentation masks of the balls. THIS IS NOT WHAT
            # HAPPENS BELOW. IT HAS TO BE IMPLEMENTED LATER.
            # find the index of slot corresponding to the background for each image
            # TODO: Note that the following only works when n_slots=n_balls+1
            
            # misaligned z and pred_z
            # background_slot_indices = torch.argmin(torch.std(attention_scores_1, dim=-1), dim=-1, keepdim=True) # [batch_size, 1]
            # background_slot_mask = torch.nn.functional.one_hot(background_slot_indices, num_classes=n_slots).bool() # [batch_size, 1, num_slots]
            # non_background_slots_mask =  ~background_slot_mask
            # non_background_obj_slots = slots_1[non_background_slots_mask.squeeze(),:].view(bs, -1, slot_dim)
            # return {"loss": loss, "true_z": z1, "obj_slots": non_background_obj_slots.detach()}


            return {"loss": loss, "true_z": z1, "obj_slots": slot_object_mapped[:, 1:, :].detach()}
            # return {"loss": loss, "true_z": z1, "obj_slots": non_background_obj_slots.view(bs,-1).detach()}


    def validation_epoch_end(self, validation_step_outputs):

        # compute the MCC score based on 3 methods: Random Projections, Regression, Principle Components
        if not self.compute_mcc:
            return
        
        z_disentanglement = [v["true_z"] for v in validation_step_outputs] # each element will be [batch_size, n_balls * z_dim]
        z_disentanglement = torch.cat(z_disentanglement, 0) # [dataset_size, n_balls * z_dim]
        n_balls = self.hparams["n_balls"]
        z_dim = z_disentanglement.shape[-1]//n_balls
        obj_slots = [v["obj_slots"] for v in validation_step_outputs] # each element will be [batch_size, n_balls, slot_dim]
        obj_slots = torch.cat(obj_slots, 0) # [dataset_size, n_balls, slot_dim]
        slot_dim = obj_slots.shape[-1]
        # random projection
        transformer = random_projection.GaussianRandomProjection(n_components=z_dim, eps=0.1)
        obj_projected = torch.tensor(transformer.fit_transform(obj_slots.reshape(-1, slot_dim).clone().cpu()), device=obj_slots.device) # [dataset_size * n_balls, z_dim]
        h_z_disentanglement = obj_projected.reshape(-1, n_balls * z_dim) # [dataset_size, n_balls * z_dim]
        self.compute_and_log_mcc(z_disentanglement, h_z_disentanglement, "RP")

        # linear regression
        reg = LinearRegression().fit(obj_slots.reshape(-1, n_balls * slot_dim).clone().cpu(), z_disentanglement.reshape(-1, n_balls * z_dim).clone().cpu())
        obj_projected =  torch.tensor(reg.predict(obj_slots.reshape(-1, n_balls * slot_dim).clone().cpu()), device=obj_slots.device) # [dataset_size * n_balls, z_dim]
        h_z_disentanglement = obj_projected.reshape(-1, n_balls * z_dim) # [dataset_size, n_balls * z_dim]
        self.compute_and_log_mcc(z_disentanglement, h_z_disentanglement, "regression")

        # principal components
        pca = PCA(n_components=z_dim)
        obj_projected = torch.tensor(pca.fit_transform(obj_slots.reshape(-1, slot_dim).clone().cpu()), device=obj_slots.device) # [dataset_size * n_balls, z_dim]
        h_z_disentanglement = obj_projected.reshape(-1, n_balls * z_dim) # [dataset_size, n_balls * z_dim]
        self.compute_and_log_mcc(z_disentanglement, h_z_disentanglement, "pca")


    def configure_optimizers(self):
        
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                   , self.parameters()
                                                                  )
            
        if self.hparams.get("scheduler_config"):
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
        else:
            # no scheduling
            return [optimizer]


    @rank_zero_only
    def log_reconstructed_samples(
        self, batch_1, attention_masks_1,
        recon_combined_1, recons_1, masks_1
        , colors
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

        figures = self.get_image_reconstructions(batch_1, attention_masks_1
                                                , recon_combined_1, recons_1, masks_1
                                                , colors
                                                , num_slots
                                                , num_samples_to_log)

        columns, data = get_img_rec_table_data(
            imgs=figures,
            step=self.trainer.global_step,
            num_samples_to_log=num_samples_to_log,
        )

        logger.log_table(table_name=table_name, train=train, columns=columns, row_list=data)


    def get_image_reconstructions(self, batch_1
                                    , attention_masks_1
                                    , recon_combined_1, recons_1, masks_1
                                    , colors
                                    , num_slots
                                    , num_samples_to_log):

        import matplotlib.pyplot as plt
        plt.cla()
        plt.close('all')

        img_width = batch_1[0].shape[-2]
        img_height = batch_1[0].shape[-1]

        right = 10
        left = 10
        top = 10
        bottom = 10
        side = 4

        def _clamp(array):
            array[array > 1.0] = 1.0
            array[array < 0.0] = 0.0
            return array

        renormalize = self.trainer.datamodule.renormalize()

        recon_combined_1 = renormalize(recon_combined_1.permute(0,2,3,1).detach().cpu().numpy())
        recons_1 = renormalize(recons_1.detach().cpu().numpy())
        masks_1 = masks_1.detach().cpu().numpy() # shape: [batch_size, num_slots, width, height, 1]
        
        figs = []

        for idx in range(num_samples_to_log):

            
            image = renormalize(batch_1[idx].permute(1,2,0).clone().detach().cpu().numpy())
            mask = image < 0.95
            
            recon_combined_1_ = recon_combined_1[idx]
            recons_1_ = recons_1[idx]            
            masks_1_ = masks_1[idx]

            colors_ = colors[idx].clone().detach().cpu().numpy()
            num_slots_ = num_slots
            
            fig, ax = plt.subplots(3, num_slots_ + 1, figsize=(24, 24))

            # t
            ax[0,0].imshow(_clamp(image))
            ax[0,0].set_title("Input Image at 't'")
            ax[1,0].imshow(_clamp(masks_1_.sum(axis=0)), vmin=0, vmax=1)
            ax[1,0].set_title("Decoder Masks sum 't'")
            # ax[2,0].imshow((recon_combined_1_ * 255).astype(np.uint8), vmin=0, vmax=255)
            ax[2,0].imshow(_clamp(recon_combined_1_), vmin=0, vmax=1)
            ax[2,0].set_title("Reconstruction 't'")

            for slot_id in range(num_slots_):

                # Attention Masks
                attn = attention_masks_1[idx].reshape(-1, img_width, img_height)[slot_id].clone().detach().cpu().numpy()
                ax[0, slot_id + 1].imshow(_clamp(attn))
                ax[0, slot_id + 1].set_title(f"Attention mask 't'")

                # Decoder Masks
                ax[1, slot_id + 1].imshow(_clamp(masks_1_[slot_id]), vmin=0, vmax=1)
                ax[1, slot_id + 1].set_title(f"Slot {slot_id} Decoder Masks 't'")

                # Reconstruction Per Slot
                rec = recons_1_[slot_id] * masks_1_[slot_id] + (1 - masks_1_[slot_id])
                ax[2, slot_id + 1].imshow(_clamp(rec), vmin=0, vmax=1)
                ax[2, slot_id + 1].set_title(f"Slot {slot_id} Recons 't'")

            for i, j in product(range(ax.shape[0]), range(ax.shape[1])):
                ax[i, j].grid(False)
                ax[i, j].axis('off')

            figs.append(fig)

        return figs


    def compute_and_log_mcc(self, z_disentanglement, h_z_disentanglement, method):

        (linear_disentanglement_score, _), _ = linear_disentanglement(
            z_disentanglement, h_z_disentanglement, mode="r2", train_test_split=True
        )

        (permutation_disentanglement_score, _), _ = permutation_disentanglement(
            z_disentanglement,
            h_z_disentanglement,
            mode="pearson",
            solver="munkres",
            rescaling=True,
        )
        mse = F.mse_loss(z_disentanglement, h_z_disentanglement).mean(0)
        self.log(f"Linear_Disentanglement_{method}", linear_disentanglement_score, prog_bar=True)
        self.log(
            f"Permutation_Disentanglement_{method}",
            permutation_disentanglement_score,
            prog_bar=True,
        )
        self.log(f"MSE_{method}", mse, prog_bar=True)
        wandb.log(
            {
                f"mse ({method})": mse,
                f"Permutation Disentanglement ({method})": permutation_disentanglement_score,
                f"Linear Disentanglement ({method})": linear_disentanglement_score,
            }
        )