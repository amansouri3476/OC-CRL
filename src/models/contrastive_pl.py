import torch
from torch.nn import functional as F
from .base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import wandb
from src.utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement


class Contrastive(BasePl):
    def __init__(
        self,
        base_architecture: str = "resnet18",
        n_frames: int = 1,
        num_slots: int = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.contrastive_params = self.hparams["contrastive"]
        
        # TODO: it seems that DistributedDataParallel is preferred over DataParallel. See below
        # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

        if num_slots is not None:
            self.hparams.encoder["num_slots"] = num_slots
        # slot_attention_autoencoder or resnet18 encoder or etc.
        # self.model = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)
        self.model = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)
        if self.hparams.get("encoder_ckpt_path", None) is not None:    
            ckpt_path = self.hparams["encoder_ckpt_path"]
            # only load the weights, i.e. HPs should be overwritten from the passed config
            # b/c maybe the ckpt has num_slots=7, but we want to test it w/ num_slots=12
            # NOTE: NEVER DO self.model = self.model.load_state_dict(...), raises _IncompatibleKey error
            self.model.load_state_dict(torch.load(ckpt_path))
            self.hparams.pop("encoder_ckpt_path") # we don't want this to be save with the ckpt, sicne it will raise key errors when we further train the model
                                                  # and load it for evaluation.

        # remove the state_dict_randomstring.ckpt to avoid cluttering the space
        import os
        import glob
        state_dicts_list = glob.glob('./state_dict_*.pth')
        # for state_dict_ckpt in state_dicts_list:
        #     try:
        #         os.remove(state_dict_ckpt)
        #     except:
        #         print("Error while deleting file: ", state_dict_ckpt)

        # freeze the parameters of encoder if needed
        if self.hparams.encoder_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        else: # if the flag is set to true we should correct the requires_grad flags, i.e. we might
              # initially freeze it for some time, but then decide to let it finetune.
            for param in self.model.parameters():
                param.requires_grad = True

        self.alpha = self.contrastive_params["alpha"]
        self.baseline = self.contrastive_params["baseline"]
        self.normalize = self.contrastive_params["normalize"]
        self.normalize_both = self.contrastive_params["normalize_both"]
        self.p = self.contrastive_params["p_norm"]
        self._load_messages()

    def m(self, z1, b):
        if self.baseline:
            return z1
        else:
            return z1 + b

    def _load_messages(self):
        print(f"Contrastive with alpha: {self.alpha}\nBaseline: {self.baseline}")

    def forward(self, x):
        return self.model(x)

    def loss(self, m_z1, z2):
        z_prime = torch.roll(z2, 1, 0)
        n = 1  # prod(m_z1.shape)
        if self.normalize or self.normalize_both:
            u1 = m_z1 / torch.norm(m_z1, p=2, dim=-1, keepdim=True)
            u2 = z_prime / torch.norm(z_prime, p=2, dim=-1, keepdim=True)
            neg = torch.norm(u1.unsqueeze(1) - u2.unsqueeze(0), p=2, dim=-1) / n
        else:
            neg = torch.norm(m_z1.unsqueeze(1) - z_prime.unsqueeze(0), p=2, dim=-1) / n
        if self.normalize_both:
            u3 = z2 / torch.norm(z2, p=2, dim=-1, keepdim=True)
            pos = torch.norm(u1 - u3, p=2, dim=-1) / n
        else:
            pos = torch.norm(m_z1 - z2, p=2, dim=-1) / n
        neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)
        loss_pos = pos
        loss_neg = torch.logsumexp(-neg_and_pos, dim=1)
        loss = loss_pos + loss_neg

        # optionally add regularization to make latents small
        # (we're only identified up to offset so we want zero offset)
        if self.alpha > 0:
            loss += (
                self.alpha * z2.abs().mean()
            )  # only penalize z2 because both z1 and use the same network

        return loss.mean()

    def configure_optimizers(self):

        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                   , params
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


    def training_step(self, train_batch, batch_idx):
        # TODO: train_batch must match dataloaders (it's now dictionary-based)
        (_, _), (x1, x2), (A, b) = train_batch
        z1 = self(x1)
        m_z1 = self.m(z1, b)
        z2 = self(x2)

        loss = self.loss(m_z1, z2)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()
        self.log("train_loss", loss)
        wandb.log({"loss": loss})
        return loss

    def validation_step(self, valid_batch, batch_idx):
        (z1, _), (x1, _), _, _ = valid_batch

        pred_z = self(x1)
        return {"true_z": z1, "pred_z": pred_z}

    def validation_epoch_end(self, validation_step_outputs):
        z_disentanglement = [v["true_z"] for v in validation_step_outputs]
        h_z_disentanglement = [v["pred_z"] for v in validation_step_outputs]
        z_disentanglement = torch.cat(z_disentanglement, 0)
        h_z_disentanglement = torch.cat(h_z_disentanglement, 0)
        # pixel_to_latent_space_norm_ratio = [v["ratio"] for v in validation_step_outputs]
        # pixel_to_latent_space_norm_ratio = torch.cat(pixel_to_latent_space_norm_ratio, 0)
        # print(f"pixel_to_latent_space_norm_ratio:\n{pixel_to_latent_space_norm_ratio.mean()}")
        # z_disentanglement = validation_step_outputs[-1]["true_z"]
        # h_z_disentanglement = validation_step_outputs[-1]["pred_z"]
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
        self.log("Linear_Disentanglement", linear_disentanglement_score, prog_bar=True)
        self.log(
            "Permutation_Disentanglement",
            permutation_disentanglement_score,
            prog_bar=True,
        )
        self.log("MSE", mse, prog_bar=True)
        wandb.log(
            {
                "mse": mse,
                "Permutation Disentanglement": permutation_disentanglement_score,
                "Linear Disentanglement": linear_disentanglement_score,
            }
        )

        from src.utils.additional_loggers import get_img_rec_table_data
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        logger = self.additional_logger
        offset =  0.
        f = plt.figure()
        plot_colors = ["red", "blue", "green", "cyan", "orange", "pink"]
        for i, property_ in enumerate(self.trainer.datamodule.hparams["dataset"]["properties_list"]):
            if property_ == "c" or property_ == "s":
                std = 0.01
            else:
                std = 0.0
            sns.regplot(z_disentanglement[:,i].detach().cpu().numpy() + std * np.random.randn(z_disentanglement.shape[0]) , i*offset + h_z_disentanglement[:,i].detach().cpu().numpy(), color=plot_colors[i], label=property_, marker='.')
        plt.legend()

        columns, data = get_img_rec_table_data(
            imgs=[f],
            step=self.trainer.global_step,
            num_samples_to_log=1,
        )

        logger.log_table(table_name=f"Disentanglement_{self.global_step}", train=True, columns=columns, row_list=data)

        # )
        # for i in range(z_disentanglement.shape[1]):
        #     data = [
        #         [x, y]
        #         for (x, y) in zip(
        #             z_disentanglement[:100, i], h_z_disentanglement[:100, i]
        #         )
        #     ]
        #     table = wandb.Table(
        #         data=data, columns=[f"predicted_z_{i}", f"actual_z_{i}"]
        #     )
        #     wandb.log(
        #         {"z_{i}": wandb.plot.scatter(table, f"predicted_z_{i}", f"actual_z_{i}")}
        #     )
    def test_epoch_end(self, test_step_outputs):
        z_disentanglement = [v["true_z"] for v in test_step_outputs]
        h_z_disentanglement = [v["pred_z"] for v in test_step_outputs]
        z_disentanglement = torch.cat(z_disentanglement, 0)
        h_z_disentanglement = torch.cat(h_z_disentanglement, 0)
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
        self.log("Linear_Disentanglement", linear_disentanglement_score, prog_bar=True)
        self.log(
            "Permutation Disentanglement",
            permutation_disentanglement_score,
            prog_bar=True,
        )
        self.log("MSE", mse, prog_bar=True)
        wandb.log(
            {
                "mse": mse,
                "Permutation Disentanglement": permutation_disentanglement_score,
                "Linear_Disentanglement": linear_disentanglement_score,
            }
        )