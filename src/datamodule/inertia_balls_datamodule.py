import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os.path as path
import hydra
import numpy as np
from src.datamodule.augmentations import ALL_AUGMENTATIONS

AUGMENTATIONS = {k: lambda x: v(x, order=1) for k, v in ALL_AUGMENTATIONS.items()}

import src.utils.general as utils
log = utils.get_logger(__name__)

# constants used for normalizing the data
# MU = np.array([0.9906, 0.9902, 0.9922])
# SIG = np.array([0.008, 0.008, 0.008])

MU = np.array([0.98, 0.98, 0.98])
SIG = np.array([0.09, 0.09, 0.09])

# MU = np.array([0.0, 0.0, 0.0])
# SIG = np.array([1.0, 1.0, 1.0])


    

class InertiaBallsDataModule(LightningDataModule):
    def __init__(self
                 , seed: int = 1234
                 , batch_size: int= 128
                 , regenerate: bool = False
                 , **kwargs):
        
        super().__init__()
        
        # So all passed parameters are accessible through self.hparams
        self.save_hyperparameters(logger=False)
        
        self.seed = seed
        self.dataset_parameters = self.hparams.dataset["dataset_parameters"]
        
        self.num_samples = {}
        for split in self.dataset_parameters.keys(): # splits: ['train','valid','test']
            self.num_samples[split] = self.dataset_parameters[split]["dataset"]["num_samples"]
        
        self.regenerate_data = regenerate
        self.path_to_files = self.hparams["data_dir"]
        self.dirname = os.path.dirname(__file__)
        self.dataset_name = self.hparams.dataset_name
        self.save_dataset = self.hparams.save_dataset
        self.load_dataset = self.hparams.load_dataset
        self.transforms = self.hparams.transforms
        self.augs = self.hparams.transform.get("augs", None)

        self.n_offsets = self.hparams["n_offsets"]
        self.n_balls = self.hparams["n_balls"]
        self.z_dim = self.hparams["z_dim"]

        assert self.z_dim == len(self.dataset_name), f"The datamodule is supposed to have {self.z_dim} latent dims, but the proposed dataset has {len(self.dataset_name)}."

        if kwargs.get("preset_latents", None) is not None:
            z_data = kwargs["preset_latents"]
            self.num_samples["train"] = len(z_data)
            self.n_balls = z_data.shape[1]
            self.n_offsets = self.n_balls * 2
            self.preset_latents = z_data
        else:
            self.preset_latents = None

        self.sparsity_degree = kwargs.get("sparsity_degree", self.n_balls)
        self.known_mechanism = kwargs.get("known_mechanism", True)

        assert self.save_dataset != self.load_dataset, "Should only load the dataset, or save it, but not both."
        
        

    def prepare_data(self):
        """
        Docs: Use this method to do things that might write to disk or that need to be done only from 
        a single process in distributed settings.
        - download
        - tokenize
        - etc.
        """
            
        if self.augs is not None:
            augmentations = [v for (k, v) in AUGMENTATIONS.items() if k in self.augs]
        else:
            augmentations = []

        log.info(f"Setting up transformations <{[k for (k, v) in AUGMENTATIONS.items() if k in self.augs]}>")
        transform = transforms.Compose([hydra.utils.instantiate(t) for _, t in self.transforms.items()])
        
        if not self.load_dataset:
            log.info(f"Generating the data sample by sample.")
            log.info(f"Using inertia dataset with {self.n_offsets} offsets, {self.n_balls} balls.")

            # sample offsets in [-0.1, 0.1] excluding anything in [-0.02, 0.02]
            if self.z_dim < 2:
                offsets = np.random.uniform(0.02, 0.55, size=(self.n_offsets,))
            else:
                offsets = np.random.uniform(0.02, 0.55, size=(self.n_offsets, self.z_dim)) # was 0.25
    #         offsets *= 1 * (np.random.rand(*offsets.shape) > 0.5)
            # randomly changing the sign of some offsets
            offsets = (1 - 2 * (np.random.rand(*offsets.shape) > 0.5)) * offsets

            self.train_dataset = hydra.utils.instantiate(self.dataset_parameters["train"]["dataset"]
                                                                , transform=transform
                                                                , offsets=offsets
                                                                , augmentations=augmentations
                                                                , n_balls = self.n_balls
                                                                , preset_latents = self.preset_latents
                                                                , sparsity_degree = self.sparsity_degree
                                                                , known_mechanism = self.known_mechanism
                                                                ) 
            
            self.test_dataset = hydra.utils.instantiate(self.dataset_parameters["test"]["dataset"]
                                                                , transform=transform
                                                                , offsets=offsets
                                                                , augmentations=augmentations
                                                                , n_balls = self.n_balls
                                                                , sparsity_degree = self.sparsity_degree
                                                                , known_mechanism = self.known_mechanism
                                                                )

            self.valid_dataset = self.test_dataset

        else:
            log.info(f"Loading the whole dataset files from {self.path_to_files}")
            self.train_dataset = torch.load(os.path.join(self.path_to_files, f"train_dataset_{self.dataset_name}_{self.num_samples['train']}.pt"))
            self.valid_dataset = torch.load(os.path.join(self.path_to_files, f"valid_dataset_{self.dataset_name}_{self.num_samples['valid']}.pt"))
            self.test_dataset = torch.load(os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}_{self.num_samples['test']}.pt"))

            screen_dim = 128
            import pygame
            if "SDL_VIDEODRIVER" not in os.environ:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                os.environ["SDL_AUDIODRIVER"] = "dsp"

            pygame.init()
            self.screen = pygame.display.set_mode((screen_dim, screen_dim))
            self.surf = pygame.Surface((screen_dim, screen_dim))

        if self.save_dataset:
            if not os.path.exists(self.path_to_files):
                os.makedirs(self.path_to_files)
            log.info(f"Saving the whole dataset files to {self.path_to_files}")
            torch.save(self.train_dataset.pickleable_dataset, os.path.join(self.path_to_files, f"train_dataset_{self.dataset_name}_{self.num_samples['train']}.pt"))
            torch.save(self.valid_dataset.pickleable_dataset, os.path.join(self.path_to_files, f"valid_dataset_{self.dataset_name}_{self.num_samples['valid']}.pt"))
            torch.save(self.test_dataset.pickleable_dataset, os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}_{self.num_samples['test']}.pt"))

                    


    def setup(self, stage=None):
        """
        Docs: There are also data operations you might want to perform on every GPU. Use setup to do 
        things like:
        - count number of classes
        - build vocabulary
        - perform train/val/test splits
        - apply transforms (defined explicitly in your datamodule)
        - etc.
        """

        # ---------------------------------------------------------------
        # Generating training and testing torch Datasets
        # ---------------------------------------------------------------

        # validation dataset is the same as the train dataset because
        # examples are constructed online.
        self.valid_dataset = self.train_dataset
        # self.test_dataset = self.train_dataset

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            dataset=self.train_dataset,
            **self.dataset_parameters['train']['dataloader'],
            generator=g,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset, **self.dataset_parameters['valid']['dataloader']
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, **self.dataset_parameters['test']['dataloader']
        )
    
    def renormalize(self):
        for _, t in self.transforms.items():
            if "Standardize" in t["_target_"]:
                """Renormalize from [-1, 1] to [0, 1]."""
                return lambda x: x / 2.0 + 0.5
            
            # TODO: add more options if required
    
    
    
    
