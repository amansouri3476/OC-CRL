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


class SparseClevrDataModule(LightningDataModule):
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

        # self.n_offsets = self.hparams["n_offsets"]
        self.n_objects = self.hparams["n_balls"]
        self.z_dim = self.hparams["z_dim"]

        assert self.z_dim == len(self.dataset_name), f"The datamodule is supposed to have {self.z_dim} latent dims, but the proposed dataset has {len(self.dataset_name)}."

        self.sparsity_degree = kwargs.get("sparsity_degree", self.n_objects)
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
        
        log.info(f"Using sparse clevr dataset with {self.n_objects} objects.")

        if not self.load_dataset:
            log.info(f"Loading the data sample by sample from {self.path_to_files}")
            self.train_dataset = hydra.utils.instantiate(self.dataset_parameters["train"]["dataset"]
                                                                , transform=transform
                                                                , augmentations=augmentations
                                                                , n_objects = self.n_objects
                                                                , sparsity_degree = self.sparsity_degree
                                                                , known_mechanism = self.known_mechanism
                                                                ) 
            
            self.valid_dataset = hydra.utils.instantiate(self.dataset_parameters["valid"]["dataset"]
                                                                , transform=transform
                                                                , augmentations=augmentations
                                                                , n_objects = self.n_objects
                                                                , sparsity_degree = self.sparsity_degree
                                                                , known_mechanism = self.known_mechanism
                                                                ) 
            
            self.test_dataset = hydra.utils.instantiate(self.dataset_parameters["test"]["dataset"]
                                                                , transform=transform
                                                                , augmentations=augmentations
                                                                , n_objects = self.n_objects
                                                                , sparsity_degree = self.sparsity_degree
                                                                , known_mechanism = self.known_mechanism
                                                                ) 

        else:
            log.info(f"Loading the whole dataset files from {self.path_to_files}")
            # self.train_dataset = torch.load(os.path.join(self.path_to_files, f"train_dataset_{self.dataset_name}.pt"))
            # self.valid_dataset = torch.load(os.path.join(self.path_to_files, f"valid_dataset_{self.dataset_name}.pt"))
            # self.test_dataset = torch.load(os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}.pt"))
            self.train_dataset = torch.load(os.path.join(self.path_to_files, f"train_dataset_{self.dataset_name}_{self.num_samples['train']}.pt"))
            self.valid_dataset = torch.load(os.path.join(self.path_to_files, f"valid_dataset_{self.dataset_name}_{self.num_samples['valid']}.pt"))
            self.test_dataset = torch.load(os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}_{self.num_samples['test']}.pt"))

            td = []
            # for i in range(self.num_samples['train']):
            for i in range(self.train_dataset.__len__()):
                if (self.train_dataset[i]["images"][0] == self.train_dataset[i]["images"][1]).all():
                    log.info(f"Removing corrupted sample {i}")
                    continue
                else:
                    td.append(self.train_dataset[i])
            self.train_dataset.data = td

            td = []
            # for i in range(self.num_samples['valid']):
            for i in range(self.valid_dataset.__len__()):
                if (self.valid_dataset[i]["images"][0] == self.valid_dataset[i]["images"][1]).all():
                    log.info(f"Removing corrupted sample {i}")
                    continue
                else:
                    td.append(self.valid_dataset[i])
            self.valid_dataset.data = td
            
            td = []
            # for i in range(self.num_samples['test']):
            for i in range(self.test_dataset.__len__()):
                if (self.test_dataset[i]["images"][0] == self.test_dataset[i]["images"][1]).all():
                    log.info(f"Removing corrupted sample {i}")
                    continue
                else:
                    td.append(self.test_dataset[i])
            self.test_dataset.data = td
            # temp
            # log.info("modifying the data")
            # self.train_dataset.modify_data()
            # self.valid_dataset.modify_data()
            # self.valid_dataset.modify_data()

        if self.save_dataset:
            log.info(f"Saving the whole dataset files to {self.path_to_files}")
            # torch.save(self.train_dataset, os.path.join(self.path_to_files, f"train_dataset_{self.dataset_name}.pt"))
            # torch.save(self.valid_dataset, os.path.join(self.path_to_files, f"valid_dataset_{self.dataset_name}.pt"))
            # torch.save(self.test_dataset, os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}.pt"))
            torch.save(self.train_dataset, os.path.join(self.path_to_files, f"train_dataset_{self.dataset_name}_{self.num_samples['train']}.pt"))
            torch.save(self.valid_dataset, os.path.join(self.path_to_files, f"valid_dataset_{self.dataset_name}_{self.num_samples['valid']}.pt"))
            torch.save(self.test_dataset, os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}_{self.num_samples['test']}.pt"))

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
            dataset=self.valid_dataset, 
            **self.dataset_parameters['valid']['dataloader']
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, 
            **self.dataset_parameters['test']['dataloader']
        )
    
    def renormalize(self):
        for _, t in self.transforms.items():
            if "Standardize" in t["_target_"]:
                """Renormalize from [-1, 1] to [0, 1]."""
                return lambda x: x / 2.0 + 0.5
            
            # TODO: add more options if required
    
    
    
    
