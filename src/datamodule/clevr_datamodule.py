import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
import os.path as path
import hydra
import numpy as np
import pickle
import src.utils.general as utils
log = utils.get_logger(__name__)
    

class CLEVRDataModule(LightningDataModule):
    def __init__(self
                 , seed: int = 1234
                 , batch_size: int = 128
                 , max_num_objects: int = 6
                 , clevr_dir: str = "../../../../scratch/CLEVR_v1.0"
                 , dictionaries_path: str = "dictionaries.pkl"
                 , **kwargs):
        
        super().__init__()
        
        # So all passed parameters are accessible through self.hparams
        self.save_hyperparameters(logger=False)
        
        self.seed = seed
        self.max_num_objects = max_num_objects
        self.dataset_parameters = self.hparams['dataset_parameters']
        self.clevr_dir = clevr_dir
        
        self.num_samples = {}
        for split in self.dataset_parameters.keys(): # splits: ['train','valid','test']
            self.num_samples[split] = self.dataset_parameters[split]['dataset']['num_samples']
        
        self.dirname = os.path.dirname(__file__)
        self.dictionaries_path = os.path.join(self.dirname, dictionaries_path)
        self.dataset_name = self.hparams.dataset_name
        self.clevr_dir = clevr_dir
        self.transforms = self.hparams.transforms
        

    def prepare_data(self):
        """
        Docs: Use this method to do things that might write to disk or that need to be done only from 
        a single process in distributed settings.
        - download
        - tokenize
        - etc.
        """
        
        log.info(f"Setting up transformations")
        transform = transforms.Compose([hydra.utils.instantiate(t) for _, t in self.transforms.items()])

#         transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 Standardize(),
#                 transforms.CenterCrop((192,192)),
#                 transforms.Resize((128,128)),
#                 TensorClip(),
#             ]
#         )


        if self.hparams.use_qa:
            with open(self.dictionaries_path, "rb") as f:
                dictionaries = pickle.load(f)
        else:
            dictionaries = None
        

        log.info(f"Using CLEVR dataset with max_num_objects: {self.max_num_objects}.")

        # load the dataset
        clevr_train_dataset = hydra.utils.instantiate(self.dataset_parameters["train"]["dataset"]
                                                      , transform=transform
                                                      , dictionaries=dictionaries)
        clevr_valid_dataset = hydra.utils.instantiate(self.dataset_parameters["valid"]["dataset"]
                                                      , transform=transform
                                                      , dictionaries=dictionaries)
        
        # not possible
#         # pick as many samples as needed for each dataset.
#         clevr_train_dataset = Subset(clevr_train_dataset, torch.tensor(range(self.num_samples["train"])))
#         clevr_valid_dataset = Subset(clevr_valid_dataset, torch.tensor(range(self.num_samples["valid"])))
        
#         self.train_dataset = Subset(self.train_dataset, torch.tensor(range(self.num_samples["train"])))
#         self.valid_dataset = Subset(self.valid_dataset, torch.tensor(range(self.num_samples["valid"])))
        
        # apply the filter based on the number of objects
        # train
        num_objects = torch.tensor([clevr_train_dataset.objects[idx].size(0) for idx in range(len(clevr_train_dataset.objects))])
        indices = np.where((num_objects <= self.max_num_objects))[0]
        self.train_dataset = Subset(clevr_train_dataset, indices)
        # valid
        num_objects = torch.tensor([clevr_valid_dataset.objects[idx].size(0) for idx in range(len(clevr_valid_dataset.objects))])
        indices = np.where((num_objects <= self.max_num_objects))[0]
        self.valid_dataset = Subset(clevr_valid_dataset, indices)
        
        # TODO: add a load functionality to avoid a lot of computation
        torch.save(self.train_dataset,f"CLEVR_train_dataset.pt")
        torch.save(self.valid_dataset,f"CLEVR_valid_dataset.pt")


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
        
        
        self.test_dataset = self.valid_dataset
        

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
        

    
    
    
    
