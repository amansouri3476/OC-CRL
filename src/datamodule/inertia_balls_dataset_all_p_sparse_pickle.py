import torch
import os
import numpy as np
import math
from typing import Callable, Optional
import colorsys
import src.utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = False

# HSV colours
COLOURS_ = [
    # [0.05, 0.6, 0.6],
    [0.15, 0.6, 0.6],
    # [0.25, 0.6, 0.6],
    # [0.35, 0.6, 0.6],
    [0.45, 0.6, 0.6],
    # [0.55, 0.6, 0.6],
    # [0.65, 0.6, 0.6],
    [0.75, 0.6, 0.6],
    # [0.85, 0.6, 0.6],
    # [0.95, 0.6, 0.6],
]

SHAPES_ = [
    # "circle",
    "square",
    "triangle",
    "heart"
]

PROPERTIES_ = [
    "x",
    "y",
    "c",
    "s",
    "l",
    "p",
]

    
class InertialBallAllPropertiesSparseOffsetPickleable(torch.utils.data.Dataset):
    """
    This class instantiates a torch.utils.data.Dataset object. This dataset returns
    pairs of images that correspond to consecutive frames. The ground truth generating
    factors of this dataset are four properties per each ball: x,y,colour,shape
    The number of balls (n) from t->t+1 does not change and is fixed by the datamodule
    config. From t->t+1, only one ball's state can be changed, and that change is also
    constrained to be sparse, meaning that only 1 out of all properties can be altered.
    """

    def __init__(
        self,
        data = None,
    ):
        super(InertialBallAllPropertiesSparseOffsetPickleable, self).__init__()

        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)
