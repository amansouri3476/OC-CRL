import torch
import os
import numpy as np
from typing import Callable, Optional
import colorsys
import pickle
from tqdm import tqdm
import src.utils.general as utils
log = utils.get_logger(__name__)


PROPERTIES = [
    "x",
    "y",
    # "z",
    "c",
    "s",
    "l",
    "p",
]

class SparseClevr(torch.utils.data.Dataset):
    """
    This class instantiates a torch.utils.data.Dataset object. This dataset returns
    pairs of images that correspond to consecutive frames. The ground truth generating
    factors of this dataset are four properties per each object: x,y,colour,shape
    The number of objects (n) from t->t+1 does not change and is fixed by the datamodule
    config. From t->t+1, only one object's state can change, and that change is also
    constrained to be sparse, meaning that only 1 out of all properties can be altered.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        offsets: np.ndarray = np.array([[0.05, -0.025]]),
        augmentations: list = [],
        n_objects: int = 1,
        num_samples: int = 20000,
        **kwargs,
    ):
        super(SparseClevr, self).__init__()

        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.n_objects = n_objects
        self.num_samples = num_samples
        self.start_idx = kwargs.get("start_idx", 0)
        self.data_dir = kwargs.get("data_dir", "./")
        self.sparsity_degree = 1
        self.known_mechanism = kwargs.get("known_mechanism", True)
        self.injective = kwargs.get("injective")
        self.same_color = kwargs.get("same_color")
        self.output_sparse_offsets = kwargs.get("output_sparse_offsets")
        self.z_dim = kwargs.get("z_dim")
        self.properties_list = kwargs.get("properties_list") # a subset of ["x","y","c","s"] preserving the order
        self.target_property_indices = [i for i,p in enumerate(PROPERTIES) if p in self.properties_list]
        self.non_target_property_indices = [i for i,p in enumerate(PROPERTIES) if p not in self.properties_list]
        assert self.z_dim == len(self.properties_list)
        self.data = self._load_data()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " + line for line in body]
        return "\n".join(lines)

    def load_sample(self, sample_idx):
        
        if sample_idx < 10:
            filename_str = f"0000{sample_idx}"
        elif sample_idx < 100:
            filename_str = f"000{sample_idx}"
        elif sample_idx < 1000:
            filename_str = f"00{sample_idx}"
        elif sample_idx < 10000:
            filename_str = f"0{sample_idx}"
        else:
            filename_str = f"{sample_idx}"

        file_path = os.path.join(self.data_dir, f"{filename_str}.pickle")
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0: # avoiding empty files
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
                # convert stuff to torch!
            return True, sample
        else:
            log.info(f"Sample {filename_str} is missing, skipping...")
            return False, 0

    def _load_data(self):
        data = []
        dict_keys = [[f"{c}+", f"{c}-"] for c in self.properties_list]
        from functools import reduce
        dict_keys = reduce(lambda xs, ys: xs + ys, dict_keys)
        data_statistics = dict.fromkeys(dict_keys)
        missing_data_statistics = dict.fromkeys(dict_keys)
        for key in data_statistics.keys():
            data_statistics[key] = 0
        for key in missing_data_statistics.keys():
            missing_data_statistics[key] = 0
        no_change_counter = 0
        change_n_object_transition = 0
        wrong_count_segmentation_masks = 0
        missing_sample_counter = 0
        for idx in tqdm(range(self.num_samples)):
            abs_idx = idx + self.start_idx
            sample_exists, sample = self.load_sample(abs_idx)
            if sample_exists:
                # if the sparse change doesn't affect the target properties
                # , we should skip this sample
                b = sample["matrices"][1] # [4]
                if (b == 0.).all():
                    log.info(f"Sample {idx} has no change between t,t+1 (b=0), skipping...")
                    no_change_counter += 1
                    continue
                changed_property_idx = (~(b == 0.)).nonzero()[0].item()
                if PROPERTIES[changed_property_idx] not in self.properties_list:
                    continue
                else:
                    # if PROPERTIES[changed_property_idx] == "c" and np.abs(b[np.abs(b) > 0.]) > 0.25:
                    #     log.info(f"Sample {idx} has {b[np.abs(b) > 0.]} change in color, skipping...")
                    #     continue
                    # if PROPERTIES[changed_property_idx] == "s" and np.abs(b[np.abs(b) > 0.]) > 0.25:
                    #     log.info(f"Sample {idx} has {b[np.abs(b) > 0.]} change in shape, skipping...")
                    #     continue
                    x1, x2 = sample["images"][0], sample["images"][1]
                    if (x1 == x2).all():
                        log.info(f"Sample {idx} has no change between t,t+1 (x1=x2), skipping...")
                        no_change_counter += 1
                        log.info(f"(b > 0.).any() {(b > 0.).any()}")
                        if (b > 0.).any():
                            missing_data_statistics[str(PROPERTIES[changed_property_idx]) + "+"] += 1
                        else:
                            missing_data_statistics[str(PROPERTIES[changed_property_idx]) + "-"] += 1
                        continue
                    if (b > 0.).any():
                        data_statistics[str(PROPERTIES[changed_property_idx]) + "+"] += 1
                    else:
                        data_statistics[str(PROPERTIES[changed_property_idx]) + "-"] += 1
                    # dividing by 255 to map them to 0,1
                    x1 = x1/255.
                    x2 = x2/255.
                    # apply the transforms
                    sample["images"] = (self.transform(x1).float(), self.transform(x2).float())

                    z1, z2 = sample["latents"][0], sample["latents"][1]
                    if (np.abs(z2-z1)<1e-4).all():
                        log.info(f"Sample {idx} has no change between t,t+1 (z1=z2), skipping...")
                        no_change_counter += 1
                        continue
                    # n = 7 if "p" in self.properties_list else 6  # the number of properties the generative model provides. (whether include shape or not)
                    n = z1.shape[0]//self.n_objects
                    n_objects = self.n_objects
                    # remove the z property from latents if not in the list of properties
                    if "z" not in self.properties_list:
                        # get object latents
                        z1 = np.concatenate([np.concatenate([z1[n * i:n * i + 2], z1[1 + 2 + n * i:n * (i+1)]], axis=0) for i in range(n_objects)], axis=0)
                        z2 = np.concatenate([np.concatenate([z2[n * i:n * i + 2], z2[1 + 2 + n * i:n * (i+1)]], axis=0) for i in range(n_objects)], axis=0)
                        # z1,z2: [n_objects * n_properties]
                        
                        # update latents and the offset matrices to only reflect the target properties
                        # property_indices: [z_dim * n_objects]
                        # jump = 6 if "p" in self.properties_list else 5
                        jump = n-1
                        property_indices = np.concatenate([[idx + jump * i for idx in self.target_property_indices] for i in range(n_objects)], axis=0)
                    else:
                        # update latents and the offset matrices to only reflect the target properties
                        # property_indices: [z_dim * n_objects]
                        # jump = 7 if "p" in self.properties_list else 6
                        jump = n
                        property_indices = np.concatenate([[idx + jump * i for idx in self.target_property_indices] for i in range(n_objects)], axis=0)

                    # z1,z2: [n_objects * z_dim]
                    z1 = z1[property_indices]
                    z2 = z2[property_indices]
                    sample["latents"] = z1, z2
                    # update matrices
                    A, b = sample["matrices"]
                    sample["matrices"] = torch.eye(b[self.target_property_indices].shape[0]).float(), torch.tensor(b[self.target_property_indices]).flatten().float()
                    # sample["matrices"] = torch.eye(b.shape[0]).float(), torch.tensor(b).flatten().float()
                    sample["mechanism_permutation"] = np.array((sample["mechanism_permutation"],))

                    # convert segmentation masks to flaots
                    seg_1, seg_2 = sample["segmentation_masks"]
                    if seg_1.shape != seg_2.shape:
                        log.info(f"The number of objects have changed across the transition, according to the segmentation ids, skipping sample {idx}")
                        change_n_object_transition += 1
                        continue
                    if seg_1.shape[0] != self.n_objects + 1 or seg_2.shape[0] != self.n_objects + 1:
                        log.info(f"{seg_1.shape[0] != self.n_objects + 1}")
                        log.info(f"{seg_2.shape[0] != self.n_objects + 1}")
                        log.info(f"The number of segmentation masks {seg_1.shape[0]},{seg_2.shape[0]} is not n_objects+1:{self.n_objects+1}, skipping sample {idx}")
                        wrong_count_segmentation_masks += 1
                        continue
                    sample["segmentation_masks"] = seg_1 * 1., seg_2 * 1.

                    data.append(sample)
            else:
                log.info("missing data sample")
                missing_sample_counter += 1
        log.info(f"The dataset being used has the following statistics for the properties being changed:\n{[(key,val) for key,val in data_statistics.items()]}\n and {no_change_counter} samples do NOT have any changes between t,t+1\nand {change_n_object_transition} samples have number of objects changing from t to t+1\nand {wrong_count_segmentation_masks} samples have wrong number of segmentation masks either at t or t+1\nand {missing_sample_counter} samples are missing. Below are the stats for missing samples:\n{[(key,val) for key,val in missing_data_statistics.items()]}")
        return data

    # temporary code
    # def modify_data(self):
    #     # only keep those that change c
    #     self.data_new = []
    #     found = False
    #     for idx in range(len(self.data)):
    #         b = self.data[idx]["matrices"][1]
    #         changed_property_idx = (~(b == 0.)).nonzero()[0].item()
    #         if PROPERTIES[changed_property_idx] != "c":
    #             if np.random.rand() > 0.7:
    #                 self.data_new.append(self.data[idx])
    #             else:
    #                 continue
    #         else:
    #             self.data_new.append(self.data[idx])
    #     self.data = self.data_new

    def __getitem__(self, idx):
        
        return self.data[idx]