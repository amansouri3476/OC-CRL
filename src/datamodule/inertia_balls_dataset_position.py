from .abstract_inertia_balls_dataset import InertialBall
import torch
import os
import numpy as np
from typing import Callable, Optional

if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
]


SCREEN_DIM = 64
Y_SHIFT = -0.9

    
class InertialBallPositionOffset(InertialBall):
    ball_rad = 0.04 # 0.04, 0.12
    screen_dim = 64

    def __init__(
        self,
        transform: Optional[Callable] = None,
        offsets: np.ndarray = np.array([[0.05, -0.025]]),
        augmentations: list = [],
        human_mode: bool = False,
        n_balls: int = 1,
        n_colours: int = 1,
        num_samples: int = 20000,
        **kwargs,
    ):
        super(InertialBallPositionOffset, self).__init__(transform
                                                , offsets
                                                , augmentations
                                                , human_mode
                                                , n_balls
                                                # , n_colours
                                                , num_samples
                                                ,**kwargs
                                                )
        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.sparsity_degree = kwargs.get("sparsity_degree", n_balls)
        self.known_mechanism = kwargs.get("known_mechanism", True)
        self.data = self._generate_data()

    def _sample_offsets(self):
        n_offsets = self.offsets.shape[0]
        
        mask = np.random.choice(range(n_offsets), size=(self.sparsity_degree,), replace=False)
        mask_onehot = np.zeros((mask.size, n_offsets))
        mask_onehot[np.arange(mask.size),mask] = 1
        mask = mask_onehot
        sampled_offsets = (mask[:, :, None] * self.offsets).sum(axis=1)

        mask_boundary = np.ones(sampled_offsets.shape)
        # resample offsets that have a duplicate among the sampled offsets
        duplicate_mechanism_threshold = 0.2
        # sampled_offsets_distance_matrix = np.linalg.norm(sampled_offsets[:, None, :] - sampled_offsets[None, :, :], axis=-1)
        sampled_offsets_distance_matrix = np.abs(np.linalg.norm(sampled_offsets, axis=-1)[None, :] - np.linalg.norm(sampled_offsets, axis=-1)[:, None])
        duplicate_mask = np.triu(sampled_offsets_distance_matrix<duplicate_mechanism_threshold).sum(-2)>1
        # mask_boundary = 1. - duplicate_mask.astype(int)
        mask_boundary[duplicate_mask] = 0.

        resample_mask = ~ (mask_boundary > 0.)

        while_loop_threshold = 1000
        while resample_mask.any() and while_loop_threshold > 0:
            while_loop_threshold -= 1
            mask = np.random.choice(range(n_offsets), size=(self.sparsity_degree,), replace=False)
            mask_onehot = np.zeros((mask.size, n_offsets))
            mask_onehot[np.arange(mask.size),mask] = 1
            mask = mask_onehot
            sampled_offsets_temp = (mask[:, :, None] * self.offsets).sum(axis=1)
            sampled_offsets[resample_mask] = sampled_offsets_temp[resample_mask]

            mask_boundary = np.ones(sampled_offsets.shape)
            # resample offsets that have a duplicate among the sampled offsets
            # sampled_offsets_distance_matrix = np.linalg.norm(sampled_offsets[:, None, :] - sampled_offsets[None, :, :], axis=-1)
            sampled_offsets_distance_matrix = np.abs(np.linalg.norm(sampled_offsets, axis=-1)[None, :] - np.linalg.norm(sampled_offsets, axis=-1)[:, None])
            duplicate_mask = np.triu(sampled_offsets_distance_matrix<duplicate_mechanism_threshold).sum(-2)>1
            mask_boundary[duplicate_mask] = 0.
            # mask_boundary = 1. - duplicate_mask.astype(int)

            resample_mask = ~ (mask_boundary > 0.)

        return sampled_offsets

    def _sample_z1_z2(self, b):
        # sample z1,z2 in a way that:
        # 1. objects aren't initialized close to each other
        # 2. objects don't fall close after the mechanisms are applied
        # 3. objects don't fall out of the frame after the mechanisms are applied

        z1 = np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))

        # find the maximum range of for each ball that doesn't push the ball out of the frame
        # upper_bound_x = 0.95 - z1[:, 0]
        # upper_bound_y = 0.95 - z1[:, 1]
        # lower_bound_x = -z1[:, 0] + 0.05
        # lower_bound_y = -z1[:, 1] + 0.05
        upper_bound_x = 0.95
        upper_bound_y = 0.95
        lower_bound_x = 0.05
        lower_bound_y = 0.05

        # sampled_offsets are [self.sparsity_degree, z_dim]; to be able to add them to z1, we need to create a z1-like zeros array
        # and fill it a with a permutation of sampled offsets
        b_ = np.zeros_like(z1) # [n_balls, z_dim]
        idx = np.random.choice(range(self.n_balls), size=(self.sparsity_degree,), replace=False)
        b_[idx, :] = b
        z2 = z1 + b_

        # prevent balls from getting initialized very close to each other
        mask = np.ones(z1.shape)
        duplicate_z1_threshold = 0.2
        sampled_z1_distance_matrix = np.linalg.norm(z1[:, None, :] - z1[None, :, :], axis=-1)
        duplicate_mask = np.triu(sampled_z1_distance_matrix<duplicate_z1_threshold).sum(-2)>1
        mask[duplicate_mask] = 0.
        # mask = 1. - duplicate_mask.astype(int)
        
        # prevent balls from falling very close to each other after the mechanisms are applied
        duplicate_z2_threshold = 0.2
        sampled_z2_distance_matrix = np.linalg.norm(z2[:, None, :] - z2[None, :, :], axis=-1)
        duplicate_mask = np.triu(sampled_z2_distance_matrix<duplicate_z2_threshold).sum(-2)>1
        mask[duplicate_mask] = 0.
        # mask = 1. - duplicate_mask.astype(int)

        # prevent balls from being pushed out of the frame
        for i in range(self.n_balls):
            if z2[i, 0] > upper_bound_x or z2[i, 0] < lower_bound_x:
                mask[i, :] = 0.

            if z2[i, 1] > upper_bound_y or z2[i, 1] < lower_bound_y:
                mask[i, :] = 0.


        resample_mask = ~ (mask > 0.)

        # under no circumstances should z2 fall out of the frame, so this loop should continue until the constraints are satisfied.
        while_loop_threshold = 1000
        while resample_mask.any() and while_loop_threshold > 0:
        # while resample_mask.any():
            while_loop_threshold -= 1
            z1_temp = np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))
            z1[resample_mask] = z1_temp[resample_mask]

            # sampled_offsets are [self.sparsity_degree, z_dim]; to be able to add them to z1, we need to create a z1-like zeros array
            # and fill it a with a permutation of sampled offsets
            b_ = np.zeros_like(z1) # [n_balls, z_dim]
            idx = np.random.choice(range(self.n_balls), size=(self.sparsity_degree,), replace=False)
            b_[idx, :] = b
            z2 = z1 + b_

            # prevent balls from getting initialized very close to each other
            mask = np.ones(z1.shape)
            sampled_z1_distance_matrix = np.linalg.norm(z1[:, None, :] - z1[None, :, :], axis=-1)
            duplicate_mask = np.triu(sampled_z1_distance_matrix<duplicate_z1_threshold).sum(-2)>1
            mask[duplicate_mask] = 0.
            # mask = 1. - duplicate_mask.astype(int)

            # prevent balls from falling very close to each other after the mechanisms are applied
            duplicate_z2_threshold = 0.1
            sampled_z2_distance_matrix = np.linalg.norm(z2[:, None, :] - z2[None, :, :], axis=-1)
            duplicate_mask = np.triu(sampled_z2_distance_matrix<duplicate_z2_threshold).sum(-2)>1
            mask[duplicate_mask] = 0.
            # mask = 1. - duplicate_mask.astype(int)

            # prevent balls from being pushed out of the frame
            for i in range(self.n_balls):
                if z2[i, 0] > upper_bound_x or z2[i, 0] < lower_bound_x:
                    mask[i, :] = 0.

                if z2[i, 1] > upper_bound_y or z2[i, 1] < lower_bound_y:
                    mask[i, :] = 0.
        
            resample_mask = ~ (mask > 0.)

        if while_loop_threshold == 0:
            for i in range(self.n_balls):
                if z2[i, 0] > upper_bound_x:
                    z2[i, 0] = upper_bound_x
                if z2[i, 0] < lower_bound_x:
                    z2[i, 0] = lower_bound_x

                if z2[i, 1] > upper_bound_y:
                    z2[i, 1] = upper_bound_y
                if z2[i, 1] < lower_bound_y:
                    z2[i, 1] = lower_bound_y

        return z1, z2, idx

    def _sample(self):
        self._setup()
        if self.generate_data:
            # we will first get the offsets, and based on the offsets we will place the balls. The reason is that offsets should be sufficiently different
            # so that the matching does not collapse.

            # sample offsets that are sufficiently different
            # sample z1,z2 in a way that:
            # 1. objects aren't initialized close to each other
            # 2. objects don't fall close after the mechanisms are applied
            # 3. objects don't fall out of the frame after the mechanisms are applied

            # sample offsets or augmentations in proportion to
            # number of offsets & number of augmentations.
            n_aug = len(self.augmentations)
            n_offsets = self.offsets.shape[0]
            total_contrasts = n_offsets + n_aug
            if n_aug > 0:
                p_offset = n_offsets / total_contrasts
                p_augment = 1 - p_offset
                constrast_id = int(
                    torch.tensor([p_offset] + [p_augment / n_aug] * n_aug).multinomial(1)
                )
            else:
                constrast_id = 0

            # selecting colors
            if self.color_selection == "cyclic_fixed":
                # different but fixed set of colors
                color_set = [COLOURS_[i % len(COLOURS_)] for i in range(self.n_balls)] # picking colours cyclically from the list of colours
                c = list(np.random.permutation(color_set)) # to avoid having the same ordering of the fixed colours used for balls
            elif self.color_selection == "same":
                # same color for all balls
                c = [COLOURS_[0] for i in range(self.n_balls)] # picking colours cyclically from the list of colours
            elif self.color_selection == "random":
                # randomized selection of colors for balls. The colors won't be fixed for different elements in a batch
                _replace = True if self.n_balls > len(COLOURS_) else False
                indices = list(np.random.choice(range(len(COLOURS_)), self.n_balls, replace=_replace))
                c = [COLOURS_[i] for i in indices]
            else:
                raise Exception(f"The color selection method '{self.color_selection}' is not supported!")


            if constrast_id == 0:
                b = self._sample_offsets()
                z1, z2, mechanism_permutation = self._sample_z1_z2(b)
                
                # segmentation_mask1: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask1[0] is the background mask
                x1, segmentation_masks1 = self.draw_scene(z1, c)
                # segmentation_mask2: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask2[0] is the background mask
                x2, segmentation_masks2 = self.draw_scene(z2, c)
                b = torch.tensor(b).float().squeeze()
                if not self.known_mechanism: # TODO: might confuse the matching? Is it ok in the sparse setting if fake changes are identical? Probably it's fine
                                             # because the real offsets aren't the same.
                                             # TODO: how about the sign of the change? does that matter?
                    b = (torch.abs(b) > 0.) * 1.
                A = torch.eye(b.shape[0]).float()
            else:
                # TODO: I don't know what this part does
                x2, A, b = self.augmentations[constrast_id - 1](x1)
                z2 = z1 @ A.numpy() + b.numpy()

        # else:
        #     z1 = self.z_data[idx]
        #     c = [[(5+5*i)%64, (100+5*i)%255, (180+5*i)%255] for i in range(self.n_balls)]
        
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        self._teardown()

        return {"latents":(z1.flatten(), z2.flatten())
                , "images":(x1, x2)
                , "segmentation_masks":(segmentation_masks1, segmentation_masks2)
                , "matrices":(A.float(), b.flatten().float())
                , "mechanism_permutation": torch.tensor(mechanism_permutation).flatten()
                , "coordinates": (z1.flatten(), z2.flatten())
                , "colors": (torch.tensor(c)/255., torch.tensor(c)/255.)}



    def _generate_data(self):
        # keys = ["latents", "images", "segmentation_masks", "matrices", "coordinates", "colors"]
        # data = dict.fromkeys(keys)
        # for key in keys:
        #     data[key] = []

        # for _ in range(self.num_samples):
        #     sample = self._sample()
        #     for key in keys:
        #         data[key].append(sample[key])
        data = []
        for _ in range(self.num_samples):
            sample = self._sample()
            data.append(sample)
        return data

    def __getitem__(self, idx):
        return self.data[idx]
