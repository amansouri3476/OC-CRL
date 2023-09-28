from copy import deepcopy
from .abstract_inertia_balls_dataset import InertialBall
import torch
import os
import numpy as np
from typing import Callable, Optional
import colorsys
import itertools

if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

# COLOURS_ = [
#     [2, 156, 154],
#     [222, 100, 100],
#     [149, 59, 123],
#     [74, 114, 179],
#     [27, 159, 119],
#     [218, 95, 2],
#     [117, 112, 180],
#     [232, 41, 139],
#     [102, 167, 30],
#     [231, 172, 2],
#     [167, 118, 29],
#     [102, 102, 102],
# ]

# HSV colours
COLOURS_ = [
    [0.05, 0.6, 0.6],
    [0.15, 0.6, 0.6],
    [0.25, 0.6, 0.6],
    [0.35, 0.6, 0.6],
    [0.45, 0.6, 0.6],
    [0.55, 0.6, 0.6],
    [0.65, 0.6, 0.6],
    [0.75, 0.6, 0.6],
    [0.85, 0.6, 0.6],
    [0.95, 0.6, 0.6],
]


SCREEN_DIM = 64
Y_SHIFT = -0.9


class InertialBallColourOffset(InertialBall):
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
        super(InertialBallColourOffset, self).__init__(transform
                                                , offsets
                                                , augmentations
                                                , human_mode
                                                , n_balls
                                                , n_colours
                                                , num_samples
                                                ,**kwargs
                                                )
        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.sparsity_degree = kwargs.get("sparsity_degree", n_balls)
        self.known_mechanism = kwargs.get("known_mechanism", True)


    def _sample_offsets(self):
        n_offsets = self.offsets.shape[0]
        
        mask = np.random.choice(range(n_offsets), size=(self.sparsity_degree,), replace=False)
        mask_onehot = np.zeros((mask.size, n_offsets))
        mask_onehot[np.arange(mask.size),mask] = 1
        mask = mask_onehot
        sampled_offsets = (mask * self.offsets).sum(axis=1)

        # resample offsets that have a duplicate among the sampled offsets
        mask_boundary = np.ones(sampled_offsets.shape)
        duplicate_mechanism_threshold = 0.05
        sampled_offsets_distance_matrix = np.abs(np.abs(sampled_offsets[:, None]) - np.abs(sampled_offsets[None, :]))
        duplicate_mask = np.triu(sampled_offsets_distance_matrix<duplicate_mechanism_threshold).sum(-1)>1
        mask_boundary = 1. - duplicate_mask.astype(int)
        # mask_boundary[duplicate_mask] = 0.

        resample_mask = ~ (mask_boundary > 0.)

        while_loop_threshold = 1000
        while resample_mask.any() and while_loop_threshold > 0:
            while_loop_threshold -= 1
            mask = np.random.choice(range(n_offsets), size=(self.sparsity_degree,), replace=False)
            mask_onehot = np.zeros((mask.size, n_offsets))
            mask_onehot[np.arange(mask.size),mask] = 1
            mask = mask_onehot
            sampled_offsets_temp = (mask * self.offsets).sum(axis=1)
            sampled_offsets[resample_mask] = sampled_offsets_temp[resample_mask]

            # resample offsets that have a duplicate among the sampled offsets
            sampled_offsets_distance_matrix = np.abs(np.abs(sampled_offsets[:, None]) - np.abs(sampled_offsets[None, :]))
            duplicate_mask = np.triu(sampled_offsets_distance_matrix<duplicate_mechanism_threshold).sum(-1)>1
            mask_boundary = 1. - duplicate_mask.astype(int)
            # mask_boundary[duplicate_mask] = 0.

            resample_mask = ~ (mask_boundary > 0.)

        return sampled_offsets


    def __getitem__(self, idx):
        self._setup()
        if self.generate_data:
            z1 = np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))

            # prevent balls from getting initialized very close to each other
            mask = np.ones(z1.shape)
            duplicate_z1_threshold = 0.1
            sampled_z1_distance_matrix = np.linalg.norm(z1[:, None, :] - z1[None, :, :], axis=-1)
            duplicate_mask = np.triu(sampled_z1_distance_matrix<duplicate_z1_threshold).sum(-2)>1
            mask = 1. - duplicate_mask.astype(int)
            # mask[duplicate_mask, :] = 0.

            resample_mask = ~ (mask > 0.)

            while_loop_threshold = 100
            while resample_mask.any() and while_loop_threshold > 0:
                while_loop_threshold -= 1
                z1_temp = np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))
                z1[resample_mask] = z1_temp[resample_mask]

                # prevent balls from getting initialized very close to each other
                mask = np.ones(z1.shape)
                sampled_z1_distance_matrix = np.linalg.norm(z1[:, None, :] - z1[None, :, :], axis=-1)
                duplicate_mask = np.triu(sampled_z1_distance_matrix<duplicate_z1_threshold).sum(-2)>1
                mask = 1. - duplicate_mask.astype(int)
                # mask[duplicate_mask, :] = 0.

                resample_mask = ~ (mask > 0.)

            if self.color_selection == "cyclic_fixed":
                # picking colours cyclically from the list of colours, different but fixed set of colors during training and test.
                # we should randomly permute c1, or else, the latent will always be the same, collapsing the computation of mcc

                # the id of the random permutation of the indices in [0,n_balls-1]
                random_permutation_id = np.random.choice(np.arange(len(list(itertools.permutations(np.arange(self.n_balls))))), 1)[0]
                # a random permutation of the indices in [0,n_balls-1]
                ids = list(itertools.permutations(np.arange(self.n_balls)))[random_permutation_id]
                c1 = [COLOURS_[ids[i]] for i in range(self.n_balls)]

                # making the hue random so that different colours are used for the image at time t. Otherwise we will have the same colour at time t for all images,
                # which is certanily not a good idea.
                hue_1 = np.array(np.random.uniform(0., 1., size=(self.n_balls,)))
                for i, c in enumerate(c1):
                    c[0] = hue_1[i]

                # now the latents won't be all identical because of the fixed ordering of the colours, hence the computation of the
                # mcc won't break

            elif self.color_selection == "same": # TODO: this method won't work, as all latents will be identical and the mcc
                                                 # computation will break.
                # same color for all balls
                c1 = [COLOURS_[0] for i in range(self.n_balls)]
            elif self.color_selection == "random":
                # randomized selection of colors for balls. The colors won't be fixed for different elements in a batch
                _replace = True if self.n_balls > len(COLOURS_) else False
                indices = list(np.random.choice(range(len(COLOURS_)), self.n_balls, replace=_replace))
                c1 = [COLOURS_[i] for i in indices]

        else:
            # TODO: for colour dataset, the following is not correct.
            z1 = self.z_data[idx]
            c1 = [[(5+5*i)%64, (100+5*i)%255, (180+5*i)%255] for i in range(self.n_balls)]
        
        # note the multiplication by 255., because draw_scene works with rgb colours in the range [0, 255.]
        rgb_c1 = [[255.*channel for channel in colorsys.hls_to_rgb(*c)] for c in c1]

        # segmentation_mask1: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask1[0] is the background mask
        x1, segmentation_masks1 = self.draw_scene(z1, rgb_c1)

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

        if constrast_id == 0:
            b = self._sample_offsets()
            # IMPORTANT: The offset needs to be added only to the hue, the rest of colour channels should remain fixed
            # so we get a one-dimensional latent.
            c1 = torch.tensor(c1) # [n_balls, num_channels]
            c2 = deepcopy(c1)

            # sampled_offsets are [self.sparsity_degree, z_dim]; to be able to add them to z1, we need to create a z1-like zeros array
            # and fill it a with a permutation of sampled offsets
            b_ = np.zeros_like(c1[:,0]) # [n_balls, z_dim]
            idx = np.random.choice(range(self.n_balls), size=(self.sparsity_degree,), replace=False)
            b_[idx] = b
            c2[:,0] += torch.tensor(b_)

            # bringing the hues back to [0,1] range, so the disentanglement computations do not break
            correction_offset = (c2[:,0] < 0.) * 1.
            c2[:,0] += correction_offset

            rgb_c2 = [[255*channel for channel in colorsys.hls_to_rgb(*c)] for c in c2]

            # segmentation_mask2: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask2[0] is the background mask
            x2, segmentation_masks2 = self.draw_scene(z1, rgb_c2)
            b = torch.tensor(b).float().squeeze()
            if not self.known_mechanism: # TODO: might confuse the matching? Is it ok in the sparse setting if fake changes are identical? Probably it's fine
                                         # because the real offsets aren't the same.
                                         # TODO: how about the sign of the change? does that matter?
                b = (torch.abs(b) > 0.) * 1.
            try:
                A = torch.eye(b.shape[0]).float()
            except:
                A = torch.eye(1).float()
        else:
            x2, A, b = self.augmentations[constrast_id - 1](x1)
            z2 = z1 @ A.numpy() + b.numpy()
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        self._teardown()
        return {"latents":(c1[:,0].flatten(), c2[:,0].flatten()) # NOTE: We should only pass hue as the latent, not all 3 hsv channels!
                                                       # hence, the slicing of c1,c2
                , "images":(x1, x2)
                , "segmentation_masks":(segmentation_masks1, segmentation_masks2)
                , "matrices":(A.float(), b.flatten().float())
                , "mechanism_permutation": torch.tensor(idx).flatten()
                , "coordinates": (z1.flatten(), z1.flatten())
                , "colors":(torch.tensor(rgb_c1)/255., torch.tensor(rgb_c2)/255.)}
