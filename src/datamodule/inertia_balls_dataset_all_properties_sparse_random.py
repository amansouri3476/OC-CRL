from .abstract_inertia_balls_dataset import InertialBall, circle
import torch
import os
import numpy as np
from typing import Callable, Optional
import pygame
from pygame import gfxdraw
import colorsys


if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dsp"

# HSV colours
COLOURS_ = [
    # [0.05, 0.6, 0.6],
    [0.15, 0.6, 0.6],
    # [0.25, 0.6, 0.6],
    # [0.35, 0.6, 0.6],
    # [0.45, 0.6, 0.6],
    # [0.55, 0.6, 0.6],
    # [0.65, 0.6, 0.6],
    # [0.75, 0.6, 0.6],
    [0.85, 0.6, 0.6],
    # [0.95, 0.6, 0.6],
]

SHAPES_ = [
    # "circle",
    "square",
    # "triangle",
    "heart"
]

PROPERTIES_ = [
    "x",
    "y",
    "c",
    "s"
]

SCREEN_DIM = 64
Y_SHIFT = 0.0

def draw_shape(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=Y_SHIFT,
    offset=None,
    shape="circle"
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    temp_surf = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)

    if shape == "circle":
        # pygame.draw.circle(surface=surf, color=color,
        #                center=(int(x), int(y - offset * y_shift)), radius=int(radius * scale))
        gfxdraw.aacircle(
            surf, int(x), int(y - offset * y_shift), int(radius * scale), color
            )
        gfxdraw.filled_circle(
            surf, int(x), int(y - offset * y_shift), int(radius * scale), color
        )

        # for segmentation mask
        gfxdraw.aacircle(
        temp_surf, int(x), int(y - offset * y_shift), int(radius * scale), color
            )
        gfxdraw.filled_circle(
            temp_surf, int(x), int(y - offset * y_shift), int(radius * scale), color
            )
    elif shape == "square":
        radius = int(radius * scale)*2
        pygame.draw.rect(surface=surf, color=color,
                        rect=(int(x) - radius//2, int(y - offset * y_shift) - radius//2, radius, radius))
        # for segmentation mask
        pygame.draw.rect(surface=temp_surf, color=color,
                        rect=(int(x) - radius//2, int(y - offset * y_shift) - radius//2, radius, radius))
    elif shape == "triangle":
        radius = (radius * scale)*2
        x, y = ((x) - radius/2, (y - offset * y_shift) - radius/2)
        pygame.draw.polygon(surface=surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(x+radius//2,y+radius), (x+radius,y), (x,y)]])
        # for segmentation mask
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(x+radius//2,y+radius), (x+radius,y), (x,y)]])
    elif shape == "heart":
        radius = (radius * scale)*2
        x, y = ((x) , (y - offset * y_shift))
        s = 3.4 # 3.5
        j = 1.33
        pygame.draw.circle(surface=surf, color=color,
                    center=(int(x+ radius /(s * j)), int(y + radius/(s * j))), radius=int(radius/s))
        pygame.draw.circle(surface=surf, color=color,
                    center=(int(x- radius/(s*j)), int(y + radius /(s*j))), radius=int(radius/s))
        pygame.draw.polygon(surface=surf, color=color,
                        points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(x,y-radius/2), (x-radius/2.0,y + radius/30), (x+radius/2.0,y+ radius/30)]])

        # for segmentation mask
        pygame.draw.circle(surface=temp_surf, color=color,
                    center=(int(x+ radius /(s * j)), int(y + radius/(s * j))), radius=int(radius/s))
        pygame.draw.circle(surface=temp_surf, color=color,
                    center=(int(x- radius/(s*j)), int(y + radius /(s*j))), radius=int(radius/s))
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(x,y-radius/2), (x-radius/2.0,y + radius/30), (x+radius/2.0,y+ radius/30)]])


    temp_surf_pos = (0,0)
    ball_mask = pygame.mask.from_surface(temp_surf)

    # mask -› surface
    new_temp_surf = ball_mask.to_surface()
    # do the same flip as the one occurring for the screen
    new_temp_surf = pygame.transform.flip(new_temp_surf, False, True)
    new_temp_surf.set_colorkey((0,0,0))

    return np.transpose(np.array(pygame.surfarray.pixels3d(new_temp_surf)), axes=(1, 0, 2))[:, :, :1] # [screen_width, screen_width, 1]

    
class InertialBallAllPropertiesSparseOffset(InertialBall):
    """
    This class instantiates a torch.utils.data.Dataset object. This dataset returns
    pairs of images that correspond to consecutive frames. The ground truth generating
    factors of this dataset are four properties per each ball: x,y,colour,shape
    The number of balls (n) from t->t+1 does not change and is fixed by the datamodule
    config. From t->t+1, only one ball's state can be changed, and that change is also
    constrained to be sparse, meaning that only 1 out of all properties can be altered.
    """

    screen_dim = 64

    def __init__(
        self,
        transform: Optional[Callable] = None,
        offsets: np.ndarray = np.array([[0.05, -0.025]]),
        augmentations: list = [],
        human_mode: bool = False,
        n_balls: int = 1,
        num_samples: int = 20000,
        **kwargs,
    ):
        super(InertialBallAllPropertiesSparseOffset, self).__init__(transform
                                                , offsets
                                                , augmentations
                                                , human_mode
                                                , n_balls
                                                , num_samples
                                                ,**kwargs
                                                )
        if transform is None:
            def transform(x):
                return x

        self.transform = transform

        # TODO: for this dataset the sparsity degree should be 1, we can think of other
        # possibilities later
        self.sparsity_degree = 1 # kwargs.get("sparsity_degree", n_balls)
        # self.known_mechanism = kwargs.get("known_mechanism", True)
        self.offset_x = kwargs.get("offset_x")
        self.offset_y = kwargs.get("offset_y")
        self.injective = kwargs.get("injective")
        self.same_color = kwargs.get("same_color")
        self.output_sparse_offsets = kwargs.get("output_sparse_offsets")
        self.signed = kwargs.get("signed")
        self.z_dim = kwargs.get("z_dim")
        self.properties_list = kwargs.get("properties_list") # a subset of ["x","y","c","s"] preserving the order
        self.target_property_indices = [i for i,p in enumerate(PROPERTIES_) if p in self.properties_list]
        self.non_target_property_indices = [i for i,p in enumerate(PROPERTIES_) if p not in self.properties_list]
        assert self.z_dim == len(self.properties_list) 
        self.z_dim_3 = kwargs.get("z_dim_3") # if 1 -> 3rd property is colour, if 0 -> 3rd property is shape
        self.ball_rad = kwargs.get("ball_radius", 0.1)
        self.data = self._generate_data()


    def _sample_offsets(self, property_idx):

        if property_idx == 0:
            offset = self.offset_x * np.random.choice([-1.,1.],1) if self.signed else self.offset_x
        elif property_idx == 1:
            offset = self.offset_y * np.random.choice([-1.,1.],1) if self.signed else self.offset_y
        elif property_idx == 2 or property_idx == 3:
            offset = int(1) if self.signed else int(np.random.choice([-1.,1.],1))
        else:
            raise Exception(f"The property index provided {property_idx} is invalid. It should be in the [0,3] range.")

        return offset

    def _sample_z1_z2_rest(self, z_all, ball_idx):

        n_balls = z_all.shape[0]
        num_colours = len(COLOURS_)
        num_shapes = len(SHAPES_)
        idx_mask = np.arange(n_balls)!=ball_idx

        # sample colours for the rest of the balls
        # if colour is among the target properties then it should be picked at random, o.w. it
        # should be fixed
        if 2 in self.target_property_indices: # colour (2) is among the targets
            colour_indices = np.random.choice(range(num_colours), n_balls-1, replace=False).astype(int)
        else:
            colour_indices = np.arange(1, n_balls) # n_balls-1 fixed colours chosen if the 3rd property is shape
        z_all[idx_mask, 2] = colour_indices
        # z_all[idx_mask, 2] = hsv_colours[idx_mask, 0]
        
        # sample shapes for the rest of the balls
        # if shape is among the target properties then it should be picked at random, o.w. it
        # should be fixed
        if 3 in self.target_property_indices: # shape (3) is among the targets
            shape_indices = np.random.choice(range(num_shapes), n_balls-1)
        else:
            shape_indices = np.zeros((n_balls-1,))
        z_all[idx_mask, 3] = shape_indices

        # TODO: Do we need random positions or should they be fixed?
        # regardless of x,y being the target of disentanglement or not, we should have random positions
        # sample ball coordinates for the rest of the balls
        # coordinates_1 = np.array([0.25, 0.25])
        coordinates_1 = np.random.uniform(0.1, 0.9, size=(n_balls-1, 2))
        # # coordinates_1 = np.array([[0.3, 0.7]])
        
        # prevent balls from getting initialized very close to each other
        mask = np.ones(coordinates_1.shape)
        duplicate_coordinates_1_threshold = self.ball_rad * 3
        sampled_coordinates_1_distance_matrix = np.linalg.norm(coordinates_1[:, None, :] - coordinates_1[None, :, :], axis=-1)
        duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
        mask[duplicate_mask] = 0.

        resample_mask = ~ (mask > 0.)

        # under no circumstances should z2 fall out of the frame, so this loop should continue until the constraints are satisfied.
        while_loop_threshold = 1000
        while resample_mask.any() and while_loop_threshold > 0:
        # while resample_mask.any():
            while_loop_threshold -= 1
            coordinates_temp = np.random.uniform(0.1, 0.9, size=(n_balls-1, 2))
            coordinates_1[resample_mask] = coordinates_temp[resample_mask]

            # prevent balls from getting initialized very close to each other
            mask = np.ones(coordinates_1.shape)
            sampled_coordinates_1_distance_matrix = np.linalg.norm(coordinates_1[:, None, :] - coordinates_1[None, :, :], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask[duplicate_mask] = 0.

            resample_mask = ~ (mask > 0.)
        
        z_all[idx_mask, :2] = coordinates_1

        return z_all


    def _sample_z1_z2(self, ball_idx, property_idx, offset, z_all):
    
        upper_bound_x = 1 - self.ball_rad
        upper_bound_y = 1 - self.ball_rad
        lower_bound_x = self.ball_rad
        lower_bound_y = self.ball_rad

        if property_idx == 0:
            # sample coordinates for the chosen ball
            coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
            # coordinates_1 = np.array([0.6, 0.3])
            coordinates_2 = coordinates_1.copy()
            coordinates_2[property_idx] += offset


            # check the constraints
            mask = False
            
            # make sure this ball isn't initialized very close to other balls
            duplicate_coordinates_1_threshold = self.ball_rad * 3
            z_all_temp1 = z_all.copy()
            z_all_temp1[ball_idx, :2] = coordinates_1
            sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

            # make sure this ball doesn't fall very close to other balls at t+1
            duplicate_coordinates_2_threshold = self.ball_rad * 3
            z_all_temp2 = z_all.copy()
            z_all_temp2[ball_idx, :2] = coordinates_2
            sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
            # if any constraint is violated, we should resmaple, should also consider the previous masks
            mask = mask or duplicate_mask.any()

            # make sure this ball doesn't fall out of the frame after the offset has been applied
            if coordinates_2[property_idx] > upper_bound_x or coordinates_2[property_idx] < lower_bound_x:
                mask = True

            resample_mask = mask

            while_loop_threshold = 1000
            while resample_mask and while_loop_threshold > 0:
                while_loop_threshold -= 1
                coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
                coordinates_2 = coordinates_1.copy()
                coordinates_2[property_idx] += offset

                # check the constraints
                mask = False
                
                # make sure this ball isn't initialized very close to other balls
                duplicate_coordinates_1_threshold = self.ball_rad * 3
                z_all_temp1 = z_all.copy()
                z_all_temp1[ball_idx, :2] = coordinates_1
                sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall very close to other balls at t+1
                duplicate_coordinates_2_threshold = self.ball_rad * 3
                z_all_temp2 = z_all.copy()
                z_all_temp2[ball_idx, :2] = coordinates_2
                sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
                mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall out of the frame after the offset has been applied
                if coordinates_2[property_idx] > upper_bound_x or coordinates_2[property_idx] < lower_bound_x:
                    mask = True

                resample_mask = mask

            if while_loop_threshold == 0:
                if coordinates_2[property_idx] > upper_bound_x:
                    coordinates_2[property_idx] = upper_bound_x
                if coordinates_2[property_idx] < lower_bound_x:
                    coordinates_2[property_idx] = lower_bound_x

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()
            z_all_1[ball_idx, :2] = coordinates_1
            z_all_2[ball_idx, :2] = coordinates_2
            if 2 in self.target_property_indices:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            if 3 in self.target_property_indices:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0
            z_all_2[ball_idx, 2] = z_all_1[ball_idx, 2].copy()
            z_all_2[ball_idx, 3] = z_all_1[ball_idx, 3].copy()


        if property_idx == 1:
            # sample coordinates for the chosen ball
            coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
            # coordinates_1 = np.array([0.6, 0.3])
            coordinates_2 = coordinates_1.copy()
            coordinates_2[property_idx] += offset


            # check the constraints
            mask = False
            
            # make sure this ball isn't initialized very close to other balls
            duplicate_coordinates_1_threshold = self.ball_rad * 3
            z_all_temp1 = z_all.copy()
            z_all_temp1[ball_idx, :2] = coordinates_1
            sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

            # make sure this ball doesn't fall very close to other balls at t+1
            duplicate_coordinates_2_threshold = self.ball_rad * 3
            z_all_temp = z_all.copy()
            z_all_temp[ball_idx, :2] = coordinates_2
            sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp[:, None, :2] - z_all_temp[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
            mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

            # make sure this ball doesn't fall out of the frame after the offset has been applied
            if coordinates_2[property_idx] > upper_bound_y or coordinates_2[property_idx] < lower_bound_y:
                mask = True

            resample_mask = mask

            while_loop_threshold = 1000
            while resample_mask and while_loop_threshold > 0:
                while_loop_threshold -= 1
                coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
                coordinates_2 = coordinates_1.copy()
                coordinates_2[property_idx] += offset

                # check the constraints
                mask = False
                
                # make sure this ball isn't initialized very close to other balls
                duplicate_coordinates_1_threshold = self.ball_rad * 3
                z_all_temp1 = z_all.copy()
                z_all_temp1[ball_idx, :2] = coordinates_1
                sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall very close to other balls at t+1
                duplicate_coordinates_2_threshold = self.ball_rad * 3
                z_all_temp = z_all.copy()
                z_all_temp[ball_idx, :2] = coordinates_2
                sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp[:, None, :2] - z_all_temp[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
                mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall out of the frame after the offset has been applied
                if coordinates_2[property_idx] > upper_bound_y or coordinates_2[property_idx] < lower_bound_y:
                    mask = True

                resample_mask = mask

            if while_loop_threshold == 0:
                if coordinates_2[property_idx] > upper_bound_y:
                    coordinates_2[property_idx] = upper_bound_y
                if coordinates_2[property_idx] < lower_bound_y:
                    coordinates_2[property_idx] = lower_bound_y

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()
            z_all_1[ball_idx, :2] = coordinates_1
            z_all_2[ball_idx, :2] = coordinates_2
            if 2 in self.target_property_indices:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            if 3 in self.target_property_indices:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0
            z_all_2[ball_idx, 2] = z_all_1[ball_idx, 2].copy()
            z_all_2[ball_idx, 3] = z_all_1[ball_idx, 3].copy()

        # if self.z_dim < 3 then the code won't reach here
        if property_idx == 2:
            colour_min_idx = 0
            colour_max_idx = len(COLOURS_)-1
            colour_idx_1 = np.random.choice(range(len(COLOURS_)), 1).astype(int)
            while colour_idx_1 + offset > colour_max_idx or colour_idx_1 + offset < colour_min_idx:
                colour_idx_1 = np.random.choice(range(len(COLOURS_)), 1).astype(int)
            colour_idx_2 = colour_idx_1 + offset

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()

            # sample coordinates for the chosen ball
            # coordinates_1 = np.array([0.5, 0.5])
            coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
            coordinates_2 = coordinates_1.copy()

            # check the constraints
            mask = False

            # make sure this ball isn't initialized very close to other balls
            duplicate_coordinates_1_threshold = self.ball_rad * 3
            z_all_temp1 = z_all.copy()
            z_all_temp1[ball_idx, :2] = coordinates_1
            sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask = duplicate_mask.any() # if any constraint is violated, we should resmaple
            resample_mask = mask

            while_loop_threshold = 1000
            while resample_mask and while_loop_threshold > 0:
                while_loop_threshold -= 1
                coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
                coordinates_2 = coordinates_1.copy()

                # check the constraints
                mask = False
                
                # make sure this ball isn't initialized very close to other balls
                duplicate_coordinates_1_threshold = self.ball_rad * 3
                z_all_temp1 = z_all.copy()
                z_all_temp1[ball_idx, :2] = coordinates_1
                sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

                resample_mask = mask


            z_all_1[ball_idx, :2] = coordinates_1
            z_all_2[ball_idx, :2] = coordinates_2
            z_all_1[ball_idx, 2] = colour_idx_1
            if 3 in self.target_property_indices:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0
            z_all_2[ball_idx, 2] = colour_idx_2
            z_all_2[ball_idx, 3] = z_all_1[ball_idx, 3].copy()
            
        # if self.z_dim < 4 then the code won't reach here
        if property_idx == 3:
            shape_min_idx = 0
            shape_max_idx = len(SHAPES_)-1
            shape_idx_1 = np.random.choice(range(len(SHAPES_)), 1).astype(int)
            while shape_idx_1 + offset > shape_max_idx or shape_idx_1 + offset < shape_min_idx:
                shape_idx_1 = np.random.choice(range(len(SHAPES_)), 1).astype(int)
            shape_idx_2 = shape_idx_1 + offset

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()

            # sample coordinates for the chosen ball
            # coordinates_1 = np.array([0.5, 0.5])
            coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
            coordinates_2 = coordinates_1.copy()

            # check the constraints
            mask = False

            # make sure this ball isn't initialized very close to other balls
            duplicate_coordinates_1_threshold = 0.2
            z_all_temp1 = z_all.copy()
            z_all_temp1[ball_idx, :2] = coordinates_1
            sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask = duplicate_mask.any() # if any constraint is violated, we should resmaple
            resample_mask = mask

            while_loop_threshold = 1000
            while resample_mask and while_loop_threshold > 0:
                while_loop_threshold -= 1
                coordinates_1 = np.random.uniform(0.1, 0.9, size=(2,))
                coordinates_2 = coordinates_1.copy()

                # check the constraints
                mask = False
                
                # make sure this ball isn't initialized very close to other balls
                duplicate_coordinates_1_threshold = self.ball_rad * 3
                z_all_temp1 = z_all.copy()
                z_all_temp1[ball_idx, :2] = coordinates_1
                sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

                resample_mask = mask


            z_all_1[ball_idx, :2] = coordinates_1
            z_all_2[ball_idx, :2] = coordinates_2
            if 2 in self.target_property_indices:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            z_all_1[ball_idx, 3] = shape_idx_1
            z_all_2[ball_idx, 2] = z_all_1[ball_idx, 2].copy()
            z_all_2[ball_idx, 3] = shape_idx_2

        return z_all_1, z_all_2



    def _sample(self):
        self._setup()
        if self.generate_data:
            # 1. we have to pick a ball m and a property i at random. The assumed order is [x,y,c,s].
            # we need ball_idx because we want to sample the latents for the rest of the balls and make
            # sure they are not initialized close to each other. Additionally, the chosen ball should
            # be placed somewhere, and we do not want it to be very close to any other balls, so if we
            # sampled the rest first, we are able to iterate and find a feasible latent for the chosen
            # ball.
            ball_idx = np.random.choice(range(self.n_balls), 1)
            # property_idx = np.random.choice(range(self.z_dim), 1)
            property_idx = np.random.choice(self.target_property_indices, 1)
            z_all = np.zeros((self.n_balls, 4))

            # 2. we can now sample the complete latent z for all other balls
            # the promise is that there are no two balls very close to one another. However, the rest of the
            # properties (colour, shape) can be identical for two or more balls.
            # note that the final z1,z2 have the same entries as z_all for all balls except one.
            z_all = self._sample_z1_z2_rest(z_all, ball_idx)

            # 3. based on the chosen property for change, we will have to choose an offset with its sign
            sampled_offset = self._sample_offsets(property_idx)

            # 4. we can now sample that property and make sure it is consistent with the chosen offset
            # and the rest of the balls.
            # z_all_1,z_all_2 are [n_balls, 4]
            z_all_1, z_all_2 = self._sample_z1_z2(ball_idx=ball_idx, property_idx=property_idx, offset=sampled_offset, z_all=z_all)
            if self.injective:
                if self.same_color:
                    hsv_colours_1 = [COLOURS_[0] for i in range(self.n_balls)]
                    hsv_colours_2 = [COLOURS_[0] for i in range(self.n_balls)]
                else:
                    hsv_colours_1 = [COLOURS_[i] for i in range(self.n_balls)]
                    hsv_colours_2 = [COLOURS_[i] for i in range(self.n_balls)]
            else:
                if self.same_color:
                    hsv_colours_1 = [COLOURS_[0] for i in range(self.n_balls)]
                    hsv_colours_2 = [COLOURS_[0] for i in range(self.n_balls)]
                else:
                    hsv_colours_1 = [COLOURS_[z_all_1[i,2].astype(int)] for i in range(z_all_1.shape[0])]
                    hsv_colours_2 = [COLOURS_[z_all_2[i,2].astype(int)] for i in range(z_all_2.shape[0])]
            # filling z_all_1,2 with colour hues at dimension 2
            z_all_1[:, 2] = np.array(hsv_colours_1)[:, 0]
            z_all_2[:, 2] = np.array(hsv_colours_2)[:, 0]
            # note the multiplication by 255., because draw_scene works with rgb colours in the range [0, 255.]
            rgb_colours_1 = [[255.*channel for channel in colorsys.hls_to_rgb(*c)] for c in hsv_colours_1]
            rgb_colours_2 = [[255.*channel for channel in colorsys.hls_to_rgb(*c)] for c in hsv_colours_2]

            # segmentation_mask1: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask1[0] is the background mask
            x1, segmentation_masks1 = self.draw_scene(z_all_1, rgb_colours_1)
            # segmentation_mask2: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask2[0] is the background mask
            x2, segmentation_masks2 = self.draw_scene(z_all_2, rgb_colours_2)

            # dividing z_all_1,2[:, -1] (shape dimension) by the number of shapes so the latent
            # becomes nicer and close to the rest of the features.
            z_all_1[:, -1] /= len(SHAPES_)
            z_all_2[:, -1] /= len(SHAPES_)

        x1 = self.transform(x1)
        x2 = self.transform(x2)
        
        z1 = z_all_1[..., self.target_property_indices].copy()
        z2 = z_all_2[..., self.target_property_indices].copy()

        # b = np.zeros((self.z_dim,))
        if self.output_sparse_offsets:
            b = np.zeros((4,))
            b[property_idx] = sampled_offset
            b = b[self.target_property_indices]
            A = torch.eye(b.shape[0]).float()
        else: # return an offset vector of the shape [n_balls * z_dim]
            b = np.zeros((self.n_balls, 4))
            b[ball_idx, property_idx] = sampled_offset
            b = b[:, self.target_property_indices]
            A = torch.eye(b.shape[0]).float()

        self._teardown()

        return {"latents":(z1.flatten(), z2.flatten())
                , "images":(x1, x2)
                , "segmentation_masks":(segmentation_masks1, segmentation_masks2)
                , "matrices":(A.float(), torch.tensor(b).flatten().float())
                , "mechanism_permutation": ball_idx
                , "coordinates": (z_all_1[:, :2].flatten(), z_all_2[:, :2].flatten())
                , "colors": (torch.tensor(rgb_colours_1)/255., torch.tensor(rgb_colours_2)/255.)}

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            sample = self._sample()
            data.append(sample)
        return data

    def __getitem__(self, idx):
        return self.data[idx]

    def draw_scene(self, z, colours=None):
        self.surf.fill((255, 255, 255))
        # getting the background segmentation mask
        self.bg_surf = pygame.Surface((self.screen_dim, self.screen_dim), pygame.SRCALPHA)

        obj_masks = []
        if z.ndim == 1:
            z = z.reshape((1, 2))
        if colours is None:
            colours = [COLOURS_[3]] * z.shape[0]
        for i in range(z.shape[0]):
            obj_masks.append(
                draw_shape(
                    z[i, 0],
                    z[i, 1],
                    self.surf,
                    color=colours[i],
                    radius=self.ball_rad,
                    screen_width=self.screen_dim,
                    y_shift=0.0,
                    offset=0.0,
                    shape=SHAPES_[int(z[i,3])]
                )
            )
            _ = draw_shape(
                z[i, 0],
                z[i, 1],
                self.bg_surf,
                color=colours[i],
                radius=self.ball_rad,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
                shape=SHAPES_[int(z[i,3])]
            )

        bg_surf_pos = (0,0)
        bg_mask = pygame.mask.from_surface(self.bg_surf)
        bg_mask.invert() # so that mask bits for balls are cleared and the bg gets set.

        # mask -› surface
        new_bg_surf = bg_mask.to_surface()
        new_bg_surf.set_colorkey((0,0,0))
        # do the same flip as the one occurring for the screen
        new_bg_surf = pygame.transform.flip(new_bg_surf, False, True)

        # print(np.array(pygame.surfarray.pixels3d(new_bg_surf)).shape)
        # bg_mask = np.array(pygame.surfarray.pixels3d(new_bg_surf))[:, :, :1] # [screen_width, screen_width, 1]
        bg_mask = np.transpose(np.array(pygame.surfarray.pixels3d(new_bg_surf)), axes=(1, 0, 2))[:, :, :1] # [screen_width, screen_width, 1]
        # ------------------------------------------ #
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.human_mode:
            pygame.display.flip()
        return (
            np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                )
            , np.array([bg_mask] + obj_masks)
        )
