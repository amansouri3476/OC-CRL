from .abstract_inertia_balls_dataset import InertialBall, circle
import torch
import os
import numpy as np
import math
from typing import Callable, Optional
import pygame
from pygame import gfxdraw
import colorsys
import src.utils.general as utils
from src.datamodule.inertia_balls_dataset_all_p_sparse_pickle import InertialBallAllPropertiesSparseOffsetPickleable
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = False

if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dsp"

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
    # "diamond",
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

SCREEN_DIM = 128
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
    shape="circle",
    rotation_angle=0.
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    temp_surf_final = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)
    if shape is not "heart" and shape is not "diamond":
        temp_surf_rotation = pygame.Surface((2 * scale * radius, 2 * scale * radius), pygame.SRCALPHA) # for rotations
        temp_surf = pygame.Surface((2 * scale * radius, 2 * scale * radius), pygame.SRCALPHA) # for rotations
    else:
        if shape is "heart":
            temp_surf_rotation = pygame.Surface((2.2 * scale * radius, 2.2 * scale * radius), pygame.SRCALPHA) # for rotations
            temp_surf = pygame.Surface((2.2 * scale * radius, 2.2 * scale * radius), pygame.SRCALPHA) # for rotations
        else:
            temp_surf_rotation = pygame.Surface((3 * scale * radius, 3 * scale * radius), pygame.SRCALPHA) # for rotations
            temp_surf = pygame.Surface((3 * scale * radius, 3 * scale * radius), pygame.SRCALPHA) # for rotations

    if shape == "circle":
        gfxdraw.aacircle(
            temp_surf_rotation, 0, 0, int(radius * scale), color
            )
        gfxdraw.filled_circle(
            temp_surf_rotation, 0, 0, int(radius * scale), color
        )

        # for segmentation mask
        gfxdraw.aacircle(
        temp_surf, 0, 0, int(radius * scale), color
            )
        gfxdraw.filled_circle(
            temp_surf, 0, 0, int(radius * scale), color
            )
        # gfxdraw.filled_circle(
        #     temp_surf, int(x), int(y), int(radius * scale), color
        #     )

    elif shape == "square":
        radius = int(radius * scale)*2
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        points=[(int(i), int(j)) for i, j in [(0,0), (radius,0), (radius,radius), (0,radius)]])
        # for segmentation mask
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(0, 0), (radius,0), (radius,radius), (0,radius)]])
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(i), int(j)) for i, j in [(int(x), int(y)), (int(x)+radius,int(y)), (int(x)+radius,int(y)+radius), (int(x),int(y)+radius)]])

    elif shape == "triangle":
        radius = (radius * scale)*2
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (0,0)]])
        # for segmentation mask
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (0, 0)]])
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(i), int(j)) for i, j in [(int(x)+radius//2,int(y)+radius), (int(x)+radius,int(y)), (int(x), int(y))]])
    
    elif shape == "diamond":
        radius = (radius * scale) * 1.4
        # pygame.draw.polygon(surface=temp_surf_rotation, color=color,
        #                 # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
        #                 points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (radius//2, 0), (0, radius)]])
        # # for segmentation mask
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0, 0)]])
        #                 points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (radius//2, 0), (0, radius)]])

        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (0, radius)]])
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 0), (radius, radius), (0, radius)]])

        # for segmentation mask
        pygame.draw.polygon(surface=temp_surf, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 2 * radius), (radius, radius), (0, radius)]])
        pygame.draw.polygon(surface=temp_surf, color=color,
                        # points=[(int(i), int(j)) for i, j in [(radius//2,radius), (radius,0), (radius//2,-radius), (0,0)]])
                        points=[(int(i), int(j)) for i, j in [(radius//2, 0), (radius, radius), (0, radius)]])

        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(i), int(j)) for i, j in [(int(x)+radius//2,int(y)+radius), (int(x)+radius,int(y)), (int(x), int(y))]])

    elif shape == "heart":
        radius = (radius * scale)*2
        s = 3.4 # 3.5
        j = 1.33
        offset_x = 3
        pygame.draw.circle(surface=temp_surf_rotation, color=color,
                    center=(offset_x + int(3 * radius /(s * j)), int(radius/(s * j) + radius/2)), radius=int(radius/s))
        pygame.draw.circle(surface=temp_surf_rotation, color=color,
                    center=(offset_x + int(radius/(s*j)), int(radius /(s*j) + radius/2)), radius=int(radius/s))
        pygame.draw.polygon(surface=temp_surf_rotation, color=color,
                        points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(offset_x + 2*radius/(s*j),0), (offset_x + 2 * radius/(s*j) - radius/2.0,radius/30 + radius/2), (offset_x + 2*radius/(s*j) + radius/2.0,radius/30 + radius/2)]])
        # for segmentation mask
        pygame.draw.circle(surface=temp_surf, color=color,
                    center=(int(3 * radius /(s * j)), int(radius/(s * j) + radius/2)), radius=int(radius/s))
        pygame.draw.circle(surface=temp_surf, color=color,
                    center=(int(radius/(s*j)), int(radius /(s*j) + radius/2)), radius=int(radius/s))
        pygame.draw.polygon(surface=temp_surf, color=color,
                        points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(2*radius/(s*j),0), (2 * radius/(s*j) - radius/2.0, radius/30 + radius/2), (2*radius/(s*j) + radius/2.0,radius/30 + radius/2)]])
        # pygame.draw.circle(surface=temp_surf, color=color,
        #             center=(int(x)+int(3 * radius /(s * j)), int(y)+int(radius/(s * j) + radius/2)), radius=int(radius/s))
        # pygame.draw.circle(surface=temp_surf, color=color,
        #             center=(int(x)+int(radius/(s*j)), int(y)+int(radius /(s*j) + radius/2)), radius=int(radius/s))
        # pygame.draw.polygon(surface=temp_surf, color=color,
        #                 points=[(int(np.floor(i)), int(np.floor(j))) for i, j in [(int(x)+2*radius/(s*j),int(y)), (int(x) + 2 * radius/(s*j) - radius/2.0,int(y) + radius/30 + radius/2), (int(x) + 2*radius/(s*j) + radius/2.0,int(y) + radius/30 + radius/2)]])

    rotated_surf = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)
    # Rotate the temporary surface with the rectangle and blit it onto the new surface
    rotated_temp_surf = pygame.transform.rotate(temp_surf_rotation, math.degrees(rotation_angle))
    rotated_temp_surf_rect = rotated_temp_surf.get_rect(center=(int(x), int(y)))
    rotated_surf.blit(rotated_temp_surf, rotated_temp_surf_rect) # now rotations will be around the center and in-place
    surf.blit(rotated_surf, (0, 0))

    temp_surf_pos = (0,0)
    ball_mask = pygame.mask.from_surface(temp_surf)

    # # mask -› surface
    # new_temp_surf = ball_mask.to_surface()
    # # do the same flip as the one occurring for the screen
    # new_temp_surf = pygame.transform.flip(new_temp_surf, False, True)
    # new_temp_surf.set_colorkey((0,0,0))
    
    # -------
    rotated_surf = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)
    # Rotate the temporary surface with the rectangle and blit it onto the new surface
    rotated_temp_surf = pygame.transform.rotate(temp_surf, math.degrees(rotation_angle))
    rotated_temp_surf.set_colorkey((0,0,0))
    rotated_temp_surf_rect = rotated_temp_surf.get_rect(center=(int(x), int(y)))
    rotated_surf.blit(rotated_temp_surf, rotated_temp_surf_rect) # now rotations will be around the center and in-place
    rotated_surf = pygame.transform.flip(rotated_surf, False, True)
    temp_surf_final.blit(rotated_surf, (0, 0))
    # -------

    return np.transpose(np.array(pygame.surfarray.pixels3d(temp_surf_final)), axes=(1, 0, 2))[:, :, :1] # [screen_width, screen_width, 1]


    
class InertialBallAllPropertiesSparseOffset(InertialBall):
    """
    This class instantiates a torch.utils.data.Dataset object. This dataset returns
    pairs of images that correspond to consecutive frames. The ground truth generating
    factors of this dataset are four properties per each ball: x,y,colour,shape
    The number of balls (n) from t->t+1 does not change and is fixed by the datamodule
    config. From t->t+1, only one ball's state can be changed, and that change is also
    constrained to be sparse, meaning that only 1 out of all properties can be altered.
    """

    screen_dim = 128

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
        self.offset_l = kwargs.get("offset_l")
        self.offset_p = kwargs.get("offset_p")
        self.min_size = kwargs.get("min_size")
        self.max_size = kwargs.get("max_size")
        self.min_phi = kwargs.get("min_phi")
        self.max_phi = kwargs.get("max_phi")
        self.injective = kwargs.get("injective")
        self.injective_property = kwargs.get("injective_property")
        self.same_color = kwargs.get("same_color")
        self.same_shape = kwargs.get("same_shape")
        self.same_size = kwargs.get("same_size")
        self.output_sparse_offsets = kwargs.get("output_sparse_offsets")
        self.signed = kwargs.get("signed")
        self.random_coordinates = kwargs.get("random_coordinates")
        self.z_dim = kwargs.get("z_dim")
        self.properties_list = kwargs.get("properties_list") # a subset of ["x","y","c","s"] preserving the order
        if self.injective:
            assert self.injective_property not in self.properties_list, f"Injectivity inducing property {self.injective_property} should not be in the list of disentanglement target properties: {self.properties_list}"
        self.target_property_indices = [i for i,p in enumerate(PROPERTIES_) if p in self.properties_list]
        self.non_target_property_indices = [i for i,p in enumerate(PROPERTIES_) if p not in self.properties_list]
        assert self.z_dim == len(self.properties_list) 
        self.z_dim_3 = kwargs.get("z_dim_3") # if 1 -> 3rd property is colour, if 0 -> 3rd property is shape
        self.ball_rad = kwargs.get("ball_radius", 0.1)
        self.data = self._generate_data()
        self.pickleable_dataset = InertialBallAllPropertiesSparseOffsetPickleable(self.data)


    def _sample_offsets(self, property_idx):

        if property_idx == 0:
            offset = self.offset_x * np.random.choice([-1.,1.],1) if self.signed else self.offset_x
        elif property_idx == 1:
            offset = self.offset_y * np.random.choice([-1.,1.],1) if self.signed else self.offset_y
        elif property_idx == 2:
            # n_colors = len(COLOURS_)
            # offset_list = list(np.arange(-(n_colors-1),n_colors))
            # offset_list.remove(0)
            # offset = int(np.random.choice(offset_list,1)) if self.signed else int(np.random.choice([1,(n_colors-1)],1))
            offset = int(np.random.choice([-1.,1.],1)) if self.signed else int(1)
        elif property_idx == 3:
            # n_shapes = len(SHAPES_)
            # offset_list = list(np.arange(-(n_shapes-1),n_shapes))
            # offset_list.remove(0)
            # offset = int(np.random.choice(offset_list,1)) if self.signed else int(np.random.choice([1,(n_shapes-1)],1))
            offset = int(np.random.choice([-2.,-1.,1.,2.],1)) if self.signed else int(1)
        elif property_idx == 4: # size
            offset = self.offset_l * (np.random.choice([-1.,1.],1) if self.signed else int(1))
        elif property_idx == 5: # rotation angle
            offset = self.offset_p * (np.random.choice([-1.,1.],1) if self.signed else int(1))
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
        if "c" in self.properties_list: # colour is among the targets
            replace = True if n_balls-1 > num_colours else False
            colour_indices = np.random.choice(range(num_colours), n_balls-1, replace=False).astype(int)
        else:
            if not self.same_color and self.injective:
                colour_indices = np.mod(np.arange(1, n_balls), len(COLOURS_)) # n_balls-1 fixed colours
            else:
                colour_indices = np.zeros(n_balls-1) # n_balls-1 same colour since we want to remove its effect
        z_all[idx_mask, 2] = colour_indices
        # z_all[idx_mask, 2] = hsv_colours[idx_mask, 0]
        
        # sample shapes for the rest of the balls
        # if shape is among the target properties then it should be picked at random, o.w. it
        # should be fixed
        if "s" in self.properties_list: # shape is among the targets
            shape_indices = np.random.choice(range(num_shapes), n_balls-1)
        else:
            if not self.same_shape and self.injective:
                shape_indices = np.mod(np.arange(1, n_balls), len(SHAPES_)) # n_balls-1 fixed shapes
            else:
                shape_indices = np.zeros((n_balls-1,)) # n_balls-1 same shapes since we want to remove its effect
            # shape_indices = np.zeros((n_balls-1,))
        z_all[idx_mask, 3] = shape_indices

        # sample sizes for the rest of the balls
        # if size is among the target properties then it should be picked at random, o.w. it
        # should be fixed
        if "l" in self.properties_list: # shape is among the targets
            sizes = np.random.uniform(self.min_size, self.max_size, n_balls-1)
        else:
            sizes = np.zeros((n_balls-1,)) + (self.min_size+self.max_size)/2.
        z_all[idx_mask, 4] = sizes
        
        # sample rotation angles for the rest of the balls
        # if rotation angle is among the target properties then it should be picked at random, o.w. it
        # should be fixed
        if "p" in self.properties_list: # rotation angle is among the targets
            rotation_angles = np.random.uniform(self.min_phi, self.max_phi, n_balls-1)
        else:
            rotation_angles = np.zeros((n_balls-1,))
        z_all[idx_mask, 5] = rotation_angles

        # TODO: Do we need random positions or should they be fixed?
        # regardless of x,y being the target of disentanglement or not, we should have random positions
        # sample ball coordinates for the rest of the balls
        # coordinates_1 = np.array([0.25, 0.25])
        max_size = np.max(z_all[:, 4])
        if log_:
            print(f"max_size:{max_size}")
            log.info(f"max_size:{max_size}")
        if self.random_coordinates:
            coordinates_1 = np.random.uniform(0.1 + max_size, 0.9 - max_size, size=(n_balls-1, 2))
            # prevent balls from getting initialized very close to each other
            mask = np.ones(coordinates_1.shape)
            duplicate_coordinates_1_threshold = max_size * 2
            sampled_coordinates_1_distance_matrix = np.linalg.norm(coordinates_1[:, None, :] - coordinates_1[None, :, :], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask[duplicate_mask] = 0.

            resample_mask = ~ (mask > 0.)
            if log_:
                print(f"{[1 for _ in range(10)]}\n{resample_mask}")
                log.info(f"{[1 for _ in range(10)]}\n{resample_mask}")
            # under no circumstances should z2 fall out of the frame, so this loop should continue until the constraints are satisfied.
            while_loop_threshold = 100
            while resample_mask.any() and while_loop_threshold > 0:
                if log_:
                    print(f"in while loop, while_loop_threshold:{while_loop_threshold}")
                    log.info(f"in while loop, while_loop_threshold:{while_loop_threshold}")
            # while resample_mask.any():
                while_loop_threshold -= 1
                coordinates_temp = np.random.uniform(0.1 + max_size, 0.9 - max_size, size=(n_balls-1, 2))
                coordinates_1[resample_mask] = coordinates_temp[resample_mask]

                # prevent balls from getting initialized very close to each other
                mask = np.ones(coordinates_1.shape)
                sampled_coordinates_1_distance_matrix = np.linalg.norm(coordinates_1[:, None, :] - coordinates_1[None, :, :], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask[duplicate_mask] = 0.

                resample_mask = ~ (mask > 0.)
                if log_:
                    print(f"{[2 for _ in range(10)]}\n{resample_mask}")
                    log.info(f"{[2 for _ in range(10)]}\n{resample_mask}")
            if while_loop_threshold == 0:
                if log_:
                    print(f"_sample_z1_z2_rest reached 100 attempts.")
                    log.info(f"_sample_z1_z2_rest reached 100 attempts.")
        else:
            coordinates_1 = np.array([(0.3 * i, 0.3 * i) for i in idx_mask])
            # coordinates_1 = np.array([0.5, 0.5])
        # # coordinates_1 = np.array([[0.3, 0.7]])
        
        z_all[idx_mask, :2] = coordinates_1 # [n_balls-1, 2]

        return z_all


    def _sample_z1_z2(self, ball_idx, property_idx, offset, z_all):
    
        ball_size = np.max(z_all[:, 4])
        upper_bound_x = 1 - ball_size
        upper_bound_y = 1 - ball_size
        lower_bound_x = ball_size
        lower_bound_y = ball_size
        consistent = True

        if property_idx == 0:
            # sample coordinates for the chosen ball
            coordinates_1 = np.random.uniform(0.1 + 2 * ball_size, 0.9 - 2 * ball_size, size=(2,))
            coordinates_2 = coordinates_1.copy()
            coordinates_2[property_idx] += offset


            # check the constraints
            mask = False
            
            # make sure this ball isn't initialized very close to other balls
            duplicate_coordinates_1_threshold = ball_size * 3
            z_all_temp1 = z_all.copy()
            z_all_temp1[ball_idx, :2] = coordinates_1.copy()
            sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

            # make sure this ball doesn't fall very close to other balls at t+1
            duplicate_coordinates_2_threshold = ball_size * 3
            z_all_temp2 = z_all.copy()
            z_all_temp2[ball_idx, :2] = coordinates_2.copy()
            sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
            # if any constraint is violated, we should resmaple, should also consider the previous masks
            mask = mask or duplicate_mask.any()

            # make sure this ball doesn't fall out of the frame after the offset has been applied
            if coordinates_2[property_idx] > upper_bound_x or coordinates_2[property_idx] < lower_bound_x:
                mask = True

            resample_mask = mask
            if log_:
                print(f"{[3 for _ in range(10)]}\n{resample_mask}")
                log.info(f"{[3 for _ in range(10)]}\n{resample_mask}")
            while_loop_threshold = 100
            while resample_mask and while_loop_threshold > 0:
                if log_:
                    print(f"in while loop, while_loop_threshold:{while_loop_threshold}")
                    log.info(f"in while loop, while_loop_threshold:{while_loop_threshold}")
                while_loop_threshold -= 1
                coordinates_1 = np.random.uniform(0.1 + 2 * ball_size, 0.9 - 2 * ball_size, size=(2,))
                coordinates_2 = coordinates_1.copy()
                coordinates_2[property_idx] += offset

                # check the constraints
                mask = False
                
                # make sure this ball isn't initialized very close to other balls
                duplicate_coordinates_1_threshold = ball_size * 3
                z_all_temp1 = z_all.copy()
                z_all_temp1[ball_idx, :2] = coordinates_1.copy()
                sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall very close to other balls at t+1
                duplicate_coordinates_2_threshold = ball_size * 3
                z_all_temp2 = z_all.copy()
                z_all_temp2[ball_idx, :2] = coordinates_2.copy()
                sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
                mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall out of the frame after the offset has been applied
                if coordinates_2[property_idx] > upper_bound_x or coordinates_2[property_idx] < lower_bound_x:
                    mask = True

                resample_mask = mask
                if log_:
                    print(f"{[4 for _ in range(10)]}\n{resample_mask}")
                    log.info(f"{[4 for _ in range(10)]}\n{resample_mask}")
            if while_loop_threshold == 0:
                consistent = False
                if log_:
                    print(f"_sample_z1_z2 property_idx=0 reached 100 attempts.")
                    log.info(f"_sample_z1_z2 property_idx=0 reached 100 attempts.")
                if coordinates_2[property_idx] > upper_bound_x:
                    coordinates_2[property_idx] = upper_bound_x
                if coordinates_2[property_idx] < lower_bound_x:
                    coordinates_2[property_idx] = lower_bound_x

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()
            z_all_1[ball_idx, :2] = coordinates_1.copy()
            z_all_2[ball_idx, :2] = coordinates_2.copy()
            if "c" in self.properties_list:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            if "s" in self.properties_list:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0
            if "l" in self.properties_list:
                z_all_1[ball_idx, 4] = np.random.uniform(self.min_size, self.max_size, 1)
            else:
                z_all_1[ball_idx, 4] = (self.min_size + self.max_size)/2.
            if "p" in self.properties_list:
                z_all_1[ball_idx, 5] = np.random.uniform(self.min_phi, self.max_phi, 1)
            else:
                z_all_1[ball_idx, 5] = 0
            
            # colour, shape, size, rotation angle
            z_all_2[ball_idx, 2:] = z_all_1[ball_idx, 2:].copy()


        if property_idx == 1:
            # sample coordinates for the chosen ball
            coordinates_1 = np.random.uniform(0.1 + 2 * ball_size, 0.9 - 2 * ball_size, size=(2,))
            coordinates_2 = coordinates_1.copy()
            coordinates_2[property_idx] += offset


            # check the constraints
            mask = False
            
            # make sure this ball isn't initialized very close to other balls
            duplicate_coordinates_1_threshold = ball_size * 3
            z_all_temp1 = z_all.copy()
            z_all_temp1[ball_idx, :2] = coordinates_1.copy()
            sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

            # make sure this ball doesn't fall very close to other balls at t+1
            duplicate_coordinates_2_threshold = ball_size * 3
            z_all_temp2 = z_all.copy()
            z_all_temp2[ball_idx, :2] = coordinates_2.copy()
            sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
            mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

            # make sure this ball doesn't fall out of the frame after the offset has been applied
            if coordinates_2[property_idx] > upper_bound_y or coordinates_2[property_idx] < lower_bound_y:
                mask = True

            resample_mask = mask
            if log_:
                print(f"{[5 for _ in range(10)]}\n{resample_mask}")
                log.info(f"{[5 for _ in range(10)]}\n{resample_mask}")
            while_loop_threshold = 100
            while resample_mask and while_loop_threshold > 0:
                if log_:
                    print(f"in while loop, while_loop_threshold:{while_loop_threshold}")
                    log.info(f"in while loop, while_loop_threshold:{while_loop_threshold}")
                while_loop_threshold -= 1
                coordinates_1 = np.random.uniform(0.1 + 2 * ball_size, 0.9 - 2 * ball_size, size=(2,))
                coordinates_2 = coordinates_1.copy()
                coordinates_2[property_idx] += offset

                # check the constraints
                mask = False
                
                # make sure this ball isn't initialized very close to other balls
                duplicate_coordinates_1_threshold = ball_size * 3
                z_all_temp1 = z_all.copy()
                z_all_temp1[ball_idx, :2] = coordinates_1.copy()
                sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall very close to other balls at t+1
                duplicate_coordinates_2_threshold = ball_size * 3
                z_all_temp2 = z_all.copy()
                z_all_temp2[ball_idx, :2] = coordinates_2.copy()
                sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
                mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

                # make sure this ball doesn't fall out of the frame after the offset has been applied
                if coordinates_2[property_idx] > upper_bound_y or coordinates_2[property_idx] < lower_bound_y:
                    mask = True

                resample_mask = mask
                if log_:
                    print(f"{[6 for _ in range(10)]}\n{resample_mask}")
                    log.info(f"{[6 for _ in range(10)]}\n{resample_mask}")
            if while_loop_threshold == 0:
                consistent = False
                if log_:
                    print(f"_sample_z1_z2 property_idx=1 reached 100 attempts.")
                    log.info(f"_sample_z1_z2 property_idx=1 reached 100 attempts.")
                if coordinates_2[property_idx] > upper_bound_y:
                    coordinates_2[property_idx] = upper_bound_y
                if coordinates_2[property_idx] < lower_bound_y:
                    coordinates_2[property_idx] = lower_bound_y

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()
            z_all_1[ball_idx, :2] = coordinates_1.copy()
            z_all_2[ball_idx, :2] = coordinates_2.copy()
            if "c" in self.properties_list:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            if "s" in self.properties_list:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0
            if "l" in self.properties_list:
                z_all_1[ball_idx, 4] = np.random.uniform(self.min_size, self.max_size, 1)
            else:
                z_all_1[ball_idx, 4] = (self.min_size + self.max_size)/2.
            if "p" in self.properties_list:
                z_all_1[ball_idx, 5] = np.random.uniform(self.min_phi, self.max_phi, 1)
            else:
                z_all_1[ball_idx, 5] = 0

            # colour, shape, size, rotation angle
            z_all_2[ball_idx, 2:] = z_all_1[ball_idx, 2:].copy()

        # if self.z_dim < 3 then the code won't reach here
        if property_idx == 2:
            colour_min_idx = 0
            assert len(COLOURS_) > 1
            colour_max_idx = len(COLOURS_)-1
            colour_idx_1 = np.random.choice(range(len(COLOURS_)), 1).astype(int)
            while colour_idx_1 + offset > colour_max_idx or colour_idx_1 + offset < colour_min_idx:
                colour_idx_1 = np.random.choice(range(len(COLOURS_)), 1).astype(int)
            colour_idx_2 = colour_idx_1 + offset

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()

            consistent, z_all_1, z_all_2 = self._sample_coordinates_rest(z_all_1, z_all_2, z_all, ball_idx)

            if "s" in self.properties_list:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0
            if "l" in self.properties_list:
                z_all_1[ball_idx, 4] = np.random.uniform(self.min_size, self.max_size, 1)
            else:
                z_all_1[ball_idx, 4] = (self.min_size + self.max_size)/2.
            if "p" in self.properties_list:
                z_all_1[ball_idx, 5] = np.random.uniform(self.min_phi, self.max_phi, 1)
            else:
                z_all_1[ball_idx, 5] = 0

            z_all_1[ball_idx, 2] = colour_idx_1
            z_all_2[ball_idx, 2] = colour_idx_2
            z_all_2[ball_idx, 3:] = z_all_1[ball_idx, 3:].copy() # shape, size, rotation angle

        # if self.z_dim < 4 then the code won't reach here
        if property_idx == 3:
            shape_min_idx = 0
            assert len(SHAPES_) > 1
            shape_max_idx = len(SHAPES_)-1
            shape_idx_1 = np.random.choice(range(len(SHAPES_)), 1).astype(int)
            while shape_idx_1 + offset > shape_max_idx or shape_idx_1 + offset < shape_min_idx:
                shape_idx_1 = np.random.choice(range(len(SHAPES_)), 1).astype(int)
            shape_idx_2 = shape_idx_1 + offset

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()

            consistent, z_all_1, z_all_2 = self._sample_coordinates_rest(z_all_1, z_all_2, z_all, ball_idx)
            
            if "c" in self.properties_list:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            if "l" in self.properties_list:
                z_all_1[ball_idx, 4] = np.random.uniform(self.min_size, self.max_size, 1)
            else:
                z_all_1[ball_idx, 4] = (self.min_size + self.max_size)/2.
            if "p" in self.properties_list:
                z_all_1[ball_idx, 5] = np.random.uniform(self.min_phi, self.max_phi, 1)
            else:
                z_all_1[ball_idx, 5] = 0

            z_all_1[ball_idx, 3] = shape_idx_1
            z_all_2[ball_idx, 2] = z_all_1[ball_idx, 2].copy()
            z_all_2[ball_idx, 3] = shape_idx_2
            z_all_2[ball_idx, 4] = z_all_1[ball_idx, 4].copy()
            z_all_2[ball_idx, 5] = z_all_1[ball_idx, 5].copy()

        if property_idx == 4: # size
            size_1 = np.random.uniform(self.min_size, self.max_size, 1)
            while size_1 + offset > self.max_size or size_1 + offset < self.min_size:
                size_1 = np.random.uniform(self.min_size, self.max_size, 1)
            size_2 = size_1 + offset

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()

            consistent, z_all_1, z_all_2 = self._sample_coordinates_rest(z_all_1, z_all_2, z_all, ball_idx)
            
            if "c" in self.properties_list:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            if "s" in self.properties_list:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0

            z_all_1[ball_idx, 4] = size_1
            z_all_2[ball_idx, 2] = z_all_1[ball_idx, 2].copy()
            z_all_2[ball_idx, 3] = z_all_1[ball_idx, 3].copy()
            z_all_2[ball_idx, 4] = size_2

            if "p" in self.properties_list:
                z_all_1[ball_idx, 5] = np.random.uniform(self.min_phi, self.max_phi, 1)
            else:
                z_all_1[ball_idx, 5] = 0


        if property_idx == 5: # rotation angle
            phi_1 = np.random.uniform(self.min_phi, self.max_phi, 1)
            while phi_1 + offset > self.max_phi or phi_1 + offset < self.min_phi:
                phi_1 = np.random.uniform(self.min_phi, self.max_phi, 1)
            phi_2 = phi_1 + offset

            # sample the rest of the properties
            z_all_1 = z_all.copy()
            z_all_2 = z_all.copy()

            consistent, z_all_1, z_all_2 = self._sample_coordinates_rest(z_all_1, z_all_2, z_all, ball_idx)
            
            if "c" in self.properties_list:
                z_all_1[ball_idx, 2] = np.random.choice(range(len(COLOURS_)), 1)
            else:
                z_all_1[ball_idx, 2] = 0
            if "s" in self.properties_list:
                z_all_1[ball_idx, 3] = np.random.choice(range(len(SHAPES_)), 1)
            else:
                z_all_1[ball_idx, 3] = 0
            if "l" in self.properties_list:
                z_all_1[ball_idx, 4] = np.random.uniform(self.min_size, self.max_size, 1)
            else:
                z_all_1[ball_idx, 4] = (self.min_size + self.max_size)/2.

            z_all_2[ball_idx, 2] = z_all_1[ball_idx, 2].copy() # colour
            z_all_2[ball_idx, 3] = z_all_1[ball_idx, 3].copy() # shape
            z_all_2[ball_idx, 4] = z_all_1[ball_idx, 4].copy() # size
            z_all_2[ball_idx, 5] = phi_2
            

        return consistent, z_all_1, z_all_2


    def _sample_coordinates_rest(self, z_all_1, z_all_2, z_all, ball_idx):
        # sample coordinates for the chosen ball
        ball_size = np.max(z_all[:, 4])
        # ball_size = z_all[ball_idx, 4]
        consistent = True
        if self.random_coordinates:
            coordinates_1 = np.random.uniform(0.1 + 2 * ball_size, 0.9 - 2 * ball_size, size=(2,))
            # check the constraints
            mask = False

            # make sure this ball isn't initialized very close to other balls
            duplicate_coordinates_1_threshold = ball_size * 3
            z_all_temp1 = z_all.copy()
            z_all_temp1[ball_idx, :2] = coordinates_1
            sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
            duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
            mask = duplicate_mask.any() # if any constraint is violated, we should resmaple
            resample_mask = mask
            if log_:
                print(f"{[7 for _ in range(10)]}\n{resample_mask}\n{sampled_coordinates_1_distance_matrix}\nduplicate_coordinates_1_threshold:{duplicate_coordinates_1_threshold}\nz_all:{z_all}\nz_all[ball_idx, 4]:{z_all[ball_idx, 4]}")
                log.info(f"{[7 for _ in range(10)]}\n{resample_mask}")
            while_loop_threshold = 100
            while resample_mask and while_loop_threshold > 0:
                if log_:
                    print(f"in while loop, while_loop_threshold:{while_loop_threshold}")
                    log.info(f"in while loop, while_loop_threshold:{while_loop_threshold}")
                while_loop_threshold -= 1
                coordinates_1 = np.random.uniform(0.1 + 2 * ball_size, 0.9 - 2 * ball_size, size=(2,))
                coordinates_2 = coordinates_1.copy()

                # check the constraints
                mask = False
                
                # make sure this ball isn't initialized very close to other balls
                duplicate_coordinates_1_threshold = ball_size * 3
                z_all_temp1 = z_all.copy()
                z_all_temp1[ball_idx, :2] = coordinates_1
                sampled_coordinates_1_distance_matrix = np.linalg.norm(z_all_temp1[:, None, :2] - z_all_temp1[None, :, :2], axis=-1)
                duplicate_mask = np.triu(sampled_coordinates_1_distance_matrix<duplicate_coordinates_1_threshold).sum(-2)>1
                mask = duplicate_mask.any() # if any constraint is violated, we should resmaple

                resample_mask = mask
                if log_:
                    print(f"{[8 for _ in range(10)]}\n{resample_mask}")
                    log.info(f"{[8 for _ in range(10)]}\n{resample_mask}")
            if while_loop_threshold == 0:
                consistent = False
                if log_:
                    print(f"_sample_coordinates_rest reached 100 attempts.")
                    log.info(f"_sample_coordinates_rest reached 100 attempts.")

        else:
            coordinates_1 = np.array([0.5, 0.5])
            # coordinates_1 = np.array([0.5, 0.25])
        coordinates_2 = coordinates_1.copy()

        z_all_1[ball_idx, :2] = coordinates_1.copy()
        z_all_2[ball_idx, :2] = coordinates_2.copy()

        return consistent, z_all_1, z_all_2

    def _sample(self):
        self._setup(SCREEN_DIM)
        if self.generate_data:
            # 1. we have to pick a ball m and a property i at random. The assumed order is [x,y,c,s].
            # we need ball_idx because we want to sample the latents for the rest of the balls and make
            # sure they are not initialized close to each other. Additionally, the chosen ball should
            # be placed somewhere, and we do not want it to be very close to any other balls, so if we
            # sampled the rest first, we are able to iterate and find a feasible latent for the chosen
            # ball.
            consistent = False
            while not consistent:
                ball_idx = np.random.choice(range(self.n_balls), 1)
                # property_idx = np.random.choice(range(self.z_dim), 1)
                property_idx = np.random.choice(self.target_property_indices, 1)
                z_all = np.zeros((self.n_balls, len(PROPERTIES_)))

                # 2. we can now sample the complete latent z for all other balls
                # the promise is that there are no two balls very close to one another. However, the rest of the
                # properties (colour, shape) can be identical for two or more balls.
                # note that the final z1,z2 have the same entries as z_all for all balls except one.
                z_all = self._sample_z1_z2_rest(z_all, ball_idx)
                if log_:
                    print(f"===========\n===========\n===========\nz_all:\n{z_all}\n===========\n===========\n===========")
                # 3. based on the chosen property for change, we will have to choose an offset with its sign
                sampled_offset = self._sample_offsets(property_idx)

                # 4. we can now sample that property and make sure it is consistent with the chosen offset
                # and the rest of the balls.
                # z_all_1,z_all_2 are [n_balls, len(PROPERTIES_)]
                consistent, z_all_1, z_all_2 = self._sample_z1_z2(ball_idx=ball_idx, property_idx=property_idx, offset=sampled_offset, z_all=z_all)
            if self.injective and self.injective_property == "c":
                if self.same_color:
                    hsv_colours_1 = [COLOURS_[0] for i in range(self.n_balls)]
                    hsv_colours_2 = [COLOURS_[0] for i in range(self.n_balls)]
                else:
                    hsv_colours_1 = [COLOURS_[np.mod(i,len(COLOURS_))] for i in range(self.n_balls)]
                    hsv_colours_2 = [COLOURS_[np.mod(i,len(COLOURS_))] for i in range(self.n_balls)]
            else:
                if self.same_color:
                    hsv_colours_1 = [COLOURS_[0] for i in range(self.n_balls)]
                    hsv_colours_2 = [COLOURS_[0] for i in range(self.n_balls)]
                else:
                    hsv_colours_1 = [COLOURS_[z_all_1[i,2].astype(int)] for i in range(z_all_1.shape[0])]
                    hsv_colours_2 = [COLOURS_[z_all_2[i,2].astype(int)] for i in range(z_all_2.shape[0])]
            if self.injective and self.injective_property == "s":
                if self.same_shape:
                    shapes_1 = [0 for i in range(self.n_balls)]
                    shapes_2 = [0 for i in range(self.n_balls)]
                else:
                    shapes_1 = [i for i in range(self.n_balls)]
                    shapes_2 = [i for i in range(self.n_balls)]
            else:
                if self.same_shape:
                    shapes_1 = [0 for i in range(self.n_balls)]
                    shapes_2 = [0 for i in range(self.n_balls)]
                else:
                    shapes_1 = [z_all_1[i,3].astype(int) for i in range(z_all_1.shape[0])]
                    shapes_2 = [z_all_2[i,3].astype(int) for i in range(z_all_2.shape[0])]
            if self.injective and self.injective_property == "l":
                if self.same_size:
                    sizes_1 = [(self.min_size+self.max_size)/2. for i in range(self.n_balls)]
                    sizes_2 = [(self.min_size+self.max_size)/2. for i in range(self.n_balls)]
                else:
                    sizes_1 = [self.min_size + i * (self.max_size - self.min_size) / self.n_balls for i in range(self.n_balls)]
                    sizes_2 = [self.min_size + i * (self.max_size - self.min_size) / self.n_balls for i in range(self.n_balls)]
            else:
                if self.same_size:
                    sizes_1 = [(self.min_size+self.max_size)/2. for i in range(self.n_balls)]
                    sizes_2 = [(self.min_size+self.max_size)/2. for i in range(self.n_balls)]
                else:
                    sizes_1 = [z_all_1[i,4] for i in range(z_all_1.shape[0])]
                    sizes_2 = [z_all_2[i,4] for i in range(z_all_2.shape[0])]
            # filling z_all_1,2 with colour hues at dimension 2
            z_all_1[:, 2] = np.array(hsv_colours_1)[:, 0]
            z_all_2[:, 2] = np.array(hsv_colours_2)[:, 0]
            # note the multiplication by 255., because draw_scene works with rgb colours in the range [0, 255.]
            rgb_colours_1 = [[255.*channel for channel in colorsys.hls_to_rgb(*c)] for c in hsv_colours_1]
            rgb_colours_2 = [[255.*channel for channel in colorsys.hls_to_rgb(*c)] for c in hsv_colours_2]

            # filling z_all_1,2 with shape indices at dimension 3
            z_all_1[:, 3] = np.array(shapes_1)
            z_all_2[:, 3] = np.array(shapes_2)
            
            # filling z_all_1,2 with size values at dimension 4
            z_all_1[:, 4] = np.array(sizes_1)
            z_all_2[:, 4] = np.array(sizes_2)

            # segmentation_mask1: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask1[0] is the background mask
            x1, segmentation_masks1 = self.draw_scene(z_all_1, rgb_colours_1)
            # segmentation_mask2: [n_balls+1, screen_width, screen_width, 1]; segmentation_mask2[0] is the background mask
            x2, segmentation_masks2 = self.draw_scene(z_all_2, rgb_colours_2)

            # dividing z_all_1,2[:, 3] (shape dimension) by the number of shapes so the latent
            # becomes nicer and close to the rest of the features.
            z_all_1[:, 3] /= len(SHAPES_)
            z_all_2[:, 3] /= len(SHAPES_)

        x1 = self.transform(x1)
        x2 = self.transform(x2)
        
        z1 = z_all_1[..., self.target_property_indices].copy()
        z2 = z_all_2[..., self.target_property_indices].copy()

        # b = np.zeros((self.z_dim,))
        if self.output_sparse_offsets:
            b = np.zeros((len(PROPERTIES_),))
            # b[property_idx] = sampled_offset
            b[property_idx] = z_all_2[ball_idx, property_idx] - z_all_1[ball_idx, property_idx]
            b = b[self.target_property_indices]
            A = torch.eye(b.shape[0]).float()
        else: # return an offset vector of the shape [n_balls, z_dim]
            b = np.zeros((self.n_balls, len(PROPERTIES_)))
            # b[ball_idx, property_idx] = sampled_offset
            b[ball_idx, property_idx] = z_all_2[ball_idx, property_idx] - z_all_1[ball_idx, property_idx]
            b = b[:, self.target_property_indices]
            A = torch.eye(b.shape[0]).float()

        # correcting the offset for shape to be small
        # b[b[:, -1] > 0.] = 0.6

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

        # data stats utilities
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

        for _ in tqdm(range(self.num_samples)):
            sample = self._sample()

        #     b = sample["matrices"][1] # [4]
        #     if (b == 0.).all():
        #         log.info(f"Sample {idx} has no change between t,t+1 (b=0), skipping...")
        #         no_change_counter += 1
        #         continue
        #     changed_property_idx = (~(b == 0.)).nonzero()[0].item()
        #     if PROPERTIES_[changed_property_idx] not in self.properties_list:
        #         continue
        #     else:
        #         x1, x2 = sample["images"][0], sample["images"][1]
        #         if (x1 == x2).all():
        #             log.info(f"Sample {idx} has no change between t,t+1 (x1=x2), skipping...")
        #             no_change_counter += 1
        #             log.info(f"(b > 0.).any() {(b > 0.).any()}")
        #             if (b > 0.).any():
        #                 missing_data_statistics[str(PROPERTIES_[changed_property_idx]) + "+"] += 1
        #             else:
        #                 missing_data_statistics[str(PROPERTIES_[changed_property_idx]) + "-"] += 1
        #             continue
        #         if (b > 0.).any():
        #             data_statistics[str(PROPERTIES_[changed_property_idx]) + "+"] += 1
        #         else:
        #             data_statistics[str(PROPERTIES_[changed_property_idx]) + "-"] += 1

        #         z1, z2 = sample["latents"][0], sample["latents"][1]
        #         if (np.abs(z2-z1)<1e-4).all():
        #             log.info(f"Sample {idx} has no change between t,t+1 (z1=z2), skipping...")
        #             no_change_counter += 1
        #             continue
        #         n = z1.shape[0]//self.n_balls
        #         n_objects = self.n_balls
                
        #         # convert segmentation masks to flaots
        #         seg_1, seg_2 = sample["segmentation_masks"]
        #         if seg_1.shape != seg_2.shape:
        #             log.info(f"The number of objects have changed across the transition, according to the segmentation ids, skipping sample {idx}")
        #             change_n_object_transition += 1
        #             continue
        #         if seg_1.shape[0] != self.n_balls + 1 or seg_2.shape[0] != self.n_balls + 1:
        #             log.info(f"{seg_1.shape[0] != self.n_balls + 1}")
        #             log.info(f"{seg_2.shape[0] != self.n_balls + 1}")
        #             log.info(f"The number of segmentation masks {seg_1.shape[0]},{seg_2.shape[0]} is not n_objects+1:{self.n_balls+1}, skipping sample {idx}")
        #             wrong_count_segmentation_masks += 1
        #             continue
        #         # sample["segmentation_masks"] = seg_1 * 1., seg_2 * 1.
            
            data.append(sample)
        
        log.info(f"The dataset being used has the following statistics for the properties being changed:\n{[(key,val) for key,val in data_statistics.items()]}\n and {no_change_counter} samples do NOT have any changes between t,t+1\nand {change_n_object_transition} samples have number of objects changing from t to t+1\nand {wrong_count_segmentation_masks} samples have wrong number of segmentation masks either at t or t+1\nand {missing_sample_counter} samples are missing. Below are the stats for missing samples:\n{[(key,val) for key,val in missing_data_statistics.items()]}")
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
                    radius=z[i,4],
                    screen_width=self.screen_dim,
                    y_shift=0.0,
                    offset=0.0,
                    shape=SHAPES_[int(z[i,3])],
                    rotation_angle=z[i,5]
                )
            )
            _ = draw_shape(
                z[i, 0],
                z[i, 1],
                self.bg_surf,
                color=colours[i],
                radius=z[i,4],
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
                shape=SHAPES_[int(z[i,3])],
                rotation_angle=z[i,5]
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
