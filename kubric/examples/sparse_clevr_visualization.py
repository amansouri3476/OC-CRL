import logging
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.renderer.blender import Blender as KubricRenderer

import os
import numpy as np
rng = np.random.default_rng()
import traitlets as tl
from kubric.core import traits as ktl
from kubric import core
import colorsys

logging.basicConfig(level="INFO")

# --- Some configuration values
# the region in which to place objects [(min), (max)]
obj_size = 0.5 # res=64 --> 0.25, res=128 --> 0.5
min_x, min_y, min_z, max_x, max_y, max_z = -1.5, -1, 0, 1.5, 1, 1
min_phi, max_phi = 0., np.pi/4 # -np.pi/4 and np.pi/4 look alike! We should constrain the range tighter
SPAWN_REGION = [(min_x, min_y, min_z), (max_x, max_y, max_z)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]
# CLEVR_OBJECTS = ("cube", "cylinder", "sphere", "cone")
# CLEVR_OBJECTS = ("cube", "cylinder", "cone") # sphere won't reflect rotations
CLEVR_OBJECTS = ["cube"] # ["cube"], ["sphere"]
rotation_axis = "z"
number_of_objects = 2
subdirectory = "all_p/config_3" # "all_p", "c_coloured/sphere", "all_p", "xyclp", "xycp", "c_grayscale/sphere", "xyl/sphere"
z_dim = 4
grayscale = False
camera_axis_aligned = True
material_name = "rubber" # "rubber", "metal"
number_of_offsets = 1
x_values = [-0.7, 0.7, 0.5] # [-0.5, -0.1, 0.5]
y_values = [-0.5, -0.5, 0.5]
c_values = [0.0, 0.5, 0.55] # [0.15, 0.35, 0.55], [0.0, 0.5, 0.55]
s_values = [0, 0, 0] # [0, 1, 2]
l_values = [0.5, 0.5, 0.5]
phi_values = [0.3, 0.7, 0.0]
sequence_length = 5
offsets = np.zeros((number_of_objects, z_dim, sequence_length))

offsets[0, :, 0] = [0.0, 0.2, 0.0, 0.0]
offsets[0, :, 1] = [0.0, 0.0, 1, 0.0]
offsets[0, :, 2] = [0.0, 0.2, 0.0, 0.0]
offsets[0, :, 3] = [0.0, 0.0, 1, 0.0]
offsets[0, :, 4] = [0.0, 0.0, 0.0, 0.0]

offsets[1, :, 0] = [0.0, 0.0, 0.0, 0.0]
offsets[1, :, 1] = [0.0, 0.2, 0.0, 0.0]
offsets[1, :, 2] = [0.0, 0.0, -1, 0.0]
offsets[1, :, 3] = [0.0, 0.2, 0.0, 0.0]
offsets[1, :, 4] = [0.0, 0.0, 2, 0.0]

# if number_of_objects > 2:
#   offsets[2, :, 0] = [0.0, 0.0, 0.0, 0.0]
#   offsets[2, :, 1] = [0.0, 0.0, 0.0, 0.0]
#   offsets[2, :, 2] = [0.0, 0.0, 0.0, 0.0]
#   offsets[2, :, 3] = [0.0, 0.0, 0.0, 0.0]
#   offsets[2, :, 4] = [0.0, 0.0, 0.0, 0.0]

# offsets[0, :, 0] = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[0, :, 1] = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[0, :, 2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
# offsets[0, :, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
# offsets[0, :, 4] = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]

# offsets[1, :, 0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[1, :, 1] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[1, :, 2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[1, :, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[1, :, 4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3]

# offsets[2, :, 0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[2, :, 1] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[2, :, 2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[2, :, 3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# offsets[2, :, 4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

assert number_of_objects > number_of_offsets, f"The number of objects that change {number_of_offsets} should be less than the number of objects {number_of_objects} in the scene."


# ONLY FOR BEAUTIFUL VISUALIZATIONS
# HSV colours
COLOURS_non_gs = [
    [0.0, 1.0, 0.5], # red
    # [0.25, 0.8, 0.8],
    [0.5, 1., 0.5], # cyan
    # [0.45, 0.8, 0.8],
    [0.83, 1, 0.5],
    # [0.65, 0.8, 0.8],
    [0.75, 0.8, 0.8],
    # # [0.85, 0.8, 0.8],
    # [0.95, 0.8, 0.8],
]

# # HSV colours
# COLOURS_non_gs = [
#     # [0.05, 0.8, 0.8],
#     [0.15, 0.8, 0.8],
#     # [0.25, 0.8, 0.8],
#     [0.35, 0.8, 0.8],
#     # [0.45, 0.8, 0.8],
#     [0.55, 0.8, 0.8],
#     # [0.65, 0.8, 0.8],
#     [0.75, 0.8, 0.8],
#     # # [0.85, 0.8, 0.8],
#     # [0.95, 0.8, 0.8],
# ]



# Grayscale colours
COLOURS_gs = [
    # [0.05, 0.8, 0.8],
    [0., 0., 0.1],
    # [0.25, 0.8, 0.8],
    [0., 0., 0.4],
    # [0.45, 0.8, 0.8],
    [0., 0., 0.7],
    # [0.65, 0.8, 0.8],
    # [0.75, 0.8, 0.8],
    # # [0.85, 0.8, 0.8],
    # [0.95, 0.8, 0.8],
]

PROPERTIES = [
    "x",
    "y",
    "c",
    "s",
    "l",
    "p",
]


COLOURS = COLOURS_gs if grayscale else COLOURS_non_gs
colour_property_idx = 2 if grayscale else 0

parser = kb.ArgumentParser()

# subdirectory = "xyl/sphere" # "c_coloured/sphere", "all_p", "xyclp", "xycp", "c_grayscale/sphere"
parser.add_argument("--subdirectory", type=str,
                    default="xyl/sphere")
parser.add_argument("--number_of_objects", type=int,
                    default=2)
parser.add_argument("--CLEVR_OBJECTS", type=list,
                    default=["sphere"])
# latent space
parser.add_argument("--properties_list", type=list,
                    default=[
                            "x",
                            "y",
                            "c",
                            "s",
                            "l",
                            "p",
                            ])
parser.add_argument("--fixed_properties_list", type=list,
                    default=[
                            # "x",
                            # "y",
                            # "z",
                            # "c",
                            # "s",
                            # "l",
                            # "p",
                            ])
parser.add_argument("--offset_x", type=float,
                    default=0.3)
parser.add_argument("--offset_y", type=float,
                    default=0.3)
parser.add_argument("--output_sparse_offsets", type=bool,
                    default=True)

# sample info
parser.add_argument("--sample_filename", type=str,
                    default="random"
                    , help="a 4 digit number")

# Configuration for the objects of the scene
parser.add_argument("--objects_set", choices=["clevr", "kubasic"],
                    default="clevr")
parser.add_argument("--min_num_objects", type=int, default=number_of_objects,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=number_of_objects,
                    help="maximum number of objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--background", choices=["clevr", "colored"],
                    default="clevr")

# Configuration for the camera
parser.add_argument("--camera", choices=["clevr", "random"], default="clevr")

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    # default="gs://kubric-public/assets/KuBasic/KuBasic.json")
                    default="/home/user/kubric/kubric/KuBasic.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                    resolution=256)
FLAGS = parser.parse_args()

assert (np.array([p not in FLAGS.properties_list for p in FLAGS.fixed_properties_list])).all(), f"There are some properties that are supposed to be fixed and changed at the same time!\ntargets:{FLAGS.properties_list}\nfixed:{FLAGS.fixed_properties_list}"

if FLAGS.min_num_objects != FLAGS.max_num_objects:
  num_objects = rng.integers(FLAGS.min_num_objects,
                            FLAGS.max_num_objects)
else:
  num_objects = FLAGS.max_num_objects
# number_of_objects = FLAGS.number_of_objects
# subdirectory = FLAGS.subdirectory
# CLEVR_OBJECTS = FLAGS.CLEVR_OBJECTS

# latent space
target_property_indices = [i for i,p in enumerate(PROPERTIES) if p in FLAGS.properties_list]
offset_x = FLAGS.offset_x
offset_y = FLAGS.offset_y
fixed_property_indices = [i for i,p in enumerate(PROPERTIES) if p in FLAGS.fixed_properties_list]

dict_keys = ["latents", "images", "segmentation_masks", "matrices", "mechanism_permutation", "coordinates", "colors"]
sample = dict.fromkeys(dict_keys)

"""
This worker file creates a sequence of samples to be used for disentanglement visualization
Each sample of the dataset batch should have the following keys:
"latents", "images", "segmentation_masks", "matrices", "mechanism_permutation", "coordinates", "colors"
"""

# create the initial scene and attach a renderer to it
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

# scene = kb.Scene(resolution=(512, 512))
simulator = PyBullet(scene, scratch_dir)
renderer = KubricRenderer(scene)

kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
floor_material = kb.PrincipledBSDFMaterial(roughness=1., specular=0.)

scene += kubasic.create("dome", name="floor", material=floor_material,
                        scale=1.0,
                        friction=FLAGS.floor_friction,
                        restitution=FLAGS.floor_restitution,
                        static=True, background=True)


if FLAGS.background == "clevr":
  floor_material.color = kb.Color.from_name("gray")
  scene.metadata["background"] = "clevr"
elif FLAGS.background == "colored":
  floor_material.color = kb.random_hue_color()
  scene.metadata["background"] = floor_material.color.hexstr

# Lights
# logging.info("Adding four (studio) lights to the scene similar to CLEVR...")
# scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)  

if camera_axis_aligned:
  scene += kb.DirectionalLight(name="sun", position=(1, 0, 1),
                            look_at=(0, 0, 0), intensity=2.5)
else:
  scene += kb.DirectionalLight(name="sun", position=(1, 1, 1),
                            look_at=(0, 0, 0), intensity=2.5)
if camera_axis_aligned:
  scene += kb.PerspectiveCamera(name="camera", position=(2.5, 0., 3),
                              look_at=(0, 0, 0))
else:
  scene += kb.PerspectiveCamera(name="camera", position=(2.5, 2.5, 3),
                              look_at=(0, 0, 0))


list_of_objects = []
colours = []
sizes = []
for i in range(num_objects):
  shape_name = CLEVR_OBJECTS[np.mod(i,len(CLEVR_OBJECTS))]
#   size_label, size = "small", (np.random.rand()-0.5)*0.3 + obj_size
  size_label, size = "small", l_values[i]
  sizes.append(size)
  # Choosing color
  hsv_color_idx = np.mod(i,len(COLOURS))
  random_color = core.color.Color.from_hsv(*COLOURS[hsv_color_idx])
  colours.append(hsv_color_idx)
  
  # specifying the coordinates. We will make sure all objects sit on the floor, only let x,y to vary
  obj_x = x_values[i]
  obj_y = y_values[i]

  if shape_name == "cone":
    obj_coordinates = (obj_x, obj_y, size/2 - 0.2)
  else:
    obj_coordinates = (obj_x, obj_y, size/2)

  obj = kubasic.create(
      asset_id=shape_name, scale=size,
      name=f"{size_label} {material_name} {shape_name}"
      , position=obj_coordinates
      )
  assert isinstance(obj, kb.FileBasedObject)

  if material_name == "metal":
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
                                            roughness=0.2, ior=2.5)
    obj.friction = 0.4
    obj.restitution = 0.3
    obj.mass *= 2.7 * size**3
  else:  # material_name == "rubber"
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
                                            ior=1.25, roughness=0.7,
                                            specular=0.33)
    obj.friction = 0.8
    obj.restitution = 0.7
    obj.mass *= 1.1 * size**3

  obj.metadata = {
      "shape": shape_name.lower(),
      "size": size,
      "size_label": size_label,
      "material": material_name.lower(),
      "color": random_color.rgb,
  }

  list_of_objects.append(obj)
  scene.add(obj)
  logging.info("    Added %s at %s", obj.asset_id, obj.position)

obj_latents = []
for i, object in enumerate(list_of_objects):
  phi = phi_values[i]
  logging.info(f"phi:    {phi}")
  
  if rotation_axis == "z":
    # rotation along z-axis
    object.quaternion = (np.cos(phi/2), 0., 0., np.sin(phi/2))
  elif rotation_axis == "y":
    # rotation along y-axis
    object.quaternion = (np.cos(phi/2), 0., np.sin(phi/2), 0.)
  elif rotation_axis == "x":
    # rotation along x-axis
    object.quaternion = (np.cos(phi/2), np.sin(phi/2), 0., 0.)
  else:
    raise(Exception(f"rotation axis {rotation_axis} should be either x or y or z. Got {rotation_axis}."))

  # getting the latents
  latents = [*list(object.position), COLOURS[colours[i]][colour_property_idx], CLEVR_OBJECTS.index(object.asset_id), sizes[i], phi]
  obj_latents.append(latents)
  
obj_latents = np.array(obj_latents)
logging.info("    Finished populating the scene")


output_path_root = "/home/user/kubric/" 
renderer.save_state(os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1.blend"))
frame_1 = renderer.render_still()
# --- save the output as pngs
kb.write_png(frame_1["rgba"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1.png"))
kb.write_palette_png(frame_1["segmentation"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1_segmentation.png"))
frame_1["segmentation"] = kb.adjust_segmentation_idxs(frame_1["segmentation"], scene.assets, [])
scale = kb.write_scaled_png(frame_1["depth"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1_depth.png"))
logging.info("Depth scale: %s", scale)

# dividing obj_latents_1,2[:, 4] (shape dimension) by the number of shapes so the latent
# becomes nicer and close to the rest of the features.
obj_latents[:, 4] /= len(CLEVR_OBJECTS)
z1 = obj_latents.copy()
x1 = frame_1["rgba"] # [screen_width, screen_width, 4] (4 being r,g,b,alpha channel)

# [num_objects+1, screen_width, screen_width, 1]; segmentation_mask_1[0] is the background mask
segmentation_ids = np.unique(frame_1["segmentation"])
# list_of_segmentation_masks = [frame_1["segmentation"] == i for i in segmentation_ids]
segmentation_masks_1 = np.array([frame_1["segmentation"] == i for i in segmentation_ids])
colors_1 = np.array([list_of_objects[i].material.color.rgb for i in range(len(list_of_objects))])

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# z_list = [z1]
# images_list = [x1]
# segmentation_masks_list = [segmentation_masks_1]
# colors_list = [colors_1]
z_list = []
images_list = []
segmentation_masks_list = []
colors_list = []

def change_obj_properties(obj_idx, perturbations):
  # perturbations: [z_dim]
  global list_of_objects
  global obj_latents
  global COLOURS
  global scene
  global z_list
  global images_list
  global segmentation_masks_list
  global colors_list

  properties_list = FLAGS.properties_list
  for property_idx, property_offset in enumerate(perturbations):

    if properties_list[property_idx] == "x" or properties_list[property_idx] == "y":
        coordinates_2 = obj_latents[obj_idx, :2].copy()
        coordinates_2[property_idx] += property_offset
        list_of_objects[obj_idx].position = (coordinates_2[0], coordinates_2[1], list_of_objects[obj_idx].position[-1])
        obj_latents[obj_idx, :3] = np.array(list_of_objects[obj_idx].position)
    if properties_list[property_idx] == "c":
        COLOUR_hues = [COLOURS[i][colour_property_idx] for i in range(len(COLOURS))]
        colour_idx_1 = COLOUR_hues.index(obj_latents[obj_idx, property_idx+1]) # +1 is because there is also z property that we don't use
        colour_idx_2 = int(colour_idx_1 + property_offset)
        assert (colour_idx_2 >= 0 and colour_idx_2 <= len(COLOURS)-1), f"The offset {sampled_offset} and the object's color property {colour_idx_1} are not consistent" 
        list_of_objects[obj_idx].material.color = core.color.Color.from_hsv(*COLOURS[colour_idx_2])
        obj_latents[obj_idx, 3] = COLOURS[colour_idx_2][colour_property_idx]
        list_of_objects[obj_idx].metadata["color"] = list_of_objects[obj_idx].material.color.rgb
    if properties_list[property_idx] == "s":
        shape_idx_1 = CLEVR_OBJECTS.index(list_of_objects[obj_idx].asset_id) # obj_latents[obj_idx, property_idx+1][0] # +1 is because there is also z property that we don't use
        shape_idx_2 = int(shape_idx_1 + property_offset)
        assert (shape_idx_2 >= 0 and shape_idx_2 <= len(CLEVR_OBJECTS)-1), f"The offset {sampled_offset} and the object's shape property {shape_idx_1} are not consistent" 
        shape_name = CLEVR_OBJECTS[shape_idx_2]
        original_size = obj_latents[obj_idx, property_idx+1+1] # +1 is because there is also z property that we don't use, +1 because we want to retrieve size
        if shape_name == "cone":
            new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], original_size/2 - 0.2)
        else:
            new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], original_size/2)
        # create the new object with the size of the object
        size_label, size = "small", original_size
        # material_name = "rubber"
        new_obj = kubasic.create(
            asset_id=shape_name, scale=size,
            # name=f"{size_label} {color_label} {material_name} {shape_name}"
            name=f"{size_label} {material_name} {shape_name}"
            , position=new_position
            # , quaternion=(1., 0., 0., 0.)
            )
        assert isinstance(new_obj, kb.FileBasedObject)

        if material_name == "metal":
            new_obj.material = kb.PrincipledBSDFMaterial(color=list_of_objects[obj_idx].material.color, metallic=1.0,
                                                        roughness=0.2, ior=2.5)
            new_obj.friction = 0.4
            new_obj.restitution = 0.3
            new_obj.mass *= 2.7 * size**3
        else:  # material_name == "rubber"
            new_obj.material = kb.PrincipledBSDFMaterial(color=list_of_objects[obj_idx].material.color, metallic=0.,
                                                        ior=1.25, roughness=0.7,
                                                        specular=0.33)
            new_obj.friction = 0.8
            new_obj.restitution = 0.7
            new_obj.mass *= 1.1 * size**3

        new_obj.metadata = {
            "shape": shape_name.lower(),
            "size": size,
            "size_label": size_label,
            "material": material_name.lower(),
            "color": list_of_objects[obj_idx].material.color.rgb,
            # "color_label": color_label,
        }
        new_obj.segmentation_id = int(obj_idx + 1) # leave 0 for the background
        # the quaternion should be preserved as well
        new_obj.quaternion = list_of_objects[obj_idx].quaternion
        scene.add(new_obj)
        scene.remove(list_of_objects[obj_idx])
        list_of_objects[obj_idx] = new_obj
        obj_latents[obj_idx, 4] = shape_idx_2 / len(CLEVR_OBJECTS)

    if properties_list[property_idx] == "l":
        shape_name = list_of_objects[obj_idx].asset_id
        new_size = obj_latents[obj_idx, 5] + property_offset

        if shape_name == "cone":
            new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], new_size/2 - 0.2)
        else:
            new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], new_size/2)

        # create the new object
        size_label, size = "small", new_size
        # material_name = "rubber"
        new_obj = kubasic.create(
            asset_id=shape_name, scale=new_size,
            # name=f"{size_label} {color_label} {material_name} {shape_name}"
            name=f"{size_label} {material_name} {shape_name}"
            , position=new_position
            # , quaternion=(1., 0., 0., 0.)
            )
        assert isinstance(new_obj, kb.FileBasedObject)
        if material_name == "metal":
            new_obj.material = kb.PrincipledBSDFMaterial(color=list_of_objects[obj_idx].material.color, metallic=1.0,
                                                        roughness=0.2, ior=2.5)
            new_obj.friction = 0.4
            new_obj.restitution = 0.3
            new_obj.mass *= 2.7 * size**3
        else:  # material_name == "rubber"
            new_obj.material = kb.PrincipledBSDFMaterial(color=list_of_objects[obj_idx].material.color, metallic=0.,
                                                        ior=1.25, roughness=0.7,
                                                        specular=0.33)
            new_obj.friction = 0.8
            new_obj.restitution = 0.7
            new_obj.mass *= 1.1 * size**3

        new_obj.metadata = {
            "shape": shape_name.lower(),
            "size": size,
            "size_label": size_label,
            "material": material_name.lower(),
            "color": list_of_objects[obj_idx].material.color.rgb,
            # "color_label": color_label,
        }
        new_obj.segmentation_id = int(obj_idx + 1) # leave 0 for the background
        # the quaternion should be preserved as well
        new_obj.quaternion = list_of_objects[obj_idx].quaternion
        scene.add(new_obj)
        scene.remove(list_of_objects[obj_idx])
        list_of_objects[obj_idx] = new_obj
        obj_latents[obj_idx, 5] = size # refer to PROPERTIES to understand dims


    if properties_list[property_idx] == "p":
        phi_1 = obj_latents[obj_idx, property_idx+1]
        phi_2 = phi_1 + property_offset
        if rotation_axis == "z":
            # rotation along z-axis
            list_of_objects[obj_idx].quaternion = (np.cos(phi_2/2), 0., 0., np.sin(phi_2/2))
        elif rotation_axis == "y":
            # rotation along y-axis
            list_of_objects[obj_idx].quaternion = (np.cos(phi_2/2), 0., np.sin(phi_2/2), 0.)
        elif rotation_axis == "x":
                # rotation along x-axis
                list_of_objects[obj_idx].quaternion = (np.cos(phi_2/2), np.sin(phi_2/2), 0., 0.)
        else:
            raise(Exception(f"rotation axis {rotation_axis} should be either x or y or z. Got {rotation_axis}."))
        obj_latents[obj_idx, property_idx+1] = phi_2


for frame_idx in range(sequence_length):
  for obj_idx in range(num_objects):
    change_obj_properties(obj_idx, offsets[obj_idx, :, frame_idx])
  
  z_new = obj_latents.copy()
  z_list.append(z_new)

  renderer.save_state(os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_{2+frame_idx}.blend"))
  new_frame = renderer.render_still()
  # --- save the output as pngs
  kb.write_png(new_frame["rgba"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_{2+frame_idx}.png"))
  new_frame["segmentation"] = kb.adjust_segmentation_idxs(new_frame["segmentation"], scene.assets, [])
  kb.write_palette_png(new_frame["segmentation"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_{2+frame_idx}_segmentation.png"))
  scale = kb.write_scaled_png(new_frame["depth"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_{2+frame_idx}_depth.png"))

  new_x = new_frame["rgba"]
  segmentation_ids = np.unique(new_frame['segmentation'])
  # list_of_segmentation_masks = [frame_2["segmentation"] == i for i in segmentation_ids]
  new_segmentation_masks = np.array([new_frame["segmentation"] == i for i in segmentation_ids])
  new_colors = np.array([list_of_objects[i].material.color.rgb for i in range(len(list_of_objects))])

  images_list.append(new_x)
  segmentation_masks_list.append(new_segmentation_masks)
  colors_list.append(new_colors)



sample = {"latents":tuple(z.flatten() for z in z_list)
        , "images":tuple(x for x in images_list)
        , "segmentation_masks":tuple(segmentation_masks for segmentation_masks in segmentation_masks_list)
        , "colors": tuple(colors for colors in colors_list)}



# save the sample to a json file
import pickle

with open(os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}.pickle"), "wb") as f:
  pickle.dump(sample, f)

# how to run the code? 
# singularity exec --nv kubruntu_latest.sif python examples/sparse_clevr_visualization.py
# to write to directories other than home, you should mount the output dir in home, to your destination
# singularity exec --nv --bind /home/user/output:/home/user/kubric/output kubruntu_latest.sif python examples/sparse_clevr_worker.py
