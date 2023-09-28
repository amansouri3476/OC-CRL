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
import ast

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
# CLEVR_OBJECTS = ["sphere"] # ["cube"], ["sphere"]
rotation_axis = "z"
# number_of_objects = 2
# subdirectory = "xyl/sphere" # "c_coloured/sphere", "all_p", "xyclp", "xycp", "c_grayscale/sphere"
grayscale = False
camera_axis_aligned = True

# HSV colours
COLOURS_non_gs = [
    # [0.05, 0.8, 0.8],
    [0.15, 0.8, 0.8],
    # [0.25, 0.8, 0.8],
    [0.35, 0.8, 0.8],
    # [0.45, 0.8, 0.8],
    [0.55, 0.8, 0.8],
    # [0.65, 0.8, 0.8],
    # [0.75, 0.8, 0.8],
    # # [0.85, 0.8, 0.8],
    # [0.95, 0.8, 0.8],
]

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
parser.add_argument("--properties_list", type=list, nargs='+',
                    default=[
                            "x",
                            "y",
                            # "c",
                            # "s",
                            "l",
                            # "p",
                            ])
parser.add_argument("--fixed_properties_list", type=list,
                    default=[
                            # "x",
                            # "y",
                            # "z",
                            "c",
                            # "s",
                            # "l",
                            "p",
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
# parser.add_argument("--min_num_objects", type=int, default=number_of_objects,
#                     help="minimum number of objects")
# parser.add_argument("--max_num_objects", type=int, default=number_of_objects,
#                     help="maximum number of objects")
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
                    resolution=64)
FLAGS = parser.parse_args()
logging.info(f"*********************\nFLAGS:{FLAGS}\n*********************s")
FLAGS.properties_list = ast.literal_eval(FLAGS.properties_list)
FLAGS.fixed_properties_list = ast.literal_eval(FLAGS.fixed_properties_list)
logging.info(f"=======-------=======\nFLAGS.properties_list:{FLAGS.properties_list}\nFLAGS.fixed_properties_list:{FLAGS.fixed_properties_list}\n=======-------=======\n")
assert (np.array([p not in FLAGS.properties_list for p in FLAGS.fixed_properties_list])).all(), f"There are some properties that are supposed to be fixed and changed at the same time!\ntargets:{FLAGS.properties_list}\nfixed:{FLAGS.fixed_properties_list}"

# if FLAGS.min_num_objects != FLAGS.max_num_objects:
#   num_objects = rng.integers(FLAGS.min_num_objects,
#                             FLAGS.max_num_objects)
# else:
#   num_objects = FLAGS.max_num_objects
number_of_objects = FLAGS.number_of_objects
subdirectory = FLAGS.subdirectory
CLEVR_OBJECTS = ast.literal_eval(FLAGS.CLEVR_OBJECTS)

# latent space
target_property_indices = [i for i,p in enumerate(PROPERTIES) if p in FLAGS.properties_list]
offset_x = FLAGS.offset_x
offset_y = FLAGS.offset_y
fixed_property_indices = [i for i,p in enumerate(PROPERTIES) if p in FLAGS.fixed_properties_list]

"""
This worker file creates a pair of samples that should be later used for disentanglement experiments
Each sample of the dataset batch should have the following keys:
"latents", "images", "segmentation_masks", "matrices", "mechanism_permutation", "coordinates", "colors"
"""

dict_keys = ["latents", "images", "segmentation_masks", "matrices", "mechanism_permutation", "coordinates", "colors"]
sample = dict.fromkeys(dict_keys)
# sample["latents"] = []



def sample_offsets(property_idx, obj_idx, z_all):
  # notice that we're making sure that the offsets are signed because of the problems we
  # have seen earlier

  upper_bound_x = max_x - obj_size
  upper_bound_y = max_y - obj_size
  lower_bound_x = min_x + obj_size
  lower_bound_y = min_y + obj_size
  closeness_threshold = obj_size * 2

  if property_idx == 0:
    offset = offset_x * np.random.choice([-1.,1.],1)

    coordinates_2 = z_all[obj_idx, :2].copy()
    coordinates_2[property_idx] += offset

    # check the constraints
    mask = False

    # make sure this ball doesn't fall very close to other balls at t+1
    duplicate_coordinates_2_threshold = closeness_threshold
    z_all_temp2 = z_all.copy()
    z_all_temp2[obj_idx, :2] = coordinates_2
    sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
    duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
    # if any constraint is violated, we should resmaple, should also consider the previous masks
    mask = mask or duplicate_mask.any()

    # make sure this ball doesn't fall out of the frame after the offset has been applied
    if coordinates_2[property_idx] > upper_bound_x or coordinates_2[property_idx] < lower_bound_x:
        mask = True

    resample_mask = mask
    if resample_mask:
      offset = -offset
      coordinates_2 = z_all[obj_idx, :2].copy()
      coordinates_2[property_idx] += offset

      # check the constraints
      mask = False
      
      # make sure this ball doesn't fall very close to other balls at t+1
      duplicate_coordinates_2_threshold = closeness_threshold
      z_all_temp2 = z_all.copy()
      z_all_temp2[obj_idx, :2] = coordinates_2
      sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
      duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
      mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

      # make sure this ball doesn't fall out of the frame after the offset has been applied
      if coordinates_2[property_idx] > upper_bound_x or coordinates_2[property_idx] < lower_bound_x:
          mask = True

      resample_mask = mask
      if resample_mask:
        return False, 0.
      else:
        return True, offset

    else:
      return True, offset

  if property_idx == 1:
    offset = offset_y * np.random.choice([-1.,1.],1)

    coordinates_2 = z_all[obj_idx, :2].copy()
    coordinates_2[property_idx] += offset

    # check the constraints
    mask = False

    # make sure this ball doesn't fall very close to other balls at t+1
    duplicate_coordinates_2_threshold = closeness_threshold
    z_all_temp2 = z_all.copy()
    z_all_temp2[obj_idx, :2] = coordinates_2
    sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
    duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
    # if any constraint is violated, we should resmaple, should also consider the previous masks
    mask = mask or duplicate_mask.any()

    # make sure this ball doesn't fall out of the frame after the offset has been applied
    if coordinates_2[property_idx] > upper_bound_x or coordinates_2[property_idx] < lower_bound_x:
        mask = True

    resample_mask = mask
    if resample_mask:
      offset = -offset
      coordinates_2 = z_all[obj_idx, :2].copy()
      coordinates_2[property_idx] += offset

      # check the constraints
      mask = False
      
      # make sure this ball doesn't fall very close to other balls at t+1
      duplicate_coordinates_2_threshold = closeness_threshold
      z_all_temp2 = z_all.copy()
      z_all_temp2[obj_idx, :2] = coordinates_2
      sampled_coordinates_2_distance_matrix = np.linalg.norm(z_all_temp2[:, None, :2] - z_all_temp2[None, :, :2], axis=-1)
      duplicate_mask = np.triu(sampled_coordinates_2_distance_matrix<duplicate_coordinates_2_threshold).sum(-2)>1
      mask = mask or duplicate_mask.any() # if any constraint is violated, we should resmaple

      # make sure this ball doesn't fall out of the frame after the offset has been applied
      if coordinates_2[property_idx] > upper_bound_y or coordinates_2[property_idx] < lower_bound_y:
          mask = True

      resample_mask = mask
      if resample_mask:
        return False, 0.
      else:
        return True, offset
      
    else:
      return True, offset

  elif property_idx == 2:
    offset = int(np.random.choice([-1.,1.],1))
    # offset = int(np.random.choice([-2., -1., 1., 2.],1))

    colour_min_idx = 0
    colour_max_idx = len(COLOURS)-1
    COLOUR_hues = [COLOURS[i][colour_property_idx] for i in range(len(COLOURS))]
    colour_idx_1 = COLOUR_hues.index(z_all[obj_idx, property_idx+1]) # +1 is because there is also z property that we don't use
    if colour_idx_1 + offset > colour_max_idx or colour_idx_1 + offset < colour_min_idx:
      # try negating the sign of offset
      offset = -offset
      if colour_idx_1 + offset > colour_max_idx or colour_idx_1 + offset < colour_min_idx:
        return False, 0.
      else:
        return True, offset
    
    else:
      return True, offset
  
  elif property_idx == 3:
    # offset = int(np.random.choice([-1.,1.],1))
    offset = int(np.random.choice([-2., -1., 1., 2.],1))

    shape_min_idx = 0
    shape_max_idx = len(CLEVR_OBJECTS)-1
    shape_idx_1 = z_all[obj_idx, property_idx+1] # +1 is because there is also z property that we don't use
    if shape_idx_1 + offset > shape_max_idx or shape_idx_1 + offset < shape_min_idx:
      # try negating the sign of offset
      offset = -offset
      if shape_idx_1 + offset > shape_max_idx or shape_idx_1 + offset < shape_min_idx:
        return False, 0.
      else:
        return True, offset
    
    else:
      return True, offset
  elif property_idx == 4:
      # size
      size_min_val = 0.3
      size_max_val = 0.7
      offset = np.random.choice([-0.15,0.15],1)[0]
      current_size = z_all[obj_idx, property_idx+1] # +1 is because there is also z property that we don't use
      if current_size + offset < size_min_val or current_size + offset > size_max_val:
        # try negating the sign of offset
        offset = -offset
        if current_size + offset < size_min_val or current_size + offset > size_max_val:
          return False, 0.
        else:
          return True, offset
      else:
        return True, offset
  elif property_idx == 5:
    # angle: There should be upper and lower bounds on the quaternion that an object
    # could take, and therefore, the offset should be chosen properly, so that the 
    # property doesn't go beyond -45,45. Importantly, the initial quaternion should
    # be all over the range since we are dealing with a continuous property, so the
    # initial value should not always be 0.

    offset = 0.3 * np.random.choice([-1.,1.],1)
    logging.info(f"offset:    {offset}")
    # offset = -np.pi/4
    phi_2 = z_all[obj_idx, property_idx+1] + offset # +1 because of z

    # check the constraints
    mask = False

    # make sure this ball doesn't fall out of the frame after the offset has been applied
    if phi_2 > max_phi or phi_2 < min_phi:
        mask = True

    resample_mask = mask
    if resample_mask:
      offset = -offset
      phi_2 = z_all[obj_idx, property_idx+1] + offset # +1 because of z

      # check the constraints
      mask = False

      # make sure this ball doesn't fall out of the frame after the offset has been applied
      if phi_2 > max_phi or phi_2 < min_phi:
        mask = True

      resample_mask = mask
      if resample_mask:
        return False, 0.
      else:
        return True, offset

    else:
      return True, offset

  else:
      raise Exception(f"The property index provided {property_idx} is invalid. It should be in the [0,{len(PROPERTIES)}] range.")

logging.info("Randomly placing %d objects:", num_objects)

def populate_scene(num_objects):
  
  global scene, rng, output_dir, scratch_dir, simulator, renderer, kubasic, floor_material, obj_size
  list_of_objects = []
  colours = []
  sizes = []
  for i in range(num_objects):
    if "s" in FLAGS.fixed_properties_list:
      shape_name = CLEVR_OBJECTS[0]
    else:
      shape_name = rng.choice(CLEVR_OBJECTS)
    
    if "l" in FLAGS.fixed_properties_list:
      size_label, size = "small", obj_size
    else:
      size_label, size = "small", (np.random.rand()-0.5)*0.3 + obj_size
    sizes.append(size)
    # Choosing color
    # the following picks one color from a set of colors
    # color_label, random_color = kb.randomness.sample_color("clevr", rng)
    # --------------
    # the following fixes saturation and value and picks a random hue, just the way
    # we want it in hsv, therefore the color will be a 1-D latent
    # _, random_color = kb.randomness.sample_color("uniform_hue", rng)
    if "c" in FLAGS.fixed_properties_list:
      hsv_color_idx = 0
    else:
      hsv_color_idx = rng.choice(len(COLOURS))
    random_color = core.color.Color.from_hsv(*COLOURS[hsv_color_idx])
    colours.append(hsv_color_idx)
    # material_name = rng.choice(["metal", "rubber"])
    # specifying the coordinates. We will make sure all objects sit on the floor, only let x,y to vary
    if "x" in FLAGS.fixed_properties_list:
      obj_x = 0.2 # -1.2 + (i+1) * 1.
    else:
      obj_x = np.random.uniform(min_x+size, max_x-size)
    if "y" in FLAGS.fixed_properties_list:
      obj_y = -1.5 + (i+1) * 1.
    else:
      obj_y = np.random.uniform(min_y+size, max_y-size)
    material_name = "rubber"
    if shape_name == "cone":
      obj_coordinates = (obj_x, obj_y, size/2 - 0.2)
    else:
      obj_coordinates = (obj_x, obj_y, size/2)

    obj = kubasic.create(
        asset_id=shape_name, scale=size,
        # name=f"{size_label} {color_label} {material_name} {shape_name}"
        name=f"{size_label} {material_name} {shape_name}"
        , position=obj_coordinates
        # , quaternion=(1., 0., 0., 0.)
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
        # "color_label": color_label,
    }

    obj.segmentation_id = i + 1 # leave 0 for the background
    list_of_objects.append(obj)
    scene.add(obj)
    
    if "x" in FLAGS.fixed_properties_list or "y" in FLAGS.fixed_properties_list:
      pass
    else:
      kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
    # initialize velocity randomly but biased towards center
  #   obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
  #                   [obj.position[0], obj.position[1], 0])

    logging.info("    Added %s at %s", obj.asset_id, obj.position)


  obj_latents = []
  # restoring orientations and heights
  for i, object in enumerate(list_of_objects):
    if "p" in FLAGS.fixed_properties_list:
      phi = (min_phi + max_phi)/2.
    else:
      phi = np.random.uniform(min_phi, max_phi)
    logging.info(f"phi:    {phi}")
    # phi = np.pi/4
    # object.quaternion = (1.0, 0., 0., 0.)
    
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
    if object.asset_id == "cone":
      object.position = (object.position[0], object.position[1], size/2 - 0.2)
    else:
      object.position = (object.position[0], object.position[1], size/2)

    # getting the latents
    # print(f"*list(object.position):{list(object.position)}\ncolorsys.rgb_to_hsv(*object.material.color.rgb)[0]:{colorsys.rgb_to_hsv(*object.material.color.rgb)[0]}\nobject.asset_id:{object.asset_id}")
    # latents = [*list(object.position), COLOURS[hsv_color_idx][0], CLEVR_OBJECTS.index(object.asset_id), size]
    latents = [*list(object.position), COLOURS[colours[i]][colour_property_idx], CLEVR_OBJECTS.index(object.asset_id), sizes[i], phi]
    obj_latents.append(latents)
    # obj_latents = [*list(obj_coordinates), colorsys.rgb_to_hsv(*random_color.rgb)[0], CLEVR_OBJECTS.index(shape_name)]
    # sample["latents"].append(obj_latents)

  obj_latents = np.array(obj_latents)
  logging.info("    Finished populating the scene")

  return obj_latents, list_of_objects


# create the scene at time t
# --- create scene and attach a renderer to it
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

# scene = kb.Scene(resolution=(512, 512))
simulator = PyBullet(scene, scratch_dir)

# of the two renderers, we opt for KubricRenderer because the Blender renderer uses the Blender software
# to render the scene, while the KubricRenderer is a built-in renderer that uses the OpenDR library to 
# render the scene. The Blender renderer can produce more realistic images with more advanced lighting
# and materials, while KubricRenderer is simpler and faster. It also depends on your use case, if you 
# want to use the output images for some visual tasks like object detection, semantic segmentation, 
# and similar then Blender renderer might be more suitable for you. But if you want to use the output 
# images for some physics-based tasks like physics simulation, motion prediction, and similar then 
# KubricRenderer might be more suitable for you. It is also worth noting that using Blender renderer 
# might be more memory and computation intensive, so if you have a resource-limited environment, it's 
# better to go with KubricRenderer.

# renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
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

obj_latents, list_of_objects = populate_scene(num_objects)


# ---------------- alter the scene with some mechanism ---------------- #

# 1. we have to pick an object m and a property i at random. The assumed order is [x,y,c,s].

property_idx = np.random.choice(target_property_indices, 1)[0]
consistent = False
list_of_available_object_indices = list(range(num_objects))

while not consistent and len(list_of_available_object_indices) > 0:
  # try and see if the selected property can be altered for any of the objects

  obj_idx = np.random.choice(list_of_available_object_indices, 1)[0]
  list_of_available_object_indices.pop(list_of_available_object_indices.index(obj_idx))
  # based on the chosen property for change, we will have to choose an offset with its sign
  consistent, sampled_offset = sample_offsets(property_idx, obj_idx, obj_latents)
  if not consistent and len(list_of_available_object_indices) == 0:
    # if all objects have been tried, repopulate the scene, else, go back to while
    # remove all already added objects from the scene
    for object in list_of_objects:
      scene.remove(object)
    # resample a new scene to be sparsely changed
    obj_latents, list_of_objects = populate_scene(num_objects)
    list_of_available_object_indices = list(range(num_objects))


# coming out of the above loop means that a consistent pair has been found
# so first we render the original scene, then apply changes, and render 
# the altered scene
output_path_root = "/home/user/kubric/"
renderer.save_state(os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1.blend"))
frame_1 = renderer.render_still()
# --- save the output as pngs
kb.write_png(frame_1["rgba"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1.png"))
frame_1["segmentation"] = kb.adjust_segmentation_idxs(frame_1["segmentation"], scene.assets, [])
kb.write_palette_png(frame_1["segmentation"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1_segmentation.png"))
scale = kb.write_scaled_png(frame_1["depth"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_1_depth.png"))
logging.info("Depth scale: %s", scale)

# dividing obj_latents_1,2[:, 4] (shape dimension) by the number of shapes so the latent
# becomes nicer and close to the rest of the features.
obj_latents[:, 4] /= len(CLEVR_OBJECTS)
z1 = obj_latents.copy()
x1 = frame_1["rgba"] # [screen_width, screen_width, 4] (4 being r,g,b,alpha channel)

# [num_objects+1, screen_width, screen_width, 1]; segmentation_mask_1[0] is the background mask
segmentation_ids = np.unique(frame_1['segmentation'])
# list_of_segmentation_masks = [frame_1["segmentation"] == i for i in segmentation_ids]
segmentation_masks_1 = np.array([frame_1["segmentation"] == i for i in segmentation_ids])
colors_1 = np.array([list_of_objects[i].material.color.rgb for i in range(len(list_of_objects))])


# --------------- transform --------------- #
# x1 = self.transform(x1)
# x2 = self.transform(x2)
for asset in scene.assets:
  try: # not all scene assets have asset_ids (like directional lightc)
    if asset.asset_id in CLEVR_OBJECTS:
      print(f"colour: {asset.metadata['color']}")
      print(f"asset: {asset}")
  except:
    pass

print("----------------------------------------")
print("----------------------------------------")

# altering the scene
if property_idx == 0 or property_idx == 1:
  coordinates_2 = obj_latents[obj_idx, :2].copy()
  coordinates_2[property_idx] += sampled_offset
  list_of_objects[obj_idx].position = (coordinates_2[0], coordinates_2[1], list_of_objects[obj_idx].position[-1])
  obj_latents[obj_idx, :3] = np.array(list_of_objects[obj_idx].position)

elif property_idx == 2:
  COLOUR_hues = [COLOURS[i][colour_property_idx] for i in range(len(COLOURS))]
  colour_idx_1 = COLOUR_hues.index(obj_latents[obj_idx, property_idx+1]) # +1 is because there is also z property that we don't use
  colour_idx_2 = int(colour_idx_1 + sampled_offset)
  assert (colour_idx_2 >= 0 and colour_idx_2 <= len(COLOURS)-1), f"The offset {sampled_offset} and the object's color property {colour_idx_1} are not consistent" 
  list_of_objects[obj_idx].material.color = core.color.Color.from_hsv(*COLOURS[colour_idx_2])
  obj_latents[obj_idx, 3] = COLOURS[colour_idx_2][colour_property_idx]
  list_of_objects[obj_idx].metadata["color"] = list_of_objects[obj_idx].material.color.rgb

elif property_idx == 3:
  shape_idx_1 = CLEVR_OBJECTS.index(list_of_objects[obj_idx].asset_id) # obj_latents[obj_idx, property_idx+1][0] # +1 is because there is also z property that we don't use
  shape_idx_2 = int(shape_idx_1 + sampled_offset)
  assert (shape_idx_2 >= 0 and shape_idx_2 <= len(CLEVR_OBJECTS)-1), f"The offset {sampled_offset} and the object's shape property {shape_idx_1} are not consistent" 
  shape_name = CLEVR_OBJECTS[shape_idx_2]
  original_size = obj_latents[obj_idx, property_idx+1+1] # +1 is because there is also z property that we don't use, +1 because we want to retrieve size
  if shape_name == "cone":
    new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], original_size/2 - 0.2)
  else:
    new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], original_size/2)
  # create the new object with the size of the object
  size_label, size = "small", original_size
  material_name = "rubber"
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

elif property_idx == 4:
  shape_name = list_of_objects[obj_idx].asset_id
  new_size = obj_latents[obj_idx, 5] + sampled_offset
  
  if shape_name == "cone":
    new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], new_size/2 - 0.2)
  else:
    new_position = (list_of_objects[obj_idx].position[0], list_of_objects[obj_idx].position[1], new_size/2)
  
  # create the new object
  size_label, size = "small", new_size
  material_name = "rubber"
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

elif property_idx == 5:
  phi_1 = obj_latents[obj_idx, property_idx+1]
  phi_2 = phi_1 + sampled_offset
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

for asset in scene.assets:
  try: # not all scene assets have asset_ids (like directional lightc)
    if asset.asset_id in CLEVR_OBJECTS:
      print(f"colour: {asset.metadata['color']}")
      print(f"asset: {asset}")
  except:
    pass

renderer.save_state(os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_2.blend"))
frame_2 = renderer.render_still()
# --- save the output as pngs
kb.write_png(frame_2["rgba"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_2.png"))
frame_2["segmentation"] = kb.adjust_segmentation_idxs(frame_2["segmentation"], scene.assets, [])
kb.write_palette_png(frame_2["segmentation"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_2_segmentation.png"))
scale = kb.write_scaled_png(frame_2["depth"], os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}_2_depth.png"))
logging.info("Depth scale: %s", scale)

z2 = obj_latents.copy()
x2 = frame_2["rgba"] # [screen_width, screen_width, 4] (4 being r,g,b,alpha channel)

assert not ((z1 == z2).all()), f"The ground truth latents at t,t+1 are the same:\nz1:\n{z1}\nz2:\n{z2}\nproperty_idx:{property_idx}\nsampled_offset:{sampled_offset}. The sample will not be saved."
assert not ((x1 == x2).all()), f"The images x1,x2 at t,t+1 are the same. This sample will not be saved."

# [num_objects+1, screen_width, screen_width, 1]; segmentation_mask_1[0] is the background mask
segmentation_ids = np.unique(frame_2['segmentation'])
# list_of_segmentation_masks = [frame_2["segmentation"] == i for i in segmentation_ids]
segmentation_masks_2 = np.array([frame_2["segmentation"] == i for i in segmentation_ids])
colors_2 = np.array([list_of_objects[i].material.color.rgb for i in range(len(list_of_objects))])

# b = np.zeros((self.z_dim,))
if FLAGS.output_sparse_offsets:
    b = np.zeros((len(PROPERTIES),))
    # b[property_idx] = sampled_offset
    if property_idx < 2:
      b[property_idx] = z2[obj_idx, property_idx] - z1[obj_idx, property_idx]
    else:
      # note the +1 below, because z's now have an extra dimension that is not going to be used
      b[property_idx] = z2[obj_idx, property_idx+1] - z1[obj_idx, property_idx+1]
    assert not (b == 0.).all(), f"The offset vector b is all zeros, so there's no change from t to t+1. This sampled will not be saved."
    # b = b[target_property_indices]
    A = np.eye(b.shape[0])
else: # return an offset vector of the shape [n_balls * z_dim]
    b = np.zeros((num_objects, len(PROPERTIES)))
    # b[obj_idx, property_idx] = sampled_offset
    b[obj_idx, property_idx] = z2[obj_idx, property_idx] - z1[obj_idx, property_idx]
    # b = b[:, target_property_indices]
    A = np.eye(b.shape[0])

logging.info(f"obj_idx:{obj_idx}, property_idx: {property_idx}, sampled_offset:{sampled_offset},\nb: {b}\nz1:{z1[obj_idx]}\nz2:{z2[obj_idx]}\n------------------\nframe_1==frame_2: {(x1==x2).all()}\n------------------\nz1_full:\n{z1}\nz2_full:\n{z2}")
# gather all values and matrices corresponding the sample
sample = {"latents":(z1.flatten(), z2.flatten())
        , "images":(x1, x2)
        , "segmentation_masks":(segmentation_masks_1, segmentation_masks_2)
        , "matrices":(A, b.flatten())
        , "mechanism_permutation": obj_idx
        , "coordinates": (z1[:, :2].flatten(), z2[:, :2].flatten())
        , "colors": (colors_1, colors_2)}



# save the sample to a json file
import pickle

with open(os.path.join(output_path_root, f"output/{num_objects}/{subdirectory}/{FLAGS.sample_filename}.pickle"), "wb") as f:
  pickle.dump(sample, f)


# how to run the code? 
# singularity exec --nv kubruntu_latest.sif python examples/sparse_clevr_worker.py
# to write to directories other than home, you should mount the output dir in home, to your destination
# singularity exec --nv --bind /home/user/output:/home/user/kubric/output kubruntu_latest.sif python examples/sparse_clevr_worker.py
