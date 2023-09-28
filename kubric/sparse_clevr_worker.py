import logging
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.renderer.blender import Blender as KubricRenderer

import numpy as np
rng = np.random.default_rng()
import traitlets as tl
from kubric.core import traits as ktl
from kubric import core
import colorsys

logging.basicConfig(level="INFO")

# --- Some configuration values
# the region in which to place objects [(min), (max)]
obj_size = 0.5
min_x, min_y, min_z, max_x, max_y, max_z = -3, -3, 0, 3, 3, 1
SPAWN_REGION = [(min_x, min_y, min_z), (max_x, max_y, max_z)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]
CLEVR_OBJECTS = ("cube", "cylinder", "sphere", "cone")

# HSV colours
COLOURS = [
    [0.05, 0.6, 0.6],
    # [0.15, 0.6, 0.6],
    [0.25, 0.6, 0.6],
    # [0.35, 0.6, 0.6],
    [0.45, 0.6, 0.6],
    # [0.55, 0.6, 0.6],
    [0.65, 0.6, 0.6],
    # [0.75, 0.6, 0.6],
    [0.85, 0.6, 0.6],
    # [0.95, 0.6, 0.6],
]

PROPERTIES = [
    "x",
    "y",
    "c",
    "s"
]


parser = kb.ArgumentParser()

# latent space
parser.add_argument("--properties_list", type=list,
                    default=["x","y","c","s"])
parser.add_argument("--offset_x", type=float,
                    default=0.3)
parser.add_argument("--offset_y", type=float,
                    default=0.3)
parser.add_argument("--output_sparse_offsets", type=bool,
                    default=True)

# sample info
parser.add_argument("--sample_filename", type=str,
                    default=["0000"]
                    , help="a 4 digit number")

# Configuration for the objects of the scene
parser.add_argument("--objects_set", choices=["clevr", "kubasic"],
                    default="clevr")
parser.add_argument("--min_num_objects", type=int, default=2,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=2,
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

# latent space
target_property_indices = [i for i,p in enumerate(PROPERTIES) if p in FLAGS.properties_list]
offset_x = FLAGS.offset_x
offset_y = FLAGS.offset_y


"""
This worker file creates a pair of samples that should be later used for disentanglement experiments
Each sample of the dataset batch should have the following keys:
"latents", "images", "segmentation_masks", "matrices", "mechanism_permutation", "coordinates", "colors"
"""

dict_keys = ["latents", "images", "segmentation_masks", "matrices", "mechanism_permutation", "coordinates", "colors"]
sample = dict.fromkeys(dict_keys)
# sample["latents"] = []

if FLAGS.min_num_objects != FLAGS.max_num_objects:
  num_objects = rng.integers(FLAGS.min_num_objects,
                            FLAGS.max_num_objects)
else:
  num_objects = FLAGS.max_num_objects

logging.info("Randomly placing %d objects:", num_objects)
list_of_objects = []

def generate_scene(num_objects):
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


  for i in range(num_objects):
    shape_name = rng.choice(CLEVR_OBJECTS)
    size_label, size = "small", obj_size

    # Choosing color
    # the following picks one color from a set of colors
    # color_label, random_color = kb.randomness.sample_color("clevr", rng)
    # --------------
    # the following fixes saturation and value and picks a random hue, just the way
    # we want it in hsv, therefore the color will be a 1-D latent
    # _, random_color = kb.randomness.sample_color("uniform_hue", rng)
    random_color = core.color.Color.from_hsv(rng.choice(COLOURS))

    # material_name = rng.choice(["metal", "rubber"])
    # specifying the coordinates. We will make sure all objects sit on the floor, only let x,y to vary
    obj_xy = np.random.uniform(min_x+obj_size, max_x-obj_size, size=(2))
    material_name = "rubber"
    obj_x, obj_y = obj_xy[0], obj_xy[1]
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

    list_of_objects.append(obj)
    scene.add(obj)
    
    kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
    # initialize velocity randomly but biased towards center
  #   obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
  #                   [obj.position[0], obj.position[1], 0])

    logging.info("    Added %s at %s", obj.asset_id, obj.position)


  obj_latents = []
  # restoring orientations and heights
  for object in list_of_objects:
    object.quaternion = (1.0, 0., 0., 0.)
    if object.asset_id == "cone":
      object.position = (object.position[0], object.position[1], size/2 - 0.2)
    else:
      object.position = (object.position[0], object.position[1], size/2)

    # getting the latents
    # print(f"*list(object.position):{list(object.position)}\ncolorsys.rgb_to_hsv(*object.material.color.rgb)[0]:{colorsys.rgb_to_hsv(*object.material.color.rgb)[0]}\nobject.asset_id:{object.asset_id}")
    latents = [*list(object.position), colorsys.rgb_to_hsv(*object.material.color.rgb)[0], CLEVR_OBJECTS.index(object.asset_id)]
    obj_latents.append(latents)
    # obj_latents = [*list(obj_coordinates), colorsys.rgb_to_hsv(*random_color.rgb)[0], CLEVR_OBJECTS.index(shape_name)]
    # sample["latents"].append(obj_latents)

  obj_latents = np.array(obj_latents)

  # ------------------------------
  scene += kb.DirectionalLight(name="sun", position=(1, 1, 1),
                              look_at=(0, 0, 0), intensity=2.5)
  scene += kb.PerspectiveCamera(name="camera", position=(4, 4, 2),
                                look_at=(1.5, 1., 0))

  return scene, renderer, obj_latents, list_of_objects


# create the scene at time t
scene, renderer, obj_latents, list_of_objects = generate_scene(num_objects)

# ---------------- alter the scene with some mechanism ---------------- #

# 1. we have to pick an object m and a property i at random. The assumed order is [x,y,c,s].

property_idx = np.random.choice(target_property_indices, 1)
consistent = False
list_of_available_object_indices = list(range(num_objects))

while ~consistent and len(list_of_available_object_indices) > 0:
  # try and see if the selected property can be altered for any of the objects

  obj_idx = np.random.choice(list_of_available_object_indices, 1)
  list_of_available_object_indices.pop(obj_idx)
  # based on the chosen property for change, we will have to choose an offset with its sign
  consistent, sampled_offset = sample_offsets(property_idx, obj_idx, obj_latents)

  if ~consistent and len(list_of_available_object_indices) == 0:
    # resample a new scene to be sparsely changed
    scene, renderer, obj_latents, list_of_objects = generate_scene(num_objects)
    list_of_available_object_indices = list(range(num_objects))


# coming out of the above loop means that a consistent pair has been found
# so first we render the original scene, then apply changes, and render 
# the altered scene

renderer.save_state(f"output/{FLAGS.sample_filename}_1.blend")
frame_1 = renderer.render_still()
# --- save the output as pngs
kb.write_png(frame_1["rgba"], f"output/{FLAGS.sample_filename}_1.png")
kb.write_palette_png(frame_1["segmentation"], f"output/{FLAGS.sample_filename}_1_segmentation.png")
scale = kb.write_scaled_png(frame_1["depth"], f"output/{FLAGS.sample_filename}_1_depth.png")
logging.info("Depth scale: %s", scale)

# dividing obj_latents_1,2[:, -1] (shape dimension) by the number of shapes so the latent
# becomes nicer and close to the rest of the features.
obj_latents[:, -1] /= len(CLEVR_OBJECTS)
z1 = obj_latents
x1 = frame_1["rgba"] # [screen_width, screen_width, 4] (4 being r,g,b,alpha channel)

# [num_objects+1, screen_width, screen_width, 1]; segmentation_mask_1[0] is the background mask
segmentation_masks_1 = np.array([frame_1["segmentation"] == i for i in range(1, num_objects+2)])
colors_1 = np.array([list_of_objects[i].material.color.rgb for i in range(len(list_of_objects))])

# b = np.zeros((self.z_dim,))
if FLAGS.output_sparse_offsets:
    b = np.zeros((4,))
    b[property_idx] = sampled_offset
    b = b[target_property_indices]
    A = np.eye(b.shape[0])
else: # return an offset vector of the shape [n_balls * z_dim]
    b = np.zeros((num_objects, 4))
    b[obj_idx, property_idx] = sampled_offset
    b = b[:, target_property_indices]
    A = np.eye(b.shape[0])


# --------------- transform --------------- #
x1 = self.transform(x1)
x2 = self.transform(x2)


# altering the scene
if property_idx == 0 or property_idx == 1:
  coordinates_2 = obj_latents[obj_idx, :2].copy()
  coordinates_2[property_idx] += sampled_offset
  list_of_objects[obj_idx].position = (coordinates_2[0], coordinates_2[1], list_of_objects[obj_idx].position[-1])

elif property_idx == 2:
  colour_idx_1 = COLOURS.index(obj_latents[obj_idx, property_idx+1]) # +1 is because there is also z property that we don't use
  colour_idx_2 = colour_idx_1 + sampled_offset
  list_of_objects[obj_idx].material.color = core.color.Color.from_hsv(*COLOURS[colour_idx_2])

elif property_idx == 3:
  shape_idx_1 = CLEVR_OBJECTS.index(obj_latents[obj_idx, property_idx+1]) # +1 is because there is also z property that we don't use
  shape_idx_2 = shape_idx_1 + sampled_offset
  shape_name = CLEVR_OBJECTS[shape_idx_2]
  # create the new object
  new_obj = kubasic.create(
        asset_id=d, scale=obj_size,
        # name=f"{size_label} {color_label} {material_name} {shape_name}"
        name=f"{size_label} {material_name} {shape_name}"
        , position=list_of_objects[obj_idx].position
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
      "color": random_color.rgb,
      # "color_label": color_label,
  }

  scene.remove(list_of_objects[obj_idx])
  scene.add(new_obj)
  list_of_objects[obj_idx] = new_obj


renderer.save_state(f"output/{FLAGS.sample_filename}_2.blend")
frame_2 = renderer.render_still()
# --- save the output as pngs
kb.write_png(frame_2["rgba"], f"output/{FLAGS.sample_filename}_2.png")
kb.write_palette_png(frame_2["segmentation"], f"output/{FLAGS.sample_filename}_2_segmentation.png")
scale = kb.write_scaled_png(frame_2["depth"], f"output/{FLAGS.sample_filename}_2_depth.png")
logging.info("Depth scale: %s", scale)

obj_latents[:, -1] /= len(CLEVR_OBJECTS)
z2 = obj_latents
x2 = frame_2["rgba"] # [screen_width, screen_width, 4] (4 being r,g,b,alpha channel)

# [num_objects+1, screen_width, screen_width, 1]; segmentation_mask_1[0] is the background mask
segmentation_masks_2 = np.array([frame_2["segmentation"] == i for i in range(1, num_objects+2)])
colors_2 = np.array([list_of_objects[i].material.color.rgb for i in range(len(list_of_objects))])

# gather all values and matrices corresponding the sample
sample = {"latents":(z1.flatten(), z2.flatten())
        , "images":(x1, x2)
        , "segmentation_masks":(segmentation_masks_1, segmentation_masks_2)
        , "matrices":(A, b.flatten())
        , "mechanism_permutation": obj_idx
        , "coordinates": (z1[:, :2].flatten(), z2[:, :2].flatten())
        , "colors": (colors_1, colors_2)}



# save the sample to a json file
import json
with open(f"{FLAGS.sample_filename}.json", "wb") as f:
  json.dump(sample, f)


def sample_offsets(property_idx, obj_idx, z_all):
  # notice that we're making sure that the offsets are signed because of the problems we
  # have seen earlier

  upper_bound_x = max_x - obj_size
  upper_bound_y = max_y - obj_size
  lower_bound_x = min_x + obj_size
  lower_bound_y = min_y + obj_size


  if property_idx == 0:
    offset = offset_x * np.random.choice([-1.,1.],1)

    coordinates_2 = z_all[obj_idx, :2].copy()
    coordinates_2[property_idx] += offset

    # check the constraints
    mask = False

    # make sure this ball doesn't fall very close to other balls at t+1
    duplicate_coordinates_2_threshold = obj_size * 2
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
      duplicate_coordinates_2_threshold = obj_size * 2
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
      return False, None
    else:
      return True, offset

  if property_idx == 1:
    offset = offset_y * np.random.choice([-1.,1.],1)

    coordinates_2 = z_all[obj_idx, :2].copy()
    coordinates_2[property_idx] += offset

    # check the constraints
    mask = False

    # make sure this ball doesn't fall very close to other balls at t+1
    duplicate_coordinates_2_threshold = obj_size * 2
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
      duplicate_coordinates_2_threshold = obj_size * 2
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
      return False, None
    else:
      return True, offset

  elif property_idx == 2:
    offset = int(np.random.choice([-1.,1.],1))

    colour_min_idx = 0
    colour_max_idx = len(COLOURS)-1
    colour_idx_1 = COLOURS.index(z_all[obj_idx, property_idx+1]) # +1 is because there is also z property that we don't use
    if colour_idx_1 + offset > colour_max_idx or colour_idx_1 + offset < colour_min_idx:
      # try negating the sign of offset
      offset = -offset
      if colour_idx_1 + offset > colour_max_idx or colour_idx_1 + offset < colour_min_idx:
        return False, None
      else:
        return True, offset
  
  elif property_idx == 3:
      offset = int(np.random.choice([-1.,1.],1))

      shape_min_idx = 0
      shape_max_idx = len(CLEVR_OBJECTS)-1
      shape_idx_1 = CLEVR_OBJECTS.index(z_all[obj_idx, property_idx+1]) # +1 is because there is also z property that we don't use
      if shape_idx_1 + offset > shape_max_idx or shape_idx_1 + offset < shape_min_idx:
        # try negating the sign of offset
        offset = -offset
        if shape_idx_1 + offset > shape_max_idx or shape_idx_1 + offset < shape_min_idx:
          return False, None
        else:
          return True, offset
  else:
      raise Exception(f"The property index provided {property_idx} is invalid. It should be in the [0,3] range.")

# how to run the code? 
# singularity exec --nv kubruntu_latest.sif python examples/sparse_clevr_worker.py