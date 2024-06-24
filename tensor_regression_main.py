from pathlib import Path
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)
from torch.nn import MSELoss
from tqdm import tqdm
import face_alignment
from skimage import io
import wandb
from key import your_api_key
from losses import landmark_loss
from myrender import lights, target_cameras, cameras, camera, silhouette_renderer, renderer_textured
from util import mesh_to_wanb, Landmark_positions
from torch import nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# STEP 0: initialize wandb:
# wandb.login(key=your_api_key)
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="3D Face Deformation"
#     # track hyperparameters and run metadata
# )

# STEP 1: Create an FaceLandmarker object.

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 2: IMPORT THE MESHES
neutral = Path('facescape_trainset_001_100/1/models_reg/1_neutral.obj')
neutral_mesh = load_objs_as_meshes([neutral], load_textures=True, device=device)
verts = neutral_mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
neutral_mesh.offset_verts_(-center)
neutral_mesh.scale_verts_((1.0 / float(scale)))

expression = Path('facescape_trainset_001_100/1/models_reg/3_mouth_stretch.obj')
expression_mesh = load_objs_as_meshes([expression], load_textures=True, device=device)
verts = expression_mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
expression_mesh.offset_verts_(-center)
expression_mesh.scale_verts_((1.0 / float(scale)))

# STEP 3: CREATE LOSSES
losses = {
    "silhouette": {"weight": 0.8, "values": []},
    "edge": {"weight": 1.0, "values": []},
    "normal": {"weight": 0.1, "values": []},
    "laplacian": {"weight": 1.0, "values": []},
    "mse": {"weight": 1.0, "values": []},
    "landmark": {"weight": 1.0, "values": []},
    "rgb": {"weight": 1.0, "values": []}

}


def update_mesh_shape_prior_losses(meshes, loss):
    loss["edge"] = mesh_edge_loss(meshes)
    loss["normal"] = mesh_normal_consistency(meshes)
    loss["laplacian"] = mesh_laplacian_smoothing(meshes, method="uniform")


#   STEP 4: optimizer
verts_shape = neutral_mesh.verts_packed().shape
deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
optimizer = torch.optim.AdamW([deform_verts], lr=0.002)

Niter = 150
num_views = 2
num_views_per_iteration = 1
plot_period = 25
loop = tqdm(range(Niter))
expression = expression_mesh.extend(num_views)
image_target = silhouette_renderer(expression_mesh, cameras=camera, lights=lights)
target_silhouette = image_target[0, ..., 3]
image_rgb = renderer_textured(expression_mesh, cameras=camera, lights=lights)
target_rgb = image_rgb[..., :3]

l2_mesh = MSELoss()

for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    new_src_mesh = neutral_mesh.offset_verts(deform_verts)

    # Losses to smooth /regularize the mesh shape
    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    # update_mesh_shape_prior_losses(new_src_mesh, loss)

    images_predicted = silhouette_renderer(new_src_mesh, cameras=camera, lights=lights)
    #
    # # Squared L2 distance between the predicted silhouette and the target
    # # silhouette from our dataset
    predicted_silhouette = images_predicted[..., 3]
    loss_silhouette = ((predicted_silhouette - target_silhouette) ** 2).mean()
    # loss["silhouette"] += loss_silhouette / num_views_per_iteration
    #
    # # Squared L2 distance between the predicted RGB image and the target
    # # image from our dataset
    images_predicted_rgb = renderer_textured(new_src_mesh, cameras=camera, lights=lights)
    predicted_rgb = images_predicted_rgb[..., :3]
    loss_rgb = ((predicted_rgb - target_rgb) ** 2).mean()
    # loss["rgb"] += loss_rgb / num_views_per_iteration
    # input = io.imread('cane1.png')
    loss['landmark'] += landmark_loss(predicted_rgb[0], target_rgb[0], detector) / num_views_per_iteration

    # Weighted sum of the losses
    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        sum_loss += l * losses[k]["weight"]
        losses[k]["values"].append(float(l.detach().cpu()))

    # Print the losses
    loop.set_description("total_loss = %.6f" % sum_loss)

    # Plot mesh
    # if i % plot_period == 0:
    #     # Plot mesh
    #     neutral, predicted, target = mesh_to_wanb(neutral_mesh, new_src_mesh, expression_mesh)
    #     wandb.log({"predicted": wandb.Plotly(predicted), "target": wandb.Plotly(target),
    #                "neutral": wandb.Plotly(neutral)})
    #     # siluette
    #     predicted_silhouette = Image.fromarray(
    #         (predicted_silhouette.cpu().detach().numpy()[0] * 255).astype(np.uint8))
    #     target_sil = Image.fromarray(
    #         (target_silhouette.cpu().detach().numpy() * 255).astype(np.uint8))
    #     # rgb
    #     predicted_rgb = Image.fromarray(
    #         (predicted_rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8))
    #     target_r = Image.fromarray(
    #         (target_rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8))
    #
    #     wandb.log({f"predicted sihoulette": wandb.Image(predicted_silhouette),
    #                "target sihoulette": wandb.Image(target_sil),
    #                f"predicted rgb": wandb.Image(predicted_rgb),
    #                f"target_rgb": wandb.Image(target_r)})

    # Optimization step
    sum_loss.backward()
    optimizer.step()
    # wandb.log({"batch_loss": sum_loss,
    #            'landmarks': loss["landmark"],
    #            "silohuette loss": loss["silhouette"],
    #            "edge": loss["edge"],
    #            "normal": loss["normal"],
    #            "laplacian": loss["laplacian"],
    #            "rgb": loss["rgb"]
    #            })
