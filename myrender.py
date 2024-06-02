import os
import torch
from PIL import Image
import torchshow as ts
import plotly.graph_objects as go
from pytorch3d.structures import Meshes
from torchvision.utils import save_image
from pytorch3d.vis import plot_scene
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pytorch3d.io import load_objs_as_meshes
import numpy as np
# Data structures and functions for rendering
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, SoftPhongShader, Textures)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# the number of different viewpoints from which we want to render the mesh.
num_views = 20

# # Get a batch of viewing angles.
# elev = torch.linspace(0, 180, num_views)
# azim = torch.linspace(0, 360, num_views)
elev = torch.linspace(0, 0, num_views)
azim = torch.linspace(0, 0, num_views)


# Place a point light in front of the object. As mentioned above, the front of
# the cow is facing the -z direction.
lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

# Initialize an OpenGL perspective camera that represents a batch of different
# viewing angles. All the cameras helper methods support mixed type inputs and
# broadcasting. So we can view the camera from the a distance of dist=2.7, and
# then specify elevation and azimuth angles for each viewpoint as tensors.

R, T = look_at_view_transform(dist=50.2, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1)
#
# We arbitrarily choose one particular view that will be used to visualize
# results

camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...],
                               T=T[None, 1, ...], fov=60)

# camera for each view
target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...],
                                        T=T[None, i, ...]) for i in range(num_views)]
# Define the settings for rasterization and shading. Here we set the output
# image to be of size 128X128. As we are rendering images for visualization
# purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
# rasterize_meshes.py for explanations of these parameters.  We also leave
# bin_size and max_faces_per_bin to their default values of None, which sets
# their values using heuristics and ensures that the faster coarse-to-fine
# rasterization method is used.  Refer to docs/notes/renderer.md for an
# explanation of the difference between naive and coarse-to-fine rasterization.
sigma = 1e-4
raster_settings = RasterizationSettings(
    image_size=1080,
    # blur_radius=0.0,
    max_faces_per_bin=100000,
    # blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
    faces_per_pixel=1,
    perspective_correct=False,

)

# Create a Phong renderer by composing a rasterizer and a shader. The textured
# Phong shader will interpolate the texture uv coordinates for each vertex,
# sample from a texture image and apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=camera,
        lights=lights
    )
)


def visualize_prediction(predicted_mesh, target_mesh, renderer=renderer,
                         title='',
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_meshes = predicted_mesh.extend(num_views)
        predicted_images = renderer(predicted_meshes, cameras=camera, lights=lights)

        target_meshes = target_mesh[0].extend(num_views)
        target_images = renderer(target_meshes, cameras=camera, lights=lights)

    target_image = [target_images[i, ..., :3] for i in range(num_views)][1]
    predicted_images = predicted_images[0, ..., inds].cpu().detach().numpy()

    # ts.show(target_image.cpu().detach(), figsize=(10, 10))

    # predicted_image = Image.fromarray(
    #     (predicted_images * 255).astype(np.uint8))
    # target_image = Image.fromarray(
    #     (target_image.cpu().detach().numpy() * 255).astype(np.uint8))
    #
    # return predicted_image, target_image

    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images)
    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()



