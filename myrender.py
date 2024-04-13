import os
import torch
from PIL import Image
import torchshow as ts
import plotly.graph_objects
from torchvision.utils import save_image
from pytorch3d.vis import plot_scene
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import numpy as np
# Data structures and functions for rendering
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, SoftPhongShader)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# the number of different viewpoints from which we want to render the mesh.
num_views = 20

# Get a batch of viewing angles.
elev = torch.linspace(0, 0, num_views)
azim = torch.linspace(180, 180, num_views)

# Place a point light in front of the object. As mentioned above, the front of
# the cow is facing the -z direction.
lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

# Initialize an OpenGL perspective camera that represents a batch of different
# viewing angles. All the cameras helper methods support mixed type inputs and
# broadcasting. So we can view the camera from the a distance of dist=2.7, and
# then specify elevation and azimuth angles for each viewpoint as tensors.
R, T = look_at_view_transform(dist=60,  elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60, znear=0.1)

# We arbitrarily choose one particular view that will be used to visualize
# results
# camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...],
#                                T=T[None, 1, ...], fov=60)


# Define the settings for rasterization and shading. Here we set the output
# image to be of size 128X128. As we are rendering images for visualization
# purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
# rasterize_meshes.py for explanations of these parameters.  We also leave
# bin_size and max_faces_per_bin to their default values of None, which sets
# their values using heuristics and ensures that the faster coarse-to-fine
# rasterization method is used.  Refer to docs/notes/renderer.md for an
# explanation of the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
    max_faces_per_bin=50000,
    bin_size=-1

)

# Create a Phong renderer by composing a rasterizer and a shader. The textured
# Phong shader will interpolate the texture uv coordinates for each vertex,
# sample from a texture image and apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)


def visualize_prediction(predicted_mesh, target_mesh, renderer=renderer,
                             title='',
                         silhouette=False):
    # fig = plot_scene({
    #     "subplot1": {
    #         "cow_mesh": target_mesh[0]
    #     }
    # })
    # fig.show()

    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_meshes = predicted_mesh.extend(num_views)
        predicted_images = renderer(predicted_meshes)

        target_meshes = target_mesh[0].extend(num_views)
        target_images = renderer(target_meshes,cameras=cameras, lights=lights)
    target_image = [target_images[i, ..., :3] for i in range(num_views)][1]

    ts.show(target_image.cpu().detach(), figsize=(10, 10))
    # im = Image.fromarray((target_image.cpu().detach().numpy() * 255).astype(np.uint8))
    # im.show()

    # plt.subplot(1, 2, 1)
    # plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
    # plt.subplot(1, 2, 2)
    # plt.imshow(target_image.cpu().detach().numpy())
    # plt.title(title)
    # plt.axis("off")
    # plt.show()



# Rasterization settings for differentiable rendering, where the blur_radius
# initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable
# Renderer for Image-based 3D Reasoning', ICCV 2019

# sigma = 1e-4
# raster_settings_soft = RasterizationSettings(
#     image_size=128,
#     blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
#     faces_per_pixel=50,
#     perspective_correct=False,
# )

# Differentiable soft renderer using per vertex RGB colors for texture
# renderer_textured = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=camera,
#         raster_settings=raster_settings_soft
#     ),
#     shader=SoftPhongShader(device=device,
#                            cameras=camera,
#                            lights=lights)
# )

