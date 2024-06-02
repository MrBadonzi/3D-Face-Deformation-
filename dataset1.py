import os
import torch
from pytorch3d.renderer import Textures
import secrets
# Data structures and functions for rendering

from pytorch3d.datasets import (
    collate_batched_meshes,
)
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

from util import create_texture

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class FaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the meshes.

        """
        self.root_dir = root_dir
        self.expressions_name = ["2_smile", "3_mouth_stretch", '4_anger', "5_jaw_left", "6_jaw_right",
                                 "7_jaw_forward", "8_mouth_left", "9_mouth_right", "10_dimpler", "11_chin_raiser",
                                 "12_lip_puckerer", "13_lip_funneler", "14_sadness", "15_lip_roll", "16_grin",
                                 "17_cheek_blowing", "18_eye_closed", "19_brow_raiser", "20_brow_lower"]

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        obj_folder = os.path.join(self.root_dir,
                                  str(idx + 1) + "/models_reg")

        # target expressions
        obj_names = [os.path.join(obj_folder, name + ".obj") for name in self.expressions_name]

        # neutral expressions: our features
        neutralObj_name = os.path.join(obj_folder, "1_neutral.obj")

        # create the meshes
        # todo: aggiungere texture
        mesh = load_objs_as_meshes([neutralObj_name], load_textures=True, device=device)
        # We scale normalize and center the target mesh to fit in a sphere of radius 1
        # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
        # to its original center and scale.  Note that normalizing the target mesh,
        # speeds up the optimization but is not necessary!
        # verts = mesh.verts_packed()
        # center = verts.mean(0)
        # scale = max((verts - center).abs().max(0)[0])
        # mesh.offset_verts_(-center)
        # mesh.scale_verts_((1.0 / float(scale)))

        # fixme: mettere espessioni random invece che sempre quella sorridente
        label = load_objs_as_meshes([obj_names[0]], load_textures=True, device=device)

        # verts = label.verts_packed()
        # center = verts.mean(0)
        # scale = max((verts - center).abs().max(0)[0])
        # label.offset_verts_(-center)
        # label.scale_verts_((1.0 / float(scale)))


        # TODO: se una texture manca prendere altra espressione

        # for i in range(samples):
        #     choice = secrets.choice(obj_names)
        #     obj_names.remove(choice)
        #     try:
        #         meshes.append(load_objs_as_meshes([choice], load_textures=False, device=device))
        #
        #     except:
        #         choice = secrets.choice(obj_names)
        #         obj_names.remove(choice)
        #         meshes.append(load_objs_as_meshes([choice], load_textures=False, device=device))
        #         print("Maledetti Cinesi")

        # take the textures of each mesh
        textures = []
        # for tex in tex_names:
        #     try:
        #         textures.append(read_image(tex))
        #     except:
        #         print("Maledetti Cinesi")

        sample = {'Mesh Feature': mesh, 'Meshes Labels': label, 'Textures Labels': textures}
        # sample = {'Neutral Mesh': mesh, 'Expressions Meshes': meshes, }

        return sample


