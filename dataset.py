import os
import secrets
import torch
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from torch.utils.data import Dataset

# Data structures and functions for rendering
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
        textures_names = [os.path.join(obj_folder, name + ".jpg") for name in self.expressions_name]

        # neutral expressions: our features
        neutral_obj_name = os.path.join(obj_folder, "1_neutral.obj")

        # create the meshes
        mesh = load_objs_as_meshes([neutral_obj_name], load_textures=True, device=device)
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))
        # We scale normalize and center the target mesh to fit in a sphere of radius 1
        # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
        # to its original center and scale.  Note that normalizing the target mesh,
        # speeds up the optimization but is not necessary!
        # verts = mesh.verts_packed()
        # center = verts.mean(0)
        # scale = max((verts - center).abs().max(0)[0])
        # mesh.offset_verts_(-center)
        # mesh.scale_verts_((1.0 / float(scale)))

        label = load_objs_as_meshes([obj_names[0]], load_textures=True, device=device)
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

        found = False

        while not found:
            index = secrets.choice(indices)
            indices.remove(index)
            # choice = obj_names[index]
            # choice_texture = textures_names[index]
            choice = obj_names[11]
            choice_texture = textures_names[11]

            if os.path.isfile(choice) and os.path.isfile(choice_texture):
                try:
                    label = load_objs_as_meshes([choice], load_textures=True, device=device)
                except:
                    print("MANNAGGIA")
                else:
                    found = True
                    num_exp = index

        verts = label.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        label.offset_verts_(-center)
        label.scale_verts_((1.0 / float(scale)))

        sample = {'Mesh Feature': mesh, 'Meshes Labels': label, "exp": num_exp}
        # sample = {'Neutral Mesh': mesh, 'Expressions Meshes': meshes, }

        return sample
