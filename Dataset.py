import os
import torch

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

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "./facescape_trainset_001_100"


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
        tex_names = [os.path.join(obj_folder, name + ".jpg") for name in self.expressions_name]

        # neutral expressions: our features
        neutralObj_name = os.path.join(obj_folder, "1_neutral.obj")

        # create the meshes
        #todo: test senza texture
        mesh = load_objs_as_meshes([neutralObj_name], load_textures=False, device=device)
        meshes = []
        for name in obj_names:
            try:
                meshes.append(load_objs_as_meshes([name], load_textures=False, device=device))
            except:
                print("Maledetti Cinesi")

        # take the textures of each mesh
        textures = []
        for tex in tex_names:
            try:
                textures.append(read_image(tex))
            except:
                print("Maledetti Cinesi")

        sample = {'Mesh Feature': mesh, 'Meshes Labels': meshes, 'Textures Labels': textures}
        # sample = {'Neutral Mesh': mesh, 'Expressions Meshes': meshes, }

        return sample


if __name__ == '__main__':
    training_data = FaceDataset(DATA_DIR)
    train_dataloader = DataLoader(training_data, batch_size=12, shuffle=True, collate_fn=collate_batched_meshes)
    for batch in tqdm(train_dataloader):
        neutral_mesh, expression_meshes, tex_labels = batch['Mesh Feature'], batch['Meshes Labels'], batch[
            'Textures Labels']

    # it = iter(train_dataloader)
    # r2n2_batch = next(it)
# batch_renderings = r2n2_batch['Neutral Mesh']  # (N, V, H, W, 3), and in this case V is 1.
#
#
# plt.figure(figsize=(7, 7))
# texture_image = batch_renderings[0][0].textures.maps_padded()
# plt.imshow(texture_image.squeeze().cpu().numpy())
# plt.axis("off")
# plt.show()
