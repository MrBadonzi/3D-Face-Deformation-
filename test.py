import torch
from pytorch3d.datasets import collate_batched_meshes
from pytorch3d.renderer import Textures
from torch.utils.data import DataLoader

from dataset1 import FaceDataset
from network import MLP_verts
from myrender import visualize_prediction

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

model = MLP_verts().to(device)
model.load_state_dict(torch.load("MLP_verts.pth"))
model.eval()

DATA_DIR = "./facescape_trainset_001_100"
full_dataset = FaceDataset(DATA_DIR)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, collate_fn=collate_batched_meshes)
train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=collate_batched_meshes)


for batch in train_dataloader:

    neutral_meshes, expression_meshes = batch['Mesh Feature'], batch['Meshes Labels']

    # take the vertices of all the mashes in the batch
    neutral_verts = torch.stack([mesh.verts_packed() for mesh in neutral_meshes])

    deform_verts = model(neutral_verts)
    # apply the new modified vertices to the original neutral mesh to modify the expression
    new_src_meshes = []
    for i in range(len(neutral_meshes)):
        mesh = neutral_meshes[i].offset_verts(deform_verts[i])
        verts_rgb = torch.ones_like(deform_verts[i])[None]
        mesh.textures = Textures(verts_rgb=verts_rgb.to(device))
        new_src_meshes.append(mesh)

    # Render the face mesh from each viewing angle

    # for mesh in expression_meshes:
    #     verts_rgb = torch.ones_like(mesh[0].verts_packed())[None]
    #     mesh[0].textures = Textures(verts_rgb=verts_rgb.to(device))

    for j in range(len(new_src_meshes)):

        visualize_prediction(new_src_meshes[j], expression_meshes[j])
        break

    break
