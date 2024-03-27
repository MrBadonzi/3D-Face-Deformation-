import torch
from pytorch3d.datasets import collate_batched_meshes
from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from Dataset import FaceDataset
from render import renderer
from network import MLP_verts

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Losses
losses = {
    "silhouette": {"weight": 1.0, "values": []},
    "edge": {"weight": 1.0, "values": []},
    "normal": {"weight": 0.01, "values": []},
    "laplacian": {"weight": 1.0, "values": []},
}


def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


# start a new wandb run to track this script
config={
        "learning_rate": 0.02,
        "architecture": "mlp",
        "epochs": 20,
    }

wandb.init(
    # set the wandb project where this run will be logged
    project="3D Face Deformation",
    # track hyperparameters and run metadata
    config= config
)


# Dataset Creation

DATA_DIR = "./facescape_trainset_001_100"
full_dataset = FaceDataset(DATA_DIR)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


if __name__ == '__main__':
    train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=collate_batched_meshes)
    test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, collate_fn=collate_batched_meshes)

    model = MLP_verts()
    optimizer = torch.optim.AdamW(params= model.parameters(), lr= config["learning_rate"])


    for epoch in tqdm(range(config["epochs"])):
        model.train()
        for batch in train_dataloader:

            optimizer.zero_grad()
            neutral_mesh, expression_meshes = batch['Mesh Feature'], batch['Meshes Labels']
                                            = model()














wandb.finish()