import torch
from pytorch3d.datasets import collate_batched_meshes
from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from key import your_api_key

from dataset1 import FaceDataset
from myrender import renderer
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


def update_mesh_shape_prior_losses(meshes, loss):
    l_edge = torch.empty(len(meshes), device=device)
    l_normal = torch.empty(len(meshes), device=device)
    l_laplacian = torch.empty(len(meshes), device=device)

    for i in range(len(meshes)):

        # and (b) the edge length of the predicted mesh
        l_edge[i]=mesh_edge_loss(meshes[i])

        # mesh normal consistency
        l_normal[i] = mesh_normal_consistency(meshes[i])

        # mesh laplacian smoothing
        l_laplacian[i] = mesh_laplacian_smoothing(meshes[i], method="uniform")

    loss["edge"] = torch.mean(l_edge)
    loss["normal"] = torch.mean(l_normal)
    loss["laplacian"] = torch.mean(l_laplacian)



wandb.login(key=your_api_key)
# # start a new wandb run to track this script
config = {
    "learning_rate": 0.02,
    "architecture": "mlp",
    "epochs": 20,
}
#
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

    model = MLP_verts().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        model.train()
        for batch in tqdm(train_dataloader):

            optimizer.zero_grad()

            neutral_meshes, expression_meshes = batch['Mesh Feature'], batch['Meshes Labels']

            # take the vertices of all the mashes in the batch
            neutral_verts = torch.stack([mesh.verts_packed() for mesh in neutral_meshes])

            deform_verts = model(neutral_verts)
            # apply the new modified vertices to the original neutral mesh to modify the expression
            new_src_meshes = [neutral_meshes[i].offset_verts(deform_verts[i]) for i in range(len(neutral_meshes))]

            loss = {k: torch.tensor(0.0, device=device) for k in losses}
            update_mesh_shape_prior_losses(new_src_meshes, loss)

            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))

            # Print the losses
            wandb.log({"total_loss": sum_loss})

            # Optimization step
            sum_loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'MLP_verts.pth')
    torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')
    wandb.finish()
