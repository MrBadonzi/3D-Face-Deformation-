import numpy as np
import torch
from pytorch3d.datasets import collate_batched_meshes
from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)
from pytorch3d.structures import join_meshes_as_batch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import wandb
from util import mesh_collate,  mesh_to_wanb
from pytorch3d.renderer import Textures
from tqdm import tqdm
from key import your_api_key
from myrender import renderer, cameras, lights, target_cameras
from dataset1 import FaceDataset
from network import MLP_verts
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Losses
losses = {
    "silhouette": {"weight": 1.0, "values": []},
    "edge": {"weight": 1.0, "values": []},
    #original weight 0.01
    "normal": {"weight": 0.01, "values": []},
    "laplacian": {"weight": 1.0, "values": []},
    "mse": {"weight": 0.5, "values": []}
}


def update_mesh_shape_prior_losses(meshes, loss):
    loss["edge"] = mesh_edge_loss(meshes)
    loss["normal"] = mesh_normal_consistency(meshes)
    loss["laplacian"] = mesh_laplacian_smoothing(meshes, method="uniform")

    wandb.log({"edge": loss["edge"]})
    wandb.log({"normal": loss["normal"]})
    wandb.log({"laplacian": loss["laplacian"]})


config = {
    "learning_rate": 0.002,
    "architecture": "mlp",
    "epochs": 20,
    "num_views": 20,
    #original 5
    "num_views_per_iteration": 1,
    "plot_period": 1,
}

DATA_DIR = "./resized_facescape"

if __name__ == '__main__':

    wandb.login(key=your_api_key)
    wandb.init(
        # set the wandb project where this run will be logged
        project="3D Face Deformation",
        # track hyperparameters and run metadata
        config=config
    )
    # Dataset Creation
    full_dataset = FaceDataset(DATA_DIR)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # # start a new wandb run to track this script
    # TODO: CERCARE DI CAPIRE PERCHE NON FUNZIONA IL RENDERING CON I WORKERS
    # fixme: resize ulteriore del dataset
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=mesh_collate,
                                  num_workers=0)

    model = MLP_verts().to(device)
    l2_mesh = MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        model.train()
        for batch in tqdm(train_dataloader, desc=f"epoch: {epoch + 1}"):

            optimizer.zero_grad()

            neutral_meshes, expression_meshes = batch['Mesh Feature'], batch['Meshes Labels']

            deform_verts = model(neutral_meshes.verts_packed())
            # apply the new modified vertices to the original neutral mesh to modify the expression
            new_src_meshes = neutral_meshes.offset_verts(deform_verts)

            loss = {k: torch.tensor(0.0, device=device) for k in losses}

            # Randomly select five views to optimize over in this iteration.  Compared
            # to using just one view, this helps resolve ambiguities between updating
            # mesh shape vs. updating mesh texture
            for idx in range(len(new_src_meshes)):
                expression = expression_meshes[idx][0].extend(config['num_views'])
                silhouette_images = renderer(expression, cameras=cameras, lights=lights)
                target_silhouette = [silhouette_images[i, ..., 3] for i in range(config['num_views'])]

                for j in np.random.permutation(config['num_views']).tolist()[:config['num_views_per_iteration']]:

                    # soft render configurations
                    images_predicted = renderer(new_src_meshes[idx], cameras=target_cameras[j], lights=lights)

                    # Squared L2 distance between the predicted silhouette and the target silhouette from our dataset

                    predicted_silhouette = images_predicted[..., 3]
                    loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
                    loss["silhouette"] += loss_silhouette / config['num_views_per_iteration']

                    # transform in pill image for the log

                    # predicted_image = Image.fromarray(
                    #     (predicted_silhouette.cpu().detach().numpy()[0] * 255).astype(np.uint8))
                    # target_image = Image.fromarray(
                    #     (target_silhouette[j].cpu().detach().numpy() * 255).astype(np.uint8))
                    # wandb.log({f"predicted sihoulette": wandb.Image(predicted_image),
                    #            "target sihoulette": wandb.Image(target_image)}, )

            loss['mse'] = l2_mesh(deform_verts, expression_meshes.verts_packed())
            wandb.log({"silohuette loss": loss["silhouette"]})
            update_mesh_shape_prior_losses(new_src_meshes, loss)

            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))

            # Print the losses
            wandb.log({"batch_loss": sum_loss, 'mse': loss['mse']})

            # Optimization step
            sum_loss.backward()
            optimizer.step()

            epoch_loss += sum_loss / len(train_dataloader)

            # Plot mesh
        if epoch % config["plot_period"] == 0:
            predicted, target, neutral = mesh_to_wanb(neutral_meshes[0], new_src_meshes[0], expression_meshes[0][0])
            wandb.log({"predicted": wandb.Plotly(predicted), "target": wandb.Plotly(target),
                        "neutral": wandb.Plotly(neutral)})

            # Print the losses
        wandb.log({"epoch_loss": epoch_loss})

    # TODO: FARE CHECKPOINTS
    torch.save(model.state_dict(), 'Models/MLP_verts.pth')
    torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')
    wandb.finish()
