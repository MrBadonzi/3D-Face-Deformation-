import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset1 import FaceDataset
from key import your_api_key
from myrender import renderer, lights, camera
from network import MLP_verts
from util import mesh_collate, mesh_to_wanb, Landmark_positions

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Losses
losses = {
    "silhouette": {"weight": 1.0, "values": []},
    "edge": {"weight": 1.0, "values": []},
    "normal": {"weight": 0.1, "values": []},
    "laplacian": {"weight": 1.0, "values": []},
    "mse": {"weight": 0.01, "values": []},
    "landmark": {"weight": 1.0, "values": []},
    "rgb": {"weight": 1.0, "values": []}

}


def update_mesh_shape_prior_losses(meshes, loss):
    loss["edge"] = mesh_edge_loss(meshes)
    loss["normal"] = mesh_normal_consistency(meshes)
    loss["laplacian"] = mesh_laplacian_smoothing(meshes, method="uniform")


config = {
    "learning_rate": 0.002,
    "architecture": "mlp",
    "epochs": 20,
    "num_views": 20,
    # original 5
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
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=mesh_collate,
                                  num_workers=0)

    model = MLP_verts().to(device)
    l2_mesh = MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"])
    NUM_ACCUMULATION_STEPS = 4

    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # The landmark positions that we don't want to modify
    Important_Face_Areas = set()
    for array in Landmark_positions.values():
        Important_Face_Areas.update(array)

    '''
    ####################################################################################################################
    #                                        START OF THE TRAIN                                                        #
    ####################################################################################################################
    '''

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader, desc=f"epoch: {epoch + 1}")):

            neutral_meshes, expression_meshes = batch["Mesh Feature"], batch['Meshes Labels']

            deform_verts = model(neutral_meshes.verts_packed())
            # apply the new modified vertices to the original neutral mesh to modify the expression
            new_src_meshes = neutral_meshes.offset_verts(deform_verts)

            loss = {k: torch.tensor(0.0, device=device) for k in losses}

            # Randomly select five views to optimize over in this iteration.  Compared
            # to using just one view, this helps resolve ambiguities between updating
            # mesh shape vs. updating mesh texture
            for i in range(len(new_src_meshes)):
                expression = expression_meshes[i][0]
                # soft render configurations
                image_target = renderer(expression, cameras=camera, lights=lights)
                images_predicted = renderer(new_src_meshes[i], cameras=camera, lights=lights)
                target_silhouette = image_target[0, ..., 3]
                predicted_silhouette = images_predicted[..., 3]

                predicted_rgb = images_predicted[..., :3]
                loss_rgb = ((predicted_rgb - image_target[..., :3]) ** 2).mean()
                loss["rgb"] += loss_rgb / config['num_views_per_iteration']

                # Squared L2 distance between the predicted silhouette and the target silhouette from our dataset
                #
                loss_silhouette = ((predicted_silhouette - target_silhouette) ** 2).mean()
                loss["silhouette"] += loss_silhouette / config['num_views_per_iteration']

                # predicted_silhouette = Image.fromarray(
                #     (predicted_silhouette.cpu().detach().numpy()[0] * 255).astype(np.uint8))
                # target_silhouette = Image.fromarray(
                #     (target_silhouette.cpu().detach().numpy() * 255).astype(np.uint8))
                # wandb.log({f"predicted sihoulette": wandb.Image(predicted_silhouette),
                #            "target sihoulette": wandb.Image(target_silhouette)}, )

                # transform in pill image
                predicted_image = Image.fromarray(
                    (images_predicted[0, ..., :3].cpu().detach().numpy() * 255).astype(np.uint8))
                target_image = Image.fromarray(
                    (image_target[0, ..., :3].cpu().detach().numpy() * 255).astype(np.uint8))

                # wandb.log({f"predicted face": wandb.Image(predicted_image),
                #            "target face": wandb.Image(target_image)})

                # Load the input image.
                predicted_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(predicted_image))
                target_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(target_image))

                # Detect face landmarks from the input image.
                loss_landmarks = []
                target_landmarks = detector.detect(target_image).face_landmarks[0]
                predicted_landmarks = detector.detect(predicted_image).face_landmarks

                if predicted_landmarks:
                    predicted_landmarks = predicted_landmarks[0]
                    for k in range(len(target_landmarks)):
                        if k in Important_Face_Areas:
                            # Squared L2 distance between the predicted landmarks and the target landmark from our mesh
                            loss_landmarks.append(np.sqrt((predicted_landmarks[k].x - target_landmarks[k].x) ** 2 +
                                                          (predicted_landmarks[k].y - target_landmarks[k].y) ** 2))
                else:
                    for k in range(len(target_landmarks)):
                        if k in Important_Face_Areas:
                            # Squared L2 distance between the predicted landmarks and the target landmark from our mesh
                            loss_landmarks.append(np.sqrt(target_landmarks[k].x ** 2 + target_landmarks[k].y ** 2))

                loss_landmarks = np.mean(loss_landmarks)
                loss['landmark'] += loss_landmarks / config['num_views_per_iteration']

            # to prevent the mesh collapse
            loss['mse'] = l2_mesh(deform_verts, expression_meshes.verts_packed())
            update_mesh_shape_prior_losses(new_src_meshes, loss)

            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))

            sum_loss = sum_loss / NUM_ACCUMULATION_STEPS
            # Optimization step
            sum_loss.backward()
            if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_dataloader)):
                # Update Optimizer
                optimizer.step()
                optimizer.zero_grad()
                # Print the losses
                wandb.log({"batch_loss": sum_loss,
                           'mse': loss['mse'],
                           'landmarks': loss["landmark"],
                           "silohuette loss": loss["silhouette"],
                           "edge": loss["edge"],
                           "normal": loss["normal"],
                           "laplacian": loss["laplacian"],
                           "rgb": loss["rgb"]})
                epoch_loss += sum_loss / (len(train_dataloader) / NUM_ACCUMULATION_STEPS)

        # Plot mesh
        if epoch % config["plot_period"] == 0:
            neutral, predicted, target = mesh_to_wanb(neutral_meshes[0], new_src_meshes[0], expression_meshes[0][0])
            wandb.log({"predicted": wandb.Plotly(predicted), "target": wandb.Plotly(target),
                       "neutral": wandb.Plotly(neutral)})
        # write the losses
        wandb.log({"epoch_loss": epoch_loss})

    # TODO: FARE CHECKPOINTS
    torch.save(model.state_dict(), 'Models/MLP_verts.pth')
    torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')
    wandb.finish()
