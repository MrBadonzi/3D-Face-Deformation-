from typing import Dict, List

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from pytorch3d.structures import join_meshes_as_batch, Meshes
from pytorch3d.vis import plot_scene

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def mesh_collate(batch: List[Dict]):
    if batch is None or len(batch) == 0:
        return None

    collated_dict = {}

    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    collated_dict['Mesh Feature'] = join_meshes_as_batch(collated_dict['Mesh Feature'], include_textures=True)
    collated_dict['Meshes Labels'] = join_meshes_as_batch(collated_dict['Meshes Labels'], include_textures=True)

    return collated_dict


def mesh_to_wanb(neutral_mesh, predicted_mesh, target_mesh, title=''):
    target = Meshes(
        verts=target_mesh.verts_list(),
        faces=target_mesh.faces_list(),
        textures=target_mesh.textures
    )
    predicted = Meshes(
        verts=predicted_mesh.verts_list(),
        faces=predicted_mesh.faces_list(),
        textures=predicted_mesh.textures
    )

    neutral = Meshes(
        verts=neutral_mesh.verts_list(),
        faces=neutral_mesh.faces_list(),
        textures=neutral_mesh.textures
    )

    # Render the plotly figure
    target_fig = plot_scene({
        "target mesh": {
            "target mesh": target
        }
    })

    predicted_fig = plot_scene({
        "predicted mesh": {
            "predicted mesh": predicted
        }
    })

    neutral_fig = plot_scene({
        "neutral mesh": {
            "neutral mesh": neutral
        }
    })

    return predicted_fig, target_fig, neutral_fig


def create_texture(verts, red_value=0.859, green_value=0.114, blue_value=0.29):
    num_vertices = verts.shape[0]
    verts_rgb = torch.ones((1, num_vertices, 3), dtype=torch.float32).to(device)  # Red color
    verts_rgb[:, 1:, 1:] = blue_value  # Set green and blue channels to 0 to make it red
    verts_rgb[1:, :, 1:] = green_value
    verts_rgb[1:, 1:, :] = red_value

    return verts_rgb


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()
