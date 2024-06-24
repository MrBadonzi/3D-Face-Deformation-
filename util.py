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
    collated_dict['exp'] = torch.tensor(collated_dict['exp'], dtype=torch.float).to(device)

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

    return neutral_fig, predicted_fig, target_fig


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


# The position of each Face Landmark via mediapipe repo :
# https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
# reference image : https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png

Landmark_positions = {
    'silhouette': [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],

    'lipsUpperOuter': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    'mouth': [391, 322, 410, 287, 432, 273, 422, 335, 424, 406, 418, 313,
              421, 18, 200, 83, 201, 182, 194, 106, 204, 43, 202, 57, 212, 186, 92, 165, 167, 164, 393, 436, 216],

    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'rightEyeUpper1': [247, 30, 29, 27, 28, 56, 190],
    'rightEyeLower1': [130, 25, 110, 24, 23, 22, 26, 112, 243],
    'rightEyeUpper2': [113, 225, 224, 223, 222, 221, 189],
    'rightEyeLower2': [226, 31, 228, 229, 230, 231, 232, 233, 244],
    'rightEyeLower3': [143, 111, 117, 118, 119, 120, 121, 128, 245],

    'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
    'rightEyebrowLower': [35, 124, 46, 53, 52, 65],

    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    'leftEyeUpper1': [467, 260, 259, 257, 258, 286, 414],
    'leftEyeLower1': [359, 255, 339, 254, 253, 252, 256, 341, 463],
    'leftEyeUpper2': [342, 445, 444, 443, 442, 441, 413],
    'leftEyeLower2': [446, 261, 448, 449, 450, 451, 452, 453, 464],
    'leftEyeLower3': [372, 340, 346, 347, 348, 349, 350, 357, 465],

    'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
    'leftEyebrowLower': [265, 353, 276, 283, 282, 295],

    'midwayBetweenEyes': [168],

    'noseTip': [1],
    'noseBottom': [2],
    'noseRightCorner': [98],
    'noseLeftCorner': [327],

    'rightCheek': [205, 36, 207, 214],
    'leftCheek': [425, 427, 266, 434]
}
