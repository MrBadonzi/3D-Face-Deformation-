# STEP 1: Import the necessary modules.
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pytorch3d.io import load_objs_as_meshes

from myrender import renderer, lights, camera
from network import MLP_verts
from util import draw_landmarks_on_image

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

model = MLP_verts().to(device)
model.load_state_dict(torch.load("Models/MLP_verts.pth"))
model.eval()

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cane = Path('resized_facescape/1/models_reg/1_neutral.obj')
label = load_objs_as_meshes([cane], load_textures=True, device=device)
deform_verts = model(label.verts_packed())
new_src_meshes = label.offset_verts(deform_verts)
image_target = renderer(new_src_meshes, cameras=camera, lights=lights)
image_target = image_target[0, ..., :3].cpu().detach().numpy()
target_image = Image.fromarray((image_target * 255).astype(np.uint8))
target_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(target_image))

detection_result = detector.detect(target_image)
print(detection_result.face_landmarks[0])
# STEP 5: Process the detection result. In this case, visualize it.
cv2.imwrite('cane1.png', cv2.cvtColor(target_image.numpy_view(), cv2.COLOR_RGB2BGR))
annotated_image = draw_landmarks_on_image(target_image.numpy_view(), detection_result)
cv2.imwrite('cane.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
