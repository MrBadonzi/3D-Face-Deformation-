import mediapipe as mp
import numpy as np
import torch
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def landmark_loss(predicted_tensor_image, target_tensor_image, detector):
    # transform in pill image
    predicted_image = Image.fromarray(
        (predicted_tensor_image.cpu().detach().numpy() * 255).astype(np.uint8))
    target_image = Image.fromarray(
        (target_tensor_image.cpu().detach().numpy() * 255).astype(np.uint8))

    # # Load the input image.
    predicted_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(predicted_image))
    target_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(target_image))

    # # Detect face landmarks from the input image.
    target_landmarks = detector.detect(target_image).face_landmarks[0]
    predicted_landmarks = detector.detect(predicted_image).face_landmarks

    # take the landmarks coordinates
    target_landmarks_coords = torch.zeros(2, len(target_landmarks), device=device)
    prediction = torch.zeros(478, 3, requires_grad=True, device=device)

    if predicted_landmarks:
        predicted_landmarks = predicted_landmarks[0]
        predicted_landmarks_coords = torch.zeros(2, len(predicted_landmarks), device=device)

        for k in range(len(target_landmarks)):
            predicted_landmarks_coords[0, k] = predicted_landmarks[k].x
            predicted_landmarks_coords[1, k] = predicted_landmarks[k].y

            target_landmarks_coords[0, k] = target_landmarks[k].x
            target_landmarks_coords[1, k] = target_landmarks[k].y

        # denormalize landmark positions
        predicted_landmarks_coords = torch.floor(predicted_landmarks_coords * predicted_tensor_image.shape[1])
        # Create a mask where elements greater than 256 are True
        mask = predicted_landmarks_coords >= 256
        # Replace elements greater than 256 with 255
        predicted_landmarks_coords[mask] = 255
        predicted_landmarks_coords = predicted_landmarks_coords.type(torch.long)

        # select the landmark point from the predicted image
        predicted_landmark_points = predicted_tensor_image[predicted_landmarks_coords[0], predicted_landmarks_coords[1]]
        prediction = torch.sigmoid(predicted_landmark_points)

    else:
        for k in range(len(target_landmarks)):
            target_landmarks_coords[0, k] = target_landmarks[k].x
            target_landmarks_coords[1, k] = target_landmarks[k].y

    # denormalize landmark positions
    target_landmarks_coords = torch.floor(target_landmarks_coords * target_tensor_image.shape[1])
    # Create a mask where elements greater than 256 are True
    mask = target_landmarks_coords > 256
    # Replace elements greater than 256 with 255
    target_landmarks_coords[mask] = 255
    target_landmarks_coords = target_landmarks_coords.type(torch.long)

    # select the landmark point from the target image
    target_landmark_points = target_tensor_image[target_landmarks_coords[0], target_landmarks_coords[1]]

    # perform intersection over union (IoU)

    target = torch.sigmoid(target_landmark_points)
    inter = (prediction * target).sum(dim=1)
    union = (prediction + target).sum(dim=1) - inter
    iou = 1 - (inter / union)
    return iou.mean()
