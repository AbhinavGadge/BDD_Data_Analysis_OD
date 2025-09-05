# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import cv2
from utils import normalize_xcycwh_conf, denormalize_xywh, xyxy_to_norm_xcycwh, denorm_xcycwh
import numpy as np

def augmentations(augment, img):
    # If img has 4 dimensions (e.g., [1, 3, H, W]), convert to [H, W, 3]
    was_4d = False
    if img.ndim == 4:
        was_4d = True
        img = img.squeeze(0)  # Shape becomes [3, H, W]
        img = np.transpose(img, (1, 2, 0))  # Shape becomes [H, W, 3]

    img = img.copy()

    if augment == 0:
        aug_img = img

    elif augment == 1:
        aug_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif augment == 2:
        aug_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif augment == 3:
        aug_img = cv2.rotate(img, cv2.ROTATE_180)

    elif augment == 4:
        aug_img = cv2.flip(img, 1)  # Horizontal flip

    elif augment == 5:
        aug_img = cv2.flip(img, 0)  # Vertical flip

    else:
        raise ValueError(f"Invalid augment value: {augment}")

    # Convert back to 4D format if original image was 4D
    if was_4d:
        aug_img = np.transpose(aug_img, (2, 0, 1))  # Back to [3, H, W]
        aug_img = aug_img[np.newaxis, ...]  # Back to [1, 3, H, W]

    return aug_img

    
def transform_detection(augment, det, img_width, img_height):
    det = normalize_xcycwh_conf(det, img_width, img_height)
    for detection in det:
        if augment == 0:
            detection = detection
        elif augment == 1:  # Clockwise 90-degree rotation
            detection[0], detection[1] = detection[1], 1 - detection[0]
            detection[2], detection[3] = detection[3], detection[2]
        elif augment == 2:  # Anti-clockwise 90-degree rotation
            detection[0], detection[1] = 1 - detection[1], detection[0]
            detection[2], detection[3] = detection[3], detection[2]
        elif augment == 3:  # 180-degree rotation
            detection[0], detection[1] = 1 - detection[0], 1 - detection[1]
        elif augment == 4:  # Horizontal flip
            detection[0] = 1 - detection[0]
        elif augment == 5:  # Vertical flip
            detection[1] = 1 - detection[1]
    det = denormalize_xywh(det, img_width, img_height)
    return det


def transform_detection_1(augment, det, img_w, img_h):
    if det.size == 0:
        return np.empty((0, 6), dtype=np.float64)
    det = xyxy_to_norm_xcycwh(det, img_w, img_h)
    xc, yc, w, h = det[:, 0], det[:, 1], det[:, 2], det[:, 3]

    if augment == 1:
        xc, yc, w, h = yc, 1 - xc, h, w
    elif augment == 2:
        xc, yc, w, h = 1 - yc, xc, h, w
    elif augment == 3:
        xc, yc = 1 - xc, 1 - yc
    elif augment == 4:
        xc = 1 - xc
    elif augment == 5:
        yc = 1 - yc

    det[:, :4] = np.stack([xc, yc, w, h], axis=1)
    return denorm_xcycwh(det, img_w, img_h)

