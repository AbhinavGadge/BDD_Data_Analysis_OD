# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import numpy as np
from typing import Tuple
import cv2

def image_to_tensor(image:np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img (np.ndarray): image for preprocessing
    Returns:
      input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

def letterbox(img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (255, 255, 255), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size


    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_image(image):
    image, ratio, (dw, dh)  = letterbox(np.array(image))
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = np.ascontiguousarray(image)  # Add batch dimension
    # image = image.astype(np.float32) / 255.0
    return image, ratio, (dw, dh)

def rescale_detections(detections, ratio, pad):
    """
    Rescales bounding boxes from letterboxed image to original image scale.

    Args:
        detections (np.ndarray): shape (N, 6), format [xc, yc, w, h, conf, class_id]
        ratio (Tuple[float, float]): (scale_w, scale_h)
        pad (Tuple[float, float]): (pad_w, pad_h)

    Returns:
        np.ndarray: rescaled detections, format [x1, y1, x2, y2, conf, class_id]
    """
    if len(detections) == 0:
        return np.empty((0, 6))

    detections = np.array(detections.copy())
    x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]

    pad_w, pad_h = pad
    gain_w, gain_h = ratio

    # Undo padding
    x1 -= pad_w
    x2 -= pad_w
    y1 -= pad_h
    y2 -= pad_h

    # Undo scaling
    x1 /= gain_w
    x2 /= gain_w
    y1 /= gain_h
    y2 /= gain_h

    # Replace bbox coords in detections
    rescaled = np.stack([x1, y1, x2, y2, detections[:, 4], detections[:, 5]], axis=1)
    return rescaled



def normalize_xcycwh_conf(boxes, img_width, img_height):
    """
    Normalize [xc, yc, w, h, conf] by image dimensions.

    Args:
        boxes (np.ndarray): shape (N, 5), format [xc, yc, w, h, conf]
        img_width (int): width of the image
        img_height (int): height of the image

    Returns:
        np.ndarray: shape (N, 5), normalized [xc/img_w, yc/img_h, w/img_w, h/img_h, conf]
    """
    boxes = boxes.copy().astype(np.float32)

    boxes[:, 0] /= img_width   # xc
    boxes[:, 1] /= img_height  # yc
    boxes[:, 2] /= img_width   # w
    boxes[:, 3] /= img_height  # h

    return boxes

def denormalize_xywh(boxes, img_width, img_height):
    """
    Convert normalized [xc, yc, w, h, conf, class_id] to non-normalized [xc, yc, w, h, conf, class_id]

    Args:
        boxes (np.ndarray): shape (N, 6), with [xc, yc, w, h, conf, class_id] normalized to [0, 1]
        img_width (int): width of the image
        img_height (int): height of the image

    Returns:
        np.ndarray: shape (N, 6), denormalized [xc, yc, w, h, conf, class_id]
    """
    boxes = boxes.copy().astype(np.float32)
    
    boxes[:, 0] *= img_width   # xc
    boxes[:, 1] *= img_height  # yc
    boxes[:, 2] *= img_width   # w
    boxes[:, 3] *= img_height  # h
    
    # conf and class_id remain unchanged
    return boxes

def xyxy_to_norm_xcycwh(det, img_w, img_h):
    det = det.copy()
    xc = (det[:, 0] + det[:, 2]) / 2 / img_w
    yc = (det[:, 1] + det[:, 3]) / 2 / img_h
    w  = (det[:, 2] - det[:, 0]) / img_w
    h  = (det[:, 3] - det[:, 1]) / img_h
    return np.stack([xc, yc, w, h, det[:, 4], det[:, 5]], axis=1)

def denorm_xcycwh(det, img_w, img_h):
    det = det.copy()
    det[:, 0] *= img_w  # xc
    det[:, 1] *= img_h  # yc
    det[:, 2] *= img_w  # w
    det[:, 3] *= img_h  # h
    return det

