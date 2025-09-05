# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import numpy as np
import cv2

def pre_class_aware_nms(output, classwise_conf, iou_threshold=0.45):
    """
    YOLO post-processing: class-aware NMS with classwise confidence threshold.

    Args:
        output (np.ndarray): shape (1, N, num_boxes), format [xc, yc, w, h, cls_probs...]
        classwise_conf (dict): {class_id: (class_name, confidence_threshold)}
        iou_threshold (float): IoU threshold for NMS

    Returns:
        List of detections in model image scale: [x1, y1, x2, y2, score, class_id]
    """
    output = output.squeeze(0)  # shape: (N, num_boxes)
    num_outputs, num_boxes = output.shape
    num_classes = num_outputs - 4

    boxes_raw = output[:4, :].T  # (num_boxes, 4)
    class_scores = output[4:, :].T  # (num_boxes, num_classes)

    # Decode to corner coordinates (still in model scale)
    xc, yc, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    all_boxes = np.stack([x1, y1, x2, y2], axis=1)

    final_detections = []

    for cls_id in range(num_classes):
        if cls_id not in classwise_conf:
            continue  # Skip if no threshold provided for this class

        class_name, conf_threshold = classwise_conf[cls_id]
        scores = class_scores[:, cls_id]
        keep = scores > conf_threshold
        if not np.any(keep):
            continue

        boxes_cls = all_boxes[keep]
        scores_cls = scores[keep]

        boxes_cv2 = np.stack([
            boxes_cls[:, 0],
            boxes_cls[:, 1],
            boxes_cls[:, 2] - boxes_cls[:, 0],
            boxes_cls[:, 3] - boxes_cls[:, 1]
        ], axis=1).astype(np.float32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_cv2.tolist(),
            scores=scores_cls.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )

        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append([
                    boxes_cls[i][0], boxes_cls[i][1],
                    boxes_cls[i][2], boxes_cls[i][3],
                    scores_cls[i], cls_id
                ])

    return final_detections

def post_class_aware_nms(output, classwise_conf, iou_threshold=0.45):
    """
    Performs class-aware NMS for YOLO output with no rescaling.

    Args:
        output (np.ndarray): shape (num_boxes, 6), format [xc, yc, w, h, conf, class_id]
        classwise_conf (dict): {class_id: (class_name, confidence_threshold)}
        iou_threshold (float): IoU threshold for NMS

    Returns:
        List of detections: [x1, y1, x2, y2, score, class_id]
    """
    if output.shape[0] == 0:
        return []

    # Convert to corner coordinates
    xc, yc, w, h, conf, cls_ids = output.T
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    all_boxes = np.stack([x1, y1, x2, y2], axis=1)
    final_detections = []

    for cls_id in np.unique(cls_ids.astype(int)):
        if cls_id not in classwise_conf:
            continue

        class_name, conf_threshold = classwise_conf[cls_id]

        # Filter detections for this class
        mask = (cls_ids == cls_id) & (conf > conf_threshold)
        if not np.any(mask):
            continue

        boxes_cls = all_boxes[mask]
        scores_cls = conf[mask]

        boxes_cv2 = np.stack([
            boxes_cls[:, 0],
            boxes_cls[:, 1],
            boxes_cls[:, 2] - boxes_cls[:, 0],
            boxes_cls[:, 3] - boxes_cls[:, 1]
        ], axis=1).astype(np.float32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_cv2.tolist(),
            scores=scores_cls.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )

        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append([
                    boxes_cls[i][0], boxes_cls[i][1],
                    boxes_cls[i][2], boxes_cls[i][3],
                    scores_cls[i], int(cls_id)
                ])

    return final_detections
