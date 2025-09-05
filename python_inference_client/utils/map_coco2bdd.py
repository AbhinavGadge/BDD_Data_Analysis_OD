import numpy as np

# Mapping dictionary
coco_to_your_id = {
    0: 0,   # person
    1: 1,   # bike
    2: 2,   # car
    3: 3,   # motor
    5: 4,   # bus
    6: 5,   # train
    7: 6,   # truck
    9: 7,   # traffic_light
    11: 8   # traffic_sign
}

def remap_detections(detections: np.ndarray) -> np.ndarray:
    """
    Remap COCO class IDs in detections to your category IDs.
    
    Args:
        detections (np.ndarray): Array of shape (N, 6) in format 
                                 [x1, y1, x2, y2, conf, coco_class_id]
    
    Returns:
        np.ndarray: Array of shape (M, 6) with mapped IDs 
                    [x1, y1, x2, y2, conf, your_class_id]
    """
    if detections.size == 0:
        return detections

    coco_ids = detections[:, 5].astype(int)

    # Keep only rows where coco_id exists in mapping
    valid_mask = np.isin(coco_ids, list(coco_to_your_id.keys()))

    # Map ids (vectorized)
    mapped_ids = np.vectorize(coco_to_your_id.get)(coco_ids[valid_mask])

    # Replace class ids with mapped ids
    mapped_detections = detections[valid_mask].copy()
    mapped_detections[:, 5] = mapped_ids

    return mapped_detections
