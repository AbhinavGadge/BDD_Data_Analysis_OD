# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import os
import cv2
import torch

def read_config(file_path):
        """Reads a config file and returns a dictionary of key-value pairs."""
        config = {}
        if os.path.exists(file_path):
            indx = 0
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # Ignore empty lines and comments
                        if "=" in line:
                            key, value = line.split("=", 1)
                        elif " " in line:
                            key, value = line.split(" ", 1)
                        else:
                            key, value = (indx, line)
                        config[key.strip() if isinstance(key, str) else key] = value.strip()
                        indx += 1
        return config

def parse_model_config(model_config):
    if isinstance(model_config, dict):
        config = model_config
    else:
        config = read_config(model_config)

    # Core settings
    framework = config.get("framework")
    input_path = config.get("input_path")
    weights = config.get("weights")
    label_file = config.get("namesFile")
    input_width = int(config.get("inputWidth", 640))
    input_height = int(config.get("inputHeight", 640))
    nms_thresh = float(config.get("nmsThreshold", 0.35))
    conf_thresh = float(config.get("confidenceThreshold", 0.50))
    network_type = config.get("networkType", "KSDetect2")
    device = 0 if bool(int(config.get("inferenceOnGPU", 0))) and torch.cuda.is_available() else 'cpu'
    class_aware_nms = bool(int(config.get("classAwareNMS", 1)))

    # Names and classwise conf
    names = read_config(label_file) if label_file else {}
    classwise_conf_file = config.get("classwiseConfFile")
    classwise_conf_raw = read_config(classwise_conf_file) if classwise_conf_file and os.path.exists(classwise_conf_file) else {}
    classwise_conf = {i: (k, float(v)) for i, (k, v) in enumerate(classwise_conf_raw.items())} if classwise_conf_raw else {}
    conf_thresh_final = min(conf_thresh, min([v[1] for v in classwise_conf.values()], default=conf_thresh))
    
    # Default fallback
    classwise_conf = classwise_conf if classwise_conf else {int(k): (v, float(conf_thresh_final)) for k, v in names.items()}

    # Image transformations
    img_trans_file = config.get("imgTransFile", "")
    img_transformation = read_config(img_trans_file) if os.path.exists(img_trans_file) else {}
    output_path = img_transformation.get("saveImagePath", None)
    save = bool(int(img_transformation.get("saveDebugImages", 0)))

    # Print info
    print(f"\033[1;31mImages Folder OL:           \033[0m {input_path}")
    print(f"\033[1;31mWeights File:               \033[0m {weights}")
    print(f"\033[1;31mFramework:                  \033[0m {framework}")
    print(f"\033[1;31mNetwork Type:               \033[0m {network_type}")
    print(f"\033[1;31mOutput Path:                \033[0m {output_path}")
    print(f"\033[1;31mCV2 Version:                \033[0m {cv2.__version__}")
    print(f"\033[1;31mModel Config:               \033[0m {config}")
    print(f"\033[1;31mImage Transformations:      \033[0m {img_transformation}")
    print(f"\033[1;31mClasswise Confidence:       \033[0m {classwise_conf}")

def split_binary_by_ratio(binary_values, ratio_str):
    ones_indices = [i for i, val in enumerate(binary_values) if val == 1]
    num_ones = len(ones_indices)

    if num_ones == 0:
        return [[] for _ in ratio_str.split(':')]  # No 1s to split

    try:
        ratio_parts = [int(x) for x in ratio_str.split(':')]
    except ValueError:
        return "Invalid ratio format. Please use numbers separated by ':'"

    total_ratio = sum(ratio_parts)

    if total_ratio == 0:
        return "Ratio cannot have all zeros."

    ones_per_unit = num_ones / total_ratio
    split_counts = [round(part * ones_per_unit) for part in ratio_parts]

    # Adjust for rounding errors to ensure the total number of 1s is correct
    diff = sum(split_counts) - num_ones
    if diff > 0:
        # Distribute extra 1s proportionally (simplistic approach)
        for _ in range(diff):
            max_index = split_counts.index(max(split_counts))
            split_counts[max_index] -= 1
    elif diff < 0:
        for _ in range(abs(diff)):
            min_index = split_counts.index(min(split_counts))
            split_counts[min_index] += 1

    groups = [[] for _ in ratio_parts]
    ones_index_counter = 0
    for i, count in enumerate(split_counts):
        groups[i] = [ones_indices[j] for j in range(ones_index_counter, ones_index_counter + count)]
        ones_index_counter += count

    return groups

# # Example Usageddd
# binary_data = [1, 0, 1, 1, 0, 1]
# user_ratio = "2:1"
# result = split_binary_by_ratio(binary_data, user_ratio)
# print(f"Original binary values: {binary_data}")
# print(f"User ratio: {user_ratio}")
# print(f"Groups of '1' indices: {result}")


from typing import List, Dict, Type

def to_outputFormat(
    detections: List[List[float]],
    output_class: Type,
    class_names: Dict[int, str]
) -> List:
    """
    Converts raw detections to output_class instances (e.g., SpatialCoordinateOutput).

    Args:
        detections (List[List[float]]): 
            List of detections where each detection is a list in the format:
            [x1, y1, x2, y2, score, cls_id]
            - x1, y1: Top-left coordinates of the bounding box
            - x2, y2: Bottom-right coordinates of the bounding box
            - score: Confidence score of the detection
            - cls_id: Class index of the detected object

        output_class (Type): 
            A class that defines the output structure, typically a dataclass like 
            SpatialCoordinateOutput with fields:
                - bounding_box (Tuple[float, float, float, float]): (x, y, w, h)
                - label (str): Human-readable class label
                - confidence (float): Detection confidence score

        class_names (Dict[int, str]): 
            A dictionary mapping class indices (int) to their corresponding string labels.
            Example: {0: "person", 1: "car", 2: "bottle"}

    Returns:
        List:
            A list of `output_class` instances, where each instance represents a processed 
            detection. The bounding box is converted from (x1, y1, x2, y2) to 
            (x, y, w, h), the label is resolved from `class_names`, and the confidence score 
            is passed through.

            Each item in the list has the following attributes:
                - bounding_box: A tuple of (x1, y1, width, height)
                - label: String name of the class
                - confidence: Float value of the confidence score

            This standardized output can then be used for downstream processing, 
            serialization, or populating API response structures.
    """
    outputs = []
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        w, h = x2 - x1, y2 - y1
        label = f"{class_names[int(cls_id)]}"
        outputs.append(
            output_class(
                bounding_box=(x1, y1, w, h),
                label=label,
                id=int(cls_id),
                confidence=float(score)
            )
        )
    return outputs
