# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import os
import cv2
import numpy as np
from PIL import Image

def draw_bbox_old(
    image,
    detections,
    save_path: str,
    class_names=None,
    box_color=(0, 255, 0),
    text_color=(0, 0, 255),
    thickness=2
):
    """
    Draw bounding boxes on the image and save it.
    Also saves annotations in YOLO-style .txt format:
    cls_id score x1 y1 x2 y2
    """
    # Convert PIL to NumPy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    img = image.copy()

    # Save directory and filename base
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base_filename = os.path.splitext(save_path)[0]
    txt_path = f"{base_filename}.txt"

    lines = []

    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{cls_id}"
        if class_names and int(cls_id) < len(class_names):
            label = f"{class_names[int(cls_id)]}"
        text = f"{label}: {score:.2f}"

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), box_color, -1)
        cv2.putText(img, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Save line for annotation file
        lines.append(f"{x1} {y1} {x2} {y2} {score:.4f} {int(cls_id)}")

    # Save the image
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save the annotations
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))



import os
import cv2
import numpy as np
from PIL import Image

def draw_bbox(
    image,
    detections,
    output_path: str,
    class_names=None,
    box_color=(0, 255, 0),
    text_color=(0, 0, 255),
    thickness=2
):
    """
    Save the original image to <folder_name>/input,
    draw bounding boxes and save annotated image + label file to <folder_name>/output.
    Format of label: x1 y1 x2 y2 score class_id
    """

    # Convert PIL to NumPy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    img = image.copy()

    # Extract filename and folder from output_path
    filename = os.path.basename(output_path)
    filename_wo_ext = os.path.splitext(filename)[0]
    folder_name = os.path.dirname(output_path).rstrip("/")

    # Define input and output folders dynamically
    input_folder = os.path.join(folder_name, "input")
    output_folder = os.path.join(folder_name, "output")

    # Ensure input/output folders exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Save original image to input_folder
    input_img_path = os.path.join(input_folder, filename)
    cv2.imwrite(input_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Prepare output paths
    output_img_path = os.path.join(output_folder, filename)
    output_txt_path = os.path.join(output_folder, f"{filename_wo_ext}.txt")

    lines = []

    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{cls_id}"
        if class_names and int(cls_id) < len(class_names):
            label = class_names[int(cls_id)]
        text = f"{label}: {score:.2f}"

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), box_color, -1)
        cv2.putText(img, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Append annotation line
        lines.append(f"{x1} {y1} {x2} {y2} {score:.4f} {int(cls_id)}")

    # Save image with bounding boxes
    cv2.imwrite(output_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Save annotation file
    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))

