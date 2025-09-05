import cv2
from pathlib import Path
import os
import json
import random
import shutil

# --- Base input paths
base_img_folder = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\data_bdd\bdd100k_images_100k\bdd100k\images\100k"
base_json_folder = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\data_bdd\bdd100k_labels_release\bdd100k\labels"

# --- Base output path
base_output_folder = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data"

# --- Class names (YOLO IDs, aligned with COCO → your mapping)
class_to_id = {
    "person": 0,
    "bike": 1,
    "car": 2,
    "motor": 3,
    "bus": 4,
    "train": 5,
    "truck": 6,
    "traffic light": 7,
    "traffic sign": 8,
}


def process_split(split: str, num_samples: int = 300):
    """
    Process a dataset split (train or val), convert BDD to YOLO format.
    """
    print(f"\n🔹 Processing split: {split}")

    inp_img_folder = os.path.join(base_img_folder, split)
    inp_json_path = os.path.join(base_json_folder, f"bdd100k_labels_images_{split}.json")

    output_image_folder = os.path.join(base_output_folder, split, "images")
    output_labels_folder = os.path.join(base_output_folder, split, "labels")

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # --- Load BDD JSON
    with open(inp_json_path, "r") as f:
        annotations = json.load(f)

    # --- Map from image name to annotation
    ann_dict = {ann["name"]: ann for ann in annotations}

    # --- Sample N images (if available)
    all_images = list(ann_dict.keys())
    sampled_images = random.sample(all_images, min(num_samples, len(all_images)))

    # --- Process each sampled image
    for img_name in sampled_images:
        ann = ann_dict[img_name]
        src_img_path = os.path.join(inp_img_folder, img_name)
        dst_img_path = os.path.join(output_image_folder, img_name)

        # Copy image
        shutil.copy(src_img_path, dst_img_path)

        # Read image to get size
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"⚠️ Warning: Could not read {src_img_path}")
            continue
        img_h, img_w = img.shape[:2]

        # Prepare YOLO label file
        yolo_lines = []
        for label in ann.get("labels", []):
            cls_name = label.get("category")
            if cls_name not in class_to_id:
                continue  # skip unknown classes

            cls_id = class_to_id[cls_name]
            box = label.get("box2d")
            if box is None:
                continue

            # Convert to YOLO format: x_center, y_center, width, height (normalized)
            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]
            x_c = ((x1 + x2) / 2) / img_w
            y_c = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # Write YOLO label file
        label_file = os.path.join(output_labels_folder, Path(img_name).stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"✅ Done! Copied {len(sampled_images)} {split} images and generated YOLO labels.")


if __name__ == "__main__":
    # Run for both splits
    process_split("train", num_samples=300)
    process_split("val", num_samples=300)
