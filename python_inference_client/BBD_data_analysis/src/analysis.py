from parser import BDDParser
from utils import save_plot, draw_bboxes
import numpy as np
from collections import Counter


def analyze_split(train_json, val_json, classes_file):
    """Compare train/val distributions."""
    train_parser = BDDParser(train_json, classes_file)
    val_parser = BDDParser(val_json, classes_file)

    return train_parser.get_class_distribution(), val_parser.get_class_distribution()


def detect_anomalies(bbox_stats):
    """Detect anomalies: very small or very large bounding boxes."""
    anomalies = {}
    for cls, boxes in bbox_stats.items():
        widths = [w for w, _ in boxes]
        heights = [h for _, h in boxes]
        areas = [w * h for w, h in boxes]

        anomalies[cls] = {
            "avg_width": float(np.mean(widths)),
            "avg_height": float(np.mean(heights)),
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "num_boxes": len(boxes),
        }
    return anomalies


def find_most_crowded_image(parser: BDDParser):
    """Find the image with maximum objects."""
    max_count, max_image, max_labels = 0, None, []
    for ann in parser.annotations:
        count = sum(1 for l in ann.get("labels", []) if l.get("category") in parser.classes)
        if count > max_count:
            max_count = count
            max_image = ann["name"]
            max_labels = ann["labels"]
    return max_image, max_labels


def find_rarest_class_images(parser: BDDParser, top_n=3):
    """Find images containing the rarest classes."""
    dist = parser.get_class_distribution()
    rare_classes = [cls for cls, _ in sorted(dist.items(), key=lambda x: x[1])[:top_n]]

    rare_images = {}
    for ann in parser.annotations:
        cats = [l.get("category") for l in ann.get("labels", [])]
        for rc in rare_classes:
            if rc in cats:
                rare_images.setdefault(rc, []).append((ann["name"], ann["labels"]))
    return rare_images

def find_interesting_samples(annotations, class_names, top_n=5):
    """
    Identify interesting/unique samples:
      - rare classes
      - extreme bbox sizes
    Returns:
      dict with keys: 'rare_classes', 'smallest', 'largest'
      each value is a list of dicts with keys: image, class, area, bbox
    """
    class_counts = {cls: 0 for cls in class_names}
    bbox_infos = []

    for ann in annotations:
        labels = ann.get("labels", [])
        for label in labels:
            cls = label.get("category")
            if cls in class_counts:
                class_counts[cls] += 1
                # bbox area
                box2d = label.get("box2d")
                if box2d:
                    x1, y1 = box2d["x1"], box2d["y1"]
                    x2, y2 = box2d["x2"], box2d["y2"]
                    w, h = x2 - x1, y2 - y1
                    area = w * h
                    bbox_infos.append({
                        "image": ann["name"],
                        "class": cls,
                        "area": area,
                        "bbox": (x1, y1, w, h)
                    })

    # Rare classes (top 3 least frequent)
    rare_classes = sorted(class_counts.items(), key=lambda x: x[1])[:3]
    rare_class_samples = []
    for cls, _ in rare_classes:
        # find one example per rare class
        for info in bbox_infos:
            if info["class"] == cls:
                rare_class_samples.append(info)
                break

    # Smallest / largest bounding boxes
    bbox_sorted = sorted(bbox_infos, key=lambda x: x["area"])
    smallest = bbox_sorted[:top_n]
    largest = bbox_sorted[-top_n:]

    return {
        "rare_classes": rare_class_samples,
        "smallest": smallest,
        "largest": largest
    }


if __name__ == "__main__":
    json_path = r"c:\Users\abhinav.gadge\Documents\KN_Docs_Induction\data_bdd\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
    image_path = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\data_bdd\bdd100k_images_100k\bdd100k\images\100k\train"
    classes_path = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\data\classes.txt"
    parser = BDDParser(json_path=json_path, classes_file=classes_path, image_dir=image_path)

    # Class distribution
    dist = parser.get_class_distribution()
    save_plot(dist, "outputs/class_distribution.png")

    # Bbox stats
    bbox_stats = parser.get_bbox_stats()
    anomalies = detect_anomalies(bbox_stats)
    print("Anomalies:", anomalies)

    # Most crowded image
    img_name, labels = find_most_crowded_image(parser)
    print(f"Most crowded image: {img_name} with {len(labels)} objects")
    draw_bboxes(parser.image_dir / img_name, labels, "outputs/most_crowded.png")

    # Rarest class samples
    rare_samples = find_rarest_class_images(parser, top_n=2)
    for cls, samples in rare_samples.items():
        for img_name, labels in samples[:1]:  # save one per class
            print(f"Rarest class {cls} → sample {img_name}")
            draw_bboxes(parser.image_dir / img_name, labels, f"outputs/{cls}_sample.png")
