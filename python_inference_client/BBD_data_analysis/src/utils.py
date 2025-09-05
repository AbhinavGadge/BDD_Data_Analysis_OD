import matplotlib.pyplot as plt
import os
import cv2


def save_plot(distribution, save_path):
    """Save class distribution bar chart to file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(distribution.keys(), distribution.values())
    plt.title("Class Distribution in BDD100K")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_bboxes(image_path, labels, save_path):
    """Draw bounding boxes on an image and save it."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = cv2.imread(str(image_path))

    for l in labels:
        if "box2d" in l:
            box = l["box2d"]
            x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
            cls = l.get("category", "unknown")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(str(save_path), img)
