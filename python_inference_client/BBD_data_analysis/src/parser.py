import json
from collections import defaultdict
from pathlib import Path


class BDDParser:
    """
    Parser for BDD100K object detection dataset.
    Loads JSON annotations, class list, and provides methods for analysis.
    """

    def __init__(self, json_path, classes_file, image_dir=None):
        self.json_path = Path(json_path)
        self.classes_file = Path(classes_file)
        self.image_dir = Path(image_dir) if image_dir else None

        # Load annotations and classes
        self.annotations = self._load_json()
        self.classes = self._load_classes()

    def _load_json(self):
        with open(self.json_path, "r") as f:
            return json.load(f)

    def _load_classes(self):
        with open(self.classes_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def get_class_distribution(self):
        """Return count of bounding boxes per class."""
        dist = defaultdict(int)
        for ann in self.annotations:
            for label in ann.get("labels", []):
                cat = label.get("category")
                if cat in self.classes:
                    dist[cat] += 1
        return dict(dist)

    def get_bbox_stats(self):
        """Return width and height of bounding boxes per class."""
        stats = defaultdict(list)
        for ann in self.annotations:
            for label in ann.get("labels", []):
                cat = label.get("category")
                if cat in self.classes and "box2d" in label:
                    box = label["box2d"]
                    w = box["x2"] - box["x1"]
                    h = box["y2"] - box["y1"]
                    stats[cat].append((w, h))
        return stats
