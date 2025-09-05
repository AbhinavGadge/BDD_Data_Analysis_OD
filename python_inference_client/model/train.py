# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import os
from ultralytics import YOLO, RTDETR

# ------------------ Config ------------------ #
# Choose model: "yolov8n.pt", "yolov8m.pt", "yolo11m.pt", "rtdetr-l.pt"
MODEL = "yolov8n.pt"

# Dataset config (YAML file)
DATA_YAML = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data\data.yaml"

# Training save directory
PROJECT_DIR = "trainingPath"

# Training hyperparameters
EPOCHS = 500
PATIENCE = 30
SAVE_PERIOD = 30
LR0 = 0.001
# -------------------------------------------- #

# Initialize model
if "rtdetr" in MODEL.lower():
    model = RTDETR(MODEL)
    batch_size = 64  # RT-DETR requires smaller batch size
else:
    model = YOLO(MODEL)
    batch_size = 2   # YOLO can handle larger batch size

# Train model
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=batch_size,
    project=PROJECT_DIR,
    name=f"BDD_{MODEL.replace('.pt', '')}",
    exist_ok=False,
    device="cpu",       # Change to "cuda" for GPU
    resume=False,
    plots=True,
    patience=PATIENCE,
    degrees=360,
    scale=0.6,
    translate=0.4,
    fliplr=0.8,
    mixup=0.5,
    copy_paste=0.3,
    save_period=SAVE_PERIOD,
    optimizer="SGD",
    lr0=LR0
)

print("✅ Training completed. Results saved in:", os.path.join(PROJECT_DIR, f"BDD_{MODEL.replace('.pt', '')}"))
