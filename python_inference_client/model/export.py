from ultralytics import YOLO
import os

# ----------------------------
# Config
# ----------------------------
# Path to your YOLOv8 trained model (.pt)
model_path = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\yolov8n.pt"

data_path = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data\data.yaml"

# ----------------------------
# Load YOLOv8 model
# ----------------------------
model = YOLO(model_path)

# ----------------------------
# Export to OpenVINO format
# ----------------------------
# Supported formats: 'onnx', 'coreml', 'tflite', 'pb', 'openvino', 'engine'
# opset can be left default or set to 16 for compatibility
model.export(format="openvino", imgsz=640, data=data_path, opset=16, dynamic=False, simplify=True, device="cpu", verbose=True)
