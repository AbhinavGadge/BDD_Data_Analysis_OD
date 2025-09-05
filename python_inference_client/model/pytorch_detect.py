# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import os
import numpy as np
import torch
from typing import Dict
import cv2

from ultralytics import YOLO
from utils import read_config, rescale_detections, draw_bbox
from model import post_class_aware_nms
from data import augmentations, transform_detection_1


class PyTorchInferencePipeline:
    def __init__(self, model_config: Dict):
        self.args = model_config
        self.device = self._setup_device(model_config.get("inferenceOnGPU", 1))
        self.weights = model_config.get("weights")
        self.input_width = int(model_config.get("inputWidth", 640))
        self.input_height = int(model_config.get("inputHeight", 640))
        self.nms_threshold = float(model_config.get("nmsThreshold", 0.70))
        self.confidence_threshold = float(model_config.get("confidenceThreshold", 0.50))
        self.network_type = model_config.get("networkType", "ksdetectv2")
        self.labelFilePath = model_config.get("namesFile")
        self.use_letterbox = bool(int(model_config.get("useLetterBox", 1)))
        self.border_color = self._parse_border_color(model_config.get("borderColor", "255,255,255"))
        self.class_aware_nms = bool(int(model_config.get("classAwareNMS", 1)))
        self.num_threads = int(model_config.get("numThreads", 2))

        # Class and label files
        self.names = read_config(self.labelFilePath) if self.labelFilePath else {}
        self.classwiseConfFile = model_config.get("classwiseConfFile")
        self.classwise_conf = read_config(self.classwiseConfFile) if self.classwiseConfFile and os.path.exists(self.classwiseConfFile) else {}
        self.classwise_conf = {i: (k, float(v)) for i, (k, v) in enumerate(self.classwise_conf.items())} if self.classwise_conf else {}
        self.confidence_threshold = min(self.confidence_threshold, min([j[1] for j in self.classwise_conf.values()], default=self.confidence_threshold))
        self.classwise_conf = self.classwise_conf if self.classwise_conf else {int(k): (v, float(self.confidence_threshold)) for k, v in self.names.items()}

        # Image transformations
        img_transformation = read_config(model_config.get("imgTransFile")) if os.path.exists(model_config.get("imgTransFile", "")) else {}
        self.transformation_list = [1]
        self.transformation_list += [int(img_transformation.get(k, 0)) for k in ["rotate90", "rotate270", "rotate180", "flipHorizontal", "flipVertical"]]
        self.save = bool(int(img_transformation.get("saveDebugImages", 0)))
        self.output_path = img_transformation.get("saveImagePath", None)
        self.output_label = img_transformation.get("saveImagePath", None)

        # Load YOLOv8 model
        self.model = self._load_yolov8_model()

        # Ensure output dirs
        if self.save:
            if self.output_path and not os.path.exists(self.output_path):
                os.makedirs(self.output_path, exist_ok=True)
            if self.output_label and not os.path.exists(self.output_label):
                os.makedirs(self.output_label, exist_ok=True)

    def _setup_device(self, inference_on_gpu):
        """Setup device based on GPU availability and config"""
        if inference_on_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("[INFO] Using CPU for inference")
        return device

    def _parse_border_color(self, border_color_str):
        """Parse border color from string format '255,255,255' to tuple"""
        try:
            return tuple(map(int, border_color_str.split(',')))
        except:
            return (255, 255, 255)

    def _load_yolov8_model(self):
        """Load YOLOv8 model using ultralytics"""
        try:
            print(f"[INFO] Loading YOLOv8 model from: {self.weights}")
            model = YOLO(self.weights)
            model.to(self.device)
            
            # Set model to evaluation mode
            model.model.eval()
            
            print(f"[✓] YOLOv8 model loaded successfully on {self.device}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8 model: {e}")
            raise e

    def _preprocess_image(self, image):
        """Preprocess image for YOLOv8 inference"""
        if self.use_letterbox:
            # YOLOv8 handles letterboxing internally, but we can customize if needed
            return image
        else:
            # Direct resize without letterboxing
            return cv2.resize(image, (self.input_width, self.input_height))

    def _yolov8_to_detection_format(self, results, original_shape):
        """Convert YOLOv8 results to the expected detection format [x1, y1, x2, y2, conf, class]"""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    detections.append([
                        box[0], box[1], box[2], box[3],  # x1, y1, x2, y2
                        conf,  # confidence
                        int(cls)  # class
                    ])
        
        return np.array(detections) if detections else np.empty((0, 6))

    def filter_detections_np(self, detections):
        """Filter detections based on class-wise confidence thresholds"""
        if len(detections) == 0:
            return detections
            
        filtered_detections = []
        
        for detection in detections:
            class_id = int(detection[5])
            confidence = detection[4]
            
            # Get class-specific threshold
            class_threshold = self.classwise_conf.get(class_id, ('', self.confidence_threshold))[1]
            
            if confidence >= class_threshold:
                filtered_detections.append(detection)
        
        return np.array(filtered_detections) if filtered_detections else np.empty((0, 6))

    def detect_objects(self, image):
        """Run YOLOv8 inference on image"""
        try:
            # YOLOv8 expects image in BGR format (OpenCV format)
            if len(image.shape) == 4:  # If batched, take first image
                image = image[0]
            
            # Convert from tensor format (C, H, W) to image format (H, W, C) if needed
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure image is in correct format
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Run YOLOv8 inference # Dont use augment=True --> to compare results with openvino
            results = self.model(
                image,
                imgsz=self.input_height,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                verbose=False,
                device=self.device
            )
            
            # Convert to detection format
            detections = self._yolov8_to_detection_format(results, image.shape[:2])
            
            return self.filter_detections_np(detections)
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return np.empty((0, 6))

    def detect_gpu(self, image, name):
        """Run detection with augmentations"""
        detections_gpu = np.empty((0, 6))
        
        for aug, flag in enumerate(self.transformation_list):
            if flag == 1:
                aug_img = augmentations(aug, image.copy())
                results = self.detect_objects(aug_img)
                
                if len(results) > 0:
                    # Transform detections back to original orientation
                    results = transform_detection_1(aug, results, self.input_width, self.input_height)
                    detections_gpu = np.vstack((detections_gpu, results))
        return detections_gpu

    def detect_single_image(self, img_data, test=False):
        """Main detection function for a single image"""
        image = img_data["tensor"]  # shape = (1, 3, 640, 640) or similar
        detections = np.empty((0, 6))
        
        # Run detection with augmentations
        detection = self.detect_gpu(image, img_data["name"])
        
        if detection is not None and len(detection) > 0:
            detections = np.vstack((detections, detection))
        
        # Apply class-aware NMS if enabled
        if self.class_aware_nms and len(detections) > 0:
            detections = post_class_aware_nms(
                detections, 
                self.classwise_conf, 
                iou_threshold=self.nms_threshold
            )
        
        # Rescale detections to original image coordinates
        detections = rescale_detections(detections, img_data["ratio"], img_data["pad"])
        
        # Draw and save results
        if not test:
            self.draw_save(img_data, detections)
        
        # Debug log
        debug_path = "/nfs/xray_object_detection/abhinav/deployment/python_server_inference/Testing/Meet_Test/IntegrationCode/tmp.txt"
        try:
            with open(debug_path, "a") as f:
                f.write(f"\n Image Name : {img_data['name']}\n")
                f.write("############## PyTorch YOLOv8 Detect ####################\n")
                f.write(str(detections) + "\n \n")
        except:
            pass  # Handle cases where debug path doesn't exist
        
        return detections

    def run_inference(self, img_data):
        """Main inference entry point"""
        return self.detect_single_image(img_data)

    def draw_save(self, img_data, detections):
        """Draw bounding boxes and save image"""
        if self.save and self.output_path:
            draw_bbox(
                img_data["orig_img"], 
                detections, 
                os.path.join(self.output_path, img_data["name"]), 
                self.names
            )

    def save_augmented_image(self, aug_img, aug, img_name):
        """Save augmented image for debugging"""
        if not self.save or not self.output_path:
            return
            
        try:
            # Convert to NumPy if it's a torch.Tensor
            if torch.is_tensor(aug_img):
                aug_img = aug_img.detach().cpu().numpy()
            
            # Handle different input formats
            if len(aug_img.shape) == 4:  # (1, 3, H, W)
                img_np = aug_img[0]
            elif len(aug_img.shape) == 3 and aug_img.shape[0] == 3:  # (3, H, W)
                img_np = aug_img
            else:  # (H, W, 3)
                img_np = aug_img
                if len(img_np.shape) == 3:
                    img_np = np.transpose(img_np, (2, 0, 1))  # Convert to (3, H, W)
            
            # Transpose to (H, W, 3)
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # Normalize to uint8 if needed
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)
            
            # Define save directory and create it
            save_dir = os.path.join(self.output_path, "YOLOv8_Debug")
            os.makedirs(save_dir, exist_ok=True)
            
            # Define the file name
            save_path = os.path.join(save_dir, f"{img_name}_{aug}.jpg")
            
            # Save the image using OpenCV
            cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"[WARN] Failed to save augmented image: {e}")

    def get_model_info(self):
        """Get model information"""
        try:
            return {
                "model_type": "YOLOv8",
                "framework": "PyTorch",
                "device": str(self.device),
                "input_size": f"{self.input_width}x{self.input_height}",
                "num_classes": len(self.names),
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold,
                "class_aware_nms": self.class_aware_nms,
                "use_letterbox": self.use_letterbox
            }
        except:
            return {"error": "Could not retrieve model info"}



# Example usage and configuration loader
def load_yolov8_pipeline(config_path: str = None, model_config: Dict = None):
    """
    Load YOLOv8 pipeline from config file or dictionary
    
    Args:
        config_path: Path to configuration file
        model_config: Configuration dictionary
    
    Returns:
        PyTorchYOLOv8Pipeline instance
    """
    if config_path:
        model_config = read_config(config_path)
    elif model_config is None:
        raise ValueError("Either config_path or model_config must be provided")
    
    return PyTorchYOLOv8Pipeline(model_config)


if __name__ == "__main__":
    # Example configuration based on your provided config
    example_config = {
        "framework": "Pytorch",
        "input_path": "./path/to/images/folder/",
        "weights": "./path/to/.pt/weights/",
        "namesFile": "./conf/classes.txt",
        "classwiseConfFile": "./conf/classwiseconf.txt",
        "inputWidth": "640",
        "inputHeight": "640",
        "nmsThreshold": "0.70",
        "confidenceThreshold": "0.50",
        "numThreads": "2",
        "networkType": "ksdetectv2",
        "inferenceOnGPU": "1",
        "useLetterBox": "1",
        "borderColor": "255,255,255",
        "classAwareNMS": "1",
        "imgTransFile": "./conf/imgTransFile.txt"
    }
    
    # Initialize pipeline
    pipeline = PyTorchInferencePipeline(example_config)
    print("YOLOv8 Pipeline initialized successfully!")
    print("Model Info:", pipeline.get_model_info())
