# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from threading import Thread
from typing import Dict

from utils import read_config, rescale_detections, draw_bbox
from model import post_class_aware_nms
from data import augmentations, transform_detection_1
import cv2

class OnnxInferencePipeline:
    def __init__(self, model_config: Dict):
        self.args = model_config
        self.device = model_config.get("device", "cpu")
        self.weights = model_config.get("weights")
        self.input_width = int(model_config.get("inputWidth", 640))
        self.input_height = int(model_config.get("inputHeight", 640))
        self.nms_threshold = float(model_config.get("nmsThreshold", 0.35))
        self.confidence_threshold = float(model_config.get("confidenceThreshold", 0.50))
        self.network_type = model_config.get("networkType", "deim")
        self.labelFilePath = model_config.get("namesFile")


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

        # Load model
        model_loader = ONNXModelLoader(self.weights, self.device, self.network_type)
        self.model = model_loader.load_model()
        self.input_name = self.model.get_inputs()[0].name

        # Ensure output dirs
        if self.save:
            if self.output_path and not os.path.exists(self.output_path):
                os.makedirs(self.output_path, exist_ok=True)
            if self.output_label and not os.path.exists(self.output_label):
                os.makedirs(self.output_label, exist_ok=True)

    def filter_detections_np(self, labels, boxes, scores):
        labels_flat = labels.reshape(-1)
        # thresholds = np.array([self.classwise_conf.get(l, self.confidence_threshold) for l in labels_flat])
        thresholds = np.array([self.classwise_conf.get(int(l), ('', self.confidence_threshold))[1] for l in labels_flat], dtype=np.float32)
        keep_mask = scores >= thresholds

        filtered_boxes = boxes[keep_mask]
        filtered_scores = scores[keep_mask]
        filtered_labels = labels_flat[keep_mask]

        return np.concatenate(
            [filtered_boxes, filtered_scores[:, None], filtered_labels[:, None]], axis=1
        )

    def detect_objects(self, image, model):
        img_size = np.array([[self.input_width, self.input_height]], dtype=np.int64)
        outs = model.run(None, {self.input_name: image, "orig_target_sizes": img_size})
        labels, boxes, scores = outs[0][0], outs[1][0], outs[2][0]
        return self.filter_detections_np(labels, boxes, scores)

    def detect_gpu(self, image, model_gpu, name):
        detections_gpu = np.empty((0, 6))
        for aug, flag in enumerate(self.transformation_list):
            if flag == 1:
                aug_img = augmentations(aug, image.copy())    # shape = (1, 3, 640, 640)
                results = self.detect_objects(aug_img, model_gpu)
                results = transform_detection_1(aug, results, self.input_width, self.input_height)
                detections_gpu = np.vstack((detections_gpu, results))
        return detections_gpu

    def detect_gpu_thread(self, image, model_gpu, detections_gpu):
        detections_gpu.extend(self.detect_gpu(image, model_gpu))

    def detect_single_image(self, img_data, test=False):
        image = img_data["tensor"]           # shape = (1, 3, 640, 640)
        detections = np.empty((0, 6)) 
        detection = self.detect_gpu(image, self.model, img_data["name"])

        if detection is not None and len(detection) > 0:
            detections = np.vstack((detections, detection))

        # Apply NMS and rescale
        detections = post_class_aware_nms(detections, self.classwise_conf, iou_threshold=self.nms_threshold)
        detections = rescale_detections(detections, img_data["ratio"], img_data["pad"])

        if not test:
            self.draw_save(img_data, detections)

        # Debug log
        with open("/nfs/xray_object_detection/abhinav/deployment/python_server_inference/Testing/Meet_Test/IntegrationCode/tmp.txt", "a") as f:
            f.write(f"\n Image Name : {img_data['name']}\n")
            f.write("############## Python Detect ####################\n")
            f.write(str(detections) + "\n \n")

        return detections

    def run_inference(self, img_data):
        return self.detect_single_image(img_data)

    def draw_save(self, img_data, detections):
        if self.save:
            draw_bbox(img_data["orig_img"], detections, os.path.join(self.output_path, img_data["name"]), self.names)

    def save_augmented_image(self, aug_img, aug, img_name):
        """
        Save the augmented image to: self.output_path/img_name/Deim_Debug/_<aug>.jpg
        
        Args:
            aug_img (torch.Tensor or np.ndarray): Augmented image of shape (1, 3, H, W)
            aug (str or int): Augmentation identifier
            img_name (str): Base name of the image
        """
        # Convert to NumPy if it's a torch.Tensor
        if not isinstance(aug_img, np.ndarray):
            aug_img = aug_img.detach().cpu().numpy()

        # Remove batch dimension (1, 3, H, W) -> (3, H, W)
        img_np = aug_img[0]

        # Transpose to (H, W, 3)
        img_np = np.transpose(img_np, (1, 2, 0))

        # Normalize to uint8 if needed
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.clip(0, 255).astype(np.uint8)

        # Define save directory and create it
        save_dir = os.path.join(self.output_path, "Deim_Debug")
        os.makedirs(save_dir, exist_ok=True)

        # Define the file name
        save_path = os.path.join(save_dir, f"{img_name}_{aug}.jpg")

        # Save the image using OpenCV
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

class ONNXModelLoader:
    def __init__(self, model_path, device="cpu", network_type="deim"):
        self.model_path = model_path
        self.device = device
        self.network_type = network_type.lower()

    def load_model(self):
        if self.network_type == "deim":
            return self.load_model_deim()
        raise NotImplementedError(f"Unsupported network_type: {self.network_type}")

    def load_model_deim(self):
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif self.device == "cuda":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
        elif self.device == "tensorrt":
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": False,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "./trt_cache",
                        "trt_timing_cache_enable": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError(f"Unsupported device: {self.device}")

        try:
            print(f"[INFO] Loading ONNX model with providers: {providers}...")
            session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"[✓] Model loaded using provider: {session.get_providers()[0]}")
            return session
        except Exception as e:
            print(f"[WARN] Failed to load with {self.device}: {e}. Falling back to CPU.")
            session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            print(f"[✓] Model loaded using provider: CPUExecutionProvider")
            return session


