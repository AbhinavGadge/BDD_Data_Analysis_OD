# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

from utils import read_config, split_binary_by_ratio
from utils import rescale_detections, draw_bbox, remap_detections
from model import pre_class_aware_nms, post_class_aware_nms
from data import augmentations, transform_detection_1
from threading import Thread
import numpy as np
import os
from pathlib import Path
import openvino as ov
import psutil


class OpenVINOInferencePipeline:
    def __init__(self, model_config):
         # Load and parse the model configuration
        self.args = model_config
        self.nw_type = model_config.get("networkType")
        self.input_path = model_config.get("input_path")
        self.weights = model_config.get("weights")
        self.labelFilePath = model_config.get("namesFile")
        self.input_width = int(model_config.get("inputWidth", 640))
        self.input_height = int(model_config.get("inputHeight", 640))
        self.nms_threshold = float(model_config.get("nmsThreshold", 0.35))
        self.confidence_threshold = float(model_config.get("confidenceThreshold", 0.50))
        self.network_type = model_config.get("networkType", "YOLOV8")
        self.class_aware_nms = bool(int(model_config.get("classAwareNMS", 1)))
        self.ratio = model_config.get("igpu_cpu_ratio", "0:6")
        
        # Class and label files
        self.names = read_config(self.labelFilePath) if self.labelFilePath else {}
        self.classwiseConfFile = model_config.get("classwiseConfFile")
        self.classwise_conf = read_config(self.classwiseConfFile) if self.classwiseConfFile and os.path.exists(self.classwiseConfFile) else {}
        self.classwise_conf = {i: (k, float(v)) for i, (k, v) in enumerate(self.classwise_conf.items())} if self.classwise_conf else {}
        self.confidence_threshold = min(self.confidence_threshold, min([j[1] for j in self.classwise_conf.values()], default=self.confidence_threshold))
        self.classwise_conf = self.classwise_conf if self.classwise_conf else {int(k): (v, float(self.confidence_threshold)) for k, v in self.names.items()}
        
        # Image transformations
        img_transformation = read_config(model_config.get("imgTransFile")) if os.path.exists(model_config.get("imgTransFile")) else {}
        self.clockwise_rota = int(img_transformation.get("rotate90", 0))
        self.aclockwise_rota = int(img_transformation.get("rotate270", 0))
        self.rotat = int(img_transformation.get("rotate180", 0))
        self.h_flip = int(img_transformation.get("flipHorizontal", 0))
        self.v_flip = int(img_transformation.get("flipVertical", 0))
        self.transformation_list = [1, self.clockwise_rota, self.aclockwise_rota, self.rotat, self.h_flip, self.v_flip]
        self.save = bool(int(img_transformation.get("saveDebugImages", 0)))
        self.output_path = img_transformation.get("saveImagePath", None)
        self.output_label = img_transformation.get("saveImagePath", None)
        self.label_format = 1
        
        # Load Model
        model_loader = OpenVINOModelLoader()
        loaded_models = model_loader.load_models_with_ratio(self.weights, self.ratio)
        if loaded_models:
            self.model_gpu = loaded_models.get("GPU", None)
            self.model_cpu = loaded_models.get("CPU", None)

            if self.model_gpu:
                print("[SUCCESS] GPU model loaded.")
            if self.model_cpu:
                print("[SUCCESS] CPU model loaded.")
        else:
            self.model_gpu = None
            self.model_cpu = None
            print("[INFO] No models loaded.")
        if self.model_cpu and not self.model_gpu:
            print("[FAILURE] GPU model NOT loaded.")
            self.ratio = f"0:{sum(map(int, self.ratio.split(':')))}"
            self.ratio = f"0:{sum(self.transformation_list)}"
            print(f"ALL inferences transfered to CPU --> Updated Ratio : {self.ratio}")
        self.split_transformations = split_binary_by_ratio(self.transformation_list, self.ratio)

        # Ensure output directories exist
        if not os.path.exists(self.output_path) and self.save:
            os.makedirs(self.output_path, exist_ok=True)
        if not os.path.exists(self.output_label) and self.save:
            os.makedirs(self.output_label, exist_ok=True)
            
    def detect_objects(self, image, model):
        """
        Detects objects in the image using the given model (either CPU or GPU).
        """
        outs = model(image)
        outs = outs[model.output(0)]  # non-normalized [xc, yc, w, h, probs...]
        detections = pre_class_aware_nms(outs, self.classwise_conf, iou_threshold=self.nms_threshold)
        return np.array(detections)

    def detect_cpu(self, image, model_cpu):
        """
        Processes a single image on CPU and performs object detection.
        """
        detections_cpu = np.empty((0, 6))
        for aug in self.split_transformations[1]:
            aug_img = augmentations(aug, image.copy())  # cv2 aug takes input in the form of (H,W,C)
            results = self.detect_objects(aug_img, model_cpu)  # input should be of (1, C, W, H) i.e., 4D
            results = transform_detection_1(aug, results, self.input_width, self.input_height)
            detections_cpu = np.vstack((detections_cpu, np.array(results)))
        return detections_cpu

    def detect_gpu(self, image, model_gpu):
        """
        Processes a single image on GPU and performs object detection.
        """
        detections_gpu = np.empty((0, 6))
        for aug in self.split_transformations[0]:
            aug_img = augmentations(aug, image.copy())
            results = self.detect_objects(aug_img, model_gpu)
            results = transform_detection_1(aug, results, self.input_width, self.input_height)
            detections_gpu = np.vstack((detections_gpu, np.array(results)))
        return detections_gpu

    def detect_cpu_thread(self, image, model_cpu, detections_cpu):
        detections_cpu.extend(self.detect_cpu(image, model_cpu))

    def detect_gpu_thread(self, image, model_gpu, detections_gpu):
        detections_gpu.extend(self.detect_gpu(image, model_gpu))

    def detect_single_image(self, img_data, test=False):
        """
        Processes a single image with both CPU and GPU in parallel using threads.
        """
        image = img_data["tensor"]
        detections = np.empty((0, 6))

        detections_cpu = []
        detections_gpu = []

        threads = []

        # Run CPU detection in a separate thread
        if self.model_cpu:
            cpu_thread = Thread(target=self.detect_cpu_thread, args=(image, self.model_cpu, detections_cpu))
            threads.append(cpu_thread)
            cpu_thread.start()

        # Run GPU detection in a separate thread
        if self.model_gpu:
            gpu_thread = Thread(target=self.detect_gpu_thread, args=(image, self.model_gpu, detections_gpu))
            threads.append(gpu_thread)
            gpu_thread.start()

        # Wait for both threads to finish
        for thread in threads:
            thread.join()

        # Combine detections from both CPU and GPU
        if detections_cpu:
            detections = np.vstack((detections, np.array(detections_cpu)))
        if detections_gpu:
            detections = np.vstack((detections, np.array(detections_gpu)))

        # Perform NMS and rescale detections
        detections = post_class_aware_nms(detections, self.classwise_conf, iou_threshold=self.nms_threshold)
        detections = rescale_detections(detections, img_data["ratio"], img_data["pad"])
        # print(detections)
        detections = remap_detections(detections)
        
        if not test:
            self.draw_save(img_data, detections)
        return detections

    def run_inference(self, img_data):
        """
        Runs inference on the image data with OpenVINO.
        """
        return self.detect_single_image(img_data)
    
    def draw_save(self, img_data, detections):
        if self.save:
            draw_bbox(img_data["orig_img"], detections, os.path.join(self.output_path, img_data["name"]), self.names)

    

class OpenVINOModelLoader:
    """
    A class to load OpenVINO models for inference on specified devices,
    optionally based on a device ratio.
    """
    def __init__(self):
        self.core = ov.Core()

    def load_model(self, model_path, device, num_threads=None, batch=1):
        print(f"[INFO] Loading model from: {model_path} on device: {device}")

        model_path = Path(model_path)
        if model_path.is_dir():
            model_path = next(model_path.glob("*.xml"), None)
        if not model_path or not model_path.is_file():
            raise FileNotFoundError(f"[ERROR] Model file not found at: {model_path}")

        model = self.core.read_model(model_path)
        available_devices = self.core.available_devices
        print(f"[INFO] Available Devices: {available_devices}")

        if device not in available_devices:
            fallback = "CPU" if device == "GPU" and "CPU" in available_devices else None
            if fallback:
                print(f"[WARNING] {device} not available. Falling back to {fallback}.")
                device = fallback
            else:
                raise ValueError(f"[ERROR] Device '{device}' not available.")

        perf_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"
        config = {"PERFORMANCE_HINT": perf_mode}

        if device == "CPU":
            if num_threads is None:
                num_threads = psutil.cpu_count(logical=True)
            config["INFERENCE_NUM_THREADS"] = num_threads
            print(f"[INFO] Using {num_threads} threads for CPU inference.")

        compiled_model = self.core.compile_model(model, device_name=device, config=config)
        print(f"[INFO] Model compiled successfully for {device}.")
        return compiled_model

    def load_models_with_ratio(self, model_path, ratio_str="1:1", num_threads=None, batch=1):
        print(f"[INFO] Requested device ratio: {ratio_str}")
        try:
            a, b = map(int, ratio_str.split(':'))
            if a < 0 or b < 0:
                raise ValueError("Ratio values must be non-negative integers.")
        except Exception as e:
            print(f"[ERROR] Invalid ratio string '{ratio_str}': {e}")
            return {}

        models = {}
        available_devices = self.core.available_devices
        print(f"[INFO] Detected devices: {available_devices}")

        if a > 0 and "GPU" in available_devices:
            try:
                print(f"[INFO] Loading GPU model (ratio {a}:{b})")
                models["GPU"] = self.load_model(model_path, "GPU", num_threads, batch)
            except Exception as e:
                print(f"[ERROR] GPU model loading failed: {e}")

        if b > 0 and "CPU" in available_devices:
            try:
                print(f"[INFO] Loading CPU model (ratio {a}:{b})")
                models["CPU"] = self.load_model(model_path, "CPU", num_threads, batch)
            except Exception as e:
                print(f"[ERROR] CPU model loading failed: {e}")

        if not models:
            print(f"[WARNING] No models loaded. Please check ratio and device availability.")

        return models
