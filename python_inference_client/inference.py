# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import os
import time
from utils import parse_model_config, read_config
from model import Autobackend  # Assuming you have a PyTorch loader defined similarly
from data import BddDataLoader
from utils import draw_bbox
import numpy as np

class MultipleViewDetection:
    def __init__(self, model_config):
        # Load and parse the model configuration
        print(model_config)
        if isinstance(model_config, dict) and "modelInfo" in model_config:
            model_config = model_config["modelInfo"]
        elif isinstance(model_config, str):  # Assuming model_config is a filepath
            model_config = read_config(model_config)
        parse_model_config(model_config)
        self.args = model_config
        self.framework = model_config.get("inferenceBackend", "openvino")
        self.input_width = int(model_config.get("inputWidth", 640))
        self.input_height = int(model_config.get("inputHeight", 640))
        self.input_path = model_config.get("input_path")
        self.class_names = model_config.get("namesFile")
        self.expect_numpy_input = self.input_path is None

            
        # Load the correct model based on the framework
        try:
            self.model_loader = Autobackend(self.framework, model_config)
        except Exception as e:
            print(f"[ERROR] Failed to initialize model loader for framework '{self.framework}': {e}")
            self.model_loader = None
            
        # DataLoader
        self.dataloader = BddDataLoader(self.input_path, self.framework, (self.input_width, self.input_height))
        #try run
        self.check_status()
        
    def check_status(self):
        try:
            test_image = self.dataloader.generate_test_image()
            self.model_loader.detect_single_image(test_image, test=True)
            print("SUCCESS : Dummy Run")
        except Exception as e:
            print(f"FAILURE: {e}")
            return


    def run_inference_local(self):
        """
        Runs inference on the image data, calls detect_objects for the selected framework.
        """
        total_time = 0
        for ic, img_data in enumerate(self.dataloader):
            print(f"{ic + 1}/{len(self.dataloader)} {img_data['name']}", end=" ")
            start_time = time.time()
            detections = self.model_loader.detect_single_image(img_data)
            total_time += time.time() - start_time
            print(f"{(time.time() - start_time):.3f} sec")
        hours = int(total_time / 3600)
        minutes = int((total_time % 3600) / 60)
        seconds = int(total_time % 60)
        print(f"\nObject Detection Took Time: {hours} hrs, {minutes} mins, {seconds} secs")
        return detections
    def run_inference_live(self, img, requestID=""):
        """
        Runs inference on the image data, calls detect_objects for the selected framework.
        """
        img_data = self.dataloader.process_numpy_image(img, requestID)
        start_time = time.time()
        detections = self.model_loader.detect_single_image(img_data)
        total_time = time.time() - start_time
        #print(f"{(time.time() - start_time):.3f} sec")
        #print(f"Time Taken : {total_time:.3f} sec | {total_time % 60} sec")
        hours = int(total_time / 3600)
        minutes = int((total_time % 3600) / 60)
        seconds = total_time % 60
        print(f"\nObject Detection Took Time: {hours} hrs, {minutes} mins, {seconds:.3f} secs")
        return detections
 

def main():
    """
    Main function to parse command-line arguments and run the general inference pipeline.
    """
    # Default paths
    model_config_path = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\conf\openvino3.conf"

    # Create the inference pipeline object
    inference_pipeline =  MultipleViewDetection(model_config=model_config_path)
    if inference_pipeline.expect_numpy_input:
        print("Running in Live/Simulator Mode")
        for _ in range(10):
            img = np.random.randint(0, 256, (724, 980, 3), dtype=np.uint8)
            inference_pipeline.run_inference_live(img)
    else:
        inference_pipeline.run_inference_local()
    
if __name__ == "__main__":
    main()
