# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

from PIL import Image
import numpy as np
import os
from typing import List, Tuple, Dict, Union
import cv2

class BddDataLoader:
    def __init__(self, img_folder: str, backend : str, img_size: Tuple[int, int] = (640, 640),):
        self.backend = backend
        self.img_folder = img_folder
        self.img_size = img_size

        if(img_folder is not None):
            self.image_paths = self._load_image_paths()
        else:
            self.image_paths = []

    def _load_image_paths(self) -> List[str]:
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        return [os.path.join(self.img_folder, f)
                for f in sorted(os.listdir(self.img_folder))
                if f.lower().endswith(supported_formats)]

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        for img_path in self.image_paths:
            image = Image.open(img_path)  # Open image using PIL
            image = image.convert('RGB')  # Convert to RGB if it's not already

            orig_w, orig_h = image.size  # Get original image size

            input_tensor, ratio, (dw, dh), (proc_w, proc_h) = self.preprocess_image(image)
            image_name = os.path.basename(img_path)

            yield {
                "orig_img": image,
                "tensor": input_tensor,
                "path": img_path,
                "name": image_name,
                "ratio": ratio,
                "pad": (dw, dh),
                "orig_size": (orig_w, orig_h),
                "processed_size": (proc_w, proc_h)
            }
    def process_numpy_image(self, np_img: np.ndarray, requestID=None) -> Dict[str, Union[np.ndarray, str, Tuple]]:
        """
        Process a numpy image like __iter__ does for file images.
        Returns a dictionary with all relevant info.
        """
        if np_img.ndim == 2:
            # Grayscale to RGB
            np_img = np.stack([np_img]*3, axis=-1)
        elif np_img.shape[2] == 4:
            # RGBA to RGB
            np_img = np_img[:, :, :3]

        image = Image.fromarray(np_img.astype('uint8')).convert('RGB')
        orig_w, orig_h = image.size

        input_tensor, ratio, (dw, dh), (proc_w, proc_h) = self.preprocess_image(image)

        return {
            "orig_img": image,
            "tensor": input_tensor,
            "path": None,
            "name": requestID + ".jpg",
            "ratio": ratio,
            "pad": (dw, dh),
            "orig_size": (orig_w, orig_h),
            "processed_size": (proc_w, proc_h)
        }
 

    def preprocess_image(self, image: Image.Image):
        # Convert to numpy array for the rest of the processing
        image = np.array(image)
        image, ratio, (dw, dh) = letterbox(image, new_shape=self.img_size)
        proc_h, proc_w = image.shape[:2]
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.ascontiguousarray(image)
        image = self.image_to_tensor(image)
        return image, ratio, (dw, dh), (proc_w, proc_h)
    
    def image_to_tensor(self, image: np.ndarray):
        if not self.backend.lower() == "onnx":
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)  # Explicit cast to float32 even without normalization
        if image.ndim == 3:
            image = np.expand_dims(image, 0)  # Add batch dim
        return image


    def generate_test_image(self) -> dict:
        # Random dimensions for the image between 500x500 and 1000x1000
        height = np.random.randint(500, 1001)
        width = np.random.randint(500, 1001)
        
        # Create a random image with values between 0 and 255 (RGB)
        random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Convert the numpy array to a PIL Image
        pil_image = Image.fromarray(random_image)
        
        # Get the original image size
        orig_w, orig_h = pil_image.size

        # Preprocess the image (this is the same as for normal images)
        input_tensor, ratio, (dw, dh), (proc_w, proc_h) = self.preprocess_image(pil_image)

        # Generate a random name for the test image
        image_name = "test_image"

        return {
            "orig_img": pil_image,
            "tensor": input_tensor,
            "path": "generated_test_image",  # Path can be a placeholder
            "name": image_name,
            "ratio": ratio,
            "pad": (dw, dh),
            "orig_size": (orig_w, orig_h),
            "processed_size": (proc_w, proc_h)
        }

# --- Helper Functions ---

def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (255, 255, 255),
              auto: bool = False, scale_fill: bool = False, scaleup: bool = False, stride: int = 32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


# Example usage
if __name__ == "__main__":
    dataloader = BddDataLoader("/nfs/xray_object_detection/ambuj/ambujTestingResults/2024-08Testing/DLCvsPYTHONvsSIMUL/images/", (320, 320))

    # To generate a test image
    test_image_data = dataloader.generate_test_image()
    print(test_image_data["name"])  # Print name of the test image
    print(test_image_data["path"])  # Print path (placeholder in this case)
    print(test_image_data["orig_img"].size)  # Print the size of the generated image

    # Example iteration over the dataset
    for data in dataloader:
        print(data["name"])
        print(data["path"])
