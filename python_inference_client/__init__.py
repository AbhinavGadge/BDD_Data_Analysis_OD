# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

# python_inference_client/__init__.py

import os
import sys

# Add the root path of `python_inference_client` to sys.path
sys.path.append(os.path.dirname(__file__))

from .inference import MultipleViewDetection
from .utils import to_outputFormat, remap_detections
