# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

class Autobackend:
    def __new__(cls, framework, model_config):
        try:
            if framework.lower() == "openvino":
                from model.openvino_detect import OpenVINOInferencePipeline
                return OpenVINOInferencePipeline(model_config)
            elif framework.lower() == "pytorch":
                from model.pytorch_detect import PyTorchInferencePipeline
                return PyTorchInferencePipeline(model_config)
            elif framework.lower() == "onnx":
                from model.onnx_detect import OnnxInferencePipeline
                return OnnxInferencePipeline(model_config)
            else:
                raise ValueError(f"Unsupported framework: {framework}")
        except ImportError as e:
            print(f"[ERROR] Failed to import module for {framework}: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to initialize inference pipeline: {e}")
            raise
