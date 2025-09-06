# BDD100K Object Detection – Data Analysis, Inference & Evaluation

This repository contains an end-to-end pipeline for analyzing the **BDD100K dataset** (object detection subset), 
performing inference with **YOLOv8 + OpenVINO**, and evaluating results with both quantitative and qualitative metrics.  
It includes dataset analysis, model preparation, inference pipeline, evaluation scripts, and reproducible Docker setup.

---

<p align="center">
  <img src="https://github.com/AbhinavGadge/BDD_Data_Analysis_OD/blob/main/OD_inference_pipeline.png" alt="Pipeline Diagram" width="700"/>
</p>



## 🚀 Quick Start with DockerHub

You can directly pull the pre-built Docker image from DockerHub:

```bash
docker pull abhigadge/bdd100k-analysis:latest
docker run -it --rm -v /path/to/data:/app/data abhigadge/bdd100k-analysis:latest
📂 Repository Structure
python
Copy code
python_inference_client/
│
├── .venv/                         # Local virtual environment
├── .vscode/                       # Editor settings
├── outputs/                       # Generated analysis plots & results
│   ├── class_distribution.png
│   ├── most_crowded.png
│   ├── motor_sample.png
│   └── train_sample.png
│
├── python_inference_client/
│   ├── BBD_data_analysis/         # Dataset parsing & analysis
│   ├── bdd_data/                  # Dataset (images + labels)
│   ├── conf/                      # Config files (OpenVINO, NMS, transformations)
│   ├── data/                      # Data loaders, utilities
│   ├── model/                     # Training scripts & notebooks
│   ├── utils/                     # Helper functions
│   ├── inference.py               # Main inference entrypoint
│   ├── __init__.py
│   └── BDD100K_Assignment_Report.pdf   # Detailed project report
│
├── yolov8n_openvino_model/        # Quantized YOLOv8n model (OpenVINO format)
│   ├── yolov8n.bin
│   ├── yolov8n.xml
│   └── metadata.yaml
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Containerized execution
└── README.md                      # Project documentation
🐳 Docker Usage
Build Image Locally
bash
Copy code
docker build -t bdd100k-analysis .
Run Container
Mount your dataset to /app/data:

bash
Copy code
docker run -it --rm -v /path/to/local/data:/app/data bdd100k-analysis
⚡ Running Inference (OpenVINO)
Update the config file: conf/openvino3.conf

Example configuration:

ini
Copy code
framework = PythonServer
inferenceBackend = Openvino

input_path = python_inference_client/bdd_data/val/images
weights = yolov8n_openvino_model
namesFile = python_inference_client/bdd_data/classes.txt
classwiseConfFile = python_inference_client/conf/classwiseconf.txt

inputWidth = 640
inputHeight = 640
nmsThreshold = 0.70
confidenceThreshold = 0.30
numThreads = 2
networkType = yolov8n
inferenceOnGPU = 1
useLetterBox = 1
borderColor = 255,255,255
classAwareNMS = 1
imgTransFile = python_inference_client/conf/imgTransFile.txt
igpu_cpu_ratio = 1:1
Run inference:

bash
Copy code
python .\python_inference_client\inference.py
🏋️ Training
To train the model (YOLOv8), use either:

.\python_inference_clientmodel\train.ipynb (Jupyter Notebook)

.\python_inference_clientmodel\train.py (Python script)

Before training
Activate your virtual environment:

bash
Copy code
# Windows (PowerShell)
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
📊 Outputs
outputs/class_distribution.png → Class imbalance visualization (validation subset)

outputs/most_crowded.png → Example crowded scene

outputs/motor_sample.png → Rare class example

outputs/train_sample.png → Train class example

# Evavaluation
python .\python_inference_client\utils\conf_mat\test.py

Evaluation metrics are saved as per-class AP, mAP@0.5, precision/recall tables.

📑 Documentation
See the full project report for details:
📄 python_inference_client/BDD100K_Assignment_Report.pdf

The report includes:

Dataset analysis (distribution, anomalies, unique samples)

Model selection & quantization (YOLOv8n + OpenVINO INT8)

End-to-end inference pipeline (CPU+iGPU multiprocessing)

Evaluation results (Detection Accuracy, Classification Accuracy, mAP)

Error analysis and improvement suggestions

✨ Key Features
Containerized pipeline → Portable with Docker.

OpenVINO integration → Efficient INT8 inference on CPU + Intel iGPU.

Real-time multi-view detection → Parallel inference on augmented inputs.

Detailed evaluation → Per-class AP, detection accuracy, misclassification analysis.

Reproducibility → Requirements, Dockerfile, and config-driven execution.

📌 Author
Abhinav Gadge
Machine Learning Engineer – Computer Vision, Deep Learning, and Model Deployment.
=======
