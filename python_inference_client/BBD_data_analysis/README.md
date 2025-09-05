# BDD100K Object Detection Analysis

This project analyzes the **BDD100K dataset** for object detection focusing on the 10 bounding box categories.

## 📂 Repo Structure

BDD100K-Analysis/
│── data/
│   ├── labels/det_train.json
│   ├── labels/det_val.json
│   ├── classes.txt
│   └── images/train/   # 100k images here (big, keep in .gitignore)
│
│── src/
│   ├── parser.py
│   ├── analysis.py
│   ├── dashboard.py
│   └── utils.py
│
│── outputs/            # generated plots & bbox visualizations
│── requirements.txt
│── Dockerfile
│── README.md
