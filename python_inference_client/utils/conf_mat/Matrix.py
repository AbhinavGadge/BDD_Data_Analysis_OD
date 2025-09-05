# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import os
import sys
import glob
import pandas as pd
import numpy as np

from confusion_matrix import *

def ConfMatrix(imgPath, savingPath, classesPath, ground_truth_path, predicted, classwiseConfidence):
    with open(classesPath, 'r') as file:
        classes = [line.strip().split()[0] for line in file if line.strip()]

    with open(classwiseConfidence, 'r') as file:
        conf_threshold = [float(line.strip().split()[1]) for line in file if line.strip()]

    conf_mat = ConfusionMatrix(num_classes=len(classes), CONF_THRESHOLD=conf_threshold, IOU_THRESHOLD=0.5)
    conf_mat, classes = confuMat(conf_mat, predicted, ground_truth_path, imgPath, classes, savingPath)

    return conf_mat, classes

def main():
     if (len(sys.argv)) < 6:
         print( "Usage :: python Matrix.py <path to images> <path to classes> <path to Ground truth> <path for Predicted labels> <path for Saving images><path for Classwise Confidence>" )
         sys.exit()

     imgPath = sys.argv[1]
     classesPath = sys.argv[2]
     ground_truth_path = sys.argv[3]
     predicted = sys.argv[4]
     savingPath = sys.argv[5]
     classwiseConfidence = sys.argv[6]

     os.makedirs(savingPath, exist_ok = True) 
     ConfMatrix(imgPath, savingPath, classesPath, ground_truth_path, predicted, classwiseConfidence)
    
if __name__ == "__main__":
    main() 
