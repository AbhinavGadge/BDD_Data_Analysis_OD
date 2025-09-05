# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import cv2
import os
import glob
import numpy as np
from tqdm import tqdm 
import pandas as pd
from display_modified import display

def Read(fileName, length):
    _file = np.empty((0, length), float)
    file = open(fileName, "r")

    while True:

        line = file.readline()
        
        if not line:
            break
        
        if len(line.split()) < (length - 1):
            continue
            
        values = np.zeros((1, length), float)
        for i in range(length):
            values[0][i] = float(line.split()[i])
        
        
        _file = np.append(_file, values, axis = 0)

    file.close()

    
    return _file


def ReadGt(fileName,length,imageWidth,imageHeight):
    labels = []
    with open(fileName, 'r') as file:
        for line in file:
            line = line.strip().split()
            if len(line) == length:
                class_id, center_x, center_y, width, height = map(float, line)
                X = (center_x-(width/2))*imageWidth
                Y = (center_y-(height/2))*imageHeight
                W = width*imageWidth
                H = height*imageHeight
                labels.append([class_id,X,Y,X+W,Y+H])
    return np.array(labels)

def findCorrespondingImgae(imgPath,labelName):
    validExtension = [".jpg",".png",".jpeg"]    
    for ext in validExtension:
        baseName = os.path.splitext(os.path.basename(labelName))[0]
        imgP = os.path.join(imgPath,baseName+ext)
        if os.path.exists(imgP):
            return baseName+ext


def confuMat(conf_mat, predicted ,groundTruth,  imgPath, classes, savingPath):
    
    files = glob.glob(os.path.join(groundTruth,'*.txt'))
    files.sort()
    for i in range(len(files)):
        # files[i] = files[i].split('/')[-1]
        files[i] =  os.path.basename(files[i])
    num_files = 0 
    for i in tqdm(range(len(files))):
        try:
            imgName = findCorrespondingImgae(imgPath,files[i])
            
            img = cv2.imread(imgPath + imgName)
            height,width,_=img.shape
            num_files += 1
        except :
            print(imgName)
            continue
        preds = []
        if not os.path.exists(predicted + files[i]):
            preds = np.empty((0, 6), float)

        else:
            preds = Read(predicted + files[i], 6)

        gt_boxes = []
        if (not os.path.exists(groundTruth + files[i])):
            gt_boxes = np.empty((0, 5), float)

        else:
            gt_boxes = ReadGt(groundTruth + files[i], 5,width,height)
            # gt_boxes = Read(groundTruth + files[i],5)
            
        conf_mat.process_batch(preds, gt_boxes, img, classes, savingPath, imgName)
    num_files = len(os.listdir(imgPath))    
    display(conf_mat, classes, num_files, savingPath)

    return conf_mat,classes
                
def box_iou_calc(boxes1, boxes2):
    
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def IntersectionOverArea(boxes1,boxes2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None])


def DrawBBox(img, label, color = (0, 0, 255)):
    
        x1 = int(label[0])
        y1 = int(label[1])
        x2 = int(label[2])
        y2 = int(label[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def PutText(img, text, x, y, font, color = (0, 0, 255)):
    
    FONT_SCALE = 2e-3  
    THICKNESS_SCALE = 2e-3
    
    height, width, _ = img.shape

    font_scale = min(width, height) * FONT_SCALE
    thickness = int(max(min(width, height) * THICKNESS_SCALE, 1))
    
    
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    
class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
        self.fpImages = 0
        self.totalImages = 0


    def process_batch(self, detections, labels: np.ndarray, img, classes, path, img_name):
        
        font = cv2.FONT_HERSHEY_PLAIN
        fpImage = False
        self.totalImages+= 1

        try:
            gt_classes = labels[:, 0].astype(np.int16)
        except Exception as e:
            # print(e)
            '''
            saving the images that are false positives.
            '''
            for i, detection in enumerate(detections):
                detection_classes = detections[:, 5].astype(np.int16)            
                detection_class = detection_classes[i]

                # if int(detection_class) == 1 and self.gunModel is not None:
                #     if test_single_image(self.gunModel, img.copy(),detections[i]) == False:
                #         continue

                    
                self.matrix[detection_class, self.num_classes] += 1

                

                
                x1 = int(detections[i][0])
                y1 = int(detections[i][1])
                    
                temp_img = img.copy()
                DrawBBox(temp_img, detections[i][:4], (128,0,128))
                
                detect_class = classes[detection_class]
                conf = detections[i][4]
                
                PutText(temp_img, detect_class, x1, (y1 + 30), font, (0, 255, 0))
                PutText(temp_img, str(conf), x1, (y1 + 15), font, (0, 255, 0))
                
#                 cv2.putText(temp_img, detect_class, (x1, y1 + 30), font, 2, (0,255,0), 2)
#                 cv2.putText(temp_img, str(conf), (x1, y1 + 15), font, 2, (0,255,0), 2)
                
                fold = 'falsePositive/' + detect_class

                if not os.path.exists(path + fold):
                    os.makedirs(path + fold)
                cv2.imwrite(path + fold + '/' +f"{i}" +img_name, temp_img)

                fpImage = True

            if fpImage == True:
                self.fpImages += 1    

            return
        
        try:
            
            '''
            Class wise confidence is implemented here checking the class 
            and its corresponding confidence thresholds.
            
            '''
            flag = np.zeros(len(detections), int)
            
            for i in range(len(detections)):
                detected_class = detections[i][5]
                if detections[i][4] > self.CONF_THRESHOLD[int(detected_class)]:
                    flag[i] = 1
                    
            detections = detections[flag[:] == 1]
            
        except IndexError or TypeError:
            '''
            if detections are empty put all the labels in missed and, end of process.
            '''
            for i, label in enumerate(labels):
                
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
                
            return

        
        '''
        Generate all the ious between the ground truth and predictions. Consider the 
        one having iou greater than a threshold.
        '''
        
        detection_classes = detections[:, 5].astype(np.int16)
        all_ious = IntersectionOverArea(labels[:, 1:], detections[:, :4])
        
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        
        '''
        all_matches have the iou index and the iou value between the ground truth and predictions greater than a threshold.
        '''
        
        if all_matches.shape[0] > 0:  # if there is a prediction having a iou greater 
                                      # than a threshold then sort them according to iou
                                      # and consider the corresponding greatest iou of that index.
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            '''
            for every ground truth label check if there is any matches and then check for the 
            category which it belongs
            
            '''
            gt_class = gt_classes[i]
                    
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
                '''
                saving the images that are missclassified
                '''
                if detection_class != gt_class: #wrong classification
                    temp_img = img.copy()
                    DrawBBox(temp_img, detections[int(all_matches[all_matches[:, 0] == i, 1][0])][:4], (0, 255, 0))
                    
                    x1 = int(detections[int(all_matches[all_matches[:, 0] == i, 1][0])][0])
                    y1 = int(detections[int(all_matches[all_matches[:, 0] == i, 1][0])][1])
                    
                    detect_class = classes[detection_class]
                    conf = detections[int(all_matches[all_matches[:, 0] == i, 1][0])][4]
                    
                    PutText(temp_img, detect_class, x1, (y1 + 30), font, (0, 255, 0))
                    PutText(temp_img, str(conf), x1, (y1 + 15), font, (0, 255, 0))
                    
#                     cv2.putText(temp_img, detect_class, (x1, y1 + 30), font, 0.5, (0,255,0), 2)
#                     cv2.putText(temp_img, str(conf), (x1, y1 + 15), font, 0.5, (0,255,0), 2)
                    
                    
                    DrawBBox(temp_img, labels[i][1:5],(0, 255, 0))
                    x1 = int(labels[i][1])
                    y2 = int(labels[i][4])
                    
                    
                    Gt_class = classes[gt_class]
                    
                    PutText(temp_img, Gt_class, x1, (y2 - 15), font, (0, 0, 255))
                    
#                     cv2.putText(temp_img, Gt_class, (x1, y2 - 15), font, 0.5, (0,0,255), 2)
                    
                    
                    fold = classes[detection_class] + classes[gt_class]
                    if not os.path.exists(path + fold):
                        os.makedirs(path + fold)
                    cv2.imwrite(path + fold + '/' + img_name, temp_img)
                    
            #saving the images that are missed.
            else:
                self.matrix[self.num_classes, gt_class] += 1
                
                temp_img = img.copy()
                DrawBBox(temp_img, labels[i][1:5],(0, 255, 0))
                x1 = int(labels[i][1])
                y1 = int(labels[i][2])
                

                Gt_class = classes[gt_class]
                
                PutText(temp_img, Gt_class, x1, (y1 - 15), font, (0, 0, 255))
                
#                 cv2.putText(temp_img, Gt_class, (x1, y1 - 15), font, 2, (0,0,255), 2)


                fold = 'missed/' + Gt_class
                if not os.path.exists(path + fold):
                    os.makedirs(path + fold)
                cv2.imwrite(path + fold + '/' +str(i)+ img_name, temp_img)

        for i, detection in enumerate(detections):
            '''
            saving the images that are false positives.
            '''
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                # if int(detected_class) == 1 and self.gunModel is not None:
                #     if test_single_image(self.gunModel, img.copy(),detections[i]) == False:
                #         continue

                self.matrix[detection_class, self.num_classes] += 1
                
                x1 = int(detections[i][0])
                y1 = int(detections[i][1])
                    
                temp_img = img.copy()
                DrawBBox(temp_img, detections[i][:4], (128,0,128))
                
                detect_class = classes[detection_class]
                conf = detections[i][4]
                
                PutText(temp_img, detect_class, x1, (y1 + 30), font, (0, 255, 0))
                PutText(temp_img, str(conf), x1, (y1 + 15), font, (0, 255, 0))
                
#                 cv2.putText(temp_img, detect_class, (x1, y1 + 30), font, 2, (0,255,0), 2)
#                 cv2.putText(temp_img, str(conf), (x1, y1 + 15), font, 2, (0,255,0), 2)
                
                fold = 'falsePositive/' + detect_class
                if not os.path.exists(path + fold):
                    os.makedirs(path + fold)
                cv2.imwrite(path + fold + '/' +f"{i}" + img_name, temp_img)

                fpImage=True

        if fpImage == True:
            self.fpImages += 1

    def return_matrix(self):
        return self.matrix
    
    def return_FP_images(self):
        return self.fpImages

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))
