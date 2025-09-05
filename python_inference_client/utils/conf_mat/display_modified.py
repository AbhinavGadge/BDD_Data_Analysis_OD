# Author: Abhinav Narayan Gadge
# Email: abhigadge12@gmail.com

import numpy as np
import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm



def display(conf_mat, classes, num_files, savingPath):
    print(conf_mat)
    count = 0
    missed = 0
    miss_class = 0
    false_positive = 0
    miss = "Missed"
    mat = conf_mat.return_matrix()
    FP_images = conf_mat.return_FP_images()
    temp = np.zeros((len(mat), len(mat[0])), int)

    # Calculate the maximum width needed for each column
    max_widths = [max(len(str(row[i])) for row in mat) for i in range(len(mat[0]))]
    max_class_width = max(len(cls) for cls in classes + [miss])

    for i in range(len(mat)):
        for j in range(len(mat[i])):
            temp[i][j] = int(mat[i][j])
            if i == j:
                count += temp[i][j]
            if i == len(mat) - 1:
                missed += temp[i][j]
            if j == len(mat) - 1:
                false_positive += temp[i][j]
            if j != len(mat) - 1 and i != len(mat) - 1 and i != j:
                miss_class += temp[i][j]
    det_acc_total = (count + miss_class) * 100 / (count + missed + miss_class)
    class_acc_total = (count * 100) / (count + missed + miss_class)
    fp_total = (false_positive * 100) / (count + missed + miss_class)
    tn_total = (missed*100)/ (count + missed + miss_class)

    _width = (max_class_width*2 + 3 + sum(max_widths))

    with open(os.path.join(savingPath, 'New_summary.txt'), 'w') as f:
        f.write('-' * _width + '\n')
        f.write(f"FP_images: {FP_images} | num_imgs: {num_files} | num_instance: {count + missed + miss_class} \n")
        f.write('-' * _width + '\n')
        f.write(f"{'Class'.ljust(max_class_width)} | {' '.join(str(i).rjust(max_widths[j]) for j, i in enumerate(range(len(mat[0]) - 1)))} {'FP'.rjust(max_widths[-1])}\n")
        f.write('-' * _width + '\n')
        for i, cls in enumerate(classes):
            f.write(f"{cls.ljust(max_class_width)} | {' '.join(str(temp[i][j]).rjust(max_widths[j]) for j in range(len(mat[i])))}\n")
        f.write(f"{miss.ljust(max_class_width)} | {' '.join(str(temp[-1][j]).rjust(max_widths[j]) for j in range(len(mat[-1])))}\n")
        f.write('-' * _width + '\n')
        f.write(f"Detection Accuracy      : {det_acc_total:.2f}%\n")
        f.write(f"Classification Accuracy : {class_acc_total:.2f}%\n")
        f.write(f"False Positive Instances: {fp_total:.2f}%\n")
        f.write(f"Missed/FN Instances     : {tn_total:.2f}%\n")
        f.write(f"False Positives Images  : {FP_images*100/num_files:.2f}%\n")

        f.write('-' * _width + '\n \n')

        # Append class-wise accuracy and false positives in a table format
        f.write('Class-wise Results:\n')
        f.write(f"{'Class'.ljust(max_class_width)} | {'Det Acc'.rjust(5)} | {'Class Acc'.rjust(10)} | {'FP'.rjust(8)} | {'Missed'.rjust(10)} | {'Precision'.rjust(10)} |{'Recall'.rjust(10)} |{'AP'.rjust(10)} |\n")
        f.write('------------------------------------------------------------------------------------------------' + '\n')

        AP_per_class = np.zeros(len(classes))
        fp = temp[:,-1]
        fn = temp[-1,:]

        for i, cls in enumerate(classes):
            tp = temp[i,i]
            actual_gt = sum(temp[:,i])
            total_fp = sum(temp[i,:]) - tp
            total_fn = actual_gt - tp
            miss_class = actual_gt - tp - fn[i]

            class_det_acc = (tp+miss_class)*100/actual_gt if actual_gt != 0 else np.nan
            class_acc = tp*100/actual_gt if actual_gt != 0 else np.nan

            class_fp = total_fp*100/actual_gt if actual_gt != 0 else np.nan

            class_missed = total_fn*100/actual_gt if actual_gt != 0 else np.nan

            recall = tp*100/(tp + total_fn)
            precision = tp*100/(tp + total_fp)

            AP_per_class[i] = recall*precision/100

            f.write(f"{cls.ljust(max_class_width)} | {f'{class_det_acc:.2f}%' if not np.isnan(class_det_acc) else 'nan%'.rjust(10)} | {f'{class_acc:.2f}%'.rjust(10)} | {f'{class_fp:.2f}%'.rjust(10)} | {f'{class_missed:.2f}%'.rjust(10)} | {f'{precision:.2f}%'.rjust(10)} | {f'{recall:.2f}%'.rjust(10)} | {f'{AP_per_class[i]:.2f}%'.rjust(10)} |\n")
        mAP = np.mean([ap for ap in AP_per_class if not np.isnan(ap)])
        f.write('-------------------------------------------------------------------------------------------------' + '\n')
        f.write(f"mAP      : {np.mean([ap for ap in AP_per_class if not np.isnan(ap)]):.2f}%\n")
        f.write('-------------------------------------------------------------------------------------------------' + '\n')
    print('-' * _width)
    print(f"FP_images: {FP_images} and num_imgs: {num_files}")
    print('-' * _width)
    print(f"{'Class'.ljust(max_class_width)} | {' '.join(str(i).rjust(max_widths[j]) for j, i in enumerate(range(len(mat[0]) - 1)))} {'FP'.rjust(max_widths[-1])}")
    print('-' * _width)
    for i, cls in enumerate(classes):
        print(f"{cls.ljust(max_class_width)} | {' '.join(str(temp[i][j]).rjust(max_widths[j]) for j in range(len(mat[i])))}")
    print(f"{miss.ljust(max_class_width)} | {' '.join(str(temp[-1][j]).rjust(max_widths[j]) for j in range(len(mat[-1])))}")
    print('-' * _width)
    print(f"Detection Accuracy      : {det_acc_total:.2f}%")
    print(f"Classification Accuracy : {class_acc_total:.2f}%")
    print(f"False Positive Instances: {fp_total:.2f}%")
    print(f"Missed/FN Instances     : {tn_total:.2f}%")
    print(f"False Positives Images  : {FP_images*100/num_files:.2f}%\n")
    print('-' * _width)

if __name__ == "__main__":
    main()
