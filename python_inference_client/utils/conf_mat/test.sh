#!/bin/bash

# === Hardcoded paths ===
XML_DIR="/nfs/xray_object_detection/ambuj/Leela_14may_analysis/version_2.7_sim_xml/sim_xmls_pred/xml_fast/"
IMG_DIR="/nfs/xray_object_detection/ambuj/Leela_14may_analysis/14MayImages/"
GT_DIR="/nfs/xray_object_detection/ambuj/Leela_14may_analysis/14MayLabels4Class/"
CONF="./../../conf/classwiseconf.txt"
NAMES="./../../conf/classes.txt"

# === Derive label directory from XML path ===
PRED_LABEL_DIR="${XML_DIR}/labels/"
BADPRED_DIR="${PRED_LABEL_DIR}/BadPred/"

# === Script paths ===
CONVERT_SCRIPT="./vehatXml2yolo.py"
MATRIX_SCRIPT="./Matrix.py"

# === Create output directory ===
echo "Creating output directory at: $BADPRED_DIR"
#mkdir -p "$BADPRED_DIR"

# === Step 1: Convert XML to YOLO annotation format ===
echo
echo "------------------------------------------------------------"
echo "Step 1: Converting XML annotations to YOLO format..."
echo "XML_DIR:        $XML_DIR"
echo "IMG_DIR:        $IMG_DIR"
echo "PRED_LABEL_DIR: $PRED_LABEL_DIR"
echo "SimulatorMode:  1"
echo "YOLO Format:    0 (non-normalized format)"
echo "------------------------------------------------------------"
python "$CONVERT_SCRIPT" "$XML_DIR" "$IMG_DIR" "$PRED_LABEL_DIR" 1 0

echo
echo "✅ Annotations saved at: $PRED_LABEL_DIR"

# === Step 2: Run confusion matrix generation ===
echo
echo "------------------------------------------------------------"
echo "Step 2: Generating confusion matrix..."
echo "IMG_DIR       : $IMG_DIR"
echo "GT_DIR        : $GT_DIR"
echo "Pred_Label_Dir: $PRED_LABEL_DIR"
echo "Output Dir    : $BADPRED_DIR"
echo "------------------------------------------------------------"
python "$MATRIX_SCRIPT" "$IMG_DIR" "$NAMES" "$GT_DIR" "$PRED_LABEL_DIR" "$BADPRED_DIR" "$CONF"

echo
echo "✅ Confusion matrix completed. Output saved to: $BADPRED_DIR"

