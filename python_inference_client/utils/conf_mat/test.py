from Matrix import ConfMatrix

# Hardcoded paths
imgPath = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data\val\images"
classesPath = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data\classes.txt"
ground_truth_path = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data\val\labels"
predicted = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data\results\output"
savingPath = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\bdd_data\results\badPred"
classwiseConfidence = r"C:\Users\abhinav.gadge\Documents\KN_Docs_Induction\python_inference_client\python_inference_client\conf\classwiseconf.txt"

# Create and run ConfMatrix
cm = ConfMatrix(imgPath, savingPath, classesPath, ground_truth_path, predicted, classwiseConfidence)
cm.save_results()
