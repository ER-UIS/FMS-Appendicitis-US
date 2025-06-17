"""
classification_evaluator.py

This module provides evaluation functionalities for the test result Excel files,
including computing a confusion matrix, accuracy, precision, recall, F1 score, AUC value, and a combined metric.
The program saves an Excel result for each file and selects the model with the highest combined metric among all files,
then copies that model file from the specified model directory to the target directory.

Additionally, this module includes functions related to image classification,
which can preprocess input images, load models, predict probabilities, and save classification results to an Excel file.
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from keras_preprocessing.image import load_img, img_to_array
from KerasModels.inception_resnet_v2 import InceptionResNetV2

# Fixed mappings: Use a fixed label mapping for specific classification tasks
LABEL_MAPPINGS = {
    "AODC": {"Thickening": 1, "No thickening": 0},
    "CC": {"Presence of fecal stones": 1, "No fecal stones": 0},
    "CAE": {"Presence of fluid accumulation": 1, "No fluid accumulation": 0},
    "AWC": {"Abnormal appendix wall": 1, "Continuous appendix wall": 0},
    "GAC": {"Presence of gas accumulation": 1, "No gas accumulation": 0},
    "FAC": {"Presence of fluid accumulation": 1, "No fluid accumulation": 0},
    "PFCC": {"Presence of fluid accumulation": 1, "No fluid accumulation": 0},
    "MSC": {"Presence of swelling": 1, "No swelling": 0},
    "BFC": {"Presence of blood flow": 1, "No blood flow": 0}
}

# Probability column mappings: The name of the positive probability column for each task
PROB_COLUMN_MAPPINGS = {
    "AODC": "thickened_prob",
    "CC": "fecal_stones_prob",
    "CAE": "fluid_accumulation_prob",
    "AWC": "appendix_wall_prob",  # Modify the name as needed
    "GAC": "gas_accumulation_prob",
    "FAC": "fluid_accumulation_prob",
    "PFCC": "fluid_accumulation_prob",
    "MSC": "swelling_prob",
    "BFC": "blood_flow_prob"
}


def find_column(columns, target):
    """
    In the given collection of columns, find the target column using case-insensitive matching.
    If an exact match is found, return that column name; otherwise return None.
    """
    for col in columns:
        if col.lower() == target.lower():
            return col
    return None


def process_excel_file(file_path, true_label_column="AODC", prob_column=None, threshold=0.5):
    """
    Process a single Excel file to compute the confusion matrix, accuracy, precision, recall, F1 score, AUC value,
    and combined metric.

    If the specified true_label_column is not found in the Excel file, the following procedures are attempted in order:
      1. Case-insensitive matching;
      2. If still no match, use the column named "main_class" (a common primary classification column) if it exists;
      3. Finally, attempt to find an intersection with the fixed mappings defined in LABEL_MAPPINGS.

    Additionally, if the prob_column parameter is not provided, it will be automatically selected based on the
    true_label_column via the probability column mappings.

    Parameters:
        file_path (str): The full path of the Excel file.
        true_label_column (str): The column name in the Excel file that contains the true labels. For dynamic mapping,
                                 this value is used as the positive class label.
        prob_column (str): The column name in the Excel file that contains the positive class probability. If None,
                           it will be associated automatically using the mapping.
        threshold (float): The probability threshold (values greater than this are considered positive).

    Returns:
        results (dict): A dictionary containing each evaluation metric.
        cm (ndarray): The confusion matrix.
    """
    data = pd.read_excel(file_path)

    # First, search for the true label column using case-insensitive matching
    found_true_label = find_column(data.columns, true_label_column)
    if found_true_label:
        true_label_column = found_true_label
    else:
        if "main_class" in data.columns:
            print(f"Warning: The specified column '{true_label_column}' was not found. Using 'main_class' as the true label column.")
            true_label_column = "main_class"
        else:
            candidate_columns = set(LABEL_MAPPINGS.keys()) & set(data.columns)
            if candidate_columns:
                chosen_column = candidate_columns.pop()
                print(f"Warning: The specified column '{true_label_column}' was not found. Using '{chosen_column}' as the true label column, "
                      f"with the mapping: {LABEL_MAPPINGS[chosen_column]}")
                true_label_column = chosen_column
            else:
                raise KeyError(
                    f"The specified true label column '{true_label_column}' was not found in the Excel file. Available columns: {list(data.columns)}"
                )
    
    # If the positive probability column is not provided, select it based on the true_label_column mapping
    if prob_column is None:
        prob_column = PROB_COLUMN_MAPPINGS.get(true_label_column, None)
        if prob_column is None:
            raise ValueError(f"Unknown classification task '{true_label_column}'. Unable to determine the positive probability column automatically. Please specify prob_column manually.")
    
    # Similarly, find the positive probability column using case-insensitive matching
    found_prob = find_column(data.columns, prob_column)
    if found_prob:
        prob_column = found_prob
    else:
        raise KeyError(f"The specified positive probability column '{prob_column}' was not found in the Excel file. Available columns: {list(data.columns)}")

    # Extract the true labels and positive class probabilities
    true_labels = data[true_label_column]
    predicted_probs = data[prob_column]

    # Convert true labels to numeric using the fixed mapping (if available)
    if true_label_column in LABEL_MAPPINGS:
        mapping = LABEL_MAPPINGS[true_label_column]
        true_labels_numeric = true_labels.apply(lambda x: mapping.get(x, 0))
    else:
        # Dynamic mapping: Map each unique label to a unique number. The specific meaning depends on the Excel file content.
        unique_labels = true_labels.unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        true_labels_numeric = true_labels.map(label_mapping)

    # Generate predicted labels based on the positive probability and threshold
    predicted_labels_numeric = (predicted_probs > threshold).astype(int)

    # Set the label order: If using a fixed mapping, assume the label order as [0, 1],
    # otherwise use the order of all unique labels from dynamic mapping.
    if true_label_column in LABEL_MAPPINGS:
        labels_order = [0, 1]
    else:
        labels_order = list(true_labels_numeric.dropna().unique())

    cm = confusion_matrix(true_labels_numeric, predicted_labels_numeric, labels=labels_order)
    overall_accuracy = cm.diagonal().sum() / cm.sum()

    # Compute TP, FN, FP (only when the positive class label 1 exists and the confusion matrix has more than one row)
    if 1 in labels_order and len(cm) > 1:
        positive_index = labels_order.index(1)
        TP = cm[positive_index, positive_index]
        FN = cm[positive_index].sum() - TP
        FP = cm[:, positive_index].sum() - TP
    else:
        TP = FN = FP = 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Compute the AUC value (it may fail if there is a data issue)
    try:
        auc_value = roc_auc_score(true_labels_numeric, predicted_probs)
    except Exception as e:
        auc_value = 0
        print(f"Warning: Unable to compute the AUC value for file {file_path}. Error: {e}")

    # Combined metric: Weighted sum of accuracy, F1 score, and AUC
    combined_metric = overall_accuracy * 0.3 + f1 * 0.4 + auc_value * 0.3

    results = {
        'File Name': os.path.basename(file_path),
        'Accuracy_Raw': f'{cm.diagonal().sum()}/{cm.sum()}',
        'Accuracy': overall_accuracy,
        'Precision_Raw': f'{TP}/{TP + FP}' if (TP + FP) > 0 else '0/0',
        'Precision': precision,
        'Recall_Raw': f'{TP}/{TP + FN}' if (TP + FN) > 0 else '0/0',
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_value,
        'Combined Metric': combined_metric
    }

    return results, cm


def process_all_files(testresults_dir,
                      output_summary_file="SummaryStatisticsforAllClassificationResults.xlsx",
                      true_label_column="",
                      prob_column=None,
                      threshold=0.5):
    """
    Process all Excel files in the specified directory. For each file, compute evaluation metrics,
    and save each file's result as an Excel file with the suffix "_ClassificationResultsStatistics.xlsx";
    at the same time, aggregate all the results into a summary Excel file.

    Parameters:
        testresults_dir (str): The directory containing the test result Excel files.
        output_summary_file (str): The name of the Excel file to save the aggregated results.
        true_label_column (str): The column name that contains the true labels (for dynamic mapping, this value is used as the positive class label).
        prob_column (str): The column name that contains the positive class probabilities. If None, it will be selected automatically based on true_label_column.
        threshold (float): The positive probability threshold.

    Returns:
        merged_results (DataFrame): A DataFrame aggregating all evaluation results.
    """
    excel_files = [f for f in os.listdir(testresults_dir) if f.endswith('.xlsx')]
    all_results = []

    for file_name in excel_files:
        file_path = os.path.join(testresults_dir, file_name)
        results, cm = process_excel_file(file_path,
                                         true_label_column=true_label_column,
                                         prob_column=prob_column,
                                         threshold=threshold)
        results_df = pd.DataFrame([results])
        output_file_path = os.path.join(testresults_dir, f'{file_name}_ClassificationResultsStatistics.xlsx')
        results_df.to_excel(output_file_path, index=False)
        all_results.append(results_df)

    merged_results = pd.concat(all_results, ignore_index=True)
    merged_summary_path = os.path.join(testresults_dir, output_summary_file)
    merged_results.to_excel(merged_summary_path, index=False)
    return merged_results


def get_best_model_file(merged_results):
    """
    Select the record with the highest combined metric from the aggregated results,
    and extract the model file name based on the file name (removing the prefix).

    For example: If the file name is "AnyPrefix_modelA.xlsx", it extracts "modelA".
    If the file name does not contain an underscore, use the file name without its extension.

    Parameters:
        merged_results (DataFrame): The aggregated evaluation results DataFrame.

    Returns:
        best_file_name (str): The original file name with the highest combined metric.
        model_file_name (str): The extracted model file name (without the extension).
    """
    max_metric = merged_results['Combined Metric'].max()
    best_result = merged_results[merged_results['Combined Metric'] == max_metric]
    best_file_name = best_result['File Name'].values[0]
    base = os.path.splitext(best_file_name)[0]
    if '_' in base:
        model_file_name = base.split('_', 1)[1]
    else:
        model_file_name = base
    return best_file_name, model_file_name


def copy_best_model(model_file_name, models_dir, best_model_dir):
    """
    Copy the model file from the specified source directory to the target directory.

    Parameters:
        model_file_name (str): The name of the model file.
        models_dir (str): The directory where model files are stored.
        best_model_dir (str): The directory where the best model file should be copied.

    Returns:
        source_file_path (str): The source file path.
        dest_file_path (str): The destination file path.
    """
    os.makedirs(best_model_dir, exist_ok=True)
    source_file_path = os.path.join(models_dir, model_file_name)
    dest_file_path = os.path.join(best_model_dir, model_file_name)
    shutil.copy(source_file_path, dest_file_path)
    return source_file_path, dest_file_path


def run_module(testresults_dir, models_dir, best_model_dir,
               true_label_column="",
               prob_column=None,
               threshold=0.5):
    """
    Execute the complete evaluation process:
      1. Process the test result Excel files, compute various metrics, and save the summary file;
      2. Select the file with the highest combined metric from the aggregated results, and obtain the best model file name;
      3. Copy the best model file from the model directory to the specified target directory.

    Parameters:
        testresults_dir (str): The directory containing the test result Excel files.
        models_dir (str): The directory where model files are stored.
        best_model_dir (str): The directory to which the best model file will be copied.
        true_label_column (str): The column in the Excel file containing the true labels (for dynamic mapping, this value is used as the positive class label).
        prob_column (str): The column in the Excel file containing the positive probabilities. If None, it will be automatically chosen based on true_label_column.
        threshold (float): The positive probability threshold.

    Returns:
        best_file_name (str): The original file name with the highest combined metric.
        model_file_name (str): The final name of the copied model file.
        merged_results (DataFrame): The aggregated evaluation results DataFrame.
    """
    merged_results = process_all_files(testresults_dir,
                                       true_label_column=true_label_column,
                                       prob_column=prob_column,
                                       threshold=threshold)
    best_file_name, model_file_name = get_best_model_file(merged_results)
    print("File with the highest combined metric:", best_file_name)
    print("Extracted model file name:", model_file_name)

    try:
        src, dst = copy_best_model(model_file_name, models_dir, best_model_dir)
        print(f"Model file successfully copied from '{src}' to '{dst}'")
    except Exception as e:
        print(f"Failed to copy the model file: {e}")

    return best_file_name, model_file_name, merged_results


# The following functions can be added for image classification purposes (e.g., image preprocessing, loading models, predicting probabilities, etc.)
# Example:
def preprocess_image(image_path, target_size=(299, 299)):
    """
    Load and preprocess an image.

    Parameters:
        image_path (str): The path of the image.
        target_size (tuple): The target image size.

    Returns:
        image_array (ndarray): The preprocessed image array.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image_array):
    """
    Predict the probability for an input image.

    Parameters:
        model: The pre-loaded model object.
        image_array (ndarray): The preprocessed image array.

    Returns:
        probs (ndarray): An array of predicted probabilities.
    """
    probs = model.predict(image_array)
    return probs


def save_classification_results(image_paths, predictions, output_excel):
    """
    Save the image classification results to an Excel file.

    Parameters:
        image_paths (list): A list of image file paths.
        predictions (list/ndarray): A list of prediction results.
        output_excel (str): The path of the Excel file where results will be saved.
    """
    df = pd.DataFrame({
        "Image Path": image_paths,
        "Prediction": predictions.squeeze()  # Adjust the squeeze logic as needed based on the actual results
    })
    df.to_excel(output_excel, index=False)


