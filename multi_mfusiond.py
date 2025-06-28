"""
multi_model_classifier.py

This module loads nine deep learning models, processes images,
predicts their classes, and saves the results and a summary report to an Excel file.
Other modules can import and use the `run_FusionDiagnosis` function.

Usage Example:
    from multi_model_classifier import run_FusionDiagnosis

    run_FusionDiagnosis(
        excel_file_path='Test_data.xlsx',
        source_directory='TestImage_all',
        output_directory='Testresults'
    )
"""

import os
import json
import numpy as np
import pandas as pd
from keras_preprocessing.image import load_img, img_to_array
from KerasModels.inception_resnet_v2 import InceptionResNetV2


def load_class_labels(model_json_path):
    """
    Load the class labels from a JSON file.

    Parameters:
        model_json_path (str): Path to the directory containing model_class.json.

    Returns:
        dict: Mapping of class indices to class names.
    """
    json_file = os.path.join(model_json_path, 'model_class.json')
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image.

    Parameters:
        img_path (str): Path of the image.
        target_size (tuple): The target size of the image.

    Returns:
        numpy.ndarray: Processed image array ready for prediction.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def predict_class(model, class_labels, img_array):
    """
    Predict the image class using a model.

    Parameters:
        model: The keras model used for prediction.
        class_labels (dict): Mapping of class indices to class names.
        img_array (numpy.ndarray): Preprocessed image array.

    Returns:
        dict: A dictionary with class names as keys and their predicted probabilities as values.
    """
    predictions = model.predict(img_array)
    predicted_probabilities = predictions[0]
    class_probabilities = {
        class_labels.get(str(index), '未知类别'): float(predicted_probabilities[index])
        for index in range(len(predicted_probabilities))
    }
    return class_probabilities


def _process_model(model_directory, model_json, feature_name, df, results, source_directory):
    """
    Process images for a given model (i.e. load the model, perform predictions, and update results).

    Parameters:
        model_directory (str): Directory which contains the .h5 model file(s).
        model_json (str): Directory which contains the JSON file with class labels.
        feature_name (str): The key (feature name) in the results dictionary that will be updated.
        df (DataFrame): The DataFrame read from the Excel file containing the App_No.
        results (list): List of current result dictionaries per App_No.
        source_directory (str): Root directory containing image subdirectories.
    """
    # Get list of .h5 model files in the directory
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.h5')]
    class_labels = load_class_labels(model_json)

    for model_filename in model_files:
        model_path = os.path.join(model_directory, model_filename)
        # Load the model architecture and weights.
        model = InceptionResNetV2(
            weights=None,
            input_shape=(224, 224, 3),
            classes=len(class_labels)
        )
        model.load_weights(model_path)

        # Iterate over all App_No entries in results
        for result in results:
            App_No = result['App_No']
            applicable_directory = os.path.join(source_directory, str(App_No))
            if os.path.exists(applicable_directory):
                max_probability = 0
                max_class = None

                # Process every .jpg image in the directory
                for img_name in os.listdir(applicable_directory):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(applicable_directory, img_name)
                        img_array = preprocess_image(img_path)
                        class_probabilities = predict_class(model, class_labels, img_array)

                        # Update max probability and corresponding class
                        for class_name, probability in class_probabilities.items():
                            if probability > max_probability:
                                max_probability = probability
                                max_class = class_name

                # Update the feature in the result dictionary
                result[feature_name] = max_class if max_class else '无'


def run_FusionDiagnosis(
    excel_file_path,
    source_directory,
    output_directory,
    models_config=None
):
    """
    Runs the classification process over all nine features, creates a summary report
    based on the predicted feature values for each App_No, and saves the result to an Excel file.

    Parameters:
        excel_file_path (str): Path to the Excel file containing the App_No column.
        source_directory (str): Directory where the image subdirectories (named as App_No) are located.
        output_directory (str): Directory where the resulting Excel file will be saved.
        models_config (list of dict): List of model configuration dictionaries with keys:
            - 'feature': The feature name (e.g., 'AODC')
            - 'model_directory': Path to the models (.h5 files)
            - 'model_json': Path to the JSON file directory for class labels
            If None, a default configuration for all nine models is used.
    """
    # Default configuration for nine models if not provided
    if models_config is None:
        models_config = [
            {
                "feature": "AODC",
                "model_directory": "model_appendix/AODC/BestModel",
                "model_json": "model_appendix/AODC/json"
            },
            {
                "feature": "CC",
                "model_directory": "model_appendix/CC/BestModel",
                "model_json": "model_appendix/CC/json"
            },
            {
                "feature": "CAE",
                "model_directory": "model_appendix/CAE/BestModel",
                "model_json": "model_appendix/CAE/json"
            },
            {
                "feature": "AWC",
                "model_directory": "model_appendix/AWC/BestModel",
                "model_json": "model_appendix/AWC/json"
            },
            {
                "feature": "GAC",
                "model_directory": "model_appendix/GAC/BestModel",
                "model_json": "model_appendix/GAC/json"
            },
            {
                "feature": "FAC",
                "model_directory": "model_appendix/FAC/BestModel",
                "model_json": "model_appendix/FAC/json"
            },
            {
                "feature": "PFCC",
                "model_directory": "model_appendix/PFCC/BestModel",
                "model_json": "model_appendix/PFCC/json"
            },
            {
                "feature": "MSC",
                "model_directory": "model_appendix/MSC/BestModel",
                "model_json": "model_appendix/MSC/json"
            },
            {
                "feature": "BFC",
                "model_directory": "model_appendix/BFC/BestModel",
                "model_json": "model_appendix/BFC/json"
            }
        ]

    # Read the Excel file
    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {excel_file_path} not found.")

    # Get the list of application numbers ("App_No")
    if 'App_No' not in df.columns:
        raise KeyError("Excel file must contain an 'App_No' column.")
    App_No_list = df['App_No'].tolist()

    # Initialize the results list.
    # Initially, every feature is set to None.
    results = []
    for App_No in App_No_list:
        results.append({
            'App_No': App_No,
            **{config['feature']: None for config in models_config}
        })

    # Process each model from the configuration.
    for config in models_config:
        _process_model(
            model_directory=config["model_directory"],
            model_json=config["model_json"],
            feature_name=config["feature"],
            df=df,
            results=results,
            source_directory=source_directory
        )

    # After obtaining each feature, generate a summary report.
    # The report now excludes the prefix like "Application 1901010955370 results ->".
    for result in results:
        report_parts = [f"{feature}: {result[feature]}" for feature in result if feature != 'App_No']
        # Create a simple descriptive report for the given application number without prefix
        report = ", ".join(report_parts)
        result["Report"] = report

    # Create the output directory if it does not exist.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the results DataFrame to an Excel file.
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_directory, "ClassificationResultsofNineFeatures.xlsx")
    results_df.to_excel(output_path, index=False)
    print(f"The classification results and report have been saved successfully to {output_path}.")


if __name__ == "__main__":
    # Example usage when running this module as a script.
    # Adjust the following paths as needed.
    EXCEL_FILE_PATH = "Test_data.xlsx"
    SOURCE_DIRECTORY = "TestImage"
    OUTPUT_DIRECTORY = "Testresults"

    run_FusionDiagnosis(
        excel_file_path=EXCEL_FILE_PATH,
        source_directory=SOURCE_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY
    )
