import os
import json
import numpy as np
import pandas as pd
from keras_preprocessing.image import load_img, img_to_array
from KerasModels.inception_resnet_v2 import InceptionResNetV2

def load_class_labels(json_path):
    """
    Load class labels from a JSON configuration file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model prediction.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_class(model, img_array, class_labels):
    """
    Predict the class probabilities for a single image array.
    Returns a dictionary mapping class names to probabilities.
    """
    predictions = model.predict(img_array)
    predicted_probabilities = predictions[0]

    class_probabilities = {
        class_labels.get(str(index), 'Unknown'): float(predicted_probabilities[index])
        for index in range(len(predicted_probabilities))
    }
    return class_probabilities

def run_classification(
    model_directory,
    model_json_path,
    excel_file_path,
    image_source_directory,
    output_directory,
    input_shape=(224, 224, 3),
    FeatureName="",
    classes=2
):
   
    # Check the type of classes parameter: if it is a list/tuple, then consider that concrete class labels are provided
    if isinstance(classes, (list, tuple)):
        num_classes = len(classes)
        override_labels = True
        provided_labels = classes
        # Construct class_labels dictionary using provided labels: index as string -> label
        class_labels = {str(i): label for i, label in enumerate(provided_labels)}
    else:
        num_classes = classes
        override_labels = False
        # Load class labels (expected format in JSON file is index: label)
        class_labels = load_class_labels(model_json_path)
    
    # Define mapping between model filename prefix and class names, if classes is numeric (non-override mode)
    label_mapping = {
        "AODC": ("Thickening", "No thickening"),
        "CC": ("Presence of fecal stones", "No fecal stones"),
        "CAE": ("Presence of fluid accumulation", "No fluid accumulation"),
        "AWC": ("Abnormal appendix wall", "Continuous appendix wall"),
        "GAC": ("Presence of gas accumulation", "No gas accumulation"),
        "FAC": ("Presence of fluid accumulation", "No fluid accumulation"),
        "PFCC": ("Presence of fluid accumulation", "No fluid accumulation"),
        "MSC": ("Presence of swelling", "No swelling"),
        "BFC": ("Presence of blood flow", "No blood flow")
    }
    
    # Load Excel file, extract application number list
    try:
        df = pd.read_excel(excel_file_path)
        application_numbers = df['App_No'].tolist()
    except FileNotFoundError:
        print(f"Error: Excel file '{excel_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Check if there is a diagnosis column
    has_diagnosis = FeatureName in df.columns

    # Get specified model weight files
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.h5')]
    if not model_files:
        print(f"No model weight files (.h5) found in directory '{model_directory}'.")
        return

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Process each model weight file
    for model_filename in model_files:
        model_path = os.path.join(model_directory, model_filename)
        print(f"Loading model weights from '{model_path}'...")

        # If labels are not overridden, determine the corresponding class name pair based on model filename
        if not override_labels:
            model_key = None
            for key in label_mapping:
                if model_filename.startswith(key):
                    model_key = key
                    break
            if model_key is not None:
                positive_label, negative_label = label_mapping[model_key]
            else:
                positive_label, negative_label = "Thickening", "No thickening"
        else:
            # In override mode, directly use the provided label pair
            positive_label, negative_label = provided_labels[0], provided_labels[1]
        
        # Initialize model and load weights
        model = InceptionResNetV2(weights=None, input_shape=input_shape, classes=num_classes)
        model.load_weights(model_path)

        results = []

        # Iterate over all application numbers from the Excel file
        for app_num in application_numbers:
            # The image path is image_source_directory/app_num/
            app_dir = os.path.join(image_source_directory, str(app_num))
            if not os.path.exists(app_dir):
                print(f"Directory for application number {app_num} does not exist, skipping.")
                continue

            # If diagnosis information exists, attempt to extract it
            diagnosis = None
            if has_diagnosis:
                row_info = df[df['App_No'] == app_num]
                if row_info.empty:
                    print(f"No diagnosis info for application number {app_num} in 'Diagnosis' column, skipping.")
                    continue
                diagnosis = row_info[FeatureName].values[0]

            max_probability = 0
            max_class = None
            max_image_name = None
            max_image_probabilities = {}

            # Iterate over all JPG images in the application number folder
            for img_name in os.listdir(app_dir):
                if not img_name.lower().endswith('.jpg'):
                    continue

                img_path = os.path.join(app_dir, img_name)
                img_array = preprocess_image(img_path)
                class_probabilities = predict_class(model, img_array, class_labels)

                # Find the class with the highest probability in the current image
                for class_name, prob in class_probabilities.items():
                    if prob > max_probability:
                        max_probability = prob
                        max_class = class_name
                        max_image_name = img_name
                        max_image_probabilities = class_probabilities

            # Skip if no valid prediction result is found
            if max_class is None:
                print(f"No valid image predictions for application number {app_num}.")
                continue

            # Extract probability for current model corresponding class name (defaulting to 0.0 if not found)
            positive_prob = max_image_probabilities.get(positive_label, 0.0)
            negative_prob = max_image_probabilities.get(negative_label, 0.0)

            results.append({
                'App_No': app_num,
                 FeatureName: diagnosis,
                'main_class': max_class,
                'main_class_prob': f"{max_probability:.2f}",
                'main_class_image': max_image_name,
                f"{positive_label}_prob": f"{positive_prob:.2f}",
                f"{negative_label}_prob": f"{negative_prob:.2f}",
            })

        # Save all application results for current model to an Excel file,
        # e.g., "Results_modelXYZ.h5.xlsx"
        if results:
            results_df = pd.DataFrame(results)
            output_file = os.path.join(output_directory, f"Results_{model_filename}.xlsx")
            results_df.to_excel(output_file, index=False)
            print(f"Results saved to '{output_file}'.")
        else:
            print(f"No results for model {model_filename}.")

    print("Classification completed successfully.")
