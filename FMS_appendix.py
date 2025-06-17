import sys, os
import time
import shutil
import psutil
import warnings
import glob
import gc
import re
import collections
import random
from io import StringIO
import decimal
import six
import packaging
import packaging.version
import packaging.specifiers
import packaging.requirements

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from KerasModels.inception_resnet_v2 import InceptionResNetV2
# Assuming that classification_module and other custom modules are already available in the environment
from classification_module import run_classification
from classification_evaluator import run_module
from ERTrain import trainModel, ERtrainModel
from multi_mfusiond import run_FusionDiagnosis

warnings.filterwarnings("ignore")

# Get the script's directory path and the current working directory
pathset = os.path.dirname(os.path.realpath(sys.argv[0]))
execution_path = os.getcwd()

# Check if a process is running (helper function)
def proc_exist(process_name):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == process_name:
            return pid

# Delete all files in the directory
def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Count the number of all files in the directory
def get_filenum(rootdir):
    file_paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            file_paths.append(os.path.join(root, file))  
    return len(file_paths)

# Get all files in the directory (including files in subdirectories)
def get_all(cwd):
    result = []
    get_dir = os.listdir(cwd)
    for i in get_dir:
        sub_dir = os.path.join(cwd, i)
        if os.path.isdir(sub_dir):
            result.extend(get_all(sub_dir))
        else:
            result.append(i)
    return result

# Complete training, classification, and evaluation tasks for a single data directory
def process_data_folder(data_folder):
    print("Starting to process directory:", data_folder)
    
    # Set the data path, using os.path.join for cross-platform compatibility
    Data_Path = os.path.join("model_appendix", data_folder)
    FeatureName=data_folder
    modelType = InceptionResNetV2

    # Perform ER training first
    ERtrainModel(Data_Path)
    
    # Construct absolute paths for all related directories
    DATASET_DIR = os.path.join(execution_path, Data_Path)
    MODELs_DIR = os.path.join(DATASET_DIR, "models")
    BestModel_DIR = os.path.join(DATASET_DIR, "BestModel")
    Json_DIR = os.path.join(DATASET_DIR, "json")
    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    DATASET_TestModel_DIR = os.path.join(DATASET_DIR, "TestModel")
    DATASET_TestModel_timeDIR = os.path.join(DATASET_TestModel_DIR, dir_time)
    
    # Get the training set directory and its statistics
    rootdir_train = os.path.join(DATASET_DIR, "train")
    if not os.path.exists(rootdir_train):
        print("Training directory does not exist:", rootdir_train)
        return
    filelist = os.listdir(rootdir_train)
    objects_num = len(filelist)
    
    filenum = get_filenum(rootdir_train)
    # Set the number of experiments based on the file count; in this example, fixed as 2 (adjust as needed)
    num_exp = 2
    
    # Perform model training; parameters can be adjusted as needed
    trainModel(modelType, num_objects=objects_num, num_experiments=num_exp, 
              enhance_data=True, continue_from_model=None, transfer_from_model=True, 
              batch_size=16, show_network_summary=False)
    
    # Define the actual label pairs for each data directory
    label_mapping = {
        "AODC": ["Thickening", "No thickening"],
        "CC": ["Presence of fecal stones", "No fecal stones"],
        "CAE": ["Presence of fluid accumulation", "No fluid accumulation"],
        "AWC": ["Abnormal appendix wall", "Continuous appendix wall"],
        "GAC": ["Presence of gas accumulation", "No gas accumulation"],
        "FAC": ["Presence of fluid accumulation", "No fluid accumulation"],
        "PFCC": ["Presence of fluid accumulation", "No fluid accumulation"],
        "MSC": ["Presence of swelling", "No swelling"],
        "BFC": ["Presence of blood flow", "No blood flow"]
    }
    # Get the labels for the current directory; if not defined, default to numeric labels
    labels = label_mapping.get(data_folder, ["class1", "class2"])
    
    # Define a mapping of probability column names for each directory; modify as needed
    prob_column_mapping = {
        "AODC": "Thickening_prob",
        "CC": "Presence of fecal stones_prob",
        "CAE": "Presence of fluid accumulation_prob",
        "AWC": "Abnormal appendix wall_prob",
        "GAC": "Presence of gas accumulation_prob",
        "FAC": "Presence of fluid accumulation_prob",
        "PFCC": "Presence of fluid accumulation_prob",
        "MSC": "Presence of swelling_prob",
        "BFC": "Presence of blood flow_prob"
    }
    
    # Retrieve the probability column name corresponding to the current directory, defaulting to "Thickening_prob"
    prob_column = prob_column_mapping.get(data_folder, "Thickening_prob")
    
    # Run the classification program. Note that the classes parameter is passed with the corresponding label pairs rather than the number 2

    run_classification(
        model_directory=os.path.join(Data_Path, "models"),
        model_json_path=os.path.join(Data_Path, "json", "model_class.json"),
        excel_file_path=os.path.join("model_appendix", "Testdata_184.xlsx"),
        image_source_directory=os.path.join("model_appendix", "TestImage_crop"),
        output_directory=os.path.join(Data_Path, "Testresults"),
        input_shape=(224, 224, 3),
        FeatureName=FeatureName,
        classes=labels    # Pass the label pairs, for example ["Thickening", "No thickening"]
    )
   
    # Run the evaluation module
    testresults_dir = os.path.join(Data_Path, "Testresults")
    models_dir = os.path.join(Data_Path, "models")
    best_model_dir = os.path.join(Data_Path, "BestModel")

    # Note: The true_label_column parameter now uses the positive label corresponding to the current data directory (the first label in the list)
    best_file_name, model_file_name, merged_results = run_module(
        testresults_dir=testresults_dir,
        models_dir=models_dir,
        best_model_dir=best_model_dir,
        true_label_column=FeatureName,  # For example, "Thickening" or "Presence of fecal stones"
        prob_column=prob_column,
        threshold=0.5
    )
    
    print("Completed processing directory {}.\n".format(data_folder))

if __name__ == "__main__":
    # Define all data directory names
    data_directories = ['AODC', 'AWC', 'BFC', 'CAE', 'CC', 'FAC', 'GAC', 'MSC', 'PFCC']
    
    # Loop through and process each directory sequentially
    for folder in data_directories:
        process_data_folder(folder)
    
   
    run_FusionDiagnosis(
        excel_file_path='Test_data.xlsx',
        source_directory='TestImage',
        output_directory='Testresults'
     )
